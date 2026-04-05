import copy

import os
import cv2
import joblib
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from phalp.utils.utils import progress_bar
from phalp.utils.utils_tracks import create_fast_tracklets, get_tracks
from phalp.utils.utils import pose_camera_vector_to_smpl
from phalp.utils.lart_utils import to_ava_labels
from .kalman_filter import AdaptiveKalmanFilter

class AdaptiveKalmanFilter:
    """专为人体运动设计的自适应卡尔曼滤波器"""
    def __init__(self, initial_state, process_noise_scale=0.5, measurement_noise=0.1):
        # 状态向量: [x, y, scale, aspect_ratio, vx, vy, v_scale, v_aspect]
        self.dim_x = 8
        self.dim_z = 4  # 测量维度: [x, y, scale, aspect_ratio]
        
        # 状态转移矩阵 (匀速模型)
        self.F = np.eye(self.dim_x)
        for i in range(4):
            self.F[i, i+4] = 1.0
        
        # 测量矩阵
        self.H = np.zeros((self.dim_z, self.dim_x))
        np.fill_diagonal(self.H[:4, :4], 1.0)
        
        # 初始状态
        self.x = np.zeros((self.dim_x, 1))
        self.x[:4, 0] = initial_state
        
        # 初始协方差
        self.P = np.eye(self.dim_x) * 10.0
        
        # 过程噪声协方差
        self.Q = np.eye(self.dim_x) * 0.01
        
        # 测量噪声协方差
        self.R = np.eye(self.dim_z) * measurement_noise
        
        # 自适应参数
        self.process_noise_scale = process_noise_scale
        self.last_measurement = None
    
    def init(self, measurement):
        """使用测量值初始化滤波器"""
        self.x[:4, 0] = measurement
        self.last_measurement = measurement
    
    def predict(self):
        """预测下一状态"""
        # 状态预测
        self.x = self.F @ self.x
        
        # 协方差预测
        self.P = self.F @ self.P @ self.F.T + self.Q
        
        return self.x[:4, 0].flatten()
    
    def update(self, measurement):
        """使用测量值更新状态"""
        # 计算残差
        z = np.array(measurement).reshape(-1, 1)
        y = z - self.H @ self.x
        
        # 残差协方差
        S = self.H @ self.P @ self.H.T + self.R
        
        # 卡尔曼增益
        K = self.P @ self.H.T @ np.linalg.inv(S)
        
        # 状态更新
        self.x = self.x + K @ y
        
        # 协方差更新
        self.P = (np.eye(self.dim_x) - K @ self.H) @ self.P
        
        # 自适应调整过程噪声
        self._adapt_process_noise(measurement)
        
        # 存储最后测量值
        self.last_measurement = measurement
        
        return self.x[:4, 0].flatten()
    
    def _adapt_process_noise(self, measurement):
        """根据运动变化自适应调整过程噪声"""
        if self.last_measurement is None:
            return
            
        # 计算位移变化
        displacement = np.linalg.norm(measurement[:2] - self.last_measurement[:2])
        
        # 计算尺寸变化
        size_change = abs(measurement[2] - self.last_measurement[2])
        
        # 综合变化量
        total_change = displacement + size_change * 10.0  # 尺寸变化权重
        
        # 调整过程噪声
        scale_factor = 1.0 + self.process_noise_scale * total_change
        self.Q *= scale_factor
        
        # 限制噪声范围
        np.clip(self.Q, 0.001, 1.0, out=self.Q)

class Postprocessor(nn.Module):
    def __init__(self, cfg, phalp_tracker):
        super(Postprocessor, self).__init__()
        self.cfg = cfg
        self.device = 'cuda'
        self.phalp_tracker = phalp_tracker

    def post_process(self, final_visuals_dic, save_fast_tracks=False, video_pkl_name=""):
        # 如果启用了卡尔曼滤波，则先进行卡尔曼滤波处理
        if self.cfg.post_process.apply_kalman:
            # 我们按轨迹进行卡尔曼滤波
            track_dict = self.organize_data_by_track(final_visuals_dic)
            # 对每条轨迹应用卡尔曼滤波
            for track_id, track_data in track_dict.items():
                print()
                smoothed_track = self.apply_kalman_to_track(track_data)
                
                # 更新原始数据字典
                for data_point in smoothed_track:
                    frame_key = data_point['frame']
                    idx = data_point['index']
                    final_visuals_dic[frame_key]['bbox'][idx] = data_point['bbox']
        
        if(self.cfg.post_process.apply_smoothing):
            final_visuals_dic_ = copy.deepcopy(final_visuals_dic)
            track_dict = get_tracks(final_visuals_dic_)

            for tid_ in track_dict.keys():
                fast_track_ = create_fast_tracklets(track_dict[tid_])
            
                with torch.no_grad():
                    smoothed_fast_track_ = self.phalp_tracker.pose_predictor.smooth_tracks(
                        fast_track_, 
                        moving_window=True, 
                        step=32, 
                        window=32
                    )

                if(save_fast_tracks):
                    frame_length = len(smoothed_fast_track_['frame_name'])
                    dict_ava_feat = {}
                    dict_ava_psudo_labels = {}
                    for idx, appe_idx in enumerate(smoothed_fast_track_['apperance_index']):
                        dict_ava_feat[appe_idx[0,0]] = smoothed_fast_track_['apperance_emb'][idx]
                        dict_ava_psudo_labels[appe_idx[0,0]] = smoothed_fast_track_['action_emb'][idx]
                    smoothed_fast_track_['action_label_gt'] = np.zeros((frame_length, 1, 80)).astype(int)
                    smoothed_fast_track_['action_label_psudo'] = dict_ava_psudo_labels
                    smoothed_fast_track_['apperance_dict'] = dict_ava_feat
                    smoothed_fast_track_['pose_shape'] = smoothed_fast_track_['pose_shape'].cpu().numpy()

                    # save the fast tracks in a pkl file
                    save_pkl_path = os.path.join(self.cfg.video.output_dir, "results_temporal_fast/", video_pkl_name + "_" + str(tid_) +  "_" + str(frame_length) + ".pkl")
                    joblib.dump(smoothed_fast_track_, save_pkl_path)

                for i_ in range(smoothed_fast_track_['pose_shape'].shape[0]):
                    f_key = smoothed_fast_track_['frame_name'][i_]
                    tids_ = np.array(final_visuals_dic_[f_key]['tid'])
                    idx_  = np.where(tids_==tid_)[0]
                    
                    if(len(idx_)>0):

                        pose_shape_ = smoothed_fast_track_['pose_shape'][i_]
                        smpl_camera = pose_camera_vector_to_smpl(pose_shape_[0])
                        smpl_ = smpl_camera[0]
                        camera = smpl_camera[1]
                        camera_ = smoothed_fast_track_['cam_smoothed'][i_][0].cpu().numpy()

                        dict_ = {}
                        for k, v in smpl_.items():
                            dict_[k] = v

                        if(final_visuals_dic[f_key]['tracked_time'][idx_[0]]>0):
                            final_visuals_dic[f_key]['camera'][idx_[0]] = np.array([camera_[0], camera_[1], 200*camera_[2]])
                            final_visuals_dic[f_key]['smpl'][idx_[0]] = copy.deepcopy(dict_)
                            final_visuals_dic[f_key]['tracked_time'][idx_[0]] = -1
                        
                        # attach ava labels
                        ava_ = smoothed_fast_track_['ava_action'][i_]
                        ava_ = ava_.cpu()
                        ava_labels, _ = to_ava_labels(ava_, self.cfg)
                        final_visuals_dic[f_key].setdefault('label', {})[tid_] = ava_labels
                        final_visuals_dic[f_key].setdefault('ava_action', {})[tid_] = ava_
                        
        
        return final_visuals_dic
    
    def organize_data_by_track(self, visuals_dic):
            """将数据按轨迹ID组织"""
            track_dict = {}
            # 遍历每一帧
            for frame_key, frame_data in visuals_dic.items():
                if 'tid' not in frame_data:
                    continue
                tids = frame_data['tid']
                # 遍历当前帧的每个轨迹
                for idx, tid in enumerate(tids):
                    if tid not in track_dict:
                        track_dict[tid] = []
                    # 收集该轨迹在当前帧的数据
                    data_point = {
                        'frame_key': frame_key,
                        'camera': frame_data['camera'][idx],
                        'smpl': frame_data['smpl'][idx],
                        'bbox': frame_data['bbox'][idx],
                        # 根据实际需要收集其他字段
                    }
                    track_dict[tid].append(data_point)
            return track_dict

    def apply_kalman_to_track(self, track_data):
        print(f"应用卡尔曼滤波到轨迹 {track_data[0]['track_id']}, 数据点: {len(track_data)}")

        """对单条轨迹应用卡尔曼滤波"""
        # 按时间顺序排序
        track_data.sort(key=lambda x: x['frame_key'])
        
        # 初始化卡尔曼滤波器
        # 初始状态：第一个数据点的位置信息（这里以bbox的中心点为例）
        first_point = track_data[0]
        # 假设bbox是[x1, y1, x2, y2]
        bbox = first_point['bbox']
        cx = (bbox[0] + bbox[2]) / 2
        cy = (bbox[1] + bbox[3]) / 2
        # 初始状态：我们使用中心点位置和bbox的宽高
        w = bbox[2] - bbox[0]
        h = bbox[3] - bbox[1]
        initial_state = [cx, cy, w, h]
        
        kf = AdaptiveKalmanFilter(
            initial_state=initial_state,
            process_noise_scale=self.cfg.post_process.kalman_params.process_noise_scale,
            measurement_noise=self.cfg.post_process.kalman_params.measurement_noise
        )
        
        smoothed_track = []
        for data_point in track_data:
            bbox = data_point['bbox']
            cx = (bbox[0] + bbox[2]) / 2
            cy = (bbox[1] + bbox[3]) / 2
            w = bbox[2] - bbox[0]
            h = bbox[3] - bbox[1]
            measurement = [cx, cy, w, h]
            
            # 预测
            kf.predict()
            # 更新
            smoothed_state = kf.update(measurement)
            
            # 使用平滑后的状态更新bbox
            smoothed_bbox = [
                smoothed_state[0] - smoothed_state[2]/2,
                smoothed_state[1] - smoothed_state[3]/2,
                smoothed_state[0] + smoothed_state[2]/2,
                smoothed_state[1] + smoothed_state[3]/2
            ]
            
            # 构建平滑后的数据点
            smoothed_point = copy.deepcopy(data_point)
            smoothed_point['bbox'] = smoothed_bbox
            # 注意：这里只平滑了bbox，如果需要平滑其他数据（如camera, smpl等），可以类似处理
            smoothed_track.append(smoothed_point)
        print(f"卡尔曼滤波完成, 最大位移变化")
        return smoothed_track

    def update_visuals_with_kalman(self, visuals_dic, track_id, smoothed_track):
        """用卡尔曼滤波后的数据更新原始数据字典"""
        for data_point in smoothed_track:
            frame_key = data_point['frame_key']
            if frame_key not in visuals_dic:
                continue
            frame_data = visuals_dic[frame_key]
            # 找到该轨迹在该帧中的索引
            if 'tid' not in frame_data:
                continue
            try:
                idx = frame_data['tid'].index(track_id)
            except ValueError:
                continue
            # 更新bbox
            frame_data['bbox'][idx] = data_point['bbox']
            # 如果需要更新其他字段，可以在这里更新

    def run_lart(self, phalp_pkl_path):
        
        # lart_output = {}
        video_pkl_name = phalp_pkl_path.split("/")[-1].split(".")[0]
        final_visuals_dic = joblib.load(phalp_pkl_path)

        os.makedirs(self.cfg.video.output_dir + "/results_temporal/", exist_ok=True)
        os.makedirs(self.cfg.video.output_dir + "/results_temporal_fast/", exist_ok=True)
        os.makedirs(self.cfg.video.output_dir + "/results_temporal_videos/", exist_ok=True)
        save_pkl_path = os.path.join(self.cfg.video.output_dir, "results_temporal/", video_pkl_name + ".pkl")
        save_video_path = os.path.join(self.cfg.video.output_dir, "results_temporal_videos/", video_pkl_name + "_.mp4")

        if(os.path.exists(save_pkl_path) and not(self.cfg.overwrite)):
            return 0
        
        # apply smoothing/action recognition etc.
        final_visuals_dic  = self.post_process(final_visuals_dic, save_fast_tracks=self.cfg.post_process.save_fast_tracks, video_pkl_name=video_pkl_name)
        
        # render the video
        if(self.cfg.render.enable):
            self.offline_render(final_visuals_dic, save_pkl_path, save_video_path)
        
        joblib.dump(final_visuals_dic, save_pkl_path)

    def run_renderer(self, phalp_pkl_path):
        
        video_pkl_name = phalp_pkl_path.split("/")[-1].split(".")[0]
        final_visuals_dic = joblib.load(phalp_pkl_path)

        os.makedirs(self.cfg.video.output_dir + "/videos/", exist_ok=True)
        os.makedirs(self.cfg.video.output_dir + "/videos_tmp/", exist_ok=True)
        save_pkl_path = os.path.join(self.cfg.video.output_dir, "videos_tmp/", video_pkl_name + ".pkl")
        save_video_path = os.path.join(self.cfg.video.output_dir, "videos/", video_pkl_name + ".mp4")

        if(os.path.exists(save_pkl_path) and not(self.cfg.overwrite)):
            return 0
        
        # render the video
        self.offline_render(final_visuals_dic, save_pkl_path, save_video_path)

    def offline_render(self, final_visuals_dic, save_pkl_path, save_video_path):
        
        video_pkl_name = save_pkl_path.split("/")[-1].split(".")[0]
        list_of_frames = list(final_visuals_dic.keys())
        
        for t_, frame_path in progress_bar(enumerate(list_of_frames), description="Rendering : " + video_pkl_name, total=len(list_of_frames), disable=False):
            
            image = self.phalp_tracker.io_manager.read_frame(frame_path)

            ################### Front view #########################
            self.cfg.render.up_scale = int(self.cfg.render.output_resolution / self.cfg.render.res)
            self.phalp_tracker.visualizer.reset_render(self.cfg.render.res*self.cfg.render.up_scale)
            final_visuals_dic[frame_path]['frame'] = image
            panel_render, f_size = self.phalp_tracker.visualizer.render_video(final_visuals_dic[frame_path])      
            del final_visuals_dic[frame_path]['frame']

            # resize the image back to render resolution
            panel_rgb = cv2.resize(image, (f_size[0], f_size[1]), interpolation=cv2.INTER_AREA)

            # save the predicted actions labels
            if('label' in final_visuals_dic[frame_path]):
                labels_to_save = []
                for tid_ in final_visuals_dic[frame_path]['label']:
                    ava_labels = final_visuals_dic[frame_path]['label'][tid_]
                    labels_to_save.append(ava_labels)
                labels_to_save = np.array(labels_to_save)

            panel_1 = np.concatenate((panel_rgb, panel_render), axis=1)
            final_panel = panel_1

            self.phalp_tracker.io_manager.save_video(save_video_path, final_panel, (final_panel.shape[1], final_panel.shape[0]), t=t_)
            t_ += 1

        self.phalp_tracker.io_manager.close_video()


