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


class Postprocessor(nn.Module):
    
    def __init__(self, cfg, phalp_tracker):
        super(Postprocessor, self).__init__()
        
        self.cfg = cfg
        self.device = 'cuda'
        self.phalp_tracker = phalp_tracker

    def post_process(self, final_visuals_dic, save_fast_tracks=False, video_pkl_name=""):

        if(self.cfg.post_process.apply_smoothing):
            final_visuals_dic_ = copy.deepcopy(final_visuals_dic)
            track_dict = get_tracks(final_visuals_dic_)

            for tid_ in track_dict.keys():
                fast_track_ = create_fast_tracklets(track_dict[tid_])
            
                with torch.no_grad():
                    smoothed_fast_track_ = self.phalp_tracker.pose_predictor.smooth_tracks(fast_track_, moving_window=True, step=32, window=32)

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
        list_of_frames = sorted(list(final_visuals_dic.keys()))
        
        # 确保输出目录存在
        os.makedirs(os.path.dirname(save_video_path), exist_ok=True)
        
        # 1. 确定视频尺寸（使用第一帧）
        first_frame_path = list_of_frames[0]
        first_image = self.phalp_tracker.io_manager.read_frame(first_frame_path)
        frame_height, frame_width, _ = first_image.shape
        
        # 渲染第一帧以获取尺寸
        self.cfg.render.up_scale = int(self.cfg.render.output_resolution / self.cfg.render.res)
        self.phalp_tracker.visualizer.reset_render(self.cfg.render.res * self.cfg.render.up_scale)
        final_visuals_dic[first_frame_path]['frame'] = first_image
        panel_render, f_size = self.phalp_tracker.visualizer.render_video(final_visuals_dic[first_frame_path])
        del final_visuals_dic[first_frame_path]['frame']
        
        # 计算最终面板尺寸
        panel_width = first_image.shape[1] + panel_render.shape[1]
        panel_height = max(first_image.shape[0], panel_render.shape[0])
        
        # 2. 创建视频写入器（关键修复）
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # 使用兼容性最好的编码
        video_writer = cv2.VideoWriter(
            save_video_path,
            fourcc,
            self.cfg.video.fps,  # 从配置获取帧率
            (panel_width, panel_height)
        )
        
        # 检查写入器是否成功创建
        if not video_writer.isOpened():
            print(f"错误：无法创建视频文件 {save_video_path}")
            print("请尝试：")
            print("1. 更换编码器：尝试 'XVID' 或 'MJPG'")
            print("2. 检查磁盘空间和权限")
            return
        
        t_ = 0
        for frame_path in progress_bar(list_of_frames, description="Rendering : " + video_pkl_name):
            try:
                # 读取原始帧
                image = self.phalp_tracker.io_manager.read_frame(frame_path)
                
                # 渲染当前帧
                final_visuals_dic[frame_path]['frame'] = image
                panel_render, f_size = self.phalp_tracker.visualizer.render_video(final_visuals_dic[frame_path])
                del final_visuals_dic[frame_path]['frame']
                
                # 调整原始图像大小
                panel_rgb = cv2.resize(image, (f_size[0], f_size[1]), interpolation=cv2.INTER_AREA)
                
                # 拼接面板
                panel_1 = np.concatenate((panel_rgb, panel_render), axis=1)
                final_panel = panel_1
                
                # 确保尺寸一致（关键修复）
                if final_panel.shape[0] != panel_height or final_panel.shape[1] != panel_width:
                    final_panel = cv2.resize(final_panel, (panel_width, panel_height))
                
                # 转换色彩空间（关键修复）
                if final_panel.shape[2] == 3:  # RGB
                    final_panel_bgr = cv2.cvtColor(final_panel, cv2.COLOR_RGB2BGR)
                else:  # RGBA
                    final_panel_bgr = cv2.cvtColor(final_panel, cv2.COLOR_RGBA2BGR)
                
                # 写入帧（使用统一的视频写入器）
                video_writer.write(final_panel_bgr)
                t_ += 1
                
            except Exception as e:
                print(f"渲染帧 {frame_path} 时出错: {str(e)}")
                # 保存问题帧用于调试
                debug_path = f"/tmp/frame_{t_}.png"
                cv2.imwrite(debug_path, final_panel if 'final_panel' in locals() else image)
                print(f"已保存问题帧到: {debug_path}")
        
        # 关闭视频写入器（关键修复）
        video_writer.release()
        print(f"视频渲染完成: {save_video_path}")
        
        # 验证视频文件
        if os.path.exists(save_video_path):
            cap = cv2.VideoCapture(save_video_path)
            if cap.isOpened():
                frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                print(f"视频验证成功: {frame_count}帧")
                cap.release()
            else:
                print("警告: 视频文件无法打开")
        else:
            print("错误: 视频文件未创建")
            
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