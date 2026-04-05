"""
Modified code from https://github.com/nwojke/deep_sort
"""

import copy
from collections import deque

import numpy as np
import scipy.signal as signal
from scipy.ndimage.filters import gaussian_filter1d

class TrackState:
    """
    Enumeration type for the single target track state. Newly created tracks are
    classified as `tentative` until enough evidence has been collected. Then,
    the track state is changed to `confirmed`. Tracks that are no longer alive
    are classified as `deleted` to mark them for removal from the set of active
    tracks.

    """

    Tentative = 1
    Confirmed = 2
    Deleted   = 3


class Track:
    """
    Mark this track as missed (no association at the current time step).
    """

    def __init__(self, cfg, track_id, n_init, max_age, detection_data, detection_id=None, dims=None):
        self.cfg               = cfg
        self.track_id          = track_id
        self.hits              = 1
        self.age               = 1
        self.time_since_update = 0
        self.time_init         = detection_data["time"]
        self.state             = TrackState.Tentative      
        # self.mean = None     
        
        self._n_init           = n_init
        self._max_age          = max_age
        
        if(dims is not None):
            self.A_dim = dims[0]
            self.P_dim = dims[1]
            self.L_dim = dims[2]
        
        self.track_data        = {"history": deque(maxlen=self.cfg.phalp.track_history) , "prediction":{}}
        for _ in range(self.cfg.phalp.track_history):
            self.track_data["history"].append(detection_data)
            
        self.track_data['prediction']['appe'] = deque([detection_data['appe']], maxlen=self.cfg.phalp.n_init+1)
        self.track_data['prediction']['loca'] = deque([detection_data['loca']], maxlen=self.cfg.phalp.n_init+1)
        self.track_data['prediction']['pose'] = deque([detection_data['pose']], maxlen=self.cfg.phalp.n_init+1)
        self.track_data['prediction']['uv']   = deque([copy.deepcopy(detection_data['uv'])], maxlen=self.cfg.phalp.n_init+1)

        # if the track is initialized by detection with annotation, then we set the track state to confirmed
        if len(detection_data['annotations'])>0:
            self.state = TrackState.Confirmed      

    def predict(self, phalp_tracker, increase_age=True):
        if(increase_age):
            self.age += 1; self.time_since_update += 1
            
    def add_predicted(self, appe=None, pose=None, loca=None, uv=None):
        appe_predicted = copy.deepcopy(appe.numpy()) if(appe is not None) else copy.deepcopy(self.track_data['history'][-1]['appe'])
        loca_predicted = copy.deepcopy(loca.numpy()) if(loca is not None) else copy.deepcopy(self.track_data['history'][-1]['loca'])
        pose_predicted = copy.deepcopy(pose.numpy()) if(pose is not None) else copy.deepcopy(self.track_data['history'][-1]['pose'])
        
        self.track_data['prediction']['appe'].append(appe_predicted)
        self.track_data['prediction']['loca'].append(loca_predicted)
        self.track_data['prediction']['pose'].append(pose_predicted)

    def _common_update(self, detection, detection_id, shot):
        """公共更新逻辑，用于卡尔曼和非卡尔曼模式"""
        self.track_data["history"].append(copy.deepcopy(detection.detection_data))
        if(shot==1): 
            for tx in range(self.cfg.phalp.track_history):
                self.track_data["history"][-1-tx]['loca'] = copy.deepcopy(detection.detection_data['loca'])

        if("T" in self.cfg.phalp.predict):
            mixing_alpha_                      = self.cfg.phalp.alpha*(detection.detection_data['conf']**2)
            ones_old                           = self.track_data['prediction']['uv'][-1][3:, :, :]==1
            ones_new                           = self.track_data['history'][-1]['uv'][3:, :, :]==1
            ones_old                           = np.repeat(ones_old, 3, 0)
            ones_new                           = np.repeat(ones_new, 3, 0)
            ones_intersect                     = np.logical_and(ones_old, ones_new)
            ones_union                         = np.logical_or(ones_old, ones_new)
            good_old_ones                      = np.logical_and(np.logical_not(ones_intersect), ones_old)
            good_new_ones                      = np.logical_and(np.logical_not(ones_intersect), ones_new)
            new_rgb_map                        = np.zeros((3, 256, 256))
            new_mask_map                       = np.zeros((1, 256, 256))-1
            new_mask_map[ones_union[:1, :, :]] = 1.0
            new_rgb_map[ones_intersect]        = (1-mixing_alpha_)*self.track_data['prediction']['uv'][-1][:3, :, :][ones_intersect] + mixing_alpha_*self.track_data['history'][-1]['uv'][:3, :, :][ones_intersect]
            new_rgb_map[good_old_ones]         = self.track_data['prediction']['uv'][-1][:3, :, :][good_old_ones] 
            new_rgb_map[good_new_ones]         = self.track_data['history'][-1]['uv'][:3, :, :][good_new_ones] 
            self.track_data['prediction']['uv'].append(np.concatenate((new_rgb_map , new_mask_map), 0))
        else:
            self.track_data['prediction']['uv'].append(self.track_data['history'][-1]['uv'])
            
        self.hits += 1
        self.time_since_update = 0
        if self.state == TrackState.Tentative and self.hits >= self._n_init:
            self.state = TrackState.Confirmed

        if len(detection.detection_data['annotations'])>0:
            self.state = TrackState.Confirmed

    def update(self, detection, detection_id, shot):
        """非卡尔曼模式更新"""
        # 非卡尔曼模式可能不需要额外的操作，直接进行公共更新
        self._common_update(detection, detection_id, shot)
     
    def mark_missed(self):
        """Mark this track as missed (no association at the current time step).
        """
        if self.state == TrackState.Tentative:
            self.state = TrackState.Deleted
        elif self.time_since_update > self._max_age:
            self.state = TrackState.Deleted

    def is_tentative(self):
        """Returns True if this track is tentative (unconfirmed).
        """
        return self.state == TrackState.Tentative

    def is_confirmed(self):
        """Returns True if this track is confirmed."""
        return self.state == TrackState.Confirmed

    def is_deleted(self):
        """Returns True if this track is dead and should be deleted."""
        return self.state == TrackState.Deleted

    def smooth_bbox(self, bbox):
        kernel_size = 5
        sigma       = 3
        bbox        = np.array(bbox)
        smoothed    = np.array([signal.medfilt(param, kernel_size) for param in bbox.T]).T
        out         = np.array([gaussian_filter1d(traj, sigma) for traj in smoothed.T]).T
        return list(out)
    
    def init_kalman_filter(self, initial_state):
        """初始化卡尔曼滤波器"""
        from .kalman_filter import AdaptiveKalmanFilter  # 根据实际路径调整
        
        self.kalman_filter = AdaptiveKalmanFilter(
            initial_state=initial_state,
            process_noise_scale=self.cfg.post_process.kalman_params.process_noise_scale,
            measurement_noise=self.cfg.post_process.kalman_params.measurement_noise
        )

    def update_with_kalman(self, detection,detection_id, shot):
        """卡尔曼模式更新"""
        # 确保卡尔曼滤波器已初始化
        if not hasattr(self, 'kalman_filter') or self.kalman_filter is None:
            self.init_kalman_filter(detection.to_xyah())
        
        # 获取测量值
        measurement = detection.to_xyah()
        
        # 更新卡尔曼滤波器
        updated_state = self.kalman_filter.update(measurement)
        
        # 使用卡尔曼滤波结果更新状态向量
        # 注意：这里假设Track类有一个mean属性来存储状态
        if not hasattr(self, 'mean'):
            self.mean = np.zeros_like(updated_state)
        self.mean[:] = updated_state
        
        # 调用公共更新逻辑
        # 注意：这里我们省略了detection_id和shot，因为卡尔曼更新可能不需要这些？
        # 但是公共更新需要，所以我们需要传递这些参数
        # 但是update_with_kalman方法目前只接收detection参数，所以我们需要调整调用方式
        # 因此，我们需要修改update_with_kalman的定义，增加detection_id和shot参数
        self._common_update(detection, detection_id, shot)
