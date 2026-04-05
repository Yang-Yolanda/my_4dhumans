import numpy as np

class AdaptiveKalmanFilter:
    """自适应卡尔曼滤波器"""
    def __init__(self, initial_state, process_noise_scale=1.0, measurement_noise=0.1):
        """
        初始化自适应卡尔曼滤波器
        
        参数:
        initial_state: 初始状态向量 [x, y, a, h] (a: 宽高比, h: 高度)
        process_noise_scale: 过程噪声缩放因子
        measurement_noise: 测量噪声
        """
        # 状态维度 (x, y, a, h, vx, vy, va, vh)
        self.dim_x = 8
        self.dim_z = 4  # 测量维度 (x, y, a, h)
        
        # 状态转移矩阵 (假设匀速运动)
        self.F = np.eye(self.dim_x)
        for i in range(4):
            self.F[i, i+4] = 1.0  # 位置和速度关系
        
        # 测量矩阵 (只能观测位置)
        self.H = np.zeros((self.dim_z, self.dim_x))
        for i in range(self.dim_z):
            self.H[i, i] = 1.0
        
        # 初始状态
        self.x = np.zeros((self.dim_x, 1))
        self.x[:4, 0] = initial_state
        
        # 初始协方差矩阵
        self.P = np.eye(self.dim_x) * 10.0
        
        # 过程噪声协方差
        self.Q = np.eye(self.dim_x) * 0.01
        
        # 测量噪声协方差
        self.R = np.eye(self.dim_z) * measurement_noise
        
        # 自适应参数
        self.process_noise_scale = process_noise_scale
        self.last_measurement = None
        self.last_prediction = None
    
    def predict(self):
        """预测下一状态"""
        # 预测状态
        self.x = self.F @ self.x
        
        # 预测协方差
        self.P = self.F @ self.P @ self.F.T + self.Q
        
        # 存储预测值（用于自适应）
        self.last_prediction = self.x[:4, 0].copy()
        
        # 返回预测状态 (位置部分)
        return self.x[:4, 0].flatten()
    
    def update(self, measurement):
        """使用测量值更新状态"""
        # 计算测量残差
        z = np.array(measurement).reshape(-1, 1)
        y = z - self.H @ self.x
        
        # 计算残差协方差
        S = self.H @ self.P @ self.H.T + self.R
        
        # 卡尔曼增益
        K = self.P @ self.H.T @ np.linalg.inv(S)
        
        # 更新状态估计
        self.x = self.x + K @ y
        
        # 更新协方差估计
        self.P = (np.eye(self.dim_x) - K @ self.H) @ self.P
        
        # 存储最后测量值
        self.last_measurement = measurement
        
        # 自适应调整过程噪声
        self._adapt_process_noise(measurement)
        
        # 返回更新后的状态 (位置部分)
        return self.x[:4, 0].flatten()
    
    def _adapt_process_noise(self, measurement):
        """根据运动变化自适应调整过程噪声"""
        if self.last_measurement is None or self.last_prediction is None:
            return
            
        # 计算实际位移
        actual_displacement = np.linalg.norm(measurement[:2] - self.last_measurement[:2])
        
        # 计算预测位移
        predicted_displacement = np.linalg.norm(self.last_prediction[:2] - measurement[:2])
        
        # 计算误差
        error = abs(actual_displacement - predicted_displacement)
        
        # 根据误差调整过程噪声
        scale_factor = 1.0 + self.process_noise_scale * error
        self.Q *= scale_factor
        
        # 限制噪声范围
        np.clip(self.Q, 0.01, 1.0, out=self.Q)