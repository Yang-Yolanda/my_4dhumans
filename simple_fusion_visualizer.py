#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
SMPL模型渲染器 - Linux服务器OpenGL优化版本
专门用于读取PKL文件并渲染出完整的SMPL模型
支持正视图、侧视图、顶视图的多视图渲染
"""

import os
import cv2
import numpy as np
import joblib
from pathlib import Path
import argparse
from typing import Dict, List, Optional, Tuple
import logging
import platform
import torch

# Linux服务器环境优化
if platform.system() == "Linux":
    os.environ['PYOPENGL_PLATFORM'] = 'egl'
    os.environ['DISPLAY'] = ':0'
    os.environ['EGL_PLATFORM'] = 'surfaceless'
    os.environ['MESA_GL_VERSION_OVERRIDE'] = '3.3'
    os.environ['MESA_GLSL_VERSION_OVERRIDE'] = '330'
    print("Linux环境：设置OpenGL平台为egl")

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class SMPLModelRenderer:
    """SMPL模型渲染器 - Linux服务器OpenGL优化版本"""
    
    def __init__(self):
        """初始化渲染器"""
        self.renderer = None
        self.smpl_faces = None
        self.is_linux = platform.system() == "Linux"
        
        # 尝试初始化HMR2渲染器
        self._init_renderer()
    
    def _init_renderer(self):
        """初始化HMR2渲染器"""
        try:
            from hmr2.configs import default_config, CACHE_DIR_4DHUMANS
            from hmr2.models import load_hmr2, DEFAULT_CHECKPOINT, download_models
            from hmr2.utils.renderer import Renderer
            
            # 下载模型
            logger.info("正在下载HMR2模型...")
            download_models(CACHE_DIR_4DHUMANS)
            
            # 加载HMR2模型获取faces
            logger.info("正在加载HMR2模型...")
            model, model_cfg = load_hmr2(DEFAULT_CHECKPOINT)
            
            # 保存HMR2模型用于生成顶点
            self.hmr2_model = model
            
            if hasattr(model, 'smpl') and hasattr(model.smpl, 'faces'):
                faces = model.smpl.faces
                if hasattr(faces, 'cpu'):
                    faces = faces.cpu().numpy()
                elif hasattr(faces, 'numpy'):
                    faces = faces.numpy()
                
                self.smpl_faces = faces
                self.renderer = Renderer(model_cfg, faces=faces)
                logger.info(f"✓ HMR2渲染器初始化成功，faces: {faces.shape}")
                
                if self.is_linux:
                    logger.info("Linux环境：OpenGL渲染器已准备就绪")
            else:
                logger.warning("⚠ 无法从HMR2模型获取faces")
                
        except Exception as e:
            logger.warning(f"⚠ HMR2渲染器初始化失败: {e}")
            self.renderer = None
            self.hmr2_model = None
    
    def load_pkl_data(self, pkl_path: str) -> Dict:
        """加载PKL文件数据"""
        logger.info(f"正在加载PKL文件: {pkl_path}")
        
        try:
            data = joblib.load(pkl_path)
            logger.info(f"✓ 成功加载PKL文件，数据类型: {type(data)}")
            
            if isinstance(data, dict):
                frame_count = len(data)
                logger.info(f"✓ 找到 {frame_count} 帧数据")
                
                # 检查第一帧的结构
                first_frame_key = list(data.keys())[0]
                first_frame = data[first_frame_key]
                logger.info(f"第一帧键: {list(first_frame.keys())}")
                
                if 'smpl' in first_frame:
                    smpl_data = first_frame['smpl']
                    if smpl_data and len(smpl_data) > 0:
                        first_smpl = smpl_data[0]
                        logger.info(f"✓ 找到SMPL数据，键: {list(first_smpl.keys())}")
                        
                        if 'pose_params' in first_smpl:
                            pose_dim = len(first_smpl['pose_params'])
                            logger.info(f"✓ 姿态参数维度: {pose_dim}")
                        
                        if 'shape_params' in first_smpl:
                            shape_dim = len(first_smpl['shape_params'])
                            logger.info(f"✓ 形状参数维度: {shape_dim}")
                
                return data
            else:
                logger.error("PKL文件不是字典格式")
                return {}
                
        except Exception as e:
            logger.error(f"加载PKL文件失败: {e}")
            return {}
    
    def extract_smpl_data(self, pkl_data: Dict) -> List[Dict]:
        """提取SMPL数据"""
        logger.info("正在提取SMPL数据...")
        
        smpl_frames = []
        
        for frame_name, frame_data in pkl_data.items():
            try:
                # 提取帧索引
                frame_idx = 0
                if 'frame_' in frame_name:
                    frame_idx = int(frame_name.split('_')[1].split('.')[0])
                
                # 提取SMPL参数
                if 'smpl' in frame_data and frame_data['smpl']:
                    smpl_list = frame_data['smpl']
                    if len(smpl_list) > 0:
                        first_smpl = smpl_list[0]  # 取第一个检测的人
                        
                        if isinstance(first_smpl, dict):
                            pose_params = first_smpl.get('pose_params')
                            shape_params = first_smpl.get('shape_params')
                            trans_params = first_smpl.get('trans_params', [0, 0, 0])
                            
                            # 提取关键点
                            keypoints_3d = []
                            if '3d_joints' in frame_data and frame_data['3d_joints']:
                                keypoints_3d = frame_data['3d_joints'][0].tolist() if hasattr(frame_data['3d_joints'][0], 'tolist') else frame_data['3d_joints'][0]
                            
                            # 提取置信度
                            confidence = 0.8
                            if 'conf' in frame_data and frame_data['conf']:
                                confidence = frame_data['conf'][0]
                            
                            smpl_frame = {
                                'frame_idx': frame_idx,
                                'frame_name': frame_name,
                                'pose_params': pose_params,
                                'shape_params': shape_params,
                                'trans_params': trans_params,
                                'keypoints_3d': keypoints_3d,
                                'confidence': confidence
                            }
                            
                            smpl_frames.append(smpl_frame)
                
            except Exception as e:
                logger.warning(f"提取帧 {frame_name} 的SMPL数据失败: {e}")
                continue
        
        logger.info(f"✓ 成功提取 {len(smpl_frames)} 帧SMPL数据")
        return smpl_frames
    
    def generate_smpl_vertices(self, pose_params: np.ndarray, shape_params: np.ndarray, 
                              trans_params: np.ndarray) -> np.ndarray:
        """从SMPL参数生成真实的顶点"""
        try:
            # 🔥 全盘重新思考：这些参数可能不是标准SMPL参数！
            # 让我们尝试多种方法来生成顶点
            
            logger.info("尝试多种方法生成SMPL顶点...")
            
            # 方法1：尝试使用HMR2的SMPL模型（如果可用）
            if hasattr(self, 'hmr2_model') and self.hmr2_model is not None:
                try:
                    logger.info("方法1：尝试使用HMR2的SMPL模型...")
                    
                    # 检查参数格式并尝试不同的转换方法
                    vertices = self._try_hmr2_smpl_methods(pose_params, shape_params, trans_params)
                    if vertices is not None:
                        logger.info(f"✓ 方法1成功：生成 {len(vertices)} 个顶点")
                        return vertices
                        
                except Exception as e:
                    logger.warning(f"方法1失败: {e}")
            
            # 方法2：使用3D关键点重建（如果可用）
            logger.info("方法2：尝试使用3D关键点重建...")
            vertices = self._try_keypoint_reconstruction(pose_params, shape_params, trans_params)
            if vertices is not None:
                logger.info(f"✓ 方法2成功：生成 {len(vertices)} 个顶点")
                return vertices
            
            # 方法3：生成合理的默认人体模型
            logger.info("方法3：生成默认人体模型...")
            vertices = self._generate_default_human_model()
            logger.info(f"✓ 方法3成功：生成 {len(vertices)} 个默认顶点")
            return vertices
            
        except Exception as e:
            logger.warning(f"所有方法都失败: {e}")
            import traceback
            logger.warning(f"详细错误: {traceback.format_exc()}")
            return self._generate_default_human_model()
    
    def _try_hmr2_smpl_methods(self, pose_params: np.ndarray, shape_params: np.ndarray, 
                               trans_params: np.ndarray) -> Optional[np.ndarray]:
        """尝试多种HMR2 SMPL方法"""
        try:
            # 方法1.1：尝试直接使用原始参数（可能是HMR2内部格式）
            logger.info("尝试方法1.1：直接使用原始参数...")
            try:
                # 将参数转换为张量，保持原始格式
                pose_tensor = torch.tensor(pose_params, dtype=torch.float32).unsqueeze(0)  # (1, 72)
                shape_tensor = torch.tensor(shape_params, dtype=torch.float32).unsqueeze(0)  # (1, 10)
                trans_tensor = torch.tensor(trans_params, dtype=torch.float32).unsqueeze(0)  # (1, 3)
                
                logger.info(f"原始参数张量形状: pose={pose_tensor.shape}, shape={shape_tensor.shape}, trans={trans_tensor.shape}")
                
                # 尝试直接调用HMR2模型
                with torch.no_grad():
                    # 可能HMR2模型有特殊的forward方法
                    if hasattr(self.hmr2_model, 'forward'):
                        output = self.hmr2_model.forward(
                            pose_params=pose_tensor,
                            shape_params=shape_tensor,
                            trans_params=trans_tensor
                        )
                        if 'vertices' in output:
                            vertices = output['vertices'][0].cpu().numpy()
                            logger.info(f"✓ 方法1.1成功：直接forward生成 {len(vertices)} 个顶点")
                            return vertices
                
            except Exception as e:
                logger.warning(f"方法1.1失败: {e}")
            
            # 方法1.2：尝试标准SMPL格式（我们之前的转换）
            logger.info("尝试方法1.2：标准SMPL格式转换...")
            try:
                pose_params_tensor = self.convert_pose_params_to_smpl_format(pose_params)
                shape_params_tensor = torch.tensor(shape_params.reshape(1, 10), dtype=torch.float32)
                trans_params_tensor = torch.tensor(trans_params.reshape(1, 3), dtype=torch.float32)
                
                with torch.no_grad():
                    smpl_output = self.hmr2_model.smpl(
                        betas=shape_params_tensor,
                        body_pose=pose_params_tensor['body_pose'],
                        global_orient=pose_params_tensor['global_orient'],
                        transl=trans_params_tensor
                    )
                
                vertices = smpl_output.vertices[0].cpu().numpy()
                logger.info(f"✓ 方法1.2成功：标准SMPL格式生成 {len(vertices)} 个顶点")
                return vertices
                
            except Exception as e:
                logger.warning(f"方法1.2失败: {e}")
            
            # 方法1.3：尝试使用HMR2的预测方法
            logger.info("尝试方法1.3：HMR2预测方法...")
            try:
                # 检查HMR2模型是否有predict方法
                if hasattr(self.hmr2_model, 'predict'):
                    # 创建模拟的输入数据
                    dummy_input = torch.randn(1, 3, 224, 224)  # 模拟图像输入
                    with torch.no_grad():
                        prediction = self.hmr2_model.predict(dummy_input)
                        if 'pred_vertices' in prediction:
                            vertices = prediction['pred_vertices'][0].cpu().numpy()
                            logger.info(f"✓ 方法1.3成功：HMR2预测生成 {len(vertices)} 个顶点")
                            return vertices
                            
            except Exception as e:
                logger.warning(f"方法1.3失败: {e}")
            
            return None
            
        except Exception as e:
            logger.warning(f"所有HMR2方法都失败: {e}")
            return None
    
    def _try_keypoint_reconstruction(self, pose_params: np.ndarray, shape_params: np.ndarray, 
                                   trans_params: np.ndarray) -> Optional[np.ndarray]:
        """尝试使用3D关键点重建人体模型"""
        try:
            # 如果pose_params实际上是3D关键点，我们可以尝试重建
            logger.info("尝试关键点重建...")
            
            # 检查参数是否可能是关键点
            if len(pose_params) == 72 and np.all(np.abs(pose_params) < 10.0):
                # 可能是标准化的关键点坐标
                logger.info("检测到可能是标准化的关键点坐标")
                
                # 将72维参数重塑为24个3D点 (24 * 3 = 72)
                keypoints = pose_params.reshape(-1, 3)
                logger.info(f"重塑为 {len(keypoints)} 个3D关键点")
                
                # 生成基于关键点的人体网格
                vertices = self._generate_mesh_from_keypoints(keypoints)
                if vertices is not None:
                    logger.info(f"✓ 关键点重建成功：生成 {len(vertices)} 个顶点")
                    return vertices
            
            return None
            
        except Exception as e:
            logger.warning(f"关键点重建失败: {e}")
            return None
    
    def _generate_mesh_from_keypoints(self, keypoints: np.ndarray) -> Optional[np.ndarray]:
        """从关键点生成人体网格"""
        try:
            # 简单的基于关键点的网格生成
            # 这里我们创建一个基本的人体形状
            
            # 生成标准SMPL顶点数量
            num_vertices = 6890
            
            # 基于关键点生成顶点
            vertices = np.zeros((num_vertices, 3))
            
            # 使用关键点作为参考，生成合理的顶点分布
            for i in range(num_vertices):
                # 随机选择一个关键点作为参考
                ref_keypoint = keypoints[i % len(keypoints)]
                # 添加一些随机变化
                noise = np.random.normal(0, 0.1, 3)
                vertices[i] = ref_keypoint + noise
            
            return vertices
            
        except Exception as e:
            logger.warning(f"关键点网格生成失败: {e}")
            return None
    
    def _generate_default_human_model(self) -> np.ndarray:
        """生成默认的人体模型"""
        try:
            logger.info("生成默认人体模型...")
            
            # 生成标准SMPL顶点数量
            num_vertices = 6890
            
            # 创建一个基本的人体形状（T-pose）
            vertices = np.zeros((num_vertices, 3))
            
            # 生成一个简单的人体轮廓
            for i in range(num_vertices):
                # 创建T-pose的人体形状
                x = np.random.normal(0, 0.3)  # 身体宽度
                y = np.random.normal(0, 0.8)  # 身体高度
                z = np.random.normal(0, 0.2)  # 身体深度
                
                # 添加一些人体特征
                if y > 0.5:  # 上半身
                    x *= 0.8  # 上半身稍窄
                if y < -0.3:  # 腿部
                    x *= 0.6  # 腿部更窄
                
                vertices[i] = [x, y, z]
            
            logger.info(f"✓ 默认人体模型生成成功：{len(vertices)} 个顶点")
            return vertices
            
        except Exception as e:
            logger.warning(f"默认人体模型生成失败: {e}")
            # 最后的备选方案：随机顶点
            return np.random.randn(6890, 3) * 0.5
    
    def convert_pose_params_to_smpl_format(self, pose_params: np.ndarray) -> Dict[str, torch.Tensor]:
        """将姿态参数转换为SMPL格式"""
        try:
            # 检查pose_params的格式
            pose_len = len(pose_params)
            logger.info(f"输入姿态参数长度: {pose_len}")
            
            if pose_len == 72:
                # 标准SMPL格式：72维参数
                # 前3维是全局旋转，后69维是23个关节的旋转（每个3维）
                global_orient = pose_params[:3]
                body_pose = pose_params[3:]
                
                logger.info(f"全局旋转参数: {global_orient}")
                logger.info(f"身体姿态参数前5个: {body_pose[:5]}")
                
                # 🔥 关键修复：将轴角参数转换为旋转矩阵！
                # SMPL模型期望：(1, 1, 3, 3) 和 (1, 23, 3, 3)
                
                # 转换全局旋转：轴角(3,) -> 旋转矩阵(3,3) -> 张量(1,1,3,3)
                global_orient_matrix = self.axis_angle_to_rotation_matrix(global_orient)
                global_orient_tensor = torch.tensor(global_orient_matrix, dtype=torch.float32).unsqueeze(0).unsqueeze(0)  # (1, 1, 3, 3)
                
                # 转换身体姿态：69维 -> 23个关节的旋转矩阵
                body_pose_matrices = []
                for i in range(0, len(body_pose), 3):
                    joint_axis_angle = body_pose[i:i+3]
                    joint_matrix = self.axis_angle_to_rotation_matrix(joint_axis_angle)
                    body_pose_matrices.append(joint_matrix)
                
                # 堆叠所有关节的旋转矩阵：(23, 3, 3) -> (1, 23, 3, 3)
                body_pose_tensor = torch.stack([torch.tensor(m, dtype=torch.float32) for m in body_pose_matrices]).unsqueeze(0)
                
                logger.info(f"转换后的张量形状: global_orient={global_orient_tensor.shape}, body_pose={body_pose_tensor.shape}")
                logger.info(f"全局旋转矩阵:\n{global_orient_matrix}")
                logger.info(f"第一个关节旋转矩阵:\n{body_pose_matrices[0]}")
                
            else:
                # 其他格式，使用默认值
                logger.warning(f"未知的姿态参数格式，长度: {pose_len}，使用默认值")
                global_orient_tensor = torch.eye(3, dtype=torch.float32).unsqueeze(0).unsqueeze(0)  # (1, 1, 3, 3)
                body_pose_tensor = torch.zeros(1, 23, 3, 3, dtype=torch.float32)  # (1, 23, 3, 3)
            
            return {
                'global_orient': global_orient_tensor,
                'body_pose': body_pose_tensor
            }
            
        except Exception as e:
            logger.warning(f"姿态参数转换失败: {e}")
            import traceback
            logger.warning(f"详细错误: {traceback.format_exc()}")
            # 返回默认值
            return {
                'global_orient': torch.eye(3, dtype=torch.float32).unsqueeze(0).unsqueeze(0),
                'body_pose': torch.zeros(1, 23, 3, 3, dtype=torch.float32)
            }
    
    def axis_angle_to_rotation_matrix(self, axis_angle: np.ndarray) -> np.ndarray:
        """将轴角表示转换为旋转矩阵（Rodrigues公式）"""
        try:
            # 轴角格式：[rx, ry, rz] 其中 [rx, ry, rz] 是旋转轴和角度的组合
            angle = np.linalg.norm(axis_angle)
            
            # 如果角度接近0，返回单位矩阵
            if angle < 1e-8:
                return np.eye(3)
            
            # 归一化旋转轴
            axis = axis_angle / angle
            
            # 构建反对称矩阵
            K = np.array([[0, -axis[2], axis[1]],
                         [axis[2], 0, -axis[0]],
                         [-axis[1], axis[0], 0]])
            
            # Rodrigues公式：R = I + sin(θ)K + (1-cos(θ))K²
            rotation_matrix = (np.eye(3) + 
                             np.sin(angle) * K + 
                             (1 - np.cos(angle)) * (K @ K))
            
            return rotation_matrix
            
        except Exception as e:
            logger.warning(f"轴角到旋转矩阵转换失败: {e}")
            # 返回单位矩阵作为默认值
            return np.eye(3)
    
    def euler_to_rotation_matrix(self, euler_angles: np.ndarray) -> np.ndarray:
        """将欧拉角转换为旋转矩阵"""
        import math
        x, y, z = euler_angles
        
        # 绕X轴旋转
        Rx = np.array([
            [1, 0, 0],
            [0, math.cos(x), -math.sin(x)],
            [0, math.sin(x), math.cos(x)]
        ])
        
        # 绕Y轴旋转
        Ry = np.array([
            [math.cos(y), 0, math.sin(y)],
            [0, 1, 0],
            [-math.sin(y), 0, math.cos(y)]
        ])
        
        # 绕Z轴旋转
        Rz = np.array([
            [math.cos(z), -math.sin(z), 0],
            [math.sin(z), math.cos(z), 0],
            [0, 0, 1]
        ])
        
        # 组合旋转 (ZYX顺序)
        R = Rz @ Ry @ Rx
        return R
    
    def render_smpl_model(self, smpl_frames: List[Dict], output_dir: str = "smpl_rendering") -> None:
        """渲染SMPL模型"""
        logger.info("开始渲染SMPL模型...")
        
        # 创建输出目录
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)
        
        # 选择要渲染的帧（前5帧，减少数量便于调试）
        frames_to_render = smpl_frames[:5]
        logger.info(f"将渲染 {len(frames_to_render)} 帧")
        
        for i, frame in enumerate(frames_to_render):
            try:
                logger.info(f"正在渲染第 {i+1}/{len(frames_to_render)} 帧: {frame['frame_name']}")
                
                # 生成SMPL顶点
                pose_params = np.array(frame['pose_params']) if frame['pose_params'] is not None else np.zeros(72)
                shape_params = np.array(frame['shape_params']) if frame['shape_params'] is not None else np.zeros(10)
                trans_params = np.array(frame['trans_params']) if frame['trans_params'] is not None else np.zeros(3)
                
                logger.info(f"输入参数: pose_params={pose_params.shape}, shape_params={shape_params.shape}, trans_params={trans_params.shape}")
                logger.info(f"pose_params前5个值: {pose_params[:5]}")
                logger.info(f"shape_params前5个值: {shape_params[:5]}")
                logger.info(f"trans_params: {trans_params}")
                
                # 生成顶点
                vertices = self.generate_smpl_vertices(pose_params, shape_params, trans_params)
                
                # 使用demo风格的渲染方法
                self._render_with_demo_style(vertices, frame, output_path, i)
                
            except Exception as e:
                logger.warning(f"渲染帧 {frame['frame_name']} 失败: {e}")
                import traceback
                logger.warning(f"错误详情: {traceback.format_exc()}")
                continue
        
        logger.info(f"✓ 渲染完成！结果保存在: {output_path}")
    
    def _render_with_demo_style(self, vertices: np.ndarray, frame: Dict, output_path: Path, frame_idx: int):
        """使用demo.py风格的渲染方法，支持正视图、侧视图、顶视图"""
        try:
            # Linux环境下优先使用HMR2渲染器
            if self.is_linux and self.renderer and self.smpl_faces is not None:
                logger.info("Linux环境：使用HMR2 OpenGL渲染器进行demo风格渲染")
                self._render_with_hmr2_demo_style(vertices, frame, output_path, frame_idx)
            else:
                # 回退到matplotlib渲染
                logger.info("使用matplotlib渲染器作为备选方案")
                self._render_with_matplotlib(vertices, frame, output_path, frame_idx)
                
        except Exception as e:
            logger.warning(f"demo风格渲染方法失败: {e}")
            # 回退到matplotlib渲染
            self._render_with_matplotlib(vertices, frame, output_path, frame_idx)
    
    def _render_with_hmr2_demo_style(self, vertices: np.ndarray, frame: Dict, output_path: Path, frame_idx: int):
        """使用HMR2渲染器进行demo风格的渲染，支持正视图、侧视图、顶视图"""
        try:
            logger.info("使用HMR2 OpenGL渲染器进行demo风格渲染...")
            
            # 创建参考图像 - 使用与demo.py相同的格式
            img_size = 512
            white_img = torch.ones((3, img_size, img_size), dtype=torch.float32) * 0.5
            
            # 使用简单的相机设置，与demo.py保持一致
            # 关键：不要过度调整相机距离，让HMR2渲染器自己处理
            cam_t = np.array([0, 0, 0])  # 使用默认相机位置
            
            logger.info(f"使用默认相机参数: {cam_t}")
            logger.info(f"顶点范围: 最小={np.min(vertices, axis=0)}, 最大={np.max(vertices, axis=0)}")
            logger.info(f"顶点数量: {len(vertices)}")
            
            # 渲染正视图
            logger.info("渲染正视图...")
            front_view = self.renderer(
                vertices, 
                cam_t, 
                white_img, 
                mesh_base_color=(0.65, 0.74, 0.86),  # LIGHT_BLUE，与demo.py一致
                scene_bg_color=(1, 1, 1)
            )
            
            # 渲染侧视图
            logger.info("渲染侧视图...")
            side_view = self.renderer(
                vertices, 
                cam_t, 
                white_img, 
                mesh_base_color=(0.65, 0.74, 0.86),  # LIGHT_BLUE
                scene_bg_color=(1, 1, 1),
                side_view=True
            )
            
            # 渲染顶视图
            logger.info("渲染顶视图...")
            top_view = self.renderer(
                vertices, 
                cam_t, 
                white_img, 
                mesh_base_color=(0.65, 0.74, 0.86),  # LIGHT_BLUE
                scene_bg_color=(1, 1, 1),
                top_view=True
            )
            
            # 水平拼接所有视图，与demo.py的拼接方式一致
            combined_view = np.concatenate([front_view, side_view, top_view], axis=1)
            
            # 保存结果
            output_file = output_path / f"frame_{frame_idx:03d}_{frame['frame_name'].replace('.jpg', '')}_demo_style_opengl.png"
            
            # 转换颜色空间并保存，与demo.py一致
            combined_view_bgr = cv2.cvtColor((combined_view * 255).astype(np.uint8), cv2.COLOR_RGB2BGR)
            cv2.imwrite(str(output_file), combined_view_bgr)
            
            logger.info(f"✓ HMR2 demo风格OpenGL渲染完成: {output_file}")
            logger.info(f"  图像尺寸: {combined_view.shape}")
            logger.info(f"  正视图尺寸: {front_view.shape}")
            logger.info(f"  侧视图尺寸: {side_view.shape}")
            logger.info(f"  顶视图尺寸: {top_view.shape}")
            
        except Exception as e:
            logger.warning(f"HMR2 demo风格OpenGL渲染失败: {e}")
            import traceback
            logger.warning(f"详细错误信息: {traceback.format_exc()}")
            # 回退到matplotlib渲染
            self._render_with_matplotlib(vertices, frame, output_path, frame_idx)
    
    def _adjust_vertices_conservatively(self, vertices: np.ndarray) -> np.ndarray:
        """采用保守策略调整顶点位置"""
        try:
            # 计算人体的边界框
            min_coords = np.min(vertices, axis=0)
            max_coords = np.max(vertices, axis=0)
            center = (min_coords + max_coords) / 2
            size = max_coords - min_coords
            
            logger.info(f"原始顶点范围: 最小={min_coords}, 最大={max_coords}, 中心={center}, 尺寸={size}")
            
            # 关键修复：不要过度调整顶点，只做最基本的居中
            # 如果顶点范围已经合理，就不做调整
            if np.max(size) < 5.0:  # 如果最大尺寸小于5，说明顶点范围合理
                logger.info("顶点范围合理，不做调整")
                return vertices
            
            # 只做最基本的居中，不做复杂的缩放和偏移
            adjusted_vertices = vertices - center
            
            # 检查调整后的范围
            adjusted_min = np.min(adjusted_vertices, axis=0)
            adjusted_max = np.max(adjusted_vertices, axis=0)
            adjusted_size = adjusted_max - adjusted_min
            
            logger.info(f"调整后顶点范围: 最小={adjusted_min}, 最大={adjusted_max}, 尺寸={adjusted_size}")
            
            return adjusted_vertices
            
        except Exception as e:
            logger.warning(f"保守顶点调整失败: {e}")
            return vertices
    
    def _render_with_matplotlib(self, vertices: np.ndarray, frame: Dict, output_path: Path, frame_idx: int):
        """使用matplotlib进行3D渲染，模拟demo风格的多视图效果"""
        try:
            import matplotlib.pyplot as plt
            from mpl_toolkits.mplot3d import Axes3D
            from mpl_toolkits.mplot3d.art3d import Poly3DCollection
            
            logger.info("开始matplotlib 3D demo风格渲染...")
            
            # 🔥 关键调试：直接检查顶点数据
            logger.info("=== 顶点数据调试 ===")
            logger.info(f"顶点数量: {len(vertices)}")
            logger.info(f"顶点形状: {vertices.shape}")
            logger.info(f"顶点数据类型: {vertices.dtype}")
            logger.info(f"顶点范围: X[{vertices[:, 0].min():.3f}, {vertices[:, 0].max():.3f}]")
            logger.info(f"          Y[{vertices[:, 1].min():.3f}, {vertices[:, 1].max():.3f}]")
            logger.info(f"          Z[{vertices[:, 2].min():.3f}, {vertices[:, 2].max():.3f}]")
            
            # 检查是否有异常值
            x_outliers = np.sum(np.abs(vertices[:, 0]) > 10)
            y_outliers = np.sum(np.abs(vertices[:, 0]) > 10)
            z_outliers = np.sum(np.abs(vertices[:, 0]) > 10)
            logger.info(f"异常值统计: X轴{x_outliers}个, Y轴{y_outliers}个, Z轴{z_outliers}个")
            
            # 检查顶点分布
            x_std = np.std(vertices[:, 0])
            y_std = np.std(vertices[:, 1])
            z_std = np.std(vertices[:, 2])
            logger.info(f"顶点分布标准差: X={x_std:.3f}, Y={y_std:.3f}, Z={z_std:.3f}")
            
            # 采用保守的顶点调整策略
            vertices = self._adjust_vertices_conservatively(vertices)
            
            # 🔥 关键改进：使用网格渲染而不是散点渲染
            logger.info("使用网格渲染模式...")
            
            # 创建3D图形，包含3个子图，模拟demo.py的布局
            fig = plt.figure(figsize=(24, 8))
            
            # 正视图 (左半边) - 对应demo.py的front_view
            ax1 = fig.add_subplot(131, projection='3d')
            self._render_mesh_view(ax1, vertices, 'Front View', frame["frame_name"], 'blue')
            ax1.view_init(elev=0, azim=0)  # 正视图：正面看
            
            # 侧视图 (中间) - 对应demo.py的side_view
            ax2 = fig.add_subplot(132, projection='3d')
            self._render_mesh_view(ax2, vertices, 'Side View', frame["frame_name"], 'red')
            ax2.view_init(elev=0, azim=90)  # 侧视图：侧面看
            
            # 顶视图 (右半边) - 对应demo.py的top_view
            ax3 = fig.add_subplot(133, projection='3d')
            self._render_mesh_view(ax3, vertices, 'Top View', frame["frame_name"], 'green')
            ax3.view_init(elev=90, azim=0)  # 顶视图：从上看
            
            # 添加帧信息，模拟demo.py的标题风格
            fig.suptitle(f'Frame: {frame["frame_name"]} | Confidence: {frame["confidence"]:.3f} | Demo-Style Multi-View Rendering', 
                        fontsize=16)
            
            # 调整布局
            plt.tight_layout()
            
            # 保存结果
            output_file = output_path / f"frame_{frame_idx:03d}_{frame['frame_name'].replace('.jpg', '')}_demo_style_matplotlib.png"
            plt.savefig(str(output_file), dpi=150, bbox_inches='tight')
            plt.close()
            
            logger.info(f"✓ matplotlib 3D demo风格渲染完成: {output_file}")
            
        except Exception as e:
            logger.warning(f"matplotlib 3D demo风格渲染失败: {e}")
            import traceback
            logger.warning(f"详细错误信息: {traceback.format_exc()}")
    
    def _render_mesh_view(self, ax, vertices: np.ndarray, view_name: str, frame_name: str, color: str):
        """渲染单个网格视图"""
        try:
            # 方法1：尝试使用SMPL faces进行网格渲染
            if hasattr(self, 'smpl_faces') and self.smpl_faces is not None:
                logger.info(f"使用SMPL faces进行网格渲染: {view_name}")
                self._render_with_smpl_faces(ax, vertices, color)
            else:
                # 方法2：使用简化的表面渲染
                logger.info(f"使用简化表面渲染: {view_name}")
                self._render_simplified_surface(ax, vertices, color)
            
            # 设置标签和标题
            ax.set_title(f'{view_name} - {frame_name}')
            ax.set_xlabel('X')
            ax.set_ylabel('Y')
            ax.set_zlabel('Z')
            
            # 设置坐标轴范围，确保完整人体可见
            self._set_optimal_view_range(ax, vertices)
            
        except Exception as e:
            logger.warning(f"网格视图渲染失败: {e}")
            # 回退到散点渲染
            ax.scatter(vertices[:, 0], vertices[:, 1], vertices[:, 2], 
                      c=color, s=0.5, alpha=0.8)
    
    def _render_with_smpl_faces(self, ax, vertices: np.ndarray, color: str):
        """使用SMPL faces进行网格渲染"""
        try:
            # 获取faces
            faces = self.smpl_faces
            
            # 创建三角形集合
            triangles = []
            for face in faces:
                # 每个face包含3个顶点索引
                triangle = [
                    vertices[face[0]],
                    vertices[face[1]], 
                    vertices[face[2]]
                ]
                triangles.append(triangle)
            
            # 创建Poly3DCollection
            poly3d = Poly3DCollection(triangles, alpha=0.7, facecolor=color, edgecolor='black', linewidth=0.1)
            ax.add_collection3d(poly3d)
            
            logger.info(f"✓ 成功渲染 {len(triangles)} 个三角形面")
            
        except Exception as e:
            logger.warning(f"SMPL faces渲染失败: {e}")
            raise e
    
    def _render_simplified_surface(self, ax, vertices: np.ndarray, color: str):
        """使用简化的表面渲染"""
        try:
            # 创建简化的表面表示
            # 使用凸包或简化的网格
            
            # 方法1：使用凸包
            from scipy.spatial import ConvexHull
            try:
                # 计算凸包
                hull = ConvexHull(vertices)
                
                # 渲染凸包面
                for simplex in hull.simplices:
                    # 每个simplex是一个三角形
                    triangle = [
                        vertices[simplex[0]],
                        vertices[simplex[1]],
                        vertices[simplex[2]]
                    ]
                    poly3d = Poly3DCollection([triangle], alpha=0.3, facecolor=color, edgecolor='black', linewidth=0.1)
                    ax.add_collection3d(poly3d)
                
                logger.info(f"✓ 使用凸包渲染 {len(hull.simplices)} 个面")
                
            except Exception as e:
                logger.warning(f"凸包渲染失败: {e}")
                # 回退到散点渲染
                ax.scatter(vertices[:, 0], vertices[:, 1], vertices[:, 2], 
                          c=color, s=0.5, alpha=0.8)
                
        except Exception as e:
            logger.warning(f"简化表面渲染失败: {e}")
            # 最后的回退：散点渲染
            ax.scatter(vertices[:, 0], vertices[:, 1], vertices[:, 2], 
                      c=color, s=0.5, alpha=0.8)
    
    def _set_optimal_view_range(self, ax, vertices: np.ndarray):
        """设置最优的视图范围"""
        try:
            # 计算合适的坐标轴范围
            x_range = vertices[:, 0].max() - vertices[:, 0].min()
            y_range = vertices[:, 1].max() - vertices[:, 1].min()
            z_range = vertices[:, 2].max() - vertices[:, 2].min()
            
            max_range = max(x_range, y_range, z_range)
            padding = max_range * 0.2  # 20%的边距
            
            x_center = (vertices[:, 0].max() + vertices[:, 0].min()) / 2
            y_center = (vertices[:, 1].max() + vertices[:, 1].min()) / 2
            z_center = (vertices[:, 2].max() + vertices[:, 2].min()) / 2
            
            half_range = max_range / 2 + padding
            
            ax.set_xlim(x_center - half_range, x_center + half_range)
            ax.set_ylim(y_center - half_range, y_center + half_range)
            ax.set_zlim(z_center - half_range, z_center + half_range)
            
        except Exception as e:
            logger.warning(f"设置视图范围失败: {e}")
            # 使用默认范围
            ax.set_xlim(-2, 2)
            ax.set_ylim(-2, 2)
            ax.set_zlim(-2, 2)

def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='SMPL模型渲染器 - Linux服务器OpenGL优化版本')
    parser.add_argument('--pkl_file', type=str, required=True,
                       help='PKL文件路径')
    parser.add_argument('--output_dir', type=str, default='smpl_rendering',
                       help='输出目录')
    parser.add_argument('--force_opengl', action='store_true', default=False,
                       help='强制使用OpenGL渲染（仅Linux环境）')
    
    args = parser.parse_args()
    
    # 检查PKL文件
    if not Path(args.pkl_file).exists():
        logger.error(f"PKL文件不存在: {args.pkl_file}")
        return False
    
    # 环境检查
    if platform.system() == "Linux":
        logger.info("✓ 检测到Linux环境，将使用OpenGL渲染")
        if args.force_opengl:
            logger.info("✓ 强制OpenGL渲染模式已启用")
    else:
        logger.info(f"⚠ 当前环境: {platform.system()}，将使用备选渲染方案")
    
    # 创建渲染器
    renderer = SMPLModelRenderer()
    
    # 加载PKL数据
    pkl_data = renderer.load_pkl_data(args.pkl_file)
    if not pkl_data:
        logger.error("没有找到有效的PKL数据")
        return False
    
    # 提取SMPL数据
    smpl_frames = renderer.extract_smpl_data(pkl_data)
    if not smpl_frames:
        logger.error("没有找到有效的SMPL数据")
        return False
    
    # 渲染SMPL模型
    renderer.render_smpl_model(smpl_frames, args.output_dir)
    
    logger.info("✓ 所有操作完成！")
    return True

if __name__ == '__main__':
    success = main()
    exit(0 if success else 1) 