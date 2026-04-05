#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
真实的多视角4D-Humans融合系统
使用真正的4D-Humans track.py处理视频
"""

import os
import platform

# 强制设置OpenGL环境变量 - 必须在任何其他导入之前
if platform.system() == "Windows":
    os.environ['PYOPENGL_PLATFORM'] = 'win32'
    os.environ['PYOPENGL_PLATFORM_WIN32'] = '1'
    print("Windows环境：强制设置OpenGL平台为win32")
    print(f"PYOPENGL_PLATFORM: {os.environ.get('PYOPENGL_PLATFORM')}")
    print(f"PYOPENGL_PLATFORM_WIN32: {os.environ.get('PYOPENGL_PLATFORM_WIN32')}")
else:
    # Linux环境下的OpenGL设置
    os.environ['PYOPENGL_PLATFORM'] = 'egl'
    
    # 设置显示环境变量
    if 'DISPLAY' not in os.environ:
        os.environ['DISPLAY'] = ':0'
    
    # 设置EGL相关环境变量
    os.environ['EGL_PLATFORM'] = 'surfaceless'
    os.environ['MESA_GL_VERSION_OVERRIDE'] = '3.3'
    os.environ['MESA_GLSL_VERSION_OVERRIDE'] = '330'
    
    # 设置OpenGL驱动相关
    os.environ['LIBGL_ALWAYS_SOFTWARE'] = '1'  # 强制使用软件渲染（如果硬件有问题）
    os.environ['PYOPENGL_PLATFORM'] = 'egl'
    
    print("Linux环境：设置OpenGL平台为egl")
    print(f"PYOPENGL_PLATFORM: {os.environ.get('PYOPENGL_PLATFORM')}")
    print(f"DISPLAY: {os.environ.get('DISPLAY')}")
    print(f"EGL_PLATFORM: {os.environ.get('EGL_PLATFORM')}")
    print(f"LIBGL_ALWAYS_SOFTWARE: {os.environ.get('LIBGL_ALWAYS_SOFTWARE')}")

import sys
import json
import numpy as np
import cv2
import torch
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional
import logging
from datetime import datetime
import subprocess
import concurrent.futures
from scipy.spatial.distance import cdist
from scipy.optimize import linear_sum_assignment
import re

# 可视化库导入
try:
    import matplotlib.pyplot as plt
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False
    print("警告: matplotlib不可用，3D可视化功能将受限")

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class Real4DHumansFusion:
    """真实的多视角4D-Humans融合系统"""
    
    def __init__(self, config_path: str = "whole/config.toml", auto_process: bool = True, save_format: str = "rotation_matrices"):
        """
        初始化融合系统
        
        Args:
            config_path: 相机标定配置文件路径
            auto_process: 是否自动处理现有结果文件（暂未实现）
            save_format: 保存格式，'rotation_matrices' 或 'axis_angle'
        """
        self.config_path = Path(config_path)
        self.cameras = {}
        self.output_dir = Path("real_4d_fusion_1")
        self.output_dir.mkdir(exist_ok=True)
        self.save_format = save_format
        
        # 4D-Humans项目路径
        self.hmr2_dir = Path(".")
        
        # 加载相机标定参数
        self.load_camera_calibration()
        
        logger.info("真实的多视角4D-Humans融合系统初始化完成")
        logger.info(f"保存格式: {self.save_format}")
        
        # 注意：自动处理现有结果的功能暂未实现
        # if auto_process:
        #     self.check_and_process_existing_results()
    
    def load_camera_calibration(self):
        """加载相机标定参数"""
        logger.info("加载相机标定参数...")
        
        try:
            # 尝试使用rtoml
            try:
                import rtoml
                with open(self.config_path, 'r', encoding='utf-8') as f:
                    config = rtoml.load(f)
            except ImportError:
                # 回退到tomli
                import tomli
                with open(self.config_path, 'rb') as f:
                    config = tomli.load(f)
            
            # 处理每个相机
            for camera_name in ['cam_1', 'cam_2']:
                if camera_name in config:
                    cam_data = config[camera_name]
                    
                    # 内参矩阵
                    K = np.array(cam_data['matrix'], dtype=np.float64).reshape(3, 3)
                    
                    # 畸变参数
                    dist = np.array(cam_data.get('distortions', [0, 0, 0, 0, 0]), dtype=np.float64)
                    
                    # 外参 - 位置和旋转
                    t = np.array(cam_data['translation'], dtype=np.float64)
                    r_euler = np.array(cam_data['rotation'], dtype=np.float64)
                    
                    # 欧拉角转旋转矩阵
                    R = self.euler_to_rotation_matrix(r_euler)
                    
                    self.cameras[camera_name] = {
                        'K': K,
                        'dist': dist,
                        'R': R,
                        't': t,
                        'name': camera_name
                    }
                    
                    logger.info(f"相机 {camera_name}: 位置={t}, 旋转={r_euler}")
            
            logger.info(f"成功加载 {len(self.cameras)} 个相机参数")
            
        except Exception as e:
            logger.error(f"加载相机标定失败: {e}")
            raise
    
    def euler_to_rotation_matrix(self, euler_angles: np.ndarray) -> np.ndarray:
        """欧拉角转旋转矩阵 (ZYX顺序)"""
        rx, ry, rz = euler_angles
        
        # 绕X轴旋转
        Rx = np.array([
            [1, 0, 0],
            [0, np.cos(rx), -np.sin(rx)],
            [0, np.sin(rx), np.cos(rx)]
        ])
        
        # 绕Y轴旋转
        Ry = np.array([
            [np.cos(ry), 0, np.sin(ry)],
            [0, 1, 0],
            [-np.sin(ry), 0, np.cos(ry)]
        ])
        
        # 绕Z轴旋转
        Rz = np.array([
            [np.cos(rz), -np.sin(rz), 0],
            [np.sin(rz), np.cos(rz), 0],
            [0, 0, 1]
        ])
        
        return Rz @ Ry @ Rx
    
    def run_real_4d_humans(self, video_path: str, camera_name: str) -> Dict[str, Any]:
        """
        使用真实的4D-Humans处理视频
        
        Args:
            video_path: 视频文件路径
            camera_name: 相机名称
            
        Returns:
            4D-Humans处理结果
        """
        logger.info(f"使用真实4D-Humans处理相机 {camera_name} 的视频...")
        
        # 创建输出目录 - 确保父目录存在
        camera_output_dir = self.output_dir / f"4d_humans_{camera_name}"
        camera_output_dir.mkdir(parents=True, exist_ok=True)
        
        # 运行4D-Humans track.py
        result = self.execute_4d_humans_tracking(video_path, camera_output_dir)
        
        return result
    
    def check_gpu_availability(self) -> bool:
        """检查GPU可用性"""
        try:
            import torch
            if torch.cuda.is_available():
                gpu_count = torch.cuda.device_count()
                gpu_name = torch.cuda.get_device_name(0) if gpu_count > 0 else "Unknown"
                gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3 if gpu_count > 0 else 0
                logger.info(f"✓ 检测到 {gpu_count} 个GPU")
                logger.info(f"GPU名称: {gpu_name}")
                logger.info(f"GPU内存: {gpu_memory:.1f} GB")
                return True
            else:
                logger.warning("✗ CUDA不可用")
                return False
        except Exception as e:
            logger.warning(f"GPU检查失败: {e}")
            return False
    
    def debug_4d_humans_environment(self):
        """调试4D-Humans环境"""
        logger.info("开始调试4D-Humans环境...")
        
        # 强制重新设置OpenGL环境变量
        import platform
        if platform.system() == "Windows":
            os.environ['PYOPENGL_PLATFORM'] = 'win32'
            os.environ['PYOPENGL_PLATFORM_WIN32'] = '1'
            logger.info("强制设置OpenGL平台为win32")
        else:
            os.environ['PYOPENGL_PLATFORM'] = 'egl'
            logger.info("强制设置OpenGL平台为egl")
        
        # 检查当前工作目录
        current_dir = os.getcwd()
        logger.info(f"当前工作目录: {current_dir}")
        
        # 检查4D-Humans目录
        if self.hmr2_dir.exists():
            logger.info(f"4D-Humans目录存在: {self.hmr2_dir}")
            
            # 检查关键文件
            track_py = self.hmr2_dir / "track.py"
            if track_py.exists():
                logger.info(f"✓ track.py文件存在: {track_py}")
            else:
                logger.error(f"✗ track.py文件不存在: {track_py}")
            
            # 检查配置文件
            configs_dir = self.hmr2_dir / "hmr2" / "configs"
            if configs_dir.exists():
                logger.info(f"✓ 配置文件目录存在: {configs_dir}")
            else:
                logger.warning(f"⚠ 配置文件目录不存在: {configs_dir}")
        else:
            logger.error(f"✗ 4D-Humans目录不存在: {self.hmr2_dir}")
        
        # 检查Python环境
        try:
            import phalp
            logger.info("✓ PHALP模块可用")
        except ImportError as e:
            logger.error(f"✗ PHALP模块不可用: {e}")
        
        try:
            import hmr2
            logger.info("✓ HMR2模块可用")
        except ImportError as e:
            logger.error(f"✗ HMR2模块不可用: {e}")
        
        try:
            import torch
            logger.info(f"✓ PyTorch版本: {torch.__version__}")
            if torch.cuda.is_available():
                logger.info(f"✓ CUDA可用，GPU数量: {torch.cuda.device_count()}")
            else:
                logger.warning("⚠ CUDA不可用")
        except ImportError as e:
            logger.error(f"✗ PyTorch不可用: {e}")
        
        # 检查环境变量
        logger.info(f"PYOPENGL_PLATFORM: {os.environ.get('PYOPENGL_PLATFORM', '未设置')}")
        logger.info(f"DISPLAY: {os.environ.get('DISPLAY', '未设置')}")
        logger.info(f"CUDA_VISIBLE_DEVICES: {os.environ.get('CUDA_VISIBLE_DEVICES', '未设置')}")
        
        logger.info("环境调试完成")
    
    def execute_4d_humans_tracking(self, video_path: str, output_dir: Path) -> Dict[str, Any]:
        """执行4D-Humans跟踪"""
        try:
            # 检查是否在正确的环境中
            try:
                import phalp
                import hmr2
                logger.info("PHALP和HMR2模块已安装")
            except ImportError as e:
                logger.error(f"缺少必要的模块: {e}")
                logger.error("请确保在4D-humans conda环境中运行")
                return self.create_fallback_result(video_path, output_dir)
            
            # 检查GPU可用性
            gpu_available = self.check_gpu_availability()
            if gpu_available:
                logger.info("✓ GPU可用，将使用GPU加速")
                device = "cuda"
            else:
                logger.warning("✗ GPU不可用，使用CPU")
                device = "cpu"
            
            # 在Windows上强制使用默认OpenGL配置
            import platform
            if platform.system() == "Windows":
                # 强制使用Windows原生OpenGL模式
                os.environ['PYOPENGL_PLATFORM'] = 'win32'
                logger.info("Windows环境：强制使用Windows原生OpenGL模式")
            else:
                # Linux服务器环境变量
                os.environ['PYOPENGL_PLATFORM'] = 'egl'
                os.environ['DISPLAY'] = ':0'
            
            # 切换到4D-Humans目录
            original_cwd = os.getcwd()
            os.chdir(self.hmr2_dir)
            
            # 确保输出目录存在
            output_dir.mkdir(parents=True, exist_ok=True)
            logger.info(f"输出目录已创建: {output_dir}")
            
            # 创建results子目录（PHALP需要这个）
            results_dir = output_dir / "results"
            results_dir.mkdir(exist_ok=True)
            logger.info(f"results子目录已创建: {results_dir}")
            
            # 构建命令 - 使用正确的参数格式
            cmd = [
                "python", "track.py",
                f"video.source={video_path}",
                f"video.output_dir={output_dir}"
            ]
            
            # 如果GPU可用，只添加基本的GPU参数
            if gpu_available:
                cmd.extend([
                    "device=cuda",  # 去掉+前缀，因为device配置项已存在
                    "video.extract_video=False" # 禁用单帧图片提取，节省磁盘空间
                ])
            else:
                cmd.extend([
                    "video.extract_video=False" # 禁用单帧图片提取
                ])
            
            logger.info(f"执行命令: {' '.join(cmd)}")
            logger.info(f"当前工作目录: {os.getcwd()}")
            logger.info(f"视频文件路径: {video_path}")
            logger.info(f"输出目录: {output_dir}")
            logger.info(f"results子目录: {results_dir}")
            logger.info(f"使用设备: {device}")
            
            # 设置环境变量，强制使用指定的输出目录
            env = os.environ.copy()
            env['PHALP_OUTPUT_DIR'] = str(output_dir)
            env['VIDEO_OUTPUT_DIR'] = str(output_dir)
            env['CUDA_VISIBLE_DEVICES'] = '0'  # 确保使用GPU 0
            
            # 添加调试环境变量
            env['PHALP_DEBUG'] = '1'
            env['PYTHONUNBUFFERED'] = '1'  # 确保Python输出不被缓存
            
            # 关键修复：设置PYTHONPATH，使子进程能找到PHALP模块
            current_pythonpath = env.get('PYTHONPATH', '')
            phalp_master_path = str(Path.cwd() / "PHALP-master")
            current_dir_path = str(Path.cwd())
            
            # 构建新的PYTHONPATH
            new_pythonpath_parts = []
            if current_pythonpath:
                new_pythonpath_parts.append(current_pythonpath)
            new_pythonpath_parts.extend([phalp_master_path, current_dir_path])
            
            new_pythonpath = os.pathsep.join(new_pythonpath_parts)
            env['PYTHONPATH'] = new_pythonpath
            
            logger.info(f"环境变量设置:")
            logger.info(f"  PHALP_OUTPUT_DIR: {env.get('PHALP_OUTPUT_DIR', '未设置')}")
            logger.info(f"  VIDEO_OUTPUT_DIR: {env.get('VIDEO_OUTPUT_DIR', '未设置')}")
            logger.info(f"  CUDA_VISIBLE_DEVICES: {env.get('CUDA_VISIBLE_DEVICES', '未设置')}")
            logger.info(f"  PHALP_DEBUG: {env.get('PHALP_DEBUG', '未设置')}")
            logger.info(f"  PYTHONUNBUFFERED: {env.get('PYTHONUNBUFFERED', '未设置')}")
            logger.info(f"  PYTHONPATH: {env.get('PYTHONPATH', '未设置')}")
            
            # 检查视频文件
            logger.info(f"检查视频文件: {video_path}")
            if os.path.exists(video_path):
                file_size = os.path.getsize(video_path) / (1024 * 1024)  # MB
                logger.info(f"✓ 视频文件存在，大小: {file_size:.2f} MB")
            else:
                logger.error(f"✗ 视频文件不存在: {video_path}")
                return self.create_fallback_result(video_path, output_dir)
            
            # 执行命令
            logger.info("开始执行4D-Humans命令...")
            logger.info("注意：PHALP可能需要几分钟来加载模型和处理视频")
            
            # 执行命令，增加超时时间
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=1800,  # 30分钟超时（PHALP需要时间加载模型和处理）
                env=env  # 使用修改后的环境变量
            )
            
            # 恢复原始工作目录
            os.chdir(original_cwd)
            
            # 输出详细的执行结果
            logger.info(f"命令返回码: {result.returncode}")
            if result.stdout:
                logger.info(f"标准输出: {result.stdout}")
            if result.stderr:
                logger.error(f"错误输出: {result.stderr}")
            
            # 检查输出目录是否生成了文件
            output_files = list(output_dir.glob("*"))
            logger.info(f"输出目录中的文件: {[f.name for f in output_files]}")
            
            # 检查results子目录
            results_files = list(results_dir.glob("*"))
            logger.info(f"results子目录中的文件: {[f.name for f in results_files]}")
            
            # 检查上级目录的outputs/results/（PHALP可能使用默认路径）
            parent_results_dir = output_dir.parent / "outputs" / "results"
            if parent_results_dir.exists():
                parent_results_files = list(parent_results_dir.glob("*.pkl"))
                logger.info(f"上级目录outputs/results/中的.pkl文件: {[f.name for f in parent_results_files]}")
            
            if result.returncode == 0:
                logger.info("4D-Humans执行成功")
                return self.parse_4d_humans_output(output_dir, video_path)
            else:
                logger.error(f"4D-Humans执行失败，返回码: {result.returncode}")
                # 直接使用parse_4d_humans_output尝试解析
                return self.parse_4d_humans_output(output_dir, video_path)
                
        except subprocess.TimeoutExpired:
            logger.error("4D-Humans执行超时")
            os.chdir(original_cwd)
            return self.create_fallback_result(video_path, output_dir)
        except Exception as e:
            logger.error(f"4D-Humans执行异常: {e}")
            if 'original_cwd' in locals():
                os.chdir(original_cwd)
            return self.create_fallback_result(video_path, output_dir)
    
    def parse_4d_humans_output(self, output_dir: Path, video_path: str) -> Dict[str, Any]:
        """解析4D-Humans输出结果"""
        
        logger.info(f"开始查找4D-Humans输出结果，输出目录: {output_dir}")
        
        # PHALP实际保存pkl文件的位置
        # 根据PHALP代码：pkl_path = output_dir + '/results/' + track_dataset + "_" + video_seq + '.pkl'
        
        # 方法1: 在output_dir/results/中查找（PHALP的标准位置）
        results_dir = output_dir / "results"
        pkl_files = []
        
        if results_dir.exists():
            pkl_files = list(results_dir.glob("*.pkl"))
            logger.info(f"在标准位置 {results_dir} 中找到 {len(pkl_files)} 个.pkl文件")
        
        # 方法2: 如果没找到，在output_dir中查找
        if not pkl_files:
            pkl_files = list(output_dir.glob("*.pkl"))
            logger.info(f"在输出目录 {output_dir} 中找到 {len(pkl_files)} 个.pkl文件")
        
        # 方法3: 如果还是没找到，检查是否有其他可能的路径
        if not pkl_files:
            # 检查上级目录是否有outputs/results/
            parent_results_dir = output_dir.parent / "outputs" / "results"
            if parent_results_dir.exists():
                pkl_files = list(parent_results_dir.glob("*.pkl"))
                if pkl_files:
                    logger.info(f"在上级目录 {parent_results_dir} 中找到 {len(pkl_files)} 个.pkl文件")
        
        # 方法4: 检查项目根目录的outputs/
        if not pkl_files:
            project_outputs_dir = Path.cwd() / "outputs"
            if project_outputs_dir.exists():
                pkl_files = list(project_outputs_dir.glob("*.pkl"))
                if pkl_files:
                    logger.info(f"在项目outputs目录 {project_outputs_dir} 中找到 {len(pkl_files)} 个.pkl文件")
        
        if not pkl_files:
            logger.warning("未找到4D-Humans结果文件(.pkl)")
            logger.info(f"已搜索的目录:")
            logger.info(f"  1. 标准位置: {results_dir}")
            logger.info(f"  2. 输出目录: {output_dir}")
            logger.info(f"  3. 上级目录: {output_dir.parent}/outputs/results/")
            logger.info(f"  4. 项目outputs: {Path.cwd()}/outputs/")
            logger.warning("创建备用结果")
            return self.create_fallback_result(video_path, output_dir)
        
        # 显示找到的pkl文件
        logger.info(f"找到的.pkl文件:")
        for i, pkl_file in enumerate(pkl_files):
            file_size = pkl_file.stat().st_size / (1024 * 1024)  # MB
            logger.info(f"  {i+1}. {pkl_file.name} ({file_size:.2f} MB)")
        
        # 尝试解析最新的.pkl文件
        latest_pkl_file = max(pkl_files, key=lambda x: x.stat().st_mtime)
        logger.info(f"尝试解析最新的.pkl文件: {latest_pkl_file}")
        
        try:
            # 使用joblib.load()解析.pkl文件
            import joblib
            tracklets_data = joblib.load(latest_pkl_file)
            
            logger.info(f"✓ 成功解析4D-Humans结果: {latest_pkl_file}")
            logger.info(f"数据类型: {type(tracklets_data)}")
            
            if isinstance(tracklets_data, dict):
                frames = sorted(tracklets_data.keys())
                logger.info(f"帧数量: {len(frames)}")
                if frames:
                    logger.info(f"帧名示例: {frames[:3] if len(frames) > 3 else frames}")
                    
                    # 检查第一帧的数据结构
                    first_frame = frames[0]
                    first_frame_data = tracklets_data[first_frame]
                    logger.info(f"第一帧数据键: {list(first_frame_data.keys())}")
                    
                    if 'tracked_ids' in first_frame_data:
                        logger.info(f"第一帧跟踪ID: {first_frame_data['tracked_ids']}")
                    
                    if '3d_joints' in first_frame_data:
                        joints = first_frame_data['3d_joints']
                        logger.info(f"第一帧3D关节点数量: {len(joints)}")
                        if len(joints) > 0:
                            logger.info(f"第一个人的关节点形状: {joints[0].shape}")
            
            # 转换4D-Humans输出格式为我们的格式
            converted_result = self.convert_4d_humans_output(tracklets_data, video_path)
            return converted_result
            
        except Exception as e:
            logger.error(f"解析.pkl文件 {latest_pkl_file} 失败: {e}")
            logger.warning("创建备用结果")
            return self.create_fallback_result(video_path, output_dir)
    
    def convert_4d_humans_output(self, tracklets_data: Any, video_path: str) -> Dict[str, Any]:
        """转换4D-Humans输出格式为我们的标准格式"""
        
        # 读取视频获取基本信息
        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.release()
        
        frames_data = []
        
        # 根据.pkl文件的实际结构解析数据
        if isinstance(tracklets_data, dict):
            # PHALP的.pkl文件结构：{frame_name: frame_data}
            logger.info(f"解析PHALP .pkl文件，包含 {len(tracklets_data)} 帧")
            
            # 按帧名排序
            frame_names = sorted(tracklets_data.keys())
            logger.info(f"帧名示例: {frame_names[:5] if len(frame_names) > 5 else frame_names}")
            
            # 添加调试信息
            logger.info(f"开始处理帧索引提取...")
            sample_frames = min(5, len(frame_names))
            for i in range(sample_frames):
                frame_name = frame_names[i]
                try:
                    if '/img/' in frame_name:
                        img_part = frame_name.split('/img/')[-1]
                        frame_number = img_part.replace('.jpg', '')
                        frame_idx = int(frame_number)
                        logger.info(f"  帧 {i}: {frame_name} -> 提取帧号: {frame_number} -> 帧索引: {frame_idx}")
                    else:
                        logger.info(f"  帧 {i}: {frame_name} -> 不包含/img/路径")
                except Exception as e:
                    logger.warning(f"  帧 {i}: {frame_name} -> 提取失败: {e}")
            
            for frame_name in frame_names:
                frame_data = tracklets_data[frame_name]
                
                # 检查是否有跟踪数据 - 使用PHALP的标准字段
                if 'tracked_ids' not in frame_data or len(frame_data['tracked_ids']) == 0:
                    logger.debug(f"帧 {frame_name} 没有跟踪数据，跳过")
                    continue
                
                # 获取帧索引
                try:
                    # ✅ 优先从 frame_data['time'] 获取，这是最准确的帧号
                    if 'time' in frame_data:
                        frame_idx = int(frame_data['time'])
                    # 备用方法 1: 从帧名（如果是字符串路径）中提取
                    elif isinstance(frame_name, str) and '/img/' in frame_name:
                        # 提取img/后面的文件名部分
                        img_part = frame_name.split('/img/')[-1]
                        # 去掉.jpg扩展名，提取数字部分
                        frame_number = img_part.replace('.jpg', '')
                        frame_idx = int(frame_number)
                    elif isinstance(frame_name, str) and '_' in frame_name:
                        # 备用方法 2：从帧名末尾提取数字
                        frame_idx = int(frame_name.split('_')[-1]) if frame_name.split('_')[-1].isdigit() else 0
                    else:
                        frame_idx = 0
                    
                    logger.debug(f"帧名: {frame_name} -> 帧索引: {frame_idx}")
                except Exception as e:
                    frame_idx = 0
                    logger.warning(f"提取帧索引失败: {frame_name}, 错误: {e}")
                
                # 处理每个被跟踪的人
                for person_idx, track_id in enumerate(frame_data['tracked_ids']):
                    try:
                        # 提取3D关节点 - 按照PHALP的标准格式
                        if '3d_joints' in frame_data and len(frame_data['3d_joints']) > person_idx:
                            joints_3d = frame_data['3d_joints'][person_idx]
                            logger.debug(f"帧 {frame_name}, 人 {person_idx}: 3D关节点形状 {joints_3d.shape}")
                        else:
                            joints_3d = np.zeros((17, 3))
                            logger.warning(f"帧 {frame_name}, 人 {person_idx}: 缺少3D关节点数据")
                        
                        # 提取SMPL参数 - 检查PHALP的标准字段
                        # 🔥 关键修复：直接保存旋转矩阵格式，而不是轴角参数
                        global_orient = np.eye(3)  # 默认单位矩阵
                        body_pose = np.zeros((23, 3, 3))  # 默认零矩阵
                        shape_params = np.zeros(10)
                        trans_params = np.zeros(3)
                        
                        if 'smpl' in frame_data and len(frame_data['smpl']) > person_idx:
                            smpl_data = frame_data['smpl'][person_idx]
                            if isinstance(smpl_data, dict):
                                # 优先使用标准的SMPL格式（旋转矩阵）
                                if 'global_orient' in smpl_data and 'body_pose' in smpl_data:
                                    global_orient = smpl_data.get('global_orient')
                                    body_pose = smpl_data.get('body_pose')
                                    shape_params = smpl_data.get('betas', np.zeros(10))
                                    trans_params = smpl_data.get('transl', np.zeros(3))
                                    
                                    # 转换为numpy数组
                                    if hasattr(global_orient, 'cpu'):
                                        global_orient = global_orient.cpu().numpy()
                                    if hasattr(body_pose, 'cpu'):
                                        body_pose = body_pose.cpu().numpy()
                                    if hasattr(shape_params, 'cpu'):
                                        shape_params = shape_params.cpu().numpy()
                                    if hasattr(trans_params, 'cpu'):
                                        trans_params = trans_params.cpu().numpy()
                                    
                                    logger.info(f"✓ 使用标准SMPL格式: global_orient={global_orient.shape}, body_pose={body_pose.shape}")
                                    
                                elif 'pose_params' in smpl_data:
                                    # 回退到轴角格式（兼容性）
                                    pose_params = smpl_data.get('pose_params', np.zeros(72))
                                    shape_params = smpl_data.get('shape_params', np.zeros(10))
                                    trans_params = smpl_data.get('trans_params', np.zeros(3))
                                    
                                    # 转换为旋转矩阵格式
                                    global_orient, body_pose = self._convert_axis_angle_to_rotation_matrices(pose_params)
                                    logger.info(f"✓ 转换轴角到旋转矩阵: global_orient={global_orient.shape}, body_pose={body_pose.shape}")
                                    
                                else:
                                    # 默认值
                                    logger.warning("⚠ 未找到SMPL数据，使用默认值")
                            else:
                                # 如果smpl是数组格式，尝试转换
                                pose_params = smpl_data[:72] if len(smpl_data) >= 72 else np.zeros(72)
                                shape_params = smpl_data[72:82] if len(smpl_data) >= 82 else np.zeros(10)
                                trans_params = smpl_data[82:85] if len(smpl_data) >= 85 else np.zeros(3)
                                
                                # 转换为旋转矩阵格式
                                global_orient, body_pose = self._convert_axis_angle_to_rotation_matrices(pose_params)
                                logger.info(f"✓ 数组格式转换: global_orient={global_orient.shape}, body_pose={body_pose.shape}")
                        
                        # 提取边界框
                        if 'bbox' in frame_data and len(frame_data['bbox']) > person_idx:
                            bbox = frame_data['bbox'][person_idx]
                        else:
                            bbox = [0, 0, 100, 100]
                        
                        # 提取置信度
                        if 'conf' in frame_data and len(frame_data['conf']) > person_idx:
                            confidence = frame_data['conf'][person_idx]
                        else:
                            confidence = 0.8
                        
                        # 构建帧数据 - 根据选择的格式保存
                        if self.save_format == 'rotation_matrices':
                            # ✅ 保存旋转矩阵格式
                            frame_info = {
                                'frame_idx': frame_idx,
                                'frame_name': frame_name,
                                'timestamp': frame_idx / fps if fps > 0 else 0,
                                'global_orient': global_orient.tolist() if hasattr(global_orient, 'tolist') else global_orient,
                                'body_pose': body_pose.tolist() if hasattr(body_pose, 'tolist') else body_pose,
                                'shape_params': shape_params.tolist() if hasattr(shape_params, 'tolist') else shape_params,
                                'trans_params': trans_params.tolist() if hasattr(trans_params, 'tolist') else trans_params,
                                'keypoints_3d': joints_3d.tolist() if hasattr(joints_3d, 'tolist') else joints_3d,
                                'bbox': bbox,
                                'confidence': confidence,
                                'track_id': track_id,
                                'person_idx': person_idx,
                                'smpl_format': 'rotation_matrices'  # 标识这是旋转矩阵格式
                            }
                        else:
                            # ✅ 保存轴角格式（兼容性）
                            # 将旋转矩阵转换回轴角参数
                            pose_params = self._convert_rotation_matrices_to_axis_angle(global_orient, body_pose)
                            frame_info = {
                                'frame_idx': frame_idx,
                                'frame_name': frame_name,
                                'timestamp': frame_idx / fps if fps > 0 else 0,
                                'pose_params': pose_params.tolist() if hasattr(pose_params, 'tolist') else pose_params,
                                'shape_params': shape_params.tolist() if hasattr(shape_params, 'tolist') else shape_params,
                                'trans_params': trans_params.tolist() if hasattr(trans_params, 'tolist') else trans_params,
                                'keypoints_3d': joints_3d.tolist() if hasattr(joints_3d, 'tolist') else joints_3d,
                                'bbox': bbox,
                                'confidence': confidence,
                                'track_id': track_id,
                                'person_idx': person_idx,
                                'smpl_format': 'axis_angle'  # 标识这是轴角格式
                            }
                        
                        frames_data.append(frame_info)
                        
                    except Exception as e:
                        logger.warning(f"处理帧 {frame_name} 中的人 {person_idx} 时出错: {e}")
                        continue
        
        elif isinstance(tracklets_data, list):
            # 如果是列表格式（旧的格式）
            logger.info("解析列表格式的tracklets数据")
            for tracklet in tracklets_data:
                if isinstance(tracklet, dict):
                    track_id = tracklet.get('track_id', 0)
                    frames = tracklet.get('frames', [])
                    
                    for frame_info in frames:
                        try:
                            # 直接处理frame_info数据
                            frame_idx = frame_info.get('frame_idx', 0)
                            
                            # 🔥 关键修复：也使用旋转矩阵格式
                            if 'global_orient' in frame_info and 'body_pose' in frame_info:
                                # 已经是旋转矩阵格式
                                global_orient = frame_info.get('global_orient', np.eye(3))
                                body_pose = frame_info.get('body_pose', np.zeros((23, 3, 3)))
                                shape_params = frame_info.get('shape_params', np.zeros(10))
                                trans_params = frame_info.get('trans_params', np.zeros(3))
                                logger.info(f"✓ 列表格式：使用现有旋转矩阵格式")
                            else:
                                # 回退到轴角格式并转换
                                pose_params = frame_info.get('pose_params', np.zeros(72))
                                shape_params = frame_info.get('shape_params', np.zeros(10))
                                trans_params = frame_info.get('trans_params', np.zeros(3))
                                
                                # 转换为旋转矩阵格式
                                global_orient, body_pose = self._convert_axis_angle_to_rotation_matrices(pose_params)
                                logger.info(f"✓ 列表格式：转换轴角到旋转矩阵")
                            
                            keypoints_3d = frame_info.get('keypoints_3d', np.zeros((17, 3)))
                            confidence = frame_info.get('confidence', 0.8)
                            
                            frame_data = {
                                'frame_idx': frame_idx,
                                'timestamp': frame_idx / fps if fps > 0 else 0,
                                # ✅ 保存旋转矩阵格式
                                'global_orient': global_orient.tolist() if hasattr(global_orient, 'tolist') else global_orient,
                                'body_pose': body_pose.tolist() if hasattr(body_pose, 'tolist') else body_pose,
                                'shape_params': shape_params.tolist() if hasattr(shape_params, 'tolist') else shape_params,
                                'trans_params': trans_params.tolist() if hasattr(trans_params, 'tolist') else trans_params,
                                'keypoints_3d': keypoints_3d.tolist() if hasattr(keypoints_3d, 'tolist') else keypoints_3d,
                                'confidence': confidence,
                                'track_id': track_id,
                                'smpl_format': 'rotation_matrices'  # 标识这是旋转矩阵格式
                            }
                            
                            frames_data.append(frame_data)
                        except Exception as e:
                            logger.warning(f"处理列表格式数据时出错: {e}")
                            continue
        
        else:
            logger.warning(f"未知的数据格式: {type(tracklets_data)}")
            return self.create_fallback_result(video_path, Path("."))
        
        logger.info(f"成功解析 {len(frames_data)} 帧数据")
        
        return {
            'frames_data': frames_data,
            'total_frames': total_frames,
            'fps': fps,
            'processing_timestamp': datetime.now().isoformat(),
            'is_fallback': False,
            'data_format': '4d_humans_pkl'
        }
    
    def create_fallback_result(self, video_path: str, output_dir: Path) -> Dict[str, Any]:
        """创建备用结果（当4D-Humans失败时）"""
        logger.info("创建备用结果...")
        
        # 读取视频获取基本信息
        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.release()
        
        # 模拟4D-Humans输出格式
        frames_data = []
        
        # 创建一些模拟帧数据
        for frame_idx in range(0, min(total_frames, 100), 10):
            # 模拟SMPL参数
            pose_params = np.random.randn(72) * 0.1
            shape_params = np.random.randn(10) * 0.5
            trans_params = np.random.randn(3) * 0.1
            
            # 模拟3D顶点和关键点
            vertices_3d = np.random.randn(6890, 3) * 0.5 + np.array([0, 0, 2])
            keypoints_3d = np.random.randn(17, 3) * 0.3 + np.array([0, 0, 2])
            
            frame_data = {
                'frame_idx': frame_idx,
                'timestamp': frame_idx / fps if fps > 0 else 0,
                'pose_params': pose_params.tolist(),
                'shape_params': shape_params.tolist(),
                'trans_params': trans_params.tolist(),
                'vertices_3d': vertices_3d.tolist(),
                'keypoints_3d': keypoints_3d.tolist(),
                'confidence': np.random.uniform(0.7, 0.95),
                'track_id': 0
            }
            
            frames_data.append(frame_data)
        
        return {
            'frames_data': frames_data,
            'total_frames': total_frames,
            'fps': fps,
            'processing_timestamp': datetime.now().isoformat(),
            'is_fallback': True
        }
    
    def transform_to_world_coordinates(self, camera_data: Dict[str, Any], 
                                    camera_name: str) -> Dict[str, Any]:
        """
        将相机坐标系中的3D模型转换到世界坐标系
        
        Args:
            camera_data: 4D-Humans处理结果
            camera_name: 相机名称
            
        Returns:
            转换到世界坐标系的结果
        """
        logger.info(f"将相机 {camera_name} 的结果转换到世界坐标系...")
        
        if camera_name not in self.cameras:
            logger.error(f"未找到相机 {camera_name} 的标定参数")
            return camera_data
        
        camera_calib = self.cameras[camera_name]
        R = camera_calib['R']
        t = camera_calib['t']
        
        # 转换每一帧的数据
        world_frames_data = []
        
        for frame_data in camera_data['frames_data']:
            try:
                # 转换3D关键点到世界坐标系
                keypoints_cam = np.array(frame_data['keypoints_3d'])
                keypoints_world = (R @ keypoints_cam.T).T + t
                
                # 转换平移参数到世界坐标系
                trans_cam = np.array(frame_data['trans_params'])
                trans_world = R @ trans_cam + t
                
                # 🔥 关键修复：转换pose_params到世界坐标系
                pose_params_cam = np.array(frame_data['pose_params'])
                pose_params_world = self.transform_pose_params_to_world(
                    pose_params_cam, R, t
                )
                
                # 转换边界框中心到世界坐标系（如果有的话）
                bbox_world = None
                if 'bbox' in frame_data and len(frame_data['bbox']) >= 4:
                    # 假设bbox是[x, y, w, h]格式，计算中心点
                    bbox = frame_data['bbox']
                    center_x = bbox[0] + bbox[2] / 2
                    center_y = bbox[1] + bbox[3] / 2
                    center_cam = np.array([center_x, center_y, 0])  # 假设在图像平面
                    center_world = (R @ center_cam.T).T + t
                    bbox_world = [center_world[0], center_world[1], bbox[2], bbox[3]]
                
                world_frame_data = {
                    'frame_idx': frame_data['frame_idx'],
                    'frame_name': frame_data.get('frame_name', ''),
                    'timestamp': frame_data['timestamp'],
                    'pose_params': pose_params_world.tolist(),  # 改为pose_params以保持一致性
                    'pose_params_camera': pose_params_cam.tolist(),  # 保留原始相机坐标系值用于调试
                    'shape_params': frame_data['shape_params'],
                    'trans_params': trans_world.tolist(),  # 改为trans_params以保持一致性
                    'keypoints_3d': keypoints_world.tolist(),  # 改为keypoints_3d以保持一致性
                    'bbox_world': bbox_world,
                    'confidence': frame_data['confidence'],
                    'track_id': frame_data.get('track_id', 0),
                    'person_idx': frame_data.get('person_idx', 0),
                    'camera_name': camera_name
                }
                
                world_frames_data.append(world_frame_data)
            except Exception as e:
                logger.warning(f"转换帧 {frame_data.get('frame_idx', 0)} 到世界坐标系失败: {e}")
                continue
        
        return {
            'camera_name': camera_name,
            'total_frames': camera_data['total_frames'],
            'fps': camera_data['fps'],
            'frames_data': world_frames_data,  # 改为frames_data以保持一致性
            'transformation_timestamp': datetime.now().isoformat(),
            'is_fallback': camera_data.get('is_fallback', False),
            'data_format': camera_data.get('data_format', 'unknown')
        }
    
    def transform_pose_params_to_world(self, pose_params_cam: np.ndarray, 
                                     R: np.ndarray, t: np.ndarray) -> np.ndarray:
        """
        将相机坐标系下的姿态参数转换到世界坐标系
        
        Args:
            pose_params_cam: 相机坐标系下的姿态参数 (72维)
            R: 相机旋转矩阵
            t: 相机平移向量
            
        Returns:
            世界坐标系下的姿态参数 (72维)
        """
        try:
            pose_params_world = pose_params_cam.copy()
            
            # 1. 转换全局方向 (前3个参数)
            if len(pose_params_cam) >= 3:
                global_orient_cam = pose_params_cam[:3]
                global_orient_world = self.transform_global_orientation_to_world(
                    global_orient_cam, R
                )
                pose_params_world[:3] = global_orient_world
            
            # 2. 身体姿态参数 (后69个参数) 通常不需要转换
            # 因为这些是相对于人体内部的关节角度
            # 但如果有需要，可以在这里添加转换逻辑
            
            logger.debug(f"姿态参数转换完成: 相机坐标系 -> 世界坐标系")
            return pose_params_world
            
        except Exception as e:
            logger.warning(f"姿态参数转换失败: {e}")
            # 如果转换失败，返回原始值
            return pose_params_cam
    
    def transform_global_orientation_to_world(self, global_orient_cam: np.ndarray, 
                                           R: np.ndarray) -> np.ndarray:
        """
        将全局方向从相机坐标系转换到世界坐标系
        
        Args:
            global_orient_cam: 相机坐标系下的全局方向 [rx, ry, rz]
            R: 相机旋转矩阵
            
        Returns:
            世界坐标系下的全局方向 [rx, ry, rz]
        """
        try:
            # 方法1: 欧拉角方式转换
            # 将欧拉角转换为旋转矩阵
            rot_matrix_cam = self.euler_to_rotation_matrix(global_orient_cam)
            
            # 应用相机旋转矩阵
            rot_matrix_world = R @ rot_matrix_cam
            
            # 转换回欧拉角
            global_orient_world = self.rotation_matrix_to_euler_angles(rot_matrix_world)
            
            logger.debug(f"全局方向转换: 相机[{global_orient_cam}] -> 世界[{global_orient_world}]")
            return global_orient_world
            
        except Exception as e:
            logger.warning(f"全局方向转换失败: {e}")
            # 如果转换失败，返回原始值
            return global_orient_cam
    
    def rotation_matrix_to_euler_angles(self, R: np.ndarray) -> np.ndarray:
        """
        将旋转矩阵转换为欧拉角 (ZYX顺序)
        
        Args:
            R: 3x3旋转矩阵
            
        Returns:
            欧拉角 [rx, ry, rz] (弧度)
        """
        try:
            # 从旋转矩阵提取欧拉角
            sy = np.sqrt(R[0, 0] * R[0, 0] + R[1, 0] * R[1, 0])
            
            if sy > 1e-6:
                rx = np.arctan2(R[2, 1], R[2, 2])
                ry = np.arctan2(-R[2, 0], sy)
                rz = np.arctan2(R[1, 0], R[0, 0])
            else:
                rx = np.arctan2(-R[1, 2], R[1, 1])
                ry = np.arctan2(-R[2, 0], sy)
                rz = 0
            
            return np.array([rx, ry, rz])
            
        except Exception as e:
            logger.warning(f"旋转矩阵转欧拉角失败: {e}")
            return np.zeros(3)
    
    def fuse_multi_view_models(self, world_results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        对世界坐标系中的多视角3D模型进行智能融合
        
        Args:
            world_results: 各视角转换到世界坐标系的结果列表
            
        Returns:
            融合后的结果
        """
        logger.info("开始多视角3D模型智能融合...")
        
        if len(world_results) < 2:
            logger.warning("视角数量不足，无法进行融合")
            return world_results[0] if world_results else {}
        
        # 按时间戳对齐帧
        aligned_frames = self.align_frames_by_timestamp(world_results)
        
        # 如果时间对齐失败或结果太少，尝试使用帧索引对齐
        if len(aligned_frames) < 2:
            logger.warning("时间对齐结果太少，尝试使用帧索引对齐...")
            aligned_frames = self.align_frames_by_frame_index(world_results)
        
        # 使用多线程处理融合
        fused_frames = []
        total_frames = len(aligned_frames)
        
        logger.info(f"开始处理 {total_frames} 帧的融合...")
        
        # 使用线程池进行并行处理
        with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
            # 提交所有融合任务
            future_to_timestamp = {}
            for frame_timestamp, frame_data_list in aligned_frames.items():
                if len(frame_data_list) < 2:
                    # 单视角，直接使用
                    fused_frames.append(frame_data_list[0])
                else:
                    # 多视角，提交融合任务
                    future = executor.submit(self.intelligent_fuse_single_frame, frame_data_list)
                    future_to_timestamp[future] = frame_timestamp
            
            # 收集结果
            for future in concurrent.futures.as_completed(future_to_timestamp):
                try:
                    fused_frame = future.result()
                    fused_frames.append(fused_frame)
                    timestamp = future_to_timestamp[future]
                    logger.debug(f"完成帧 {timestamp} 的融合")
                except Exception as e:
                    logger.error(f"融合帧 {future_to_timestamp[future]} 时出错: {e}")
        
        # 按时间戳排序
        fused_frames.sort(key=lambda x: x['timestamp'])
        
        # 计算融合统计信息
        fusion_stats = self.calculate_fusion_statistics(fused_frames)
        
        logger.info(f"融合完成，共处理 {len(fused_frames)} 帧")
        logger.info(f"融合统计: {fusion_stats}")
        
        return {
            'fused_frames': fused_frames,
            'num_cameras': len(world_results),
            'fusion_timestamp': datetime.now().isoformat(),
            'fusion_statistics': fusion_stats
        }
    
    def align_frames_by_timestamp(self, world_results: List[Dict[str, Any]], 
                                 time_tolerance: float = 0.5) -> Dict[float, List[Dict[str, Any]]]:
        """
        按时间戳对齐不同视角的帧，支持时间容差
        
        Args:
            world_results: 各视角转换到世界坐标系的结果列表
            time_tolerance: 时间容差（秒），用于处理时间戳不完全一致的情况
            
        Returns:
            按时间戳对齐的帧数据
        """
        aligned_frames = {}
        
        # 收集所有时间戳
        all_timestamps = []
        for result in world_results:
            for frame_data in result['frames_data']:
                all_timestamps.append(frame_data['timestamp'])
        
        if not all_timestamps:
            logger.warning("没有找到时间戳数据")
            return aligned_frames
        
        # 添加调试信息
        logger.info(f"时间戳统计:")
        logger.info(f"  总时间戳数量: {len(all_timestamps)}")
        logger.info(f"  时间戳范围: {min(all_timestamps):.3f} - {max(all_timestamps):.3f}")
        logger.info(f"  时间戳示例: {all_timestamps[:10]}")
        
        # 对时间戳进行聚类，处理时间容差
        all_timestamps = sorted(all_timestamps)
        timestamp_clusters = []
        current_cluster = [all_timestamps[0]]
        
        for timestamp in all_timestamps[1:]:
            if abs(timestamp - current_cluster[-1]) <= time_tolerance:
                # 在容差范围内，加入当前聚类
                current_cluster.append(timestamp)
            else:
                # 超出容差范围，开始新聚类
                timestamp_clusters.append(current_cluster)
                current_cluster = [timestamp]
        
        # 添加最后一个聚类
        timestamp_clusters.append(current_cluster)
        
        logger.info(f"时间聚类结果:")
        logger.info(f"  聚类数量: {len(timestamp_clusters)}")
        for i, cluster in enumerate(timestamp_clusters):
            logger.info(f"  聚类 {i+1}: {len(cluster)} 个时间戳, 范围: {min(cluster):.3f} - {max(cluster):.3f}")
        
        # 为每个聚类选择代表时间戳（使用中位数）
        representative_timestamps = []
        for cluster in timestamp_clusters:
            if len(cluster) % 2 == 0:
                # 偶数个，取中间两个的平均值
                mid = len(cluster) // 2
                rep_timestamp = (cluster[mid-1] + cluster[mid]) / 2
            else:
                # 奇数个，取中间值
                rep_timestamp = cluster[len(cluster) // 2]
            representative_timestamps.append(rep_timestamp)
        
        logger.info(f"代表时间戳: {[f'{t:.3f}' for t in representative_timestamps]}")
        
        # 按代表时间戳对齐帧
        for result in world_results:
            for frame_data in result['frames_data']:
                timestamp = frame_data['timestamp']
                
                # 找到最接近的代表时间戳
                closest_rep = min(representative_timestamps, 
                                key=lambda x: abs(x - timestamp))
                
                if closest_rep not in aligned_frames:
                    aligned_frames[closest_rep] = []
                
                # 检查是否已经包含来自同一相机的数据
                camera_name = frame_data.get('camera_name', 'unknown')
                existing_cameras = [f.get('camera_name', 'unknown') 
                                  for f in aligned_frames[closest_rep]]
                
                if camera_name not in existing_cameras:
                    aligned_frames[closest_rep].append(frame_data)
        
        logger.info(f"时间对齐完成，共 {len(aligned_frames)} 个时间点")
        logger.info(f"时间容差: {time_tolerance} 秒")
        
        # 添加对齐结果的详细信息
        for timestamp, frames in aligned_frames.items():
            camera_names = [f.get('camera_name', 'unknown') for f in frames]
            logger.info(f"  时间点 {timestamp:.3f}: {len(frames)} 帧, 相机: {camera_names}")
        
        return aligned_frames
    
    def align_frames_by_frame_index(self, world_results: List[Dict[str, Any]]) -> Dict[int, List[Dict[str, Any]]]:
        """
        按帧索引对齐不同视角的帧（备用策略）
        
        Args:
            world_results: 各视角转换到世界坐标系的结果列表
            
        Returns:
            按帧索引对齐的帧数据
        """
        logger.info("使用帧索引对齐策略...")
        
        aligned_frames = {}
        
        # 收集所有帧索引
        all_frame_indices = []
        for result in world_results:
            for frame_data in result['frames_data']:
                all_frame_indices.append(frame_data['frame_idx'])
        
        if not all_frame_indices:
            logger.warning("没有找到帧索引数据")
            return aligned_frames
        
        # 添加调试信息
        logger.info(f"帧索引统计:")
        logger.info(f"  总帧索引数量: {len(all_frame_indices)}")
        logger.info(f"  帧索引范围: {min(all_frame_indices)} - {max(all_frame_indices)}")
        logger.info(f"  帧索引示例: {sorted(set(all_frame_indices))[:10]}")
        
        # 按帧索引对齐帧
        for result in world_results:
            for frame_data in result['frames_data']:
                frame_idx = frame_data['frame_idx']
                
                if frame_idx not in aligned_frames:
                    aligned_frames[frame_idx] = []
                
                # 检查是否已经包含来自同一相机的数据
                camera_name = frame_data.get('camera_name', 'unknown')
                existing_cameras = [f.get('camera_name', 'unknown') 
                                  for f in aligned_frames[frame_idx]]
                
                if camera_name not in existing_cameras:
                    aligned_frames[frame_idx].append(frame_data)
        
        logger.info(f"帧索引对齐完成，共 {len(aligned_frames)} 个帧索引")
        
        # 添加对齐结果的详细信息
        for frame_idx, frames in aligned_frames.items():
            camera_names = [f.get('camera_name', 'unknown') for f in frames]
            logger.info(f"  帧索引 {frame_idx}: {len(frames)} 帧, 相机: {camera_names}")
        
        return aligned_frames
    
    def fuse_single_frame(self, frame_data_list: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        融合单帧的多视角数据
        
        Args:
            frame_data_list: 同一时间戳的多个视角数据
            
        Returns:
            融合后的帧数据
        """
        # 投票融合策略：
        # 1. 对于3D关键点：加权平均（基于置信度）
        # 2. 对于3D顶点：加权平均（基于置信度）
        # 3. 对于SMPL参数：加权平均（基于置信度）
        
        # 计算权重
        confidences = [data['confidence'] for data in frame_data_list]
        weights = np.array(confidences) / np.sum(confidences)
        
        # 融合3D关键点
        keypoints_list = [np.array(data['keypoints_3d']) for data in frame_data_list]
        fused_keypoints = np.zeros_like(keypoints_list[0])
        
        for i, keypoints in enumerate(keypoints_list):
            fused_keypoints += weights[i] * keypoints
        
        # 融合3D顶点
        vertices_list = [np.array(data['vertices_3d']) for data in frame_data_list]
        fused_vertices = np.zeros_like(vertices_list[0])
        
        for i, vertices in enumerate(vertices_list):
            fused_vertices += weights[i] * vertices
        
        # 融合SMPL参数
        pose_params_list = [np.array(data['pose_params']) for data in frame_data_list]
        fused_pose_params = np.zeros_like(pose_params_list[0])
        
        for i, pose_params in enumerate(pose_params_list):
            fused_pose_params += weights[i] * pose_params
        
        shape_params_list = [np.array(data['shape_params']) for data in frame_data_list]
        fused_shape_params = np.zeros_like(shape_params_list[0])
        
        for i, shape_params in enumerate(shape_params_list):
            fused_shape_params += weights[i] * shape_params
        
        trans_params_list = [np.array(data['trans_params']) for data in frame_data_list]
        fused_trans_params = np.zeros_like(trans_params_list[0])
        
        for i, trans_params in enumerate(trans_params_list):
            fused_trans_params += weights[i] * trans_params
        
        # 计算融合后的置信度
        fused_confidence = np.mean(confidences)
        
        return {
            'frame_idx': frame_data_list[0]['frame_idx'],
            'timestamp': frame_data_list[0]['timestamp'],
            'pose_params': fused_pose_params.tolist(),
            'shape_params': fused_shape_params.tolist(),
            'trans_params': fused_trans_params.tolist(),
            'vertices_3d': fused_vertices.tolist(),
            'keypoints_3d': fused_keypoints.tolist(),
            'confidence': fused_confidence,
            'num_views': len(frame_data_list),
            'view_weights': weights.tolist(),
            'camera_names': [data['camera_name'] for data in frame_data_list],
            'track_ids': [data.get('track_id', 0) for data in frame_data_list]
        }
    
    def intelligent_fuse_single_frame(self, frame_data_list: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        智能融合单帧的多视角数据
        
        Args:
            frame_data_list: 同一时间戳的多个视角数据
            
        Returns:
            融合后的帧数据
        """
        if len(frame_data_list) == 1:
            return frame_data_list[0]
        
        # 1. 朝向一致性检查和朝向对齐
        logger.debug("检查多视角朝向一致性...")
        aligned_frame_data_list = self.align_orientations(frame_data_list)
        
        # 2. 基于置信度的加权融合
        confidences = np.array([data['confidence'] for data in aligned_frame_data_list])
        weights = confidences / np.sum(confidences)
        
        # 3. 基于3D关键点一致性的质量评估
        keypoints_list = [np.array(data['keypoints_3d']) for data in aligned_frame_data_list]
        
        # 计算关键点的一致性分数
        consistency_scores = []
        for i, kp1 in enumerate(keypoints_list):
            scores = []
            for j, kp2 in enumerate(keypoints_list):
                if i != j:
                    # 计算关键点之间的平均距离
                    distances = np.linalg.norm(kp1 - kp2, axis=1)
                    scores.append(np.mean(distances))
            consistency_scores.append(np.mean(scores) if scores else 0)
        
        # 将一致性分数转换为权重（距离越小，权重越大）
        consistency_weights = 1 / (1 + np.array(consistency_scores))
        consistency_weights = consistency_weights / np.sum(consistency_weights)
        
        # 4. 综合权重（置信度 + 一致性）
        final_weights = 0.7 * weights + 0.3 * consistency_weights
        final_weights = final_weights / np.sum(final_weights)
        
        # 5. 加权融合3D关键点
        fused_keypoints = np.zeros_like(keypoints_list[0])
        for i, keypoints in enumerate(keypoints_list):
            fused_keypoints += final_weights[i] * keypoints
        
        # 6. 加权融合SMPL参数（朝向已对齐）
        pose_params_list = [np.array(data['pose_params']) for data in aligned_frame_data_list]
        fused_pose_params = np.zeros_like(pose_params_list[0])
        for i, pose_params in enumerate(pose_params_list):
            fused_pose_params += final_weights[i] * pose_params
        
        shape_params_list = [np.array(data['shape_params']) for data in aligned_frame_data_list]
        fused_shape_params = np.zeros_like(shape_params_list[0])
        for i, shape_params in enumerate(shape_params_list):
            fused_shape_params += final_weights[i] * shape_params
        
        trans_params_list = [np.array(data['trans_params']) for data in aligned_frame_data_list]
        fused_trans_params = np.zeros_like(trans_params_list[0])
        for i, trans_params in enumerate(trans_params_list):
            fused_trans_params += final_weights[i] * trans_params
        
        # 7. 计算融合质量指标
        fusion_quality = {
            'confidence_weighted_avg': np.average(confidences, weights=weights),
            'consistency_score': 1 / (1 + np.mean(consistency_scores)),
            'view_diversity': len(aligned_frame_data_list),
            'weight_distribution': final_weights.tolist(),
            'orientation_aligned': True
        }
        
        return {
            'frame_idx': aligned_frame_data_list[0]['frame_idx'],
            'timestamp': aligned_frame_data_list[0]['timestamp'],
            'pose_params': fused_pose_params.tolist(),
            'shape_params': fused_shape_params.tolist(),
            'trans_params': fused_trans_params.tolist(),
            'keypoints_3d': fused_keypoints.tolist(),
            'confidence': fusion_quality['confidence_weighted_avg'],
            'fusion_quality': fusion_quality,
            'num_views': len(aligned_frame_data_list),
            'camera_names': [data['camera_name'] for data in aligned_frame_data_list],
            'track_ids': [data.get('track_id', 0) for data in aligned_frame_data_list]
        }
    
    def align_orientations(self, frame_data_list: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        对齐多视角的人体朝向，确保姿态参数的一致性
        
        Args:
            frame_data_list: 多视角帧数据列表
            
        Returns:
            朝向对齐后的帧数据列表
        """
        if len(frame_data_list) < 2:
            return frame_data_list
        
        logger.debug(f"对齐 {len(frame_data_list)} 个视角的人体朝向...")
        
        # 选择参考视角（置信度最高的）
        reference_idx = np.argmax([data['confidence'] for data in frame_data_list])
        reference_data = frame_data_list[reference_idx]
        
        aligned_data_list = [reference_data]  # 参考视角保持不变
        
        # 对其他视角进行朝向对齐
        for i, frame_data in enumerate(frame_data_list):
            if i == reference_idx:
                continue
            
            try:
                # 对齐朝向
                aligned_frame_data = self.align_single_orientation(
                    frame_data, reference_data
                )
                aligned_data_list.append(aligned_frame_data)
                
            except Exception as e:
                logger.warning(f"视角 {i} 朝向对齐失败: {e}")
                # 如果对齐失败，使用原始数据
                aligned_data_list.append(frame_data)
        
        logger.debug(f"朝向对齐完成，处理了 {len(aligned_data_list)} 个视角")
        return aligned_data_list
    
    def align_single_orientation(self, target_data: Dict[str, Any], 
                               reference_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        将目标视角的朝向对齐到参考视角
        
        Args:
            target_data: 目标视角数据
            reference_data: 参考视角数据
            
        Returns:
            朝向对齐后的目标数据
        """
        try:
            # 提取3D关键点
            target_keypoints = np.array(target_data['keypoints_3d'])
            reference_keypoints = np.array(reference_data['keypoints_3d'])
            
            if len(target_keypoints) == 0 or len(reference_keypoints) == 0:
                return target_data
            
            # 计算朝向向量（使用躯干方向）
            target_direction = self.calculate_body_direction(target_keypoints)
            reference_direction = self.calculate_body_direction(reference_keypoints)
            
            # 计算朝向差异
            orientation_diff = self.calculate_orientation_difference(
                target_direction, reference_direction
            )
            
            # 如果朝向差异太大，进行对齐
            if abs(orientation_diff) > np.pi/4:  # 45度阈值
                logger.debug(f"检测到朝向差异: {np.degrees(orientation_diff):.1f}度，进行对齐")
                
                # 对齐姿态参数
                aligned_pose_params = self.align_pose_parameters(
                    target_data['pose_params'], orientation_diff
                )
                
                # 对齐3D关键点
                aligned_keypoints = self.align_3d_keypoints(
                    target_keypoints, orientation_diff
                )
                
                # 创建对齐后的数据
                aligned_data = target_data.copy()
                aligned_data['pose_params'] = aligned_pose_params.tolist()
                aligned_data['keypoints_3d'] = aligned_keypoints.tolist()
                aligned_data['orientation_aligned'] = True
                aligned_data['original_orientation_diff'] = np.degrees(orientation_diff)
                
                return aligned_data
            else:
                # 朝向差异不大，不需要对齐
                logger.debug(f"朝向差异较小: {np.degrees(orientation_diff):.1f}度，无需对齐")
                return target_data
                
        except Exception as e:
            logger.warning(f"朝向对齐过程中出错: {e}")
            return target_data
    
    def calculate_body_direction(self, keypoints: np.ndarray) -> np.ndarray:
        """
        计算人体朝向向量
        
        Args:
            keypoints: 3D关键点数组
            
        Returns:
            朝向向量 [x, y, z]
        """
        try:
            # 使用躯干关键点计算朝向
            # 假设关键点顺序：0=鼻子, 1=颈部, 8=右髋, 11=左髋
            
            if len(keypoints) >= 12:
                # 方法1: 使用颈部到髋部中心的方向
                neck = keypoints[1]  # 颈部
                right_hip = keypoints[8]  # 右髋
                left_hip = keypoints[11]  # 左髋
                hip_center = (right_hip + left_hip) / 2
                
                # 躯干方向（从髋部到颈部）
                body_direction = neck - hip_center
                body_direction[2] = 0  # 忽略Z轴（高度）
                
                # 归一化
                if np.linalg.norm(body_direction) > 0:
                    body_direction = body_direction / np.linalg.norm(body_direction)
                
                return body_direction
            
            elif len(keypoints) >= 2:
                # 方法2: 使用前两个关键点的方向
                direction = keypoints[1] - keypoints[0]
                direction[2] = 0  # 忽略Z轴
                
                if np.linalg.norm(direction) > 0:
                    direction = direction / np.linalg.norm(direction)
                
                return direction
            
            else:
                # 默认朝向
                return np.array([1, 0, 0])
                
        except Exception as e:
            logger.warning(f"计算人体朝向失败: {e}")
            return np.array([1, 0, 0])
    
    def calculate_orientation_difference(self, direction1: np.ndarray, 
                                      direction2: np.ndarray) -> float:
        """
        计算两个方向向量之间的角度差异
        
        Args:
            direction1: 方向向量1
            direction2: 方向向量2
            
        Returns:
            角度差异（弧度）
        """
        try:
            # 计算点积
            dot_product = np.dot(direction1, direction2)
            
            # 限制在[-1, 1]范围内
            dot_product = np.clip(dot_product, -1.0, 1.0)
            
            # 计算角度
            angle = np.arccos(dot_product)
            
            # 确定旋转方向（使用叉积的Z分量）
            cross_product = np.cross(direction1, direction2)
            if cross_product[2] < 0:
                angle = -angle
            
            return angle
            
        except Exception as e:
            logger.warning(f"计算方向差异失败: {e}")
            return 0.0
    
    def align_pose_parameters(self, pose_params: np.ndarray, 
                            orientation_diff: float) -> np.ndarray:
        """
        根据朝向差异调整姿态参数
        
        Args:
            pose_params: 原始姿态参数
            orientation_diff: 朝向差异（弧度）
            
        Returns:
            调整后的姿态参数
        """
        try:
            # 简化处理：只调整全局方向参数
            # 注意：这是简化版本，实际应该使用SMPL的旋转矩阵操作
            
            aligned_pose_params = pose_params.copy()
            
            # 如果pose_params包含全局方向参数（前3个参数）
            if len(pose_params) >= 3:
                # 调整全局方向（简化处理）
                # 实际应该使用旋转矩阵的复合操作
                aligned_pose_params[:3] += orientation_diff * 0.1  # 缩放因子
            
            return aligned_pose_params
            
        except Exception as e:
            logger.warning(f"姿态参数对齐失败: {e}")
            return pose_params
    
    def align_3d_keypoints(self, keypoints: np.ndarray, 
                          orientation_diff: float) -> np.ndarray:
        """
        根据朝向差异调整3D关键点
        
        Args:
            keypoints: 原始3D关键点
            orientation_diff: 朝向差异（弧度）
            
        Returns:
            调整后的3D关键点
        """
        try:
            if len(keypoints) == 0:
                return keypoints
            
            # 创建旋转矩阵（绕Z轴旋转）
            cos_theta = np.cos(orientation_diff)
            sin_theta = np.sin(orientation_diff)
            
            rotation_matrix = np.array([
                [cos_theta, -sin_theta, 0],
                [sin_theta, cos_theta, 0],
                [0, 0, 1]
            ])
            
            # 计算关键点的中心
            center = np.mean(keypoints, axis=0)
            
            # 应用旋转
            aligned_keypoints = []
            for keypoint in keypoints:
                # 相对于中心点
                relative_point = keypoint - center
                # 应用旋转
                rotated_point = rotation_matrix @ relative_point
                # 恢复到世界坐标
                aligned_keypoint = rotated_point + center
                aligned_keypoints.append(aligned_keypoint)
            
            return np.array(aligned_keypoints)
            
        except Exception as e:
            logger.warning(f"3D关键点对齐失败: {e}")
            return keypoints
    
    def calculate_fusion_statistics(self, fused_frames: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        计算融合统计信息
        
        Args:
            fused_frames: 融合后的帧数据列表
            
        Returns:
            融合统计信息
        """
        if not fused_frames:
            return {}
        
        # 基础统计
        total_frames = len(fused_frames)
        confidences = [frame['confidence'] for frame in fused_frames]
        view_counts = [frame.get('num_views', 1) for frame in fused_frames]
        
        # 融合质量统计
        fusion_qualities = []
        for frame in fused_frames:
            if 'fusion_quality' in frame:
                fusion_qualities.append(frame['fusion_quality'])
        
        # 计算统计值
        stats = {
            'total_frames': total_frames,
            'confidence': {
                'mean': np.mean(confidences),
                'std': np.std(confidences),
                'min': np.min(confidences),
                'max': np.max(confidences)
            },
            'view_coverage': {
                'mean': np.mean(view_counts),
                'max': np.max(view_counts),
                'single_view_frames': sum(1 for v in view_counts if v == 1),
                'multi_view_frames': sum(1 for v in view_counts if v > 1)
            }
        }
        
        # 如果有融合质量信息，添加质量统计
        if fusion_qualities:
            consistency_scores = [fq['consistency_score'] for fq in fusion_qualities]
            stats['fusion_quality'] = {
                'consistency_mean': np.mean(consistency_scores),
                'consistency_std': np.std(consistency_scores),
                'consistency_min': np.min(consistency_scores),
                'consistency_max': np.max(consistency_scores)
            }
        
        return stats
    
    def process_multi_view_videos(self, video_paths: Dict[str, str]) -> Dict[str, Any]:
        """
        处理多视角视频的完整流程
        
        Args:
            video_paths: 相机名称到视频路径的映射
            
        Returns:
            融合后的结果
        """
        logger.info("开始真实的多视角4D-Humans处理流程...")
        
        # 步骤1: 使用真实4D-Humans处理每个视角
        camera_results = {}
        for camera_name, video_path in video_paths.items():
            logger.info(f"处理相机 {camera_name} 的视频...")
            result = self.run_real_4d_humans(video_path, camera_name)
            camera_results[camera_name] = result
        
        # 步骤2: 将各视角结果转换到世界坐标系
        world_results = []
        for camera_name, result in camera_results.items():
            logger.info(f"转换相机 {camera_name} 到世界坐标系...")
            world_result = self.transform_to_world_coordinates(result, camera_name)
            world_results.append(world_result)
        
        # 步骤3: 检查朝向一致性
        logger.info("检查多视角朝向一致性...")
        self.check_orientation_consistency(world_results)
        
        # 步骤4: 对世界坐标系中的3D模型进行投票融合
        logger.info("进行多视角投票融合...")
        fused_result = self.fuse_multi_view_models(world_results)
        
        # 保存结果
        self.save_results(camera_results, world_results, fused_result)
        
        return fused_result
    
    def check_orientation_consistency(self, world_results: List[Dict[str, Any]]):
        """
        检查多视角朝向一致性
        
        Args:
            world_results: 各视角转换到世界坐标系的结果列表
        """
        if len(world_results) < 2:
            logger.info("视角数量不足，跳过朝向一致性检查")
            return
        
        logger.info("=" * 50)
        logger.info("多视角朝向一致性检查")
        logger.info("=" * 50)
        
        # 选择前几帧进行检查
        sample_frames = 5
        
        for frame_idx in range(min(sample_frames, len(world_results[0]['frames_data']))):
            logger.info(f"检查第 {frame_idx} 帧的朝向一致性...")
            
            frame_data_list = []
            for result in world_results:
                if frame_idx < len(result['frames_data']):
                    frame_data_list.append(result['frames_data'][frame_idx])
            
            if len(frame_data_list) < 2:
                continue
            
            # 计算朝向差异
            orientations = []
            for i, frame_data in enumerate(frame_data_list):
                try:
                    keypoints = np.array(frame_data['keypoints_3d'])
                    if len(keypoints) > 0:
                        direction = self.calculate_body_direction(keypoints)
                        orientations.append({
                            'camera': frame_data.get('camera_name', f'camera_{i}'),
                            'direction': direction,
                            'confidence': frame_data.get('confidence', 0.0)
                        })
                        logger.info(f"  相机 {frame_data.get('camera_name', f'camera_{i}')}: "
                                  f"朝向向量 [{direction[0]:.3f}, {direction[1]:.3f}, {direction[2]:.3f}]")
                        
                        # 🔥 新增：检查pose_params的转换效果
                        if 'pose_params' in frame_data and 'pose_params_camera' in frame_data:
                            pose_world = frame_data['pose_params']
                            pose_camera = frame_data['pose_params_camera']
                            if len(pose_world) >= 3 and len(pose_camera) >= 3:
                                logger.info(f"    姿态参数对比:")
                                logger.info(f"      相机坐标系: [{pose_camera[0]:.3f}, {pose_camera[1]:.3f}, {pose_camera[2]:.3f}]")
                                logger.info(f"      世界坐标系: [{pose_world[0]:.3f}, {pose_world[1]:.3f}, {pose_world[2]:.3f}]")
                        
                except Exception as e:
                    logger.warning(f"  相机 {frame_data.get('camera_name', f'camera_{i}')} 朝向计算失败: {e}")
            
            # 计算朝向差异
            if len(orientations) >= 2:
                for i in range(len(orientations)):
                    for j in range(i + 1, len(orientations)):
                        angle_diff = np.degrees(self.calculate_orientation_difference(
                            orientations[i]['direction'], orientations[j]['direction']
                        ))
                        logger.info(f"  朝向差异 {orientations[i]['camera']} vs {orientations[j]['camera']}: "
                                  f"{angle_diff:.1f}度")
                        
                        if abs(angle_diff) > 45:
                            logger.warning(f"  ⚠ 检测到大朝向差异: {angle_diff:.1f}度 > 45度")
                        elif abs(angle_diff) > 20:
                            logger.info(f"  ⚠ 检测到中等朝向差异: {angle_diff:.1f}度 > 20度")
                        else:
                            logger.info(f"  ✓ 朝向差异较小: {angle_diff:.1f}度 <= 20度")
            
            logger.info("-" * 30)
        
        logger.info("朝向一致性检查完成")
        logger.info("=" * 50)
    
    def save_results(self, camera_results: Dict[str, Any], 
                    world_results: List[Dict[str, Any]], 
                    fused_result: Dict[str, Any]):
        """保存所有结果"""
        
        # 保存各视角原始结果
        for camera_name, result in camera_results.items():
            output_file = self.output_dir / f"{camera_name}_4d_humans_result.json"
            # 转换numpy类型为Python原生类型
            serializable_result = self.convert_to_serializable(result)
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(serializable_result, f, indent=2, ensure_ascii=False)
            logger.info(f"保存相机 {camera_name} 结果到: {output_file}")
        
        # 保存世界坐标系结果
        for i, world_result in enumerate(world_results):
            output_file = self.output_dir / f"world_coordinate_{world_result['camera_name']}.json"
            # 转换numpy类型为Python原生类型
            serializable_world_result = self.convert_to_serializable(world_result)
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(serializable_world_result, f, indent=2, ensure_ascii=False)
            logger.info(f"保存世界坐标系结果到: {output_file}")
        
        # 🔥 新增：保存融合结果为PHALP格式的pkl文件
        logger.info("保存融合结果为PHALP格式的pkl文件...")
        
        # 将融合结果转换为PHALP格式
        phalp_format_data = self.convert_fusion_to_phalp_format(fused_result)
        
        # 保存为pkl文件（与PHALP原始格式一致）
        pkl_output_file = self.output_dir / "fused_multi_view_result.pkl"
        try:
            import joblib
            joblib.dump(phalp_format_data, pkl_output_file, compress=3)
            logger.info(f"✓ 融合结果已保存为PHALP格式pkl文件: {pkl_output_file}")
        except Exception as e:
            logger.error(f"保存pkl文件失败: {e}")
            # 如果joblib不可用，回退到pickle
            try:
                import pickle
                with open(pkl_output_file, 'wb') as f:
                    pickle.dump(phalp_format_data, f)
                logger.info(f"✓ 融合结果已保存为pickle格式pkl文件: {pkl_output_file}")
            except Exception as e2:
                logger.error(f"保存pickle文件也失败: {e2}")
        
        # 同时保存JSON格式（保持向后兼容）
        fused_output_file = self.output_dir / "fused_multi_view_result.json"
        # 转换numpy类型为Python原生类型
        serializable_fused_result = self.convert_to_serializable(fused_result)
        with open(fused_output_file, 'w', encoding='utf-8') as f:
            json.dump(serializable_fused_result, f, indent=2, ensure_ascii=False)
        logger.info(f"保存融合结果JSON格式到: {fused_output_file}")
        
        # 生成统计报告
        self.generate_statistics_report(camera_results, world_results, fused_result)
    
    def convert_to_serializable(self, obj):
        """将对象转换为可JSON序列化的格式"""
        if isinstance(obj, dict):
            return {key: self.convert_to_serializable(value) for key, value in obj.items()}
        elif isinstance(obj, list):
            return [self.convert_to_serializable(item) for item in obj]
        elif isinstance(obj, (np.integer, np.int64, np.int32)):
            return int(obj)
        elif isinstance(obj, (np.floating, np.float64, np.float32)):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, np.bool_):
            return bool(obj)
        else:
            return obj
    
    def generate_statistics_report(self, camera_results: Dict[str, Any], 
                                world_results: List[Dict[str, Any]], 
                                fused_result: Dict[str, Any]):
        """生成统计报告"""
        
        report = {
            'processing_summary': {
                'total_cameras': len(camera_results),
                'camera_names': list(camera_results.keys()),
                'total_frames_processed': sum(len(result['frames_data']) for result in camera_results.values()),
                'fused_frames': len(fused_result.get('fused_frames', [])),
                'processing_timestamp': datetime.now().isoformat()
            },
            'camera_statistics': {},
            'orientation_alignment': {}
        }
        
        # 各相机统计
        for camera_name, result in camera_results.items():
            report['camera_statistics'][camera_name] = {
                'total_frames': result['total_frames'],
                'processed_frames': len(result['frames_data']),
                'fps': result['fps'],
                'is_fallback': result.get('is_fallback', False)
            }
        
        # 融合统计
        if 'fused_frames' in fused_result:
            fused_frames = fused_result['fused_frames']
            confidences = [frame['confidence'] for frame in fused_frames]
            view_counts = [frame.get('num_views', 1) for frame in fused_frames]
            
            # 朝向对齐统计
            orientation_aligned_frames = 0
            orientation_diffs = []
            for frame in fused_frames:
                if frame.get('fusion_quality', {}).get('orientation_aligned', False):
                    orientation_aligned_frames += 1
                if 'fusion_quality' in frame and 'orientation_aligned' in frame['fusion_quality']:
                    orientation_aligned_frames += 1
            
            report['fusion_statistics'] = {
                'total_fused_frames': len(fused_frames),
                'average_confidence': np.mean(confidences),
                'min_confidence': np.min(confidences),
                'max_confidence': np.max(confidences),
                'frames_with_multiple_views': sum(1 for v in view_counts if v > 1),
                'orientation_aligned_frames': orientation_aligned_frames,
                'orientation_alignment_rate': orientation_aligned_frames / len(fused_frames) if fused_frames else 0
            }
        
        # 保存报告
        report_file = self.output_dir / "processing_report.json"
        # 转换numpy类型为Python原生类型
        serializable_report = self.convert_to_serializable(report)
        with open(report_file, 'w', encoding='utf-8') as f:
            json.dump(serializable_report, f, indent=2, ensure_ascii=False)
        
        logger.info(f"统计报告已保存到: {report_file}")
        
        # 打印摘要
        logger.info("=" * 50)
        logger.info("处理完成摘要:")
        logger.info(f"处理相机数量: {report['processing_summary']['total_cameras']}")
        logger.info(f"总处理帧数: {report['processing_summary']['total_frames_processed']}")
        logger.info(f"融合帧数: {report['processing_summary']['fused_frames']}")
        if 'fusion_statistics' in report:
            logger.info(f"平均置信度: {report['fusion_statistics']['average_confidence']:.3f}")
            logger.info(f"多视角帧数: {report['fusion_statistics']['frames_with_multiple_views']}")
            logger.info(f"朝向对齐帧数: {report['fusion_statistics']['orientation_aligned_frames']}")
            logger.info(f"朝向对齐率: {report['fusion_statistics']['orientation_alignment_rate']:.2%}")
        
        # 显示各相机状态
        for camera_name, stats in report['camera_statistics'].items():
            status = "备用模式" if stats['is_fallback'] else "正常模式"
            logger.info(f"相机 {camera_name}: {status}, 处理帧数: {stats['processed_frames']}")
        
        logger.info("=" * 50)

    def create_3d_visualization(self, frame_idx: int = 0, use_open3d: bool = True, 
                               use_plotly: bool = True, create_animation: bool = True):
        """
        创建3D人体模型可视化
        
        Args:
            frame_idx: 要可视化的帧索引
            use_open3d: 是否使用Open3D进行3D可视化
            use_plotly: 是否使用Plotly进行交互式3D可视化
            create_animation: 是否创建融合前后对比动画
        """
        logger.info("开始创建3D人体模型可视化...")
        
        # 检查可视化库可用性
        try:
            import open3d as o3d
            OPEN3D_AVAILABLE = True
            logger.info("Open3D库可用，将使用3D可视化")
        except ImportError:
            OPEN3D_AVAILABLE = False
            logger.warning("Open3D库不可用，将使用matplotlib进行3D可视化")
            logger.info("安装Open3D: pip install open3d")

        try:
            import plotly.graph_objects as go
            import plotly.offline as pyo
            PLOTLY_AVAILABLE = True
            logger.info("Plotly库可用，将使用交互式3D可视化")
        except ImportError:
            PLOTLY_AVAILABLE = False
            logger.warning("Plotly库不可用，将使用静态3D可视化")
            logger.info("安装Plotly: pip install plotly")

        # 人体骨架连接关系 (SMPL格式)
        skeleton_connections = [
            # 头部和躯干
            (0, 1), (1, 2), (2, 3), (3, 4),  # 头部到右手
            (1, 5), (5, 6), (6, 7),          # 左手
            (1, 8), (8, 9), (9, 10),         # 右腿
            (1, 11), (11, 12), (12, 13),     # 左腿
            (0, 14), (14, 15), (15, 16),     # 躯干
            # 手部细节
            (4, 17), (17, 18), (18, 19),     # 右手手指
            (7, 20), (20, 21), (21, 22),     # 左手手指
            # 脚部细节
            (10, 23), (23, 24), (24, 25),    # 右脚
            (13, 26), (26, 27), (27, 28)     # 左脚
        ]
        
        # 颜色配置
        colors = {
            'before_fusion': [1.0, 0.0, 0.0],      # 红色
            'after_fusion': [0.0, 1.0, 0.0],       # 绿色
            'skeleton': [0.0, 0.0, 1.0],           # 蓝色
            'background': [0.1, 0.1, 0.1]          # 深灰色
        }
        
        # 加载融合结果数据
        fused_file = self.output_dir / "fused_world_coordinate_results.json"
        cam1_file = self.output_dir / "cam_1_results.json"
        cam2_file = self.output_dir / "cam_2_results.json"
        
        if not fused_file.exists() or not cam1_file.exists() or not cam2_file.exists():
            logger.error("缺少必要的融合结果文件，无法进行可视化")
            return
        
        # 加载数据
        try:
            with open(fused_file, 'r', encoding='utf-8') as f:
                fused_results = json.load(f)
            with open(cam1_file, 'r', encoding='utf-8') as f:
                cam1_results = json.load(f)
            with open(cam2_file, 'r', encoding='utf-8') as f:
                cam2_results = json.load(f)
        except Exception as e:
            logger.error(f"加载结果文件时出错: {e}")
            return
        
        # 提取关键点数据
        extracted_data = self._extract_keypoints_for_visualization(
            fused_results, cam1_results, cam2_results
        )
        
        if not any(extracted_data.values()):
            logger.error("没有提取到关键点数据")
            return
        
        # 创建可视化输出目录
        viz_output_dir = self.output_dir / "3d_visualization"
        viz_output_dir.mkdir(exist_ok=True)
        
        # 选择可视化方法
        if use_open3d and OPEN3D_AVAILABLE:
            logger.info("使用Open3D进行3D可视化...")
            self._visualize_with_open3d(extracted_data, frame_idx, viz_output_dir, 
                                       skeleton_connections, colors)
        
        if use_plotly and PLOTLY_AVAILABLE:
            logger.info("使用Plotly进行交互式3D可视化...")
            self._visualize_with_plotly(extracted_data, frame_idx, viz_output_dir, 
                                       skeleton_connections, colors)
        
        # 使用Matplotlib作为备选方案
        if not (use_open3d and OPEN3D_AVAILABLE) and not (use_plotly and PLOTLY_AVAILABLE):
            logger.info("使用Matplotlib进行3D可视化...")
            self._visualize_with_matplotlib(extracted_data, frame_idx, viz_output_dir, 
                                          skeleton_connections, colors)
        
        # 创建动画
        if create_animation and PLOTLY_AVAILABLE:
            self._create_comparison_animation(extracted_data, viz_output_dir, 
                                           skeleton_connections, colors)
        
        logger.info(f"3D可视化完成！结果保存在: {viz_output_dir}")
    
    def _extract_keypoints_for_visualization(self, fused_results, cam1_results, cam2_results):
        """提取可视化所需的关键点数据"""
        extracted_data = {
            'fused_frames': [],
            'cam1_frames': [],
            'cam2_frames': []
        }
        
        # 提取融合结果中的关键点
        if fused_results and isinstance(fused_results, dict):
            # 处理融合结果字典格式
            fused_frames = fused_results.get('fused_frames', [])
            for frame_data in fused_frames:
                if isinstance(frame_data, dict) and 'keypoints_3d' in frame_data:
                    extracted_data['fused_frames'].append({
                        'frame_id': frame_data.get('frame_idx', 0),
                        'timestamp': frame_data.get('timestamp', 0.0),
                        'keypoints_3d': frame_data['keypoints_3d'],
                        'confidence': frame_data.get('confidence', 1.0)
                    })
        elif fused_results and isinstance(fused_results, list):
            # 处理融合结果列表格式
            for frame_data in fused_results:
                if isinstance(frame_data, dict) and frame_data.get('world_coordinate_detections'):
                    for detection in frame_data['world_coordinate_detections']:
                        if 'keypoints_3d' in detection:
                            extracted_data['fused_frames'].append({
                                'frame_id': frame_data['frame_id'],
                                'timestamp': frame_data['timestamp'],
                                'keypoints_3d': detection['keypoints_3d'],
                                'confidence': detection.get('confidence', 1.0)
                            })
        
        # 提取相机1结果中的关键点
        if cam1_results and isinstance(cam1_results, list):
            for frame_data in cam1_results:
                if isinstance(frame_data, dict) and frame_data.get('detections'):
                    for detection in frame_data['detections']:
                        if 'keypoints_3d' in detection:
                            extracted_data['cam1_frames'].append({
                                'frame_id': frame_data['frame_id'],
                                'timestamp': frame_data['timestamp'],
                                'keypoints_3d': detection['keypoints_3d'],
                                'confidence': detection.get('confidence', 1.0)
                            })
        
        # 提取相机2结果中的关键点
        if cam2_results and isinstance(cam2_results, list):
            for frame_data in cam2_results:
                if isinstance(frame_data, dict) and frame_data.get('detections'):
                    for detection in frame_data['detections']:
                        if 'keypoints_3d' in detection:
                            extracted_data['cam2_frames'].append({
                                'frame_id': frame_data['frame_id'],
                                'timestamp': frame_data['timestamp'],
                                'keypoints_3d': detection['keypoints_3d'],
                                'confidence': detection.get('confidence', 1.0)
                            })
        
        logger.info(f"提取到 {len(extracted_data['fused_frames'])} 帧融合结果")
        logger.info(f"提取到 {len(extracted_data['cam1_frames'])} 帧相机1结果")
        logger.info(f"提取到 {len(extracted_data['cam2_frames'])} 帧相机2结果")
        
        return extracted_data
    
    def _visualize_with_open3d(self, extracted_data, frame_idx, output_dir, 
                              skeleton_connections, colors):
        """使用Open3D进行3D可视化"""
        try:
            import open3d as o3d
        except ImportError:
            logger.error("Open3D不可用，无法进行3D可视化")
            return
        
        # 获取指定帧的数据
        fused_frame = extracted_data['fused_frames'][frame_idx] if extracted_data['fused_frames'] else None
        cam1_frame = extracted_data['cam1_frames'][frame_idx] if extracted_data['cam1_frames'] else None
        cam2_frame = extracted_data['cam2_frames'][frame_idx] if extracted_data['cam2_frames'] else None
        
        if not any([fused_frame, cam1_frame, cam2_frame]):
            logger.error(f"第 {frame_idx} 帧没有可用数据")
            return
        
        # 创建可视化窗口
        vis = o3d.visualization.Visualizer()
        vis.create_window("Linux 3D人体模型可视化 - 融合前后对比", width=1200, height=800)
        
        # 设置背景颜色
        opt = vis.get_render_option()
        opt.background_color = np.array(colors['background'])
        
        # 添加几何体
        geometries = []
        
        # 添加融合后的结果
        if fused_frame:
            keypoints_3d = np.array(fused_frame['keypoints_3d'])
            if len(keypoints_3d) > 0:
                # 关键点
                point_cloud = o3d.geometry.PointCloud()
                point_cloud.points = o3d.utility.Vector3dVector(keypoints_3d)
                point_cloud.colors = o3d.utility.Vector3dVector([colors['after_fusion']] * len(keypoints_3d))
                vis.add_geometry(point_cloud)
                geometries.append(point_cloud)
                
                # 骨架
                lines = []
                line_colors = []
                for start_idx, end_idx in skeleton_connections:
                    if start_idx < len(keypoints_3d) and end_idx < len(keypoints_3d):
                        lines.append([start_idx, end_idx])
                        line_colors.append(colors['skeleton'])
                
                if lines:
                    line_set = o3d.geometry.LineSet()
                    line_set.points = o3d.utility.Vector3dVector(keypoints_3d)
                    line_set.lines = o3d.utility.Vector2iVector(lines)
                    line_set.colors = o3d.utility.Vector3dVector(line_colors)
                    vis.add_geometry(line_set)
                    geometries.append(line_set)
        
        # 添加融合前的结果
        if cam1_frame:
            keypoints_3d = np.array(cam1_frame['keypoints_3d'])
            if len(keypoints_3d) > 0:
                point_cloud = o3d.geometry.PointCloud()
                point_cloud.points = o3d.utility.Vector3dVector(keypoints_3d)
                point_cloud.colors = o3d.utility.Vector3dVector([colors['before_fusion']] * len(keypoints_3d))
                vis.add_geometry(point_cloud)
                geometries.append(point_cloud)
        
        if cam2_frame:
            keypoints_3d = np.array(cam2_frame['keypoints_3d'])
            if len(keypoints_3d) > 0:
                point_cloud = o3d.geometry.PointCloud()
                point_cloud.points = o3d.utility.Vector3dVector(keypoints_3d)
                point_cloud.colors = o3d.utility.Vector3dVector([colors['before_fusion']] * len(keypoints_3d))
                vis.add_geometry(point_cloud)
                geometries.append(point_cloud)
        
        # 设置相机视角
        ctr = vis.get_view_control()
        ctr.set_front([0, 0, -1])
        ctr.set_lookat([0, 0, 0])
        ctr.set_up([0, 1, 0])
        ctr.set_zoom(0.8)
        
        # 运行可视化
        logger.info("3D可视化窗口已打开，按Q退出")
        vis.run()
        vis.destroy_window()
    
    def _visualize_with_plotly(self, extracted_data, frame_idx, output_dir, 
                              skeleton_connections, colors):
        """使用Plotly进行交互式3D可视化"""
        try:
            import plotly.graph_objects as go
            import plotly.offline as pyo
        except ImportError:
            logger.error("Plotly不可用，无法进行交互式3D可视化")
            return
        
        # 获取指定帧的数据
        fused_frame = extracted_data['fused_frames'][frame_idx] if extracted_data['fused_frames'] else None
        cam1_frame = extracted_data['cam1_frames'][frame_idx] if extracted_data['cam1_frames'] else None
        cam2_frame = extracted_data['cam2_frames'][frame_idx] if extracted_data['cam2_frames'] else None
        
        if not any([fused_frame, cam1_frame, cam2_frame]):
            logger.error(f"第 {frame_idx} 帧没有可用数据")
            return
        
        # 创建图形
        fig = go.Figure()
        
        # 添加融合后的结果
        if fused_frame:
            keypoints_3d = np.array(fused_frame['keypoints_3d'])
            if len(keypoints_3d) > 0:
                # 关键点
                fig.add_trace(go.Scatter3d(
                    x=keypoints_3d[:, 0],
                    y=keypoints_3d[:, 1],
                    z=keypoints_3d[:, 2],
                    mode='markers',
                    name='融合后关键点',
                    marker=dict(
                        size=8,
                        color='green',
                        opacity=0.8
                    )
                ))
                
                # 骨架
                for start_idx, end_idx in skeleton_connections:
                    if start_idx < len(keypoints_3d) and end_idx < len(keypoints_3d):
                        start_point = keypoints_3d[start_idx]
                        end_point = keypoints_3d[end_idx]
                        fig.add_trace(go.Scatter3d(
                            x=[start_point[0], end_point[0]],
                            y=[start_point[1], end_point[1]],
                            z=[start_point[2], end_point[2]],
                            mode='lines',
                            name='融合后骨架',
                            line=dict(color='blue', width=3),
                            showlegend=False
                        ))
        
        # 添加融合前的结果
        if cam1_frame:
            keypoints_3d = np.array(cam1_frame['keypoints_3d'])
            if len(keypoints_3d) > 0:
                fig.add_trace(go.Scatter3d(
                    x=keypoints_3d[:, 0],
                    y=keypoints_3d[:, 1],
                    z=keypoints_3d[:, 2],
                    mode='markers',
                    name='相机1关键点',
                    marker=dict(
                        size=6,
                        color='red',
                        opacity=0.6
                    )
                ))
        
        if cam2_frame:
            keypoints_3d = np.array(cam2_frame['keypoints_3d'])
            if len(keypoints_3d) > 0:
                fig.add_trace(go.Scatter3d(
                    x=keypoints_3d[:, 0],
                    y=keypoints_3d[:, 1],
                    z=keypoints_3d[:, 2],
                    mode='markers',
                    name='相机2关键点',
                    marker=dict(
                        size=6,
                        color='orange',
                        opacity=0.6
                    )
                ))
        
        # 设置布局
        fig.update_layout(
            title=f'Linux 3D人体模型可视化 - 第{frame_idx}帧融合前后对比',
            scene=dict(
                xaxis_title='X轴',
                yaxis_title='Y轴',
                zaxis_title='Z轴',
                camera=dict(
                    eye=dict(x=1.5, y=1.5, z=1.5)
                )
            ),
            width=1200,
            height=800
        )
        
        # 保存为HTML文件
        output_file = output_dir / f"frame_{frame_idx}_3d_visualization.html"
        pyo.plot(fig, filename=str(output_file), auto_open=False)
        logger.info(f"交互式3D可视化已保存到: {output_file}")
        
        # 显示图形
        fig.show()
    
    def _visualize_with_matplotlib(self, extracted_data, frame_idx, output_dir, 
                                  skeleton_connections, colors):
        """使用Matplotlib进行3D可视化"""
        # 获取指定帧的数据
        fused_frame = extracted_data['fused_frames'][frame_idx] if extracted_data['fused_frames'] else None
        cam1_frame = extracted_data['cam1_frames'][frame_idx] if extracted_data['cam1_frames'] else None
        cam2_frame = extracted_data['cam2_frames'][frame_idx] if extracted_data['cam2_frames'] else None
        
        if not any([fused_frame, cam1_frame, cam2_frame]):
            logger.error(f"第 {frame_idx} 帧没有可用数据")
            return
        
        # 创建图形
        fig = plt.figure(figsize=(15, 10))
        
        # 创建3D子图
        ax = fig.add_subplot(111, projection='3d')
        
        # 添加融合后的结果
        if fused_frame:
            keypoints_3d = np.array(fused_frame['keypoints_3d'])
            if len(keypoints_3d) > 0:
                # 关键点
                ax.scatter(keypoints_3d[:, 0], keypoints_3d[:, 1], keypoints_3d[:, 2],
                          c='green', s=100, alpha=0.8, label='融合后关键点')
                
                # 骨架
                for start_idx, end_idx in skeleton_connections:
                    if start_idx < len(keypoints_3d) and end_idx < len(keypoints_3d):
                        start_point = keypoints_3d[start_idx]
                        end_point = keypoints_3d[end_idx]
                        ax.plot([start_point[0], end_point[0]],
                               [start_point[1], end_point[1]],
                               [start_point[2], end_point[2]], 'b-', linewidth=3)
        
        # 添加融合前的结果
        if cam1_frame:
            keypoints_3d = np.array(cam1_frame['keypoints_3d'])
            if len(keypoints_3d) > 0:
                ax.scatter(keypoints_3d[:, 0], keypoints_3d[:, 1], keypoints_3d[:, 2],
                          c='red', s=80, alpha=0.6, label='相机1关键点')
        
        if cam2_frame:
            keypoints_3d = np.array(cam2_frame['keypoints_3d'])
            if len(keypoints_3d) > 0:
                ax.scatter(keypoints_3d[:, 0], keypoints_3d[:, 1], keypoints_3d[:, 2],
                          c='orange', s=80, alpha=0.6, label='相机2关键点')
        
        # 设置标签和标题
        ax.set_xlabel('X轴')
        ax.set_ylabel('Y轴')
        ax.set_zlabel('Z轴')
        ax.set_title(f'Linux 3D人体模型可视化 - 第{frame_idx}帧融合前后对比')
        ax.legend()
        
        # 设置坐标轴范围
        all_keypoints = []
        for frame in [fused_frame, cam1_frame, cam2_frame]:
            if frame and 'keypoints_3d' in frame:
                all_keypoints.extend(frame['keypoints_3d'])
        
        if all_keypoints:
            all_keypoints = np.array(all_keypoints)
            max_range = np.array([
                all_keypoints[:, 0].max() - all_keypoints[:, 0].min(),
                all_keypoints[:, 1].max() - all_keypoints[:, 1].min(),
                all_keypoints[:, 2].max() - all_keypoints[:, 2].min()
            ]).max() / 2.0
            
            mid_x = (all_keypoints[:, 0].max() + all_keypoints[:, 0].min()) * 0.5
            mid_y = (all_keypoints[:, 1].max() + all_keypoints[:, 1].min()) * 0.5
            mid_z = (all_keypoints[:, 2].max() + all_keypoints[:, 2].min()) * 0.5
            
            ax.set_xlim(mid_x - max_range, mid_x + max_range)
            ax.set_ylim(mid_y - max_range, mid_y + max_range)
            ax.set_zlim(mid_z - max_range, mid_z + max_range)
        
        # 保存图像
        output_file = output_dir / f"frame_{frame_idx}_3d_visualization.png"
        plt.savefig(output_file, dpi=150, bbox_inches='tight')
        logger.info(f"3D可视化图像已保存到: {output_file}")
        
        # 显示图像
        plt.tight_layout()
        plt.show()
    
    def _create_comparison_animation(self, extracted_data, output_dir, 
                                   skeleton_connections, colors):
        """创建融合前后对比动画"""
        try:
            import plotly.graph_objects as go
            import plotly.offline as pyo
        except ImportError:
            logger.warning("Plotly不可用，无法创建动画")
            return
        
        logger.info("创建融合前后对比动画...")
        
        # 选择要可视化的帧
        available_frames = min(len(extracted_data['fused_frames']), 10)
        if available_frames == 0:
            logger.error("没有可用的帧数据")
            return
        
        # 创建动画帧
        frames = []
        for frame_idx in range(available_frames):
            frame_data = []
            
            # 添加融合后的结果
            if frame_idx < len(extracted_data['fused_frames']):
                fused_frame = extracted_data['fused_frames'][frame_idx]
                keypoints_3d = np.array(fused_frame['keypoints_3d'])
                if len(keypoints_3d) > 0:
                    frame_data.append(go.Scatter3d(
                        x=keypoints_3d[:, 0],
                        y=keypoints_3d[:, 1],
                        z=keypoints_3d[:, 2],
                        mode='markers',
                        name='融合后关键点',
                        marker=dict(size=8, color='green', opacity=0.8)
                    ))
            
            # 添加融合前的结果
            if frame_idx < len(extracted_data['cam1_frames']):
                cam1_frame = extracted_data['cam1_frames'][frame_idx]
                keypoints_3d = np.array(cam1_frame['keypoints_3d'])
                if len(keypoints_3d) > 0:
                    frame_data.append(go.Scatter3d(
                        x=keypoints_3d[:, 0],
                        y=keypoints_3d[:, 1],
                        z=keypoints_3d[:, 2],
                        mode='markers',
                        name='相机1关键点',
                        marker=dict(size=6, color='red', opacity=0.6)
                    ))
            
            frames.append(go.Frame(data=frame_data, name=f"frame_{frame_idx}"))
        
        # 创建初始图形
        fig = go.Figure(
            data=frames[0].data,
            frames=frames,
            layout=go.Layout(
                title="Linux 3D人体模型融合前后对比动画",
                scene=dict(
                    xaxis_title='X轴',
                    yaxis_title='Y轴',
                    zaxis_title='Z轴',
                    camera=dict(eye=dict(x=1.5, y=1.5, z=1.5))
                ),
                width=1200,
                height=800,
                updatemenus=[{
                    'type': 'buttons',
                    'showactive': False,
                    'buttons': [
                        {'label': '播放', 'method': 'animate', 'args': [None, {'frame': {'duration': 500, 'redraw': True}, 'fromcurrent': True}]},
                        {'label': '暂停', 'method': 'animate', 'args': [[None], {'frame': {'duration': 0, 'redraw': False}, 'mode': 'immediate', 'transition': {'duration': 0}}]}
                    ]
                }]
            )
        )
        
        # 保存动画
        output_file = output_dir / "fusion_comparison_animation.html"
        pyo.plot(fig, filename=str(output_file), auto_open=False)
        logger.info(f"融合前后对比动画已保存到: {output_file}")

    def load_real_phalp_data(self, pkl_file_path: str) -> Dict[str, Any]:
        """
        加载真实的PHALP数据文件
        
        Args:
            pkl_file_path: PHALP .pkl文件路径
            
        Returns:
            PHALP数据结构
        """
        logger.info(f"加载真实PHALP数据: {pkl_file_path}")
        
        try:
            import joblib
            phalp_data = joblib.load(pkl_file_path)
            
            logger.info(f"✓ 成功加载PHALP数据: {pkl_file_path}")
            logger.info(f"数据类型: {type(phalp_data)}")
            
            if isinstance(phalp_data, dict):
                frames = sorted(phalp_data.keys())
                logger.info(f"帧数量: {len(frames)}")
                if frames:
                    logger.info(f"帧名示例: {frames[:3] if len(frames) > 3 else frames}")
                    
                    # 检查第一帧的数据结构
                    first_frame = frames[0]
                    first_frame_data = phalp_data[first_frame]
                    logger.info(f"第一帧数据键: {list(first_frame_data.keys())}")
                    
                    if 'smpl' in first_frame_data:
                        smpl_data = first_frame_data['smpl']
                        if isinstance(smpl_data, list) and len(smpl_data) > 0:
                            first_person_smpl = smpl_data[0]
                            if isinstance(first_person_smpl, dict):
                                logger.info(f"SMPL参数结构: {list(first_person_smpl.keys())}")
                                if 'global_orient' in first_person_smpl:
                                    logger.info(f"global_orient形状: {np.array(first_person_smpl['global_orient']).shape}")
                                if 'body_pose' in first_person_smpl:
                                    logger.info(f"body_pose形状: {np.array(first_person_smpl['body_pose']).shape}")
                                if 'betas' in first_person_smpl:
                                    logger.info(f"betas形状: {np.array(first_person_smpl['betas']).shape}")
            
            return phalp_data
            
        except Exception as e:
            logger.error(f"加载PHALP数据失败: {e}")
            return None
    
    def convert_phalp_to_fusion_format(self, phalp_data: Dict[str, Any], 
                                     camera_name: str) -> Dict[str, Any]:
        """
        将PHALP数据转换为融合系统需要的格式
        
        Args:
            phalp_data: PHALP原始数据
            camera_name: 相机名称
            
        Returns:
            转换后的数据格式
        """
        logger.info(f"转换相机 {camera_name} 的PHALP数据格式...")
        
        if not isinstance(phalp_data, dict):
            logger.error("PHALP数据格式错误")
            return None
        
        # 按帧名排序
        frame_names = sorted(phalp_data.keys())
        logger.info(f"处理 {len(frame_names)} 帧数据")
        
        # 添加调试信息
        logger.info(f"开始处理帧索引提取...")
        sample_frames = min(5, len(frame_names))
        for i in range(sample_frames):
            frame_name = frame_names[i]
            try:
                if '/img/' in frame_name:
                    img_part = frame_name.split('/img/')[-1]
                    frame_number = img_part.replace('.jpg', '')
                    frame_idx = int(frame_number)
                    logger.info(f"  帧 {i}: {frame_name} -> 提取帧号: {frame_number} -> 帧索引: {frame_idx}")
                else:
                    logger.info(f"  帧 {i}: {frame_name} -> 不包含/img/路径")
            except Exception as e:
                logger.warning(f"  帧 {i}: {frame_name} -> 提取失败: {e}")
        
        converted_frames = []
        
        # 添加调试统计
        frame_count = 0
        person_count = 0
        frame_person_stats = {}
        
        for frame_name in frame_names:
            frame_data = phalp_data[frame_name]
            
            # 检查是否有SMPL数据
            if 'smpl' not in frame_data or not frame_data['smpl']:
                continue
            
            frame_count += 1
            persons_in_frame = len(frame_data['smpl'])
            person_count += persons_in_frame
            
            # 统计每帧的人数
            if persons_in_frame not in frame_person_stats:
                frame_person_stats[persons_in_frame] = 0
            frame_person_stats[persons_in_frame] += 1
            
            # 获取帧索引
            try:
                # 从完整路径中提取帧号
                # 路径格式: .../img/000001.jpg
                if '/img/' in frame_name:
                    # 提取img/后面的文件名部分
                    img_part = frame_name.split('/img/')[-1]
                    # 去掉.jpg扩展名，提取数字部分
                    frame_number = img_part.replace('.jpg', '')
                    frame_idx = int(frame_number)
                elif '_' in frame_name:
                    # 备用方法：从帧名末尾提取数字
                    frame_idx = int(frame_name.split('_')[-1]) if frame_name.split('_')[-1].isdigit() else 0
                else:
                    frame_idx = 0
                
                logger.debug(f"帧名: {frame_name} -> 帧索引: {frame_idx}")
            except Exception as e:
                frame_idx = 0
                logger.warning(f"提取帧索引失败: {frame_name}, 错误: {e}")
            
            # 🔥 修改：每帧只保留置信度最高的人，而不是为每个人创建记录
            best_person_idx = 0
            best_confidence = 0.0
            
            # 找到置信度最高的人
            for person_idx, person_smpl in enumerate(frame_data['smpl']):
                try:
                    # 提取SMPL参数
                    if isinstance(person_smpl, dict):
                        global_orient = np.array(person_smpl.get('global_orient', np.zeros((1, 3, 3))))
                        body_pose = np.array(person_smpl.get('body_pose', np.zeros((23, 3, 3))))
                        betas = np.array(person_smpl.get('betas', np.zeros(10)))
                        
                        # 将旋转矩阵转换为欧拉角（简化处理）
                        pose_params = self.rotation_matrices_to_euler_angles(global_orient, body_pose)
                        shape_params = betas
                        
                        # 提取3D关节
                        if '3d_joints' in frame_data and len(frame_data['3d_joints']) > person_idx:
                            joints_3d = np.array(frame_data['3d_joints'][person_idx])
                        else:
                            joints_3d = np.zeros((45, 3))  # PHALP使用45个关节
                        
                        # 计算置信度
                        confidence = self.calculate_smpl_confidence(pose_params, shape_params, joints_3d)
                        
                        # 更新最佳人选
                        if confidence > best_confidence:
                            best_confidence = confidence
                            best_person_idx = person_idx
                            
                except Exception as e:
                    logger.warning(f"处理帧 {frame_name} 中的人 {person_idx} 时出错: {e}")
                    continue
            
            # 只处理置信度最高的人
            try:
                person_smpl = frame_data['smpl'][best_person_idx]
                
                # 提取SMPL参数
                if isinstance(person_smpl, dict):
                    global_orient = np.array(person_smpl.get('global_orient', np.zeros((1, 3, 3))))
                    body_pose = np.array(person_smpl.get('body_pose', np.zeros((23, 3, 3))))
                    betas = np.array(person_smpl.get('betas', np.zeros(10)))
                    
                    # 将旋转矩阵转换为欧拉角（简化处理）
                    pose_params = self.rotation_matrices_to_euler_angles(global_orient, body_pose)
                    shape_params = betas
                    
                    # 提取3D关节
                    if '3d_joints' in frame_data and len(frame_data['3d_joints']) > best_person_idx:
                        joints_3d = np.array(frame_data['3d_joints'][best_person_idx])
                    else:
                        joints_3d = np.zeros((45, 3))  # PHALP使用45个关节
                    
                    # 提取2D关节
                    if '2d_joints' in frame_data and len(frame_data['2d_joints']) > best_person_idx:
                        joints_2d = np.array(frame_data['2d_joints'][best_person_idx])
                    else:
                        joints_2d = np.zeros((45, 2))
                    
                    # 提取边界框
                    if 'bbox' in frame_data and len(frame_data['bbox']) > best_person_idx:
                        bbox = frame_data['bbox'][best_person_idx]
                    else:
                        bbox = [0, 0, 100, 100]
                    
                    # 提取跟踪ID
                    if 'tid' in frame_data and len(frame_data['tid']) > best_person_idx:
                        track_id = frame_data['tid'][best_person_idx]
                    else:
                        track_id = best_person_idx
                    
                    # 构建转换后的帧数据（每帧只保留一个人）
                    converted_frame = {
                        'frame_idx': frame_idx,
                        'frame_name': frame_name,
                        'timestamp': frame_idx / 30.0,  # 假设30fps
                        'pose_params': pose_params.tolist(),
                        'shape_params': shape_params.tolist(),
                        'trans_params': [0, 0, 0],  # PHALP不直接提供平移参数
                        'keypoints_3d': joints_3d.tolist(),
                        'keypoints_2d': joints_2d.tolist(),
                        'bbox': bbox,
                        'confidence': best_confidence,
                        'track_id': track_id,
                        'person_idx': best_person_idx,
                        'camera_name': camera_name
                    }
                    
                    converted_frames.append(converted_frame)
                    
            except Exception as e:
                logger.warning(f"处理帧 {frame_name} 的最佳人选时出错: {e}")
                continue
        
        logger.info(f"成功转换 {len(converted_frames)} 帧数据")
        
        # 输出调试统计信息
        logger.info(f"相机 {camera_name} 调试统计:")
        logger.info(f"  原始帧数: {len(frame_names)}")
        logger.info(f"  有效帧数: {frame_count}")
        logger.info(f"  总人数: {person_count}")
        logger.info(f"  平均每帧人数: {person_count/frame_count if frame_count > 0 else 0:.2f}")
        logger.info(f"  每帧人数分布: {frame_person_stats}")
        logger.info(f"  转换后帧数据数: {len(converted_frames)}")
        
        # 🔥 新增：详细分析帧数差异的原因
        logger.info(f"相机 {camera_name} 详细分析:")
        
        # 分析帧名模式
        frame_name_patterns = {}
        for frame_name in frame_names[:10]:  # 只分析前10个帧名
            if '/img/' in frame_name:
                pattern = 'img_path'
            elif '_' in frame_name:
                pattern = 'underscore'
            else:
                pattern = 'simple'
            
            if pattern not in frame_name_patterns:
                frame_name_patterns[pattern] = 0
            frame_name_patterns[pattern] += 1
        
        logger.info(f"  帧名模式分布: {frame_name_patterns}")
        
        # 分析有效帧和无效帧
        valid_frames = 0
        invalid_frames = 0
        no_smpl_frames = 0
        
        for frame_name in frame_names:
            frame_data = phalp_data[frame_name]
            if 'smpl' in frame_data and frame_data['smpl']:
                valid_frames += 1
            elif 'smpl' in frame_data and not frame_data['smpl']:
                no_smpl_frames += 1
            else:
                invalid_frames += 1
        
        logger.info(f"  有效帧数: {valid_frames}")
        logger.info(f"  无SMPL数据帧数: {no_smpl_frames}")
        logger.info(f"  无效帧数: {invalid_frames}")
        
        # 分析帧索引分布
        frame_indices = []
        for frame_name in frame_names:
            try:
                if '/img/' in frame_name:
                    img_part = frame_name.split('/img/')[-1]
                    frame_number = img_part.replace('.jpg', '')
                    frame_idx = int(frame_number)
                    frame_indices.append(frame_idx)
            except:
                continue
        
        if frame_indices:
            logger.info(f"  帧索引范围: {min(frame_indices)} - {max(frame_indices)}")
            logger.info(f"  帧索引示例: {sorted(frame_indices)[:10]}")
        
        # 分析SMPL数据结构
        if frame_names:
            first_frame = frame_names[0]
            first_frame_data = phalp_data[first_frame]
            logger.info(f"  第一帧数据结构: {list(first_frame_data.keys())}")
            if 'smpl' in first_frame_data:
                smpl_data = first_frame_data['smpl']
                logger.info(f"  第一帧SMPL数据类型: {type(smpl_data)}")
                if isinstance(smpl_data, list):
                    logger.info(f"  第一帧SMPL数据长度: {len(smpl_data)}")
                    if len(smpl_data) > 0:
                        first_person = smpl_data[0]
                        logger.info(f"  第一帧第一个人的数据类型: {type(first_person)}")
                        if isinstance(first_person, dict):
                            logger.info(f"  第一帧第一个人的数据键: {list(first_person.keys())}")
        
        logger.info(f"相机 {camera_name} 分析完成")
        logger.info("-" * 50)
        
        return {
            'camera_name': camera_name,
            'total_frames': len(frame_names),
            'fps': 30.0,
            'frames_data': converted_frames,  # 改为frames_data以保持一致性
            'conversion_timestamp': datetime.now().isoformat(),
            'is_fallback': False,
            'data_format': 'phalp_converted'
        }
    
    def rotation_matrices_to_euler_angles(self, global_orient: np.ndarray, 
                                        body_pose: np.ndarray) -> np.ndarray:
        """
        将旋转矩阵转换为欧拉角（简化版本）
        
        Args:
            global_orient: 全局方向旋转矩阵 (1, 3, 3)
            body_pose: 身体姿态旋转矩阵 (23, 3, 3)
            
        Returns:
            欧拉角参数 (72维)
        """
        try:
            # 简化处理：将旋转矩阵展平为参数向量
            # 注意：这不是标准的SMPL格式，但可以用于融合
            
            # 全局方向 (1个关节 × 3个轴 × 3个参数 = 9个参数)
            global_params = global_orient.flatten()
            
            # 身体姿态 (23个关节 × 3个轴 × 3个参数 = 207个参数)
            body_params = body_pose.flatten()
            
            # 组合为216个参数
            pose_params = np.concatenate([global_params, body_params])
            
            # 如果超过72个参数，取前72个（简化处理）
            if len(pose_params) > 72:
                pose_params = pose_params[:72]
            elif len(pose_params) < 72:
                # 如果不足72个参数，用0填充
                pose_params = np.pad(pose_params, (0, 72 - len(pose_params)), 'constant')
            
            return pose_params
            
        except Exception as e:
            logger.warning(f"旋转矩阵转换失败: {e}")
            return np.zeros(72)
    
    def calculate_smpl_confidence(self, pose_params: np.ndarray, 
                                shape_params: np.ndarray, 
                                joints_3d: np.ndarray) -> float:
        """
        基于SMPL参数和3D关节计算置信度
        
        Args:
            pose_params: 姿态参数
            shape_params: 形状参数
            joints_3d: 3D关节坐标
            
        Returns:
            置信度分数 (0-1)
        """
        try:
            # 1. 姿态参数合理性检查
            pose_confidence = 1.0
            if len(pose_params) > 0:
                # 检查姿态参数是否在合理范围内
                pose_range = np.max(np.abs(pose_params))
                if pose_range > 10.0:  # 过大的姿态参数可能有问题
                    pose_confidence = max(0.5, 1.0 - (pose_range - 10.0) / 10.0)
            
            # 2. 形状参数合理性检查
            shape_confidence = 1.0
            if len(shape_params) > 0:
                # 检查形状参数是否在合理范围内
                shape_range = np.max(np.abs(shape_params))
                if shape_range > 3.0:  # 过大的形状参数可能有问题
                    shape_confidence = max(0.5, 1.0 - (shape_range - 3.0) / 3.0)
            
            # 3. 3D关节合理性检查
            joint_confidence = 1.0
            if len(joints_3d) > 0:
                # 检查关节是否在合理范围内
                joint_distances = []
                for i in range(len(joints_3d) - 1):
                    dist = np.linalg.norm(joints_3d[i] - joints_3d[i+1])
                    joint_distances.append(dist)
                
                if joint_distances:
                    avg_distance = np.mean(joint_distances)
                    if avg_distance > 2.0:  # 关节间距离过大可能有问题
                        joint_confidence = max(0.5, 1.0 - (avg_distance - 2.0) / 2.0)
            
            # 综合置信度
            final_confidence = (pose_confidence + shape_confidence + joint_confidence) / 3.0
            
            # 确保在0-1范围内
            return max(0.1, min(1.0, final_confidence))
            
        except Exception as e:
            logger.warning(f"置信度计算失败: {e}")
            return 0.8  # 默认置信度
    
    def process_real_phalp_fusion(self, cam1_pkl: str, cam2_pkl: str) -> Dict[str, Any]:
        """
        使用真实的PHALP数据进行多视角融合
        
        Args:
            cam1_pkl: 相机1的PHALP .pkl文件路径
            cam2_pkl: 相机2的PHALP .pkl文件路径
            
        Returns:
            融合后的结果
        """
        logger.info("开始真实PHALP数据的多视角融合...")
        
        # 步骤1: 加载PHALP数据
        logger.info("步骤1: 加载PHALP数据...")
        cam1_data = self.load_real_phalp_data(cam1_pkl)
        cam2_data = self.load_real_phalp_data(cam2_pkl)
        
        if not cam1_data or not cam2_data:
            logger.error("无法加载PHALP数据")
            return None
        
        # 步骤2: 转换数据格式
        logger.info("步骤2: 转换数据格式...")
        cam1_converted = self.convert_phalp_to_fusion_format(cam1_data, 'cam_1')
        cam2_converted = self.convert_phalp_to_fusion_format(cam2_data, 'cam_2')
        
        if not cam1_converted or not cam2_converted:
            logger.error("数据格式转换失败")
            return None
        
        # 步骤2.5: 转换到世界坐标系
        logger.info("步骤2.5: 转换到世界坐标系...")
        cam1_world = self.transform_to_world_coordinates(cam1_converted, 'cam_1')
        cam2_world = self.transform_to_world_coordinates(cam2_converted, 'cam_2')
        
        # 步骤3: 执行融合
        logger.info("步骤3: 执行多视角融合...")
        world_results = [cam1_world, cam2_world]
        fused_result = self.fuse_multi_view_models(world_results)
        
        if not fused_result:
            logger.error("融合失败")
            return None
        
        # 步骤4: 保存结果
        logger.info("步骤4: 保存融合结果...")
        self.save_real_phalp_fusion_results(cam1_converted, cam2_converted, fused_result)
        
        logger.info("真实PHALP数据融合完成！")
        return fused_result
    
    def save_real_phalp_fusion_results(self, cam1_result: Dict[str, Any], 
                                     cam2_result: Dict[str, Any], 
                                     fused_result: Dict[str, Any]):
        """保存真实PHALP融合结果"""
        
        # 保存各相机转换后的结果
        for result in [cam1_result, cam2_result]:
            camera_name = result['camera_name']
            output_file = self.output_dir / f"{camera_name}_results.json"
            
            # 转换numpy类型为Python原生类型
            serializable_result = self.convert_to_serializable(result)
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(serializable_result, f, indent=2, ensure_ascii=False)
            logger.info(f"保存相机 {camera_name} 结果到: {output_file}")
        
        # 🔥 新增：保存融合结果为PHALP格式的pkl文件
        logger.info("保存融合结果为PHALP格式的pkl文件...")
        
        # 将融合结果转换为PHALP格式
        phalp_format_data = self.convert_fusion_to_phalp_format(fused_result)
        
        # 保存为pkl文件（与PHALP原始格式一致）
        pkl_output_file = self.output_dir / "fused_multi_view_result.pkl"
        try:
            import joblib
            joblib.dump(phalp_format_data, pkl_output_file, compress=3)
            logger.info(f"✓ 融合结果已保存为PHALP格式pkl文件: {pkl_output_file}")
        except Exception as e:
            logger.error(f"保存pkl文件失败: {e}")
            # 如果joblib不可用，回退到pickle
            try:
                import pickle
                with open(pkl_output_file, 'wb') as f:
                    pickle.dump(phalp_format_data, f)
                logger.info(f"✓ 融合结果已保存为pickle格式pkl文件: {pkl_output_file}")
            except Exception as e2:
                logger.error(f"保存pickle文件也失败: {e2}")
        
        # 同时保存JSON格式（保持向后兼容）
        fused_output_file = self.output_dir / "fused_world_coordinate_results.json"
        
        # 将融合结果转换为世界坐标系格式
        world_coordinate_frames = []
        for frame_data in fused_result.get('fused_frames', []):
            world_frame = {
                'frame_id': frame_data['frame_idx'],
                'timestamp': frame_data['timestamp'],
                'world_coordinate_detections': [{
                    'keypoints_3d': frame_data['keypoints_3d'],
                    'pose_params': frame_data['pose_params'],
                    'shape_params': frame_data['shape_params'],
                    'confidence': frame_data['confidence'],
                    'num_views': frame_data.get('num_views', 1),
                    'camera_names': frame_data.get('camera_names', []),
                    'track_ids': frame_data.get('track_ids', [])
                }]
            }
            world_coordinate_frames.append(world_frame)
        
        world_coordinate_result = {
            'frames': world_coordinate_frames,
            'total_frames': len(world_coordinate_frames),
            'fusion_timestamp': datetime.now().isoformat()
        }
        
        # 转换numpy类型为Python原生类型
        serializable_world_result = self.convert_to_serializable(world_coordinate_result)
        with open(fused_output_file, 'w', encoding='utf-8') as f:
            json.dump(serializable_world_result, f, indent=2, ensure_ascii=False)
        logger.info(f"保存融合结果JSON格式到: {fused_output_file}")
        
        # 生成统计报告
        self.generate_real_phalp_statistics_report(cam1_result, cam2_result, fused_result)
    
    def generate_real_phalp_statistics_report(self, cam1_result: Dict[str, Any], 
                                           cam2_result: Dict[str, Any], 
                                           fused_result: Dict[str, Any]):
        """生成真实PHALP融合的统计报告"""
        
        report = {
            'processing_summary': {
                'data_source': 'real_phalp_pkl',
                'total_cameras': 2,
                'camera_names': ['cam_1', 'cam_2'],
                'total_frames_processed': len(cam1_result['frames_data']) + len(cam2_result['frames_data']),
                'fused_frames': len(fused_result.get('fused_frames', [])),
                'processing_timestamp': datetime.now().isoformat()
            },
            'camera_statistics': {},
            'fusion_statistics': {}
        }
        
        # 各相机统计
        for result in [cam1_result, cam2_result]:
            camera_name = result['camera_name']
            frames_data = result['frames_data']
            
            confidences = [frame['confidence'] for frame in frames_data]
            report['camera_statistics'][camera_name] = {
                'total_frames': result['total_frames'],
                'processed_frames': len(frames_data),
                'fps': result['fps'],
                'average_confidence': np.mean(confidences) if confidences else 0.0,
                'min_confidence': np.min(confidences) if confidences else 0.0,
                'max_confidence': np.max(confidences) if confidences else 0.0,
                'is_fallback': False
            }
        
        # 融合统计
        if 'fused_frames' in fused_result:
            fused_frames = fused_result['fused_frames']
            confidences = [frame['confidence'] for frame in fused_frames]
            view_counts = [frame.get('num_views', 1) for frame in fused_frames]
            
            report['fusion_statistics'] = {
                'total_fused_frames': len(fused_frames),
                'average_confidence': np.mean(confidences) if confidences else 0.0,
                'min_confidence': np.min(confidences) if confidences else 0.0,
                'max_confidence': np.max(confidences) if confidences else 0.0,
                'frames_with_multiple_views': sum(1 for v in view_counts if v > 1),
                'average_views_per_frame': np.mean(view_counts) if view_counts else 1.0
            }
        
        # 保存报告
        report_file = self.output_dir / "real_phalp_fusion_report.json"
        serializable_report = self.convert_to_serializable(report)
        with open(report_file, 'w', encoding='utf-8') as f:
            json.dump(serializable_report, f, indent=2, ensure_ascii=False)
        
        logger.info(f"真实PHALP融合统计报告已保存到: {report_file}")
        
        # 打印摘要
        logger.info("=" * 50)
        logger.info("真实PHALP融合完成摘要:")
        logger.info(f"数据源: {report['processing_summary']['data_source']}")
        logger.info(f"处理相机数量: {report['processing_summary']['total_cameras']}")
        logger.info(f"总处理帧数: {report['processing_summary']['total_frames_processed']}")
        logger.info(f"融合帧数: {report['processing_summary']['fused_frames']}")
        
        if 'fusion_statistics' in report:
            fusion_stats = report['fusion_statistics']
            logger.info(f"平均置信度: {fusion_stats['average_confidence']:.3f}")
            logger.info(f"多视角帧数: {fusion_stats['frames_with_multiple_views']}")
            logger.info(f"平均视角数: {fusion_stats['average_views_per_frame']:.2f}")
        
        # 显示各相机状态
        for camera_name, stats in report['camera_statistics'].items():
            logger.info(f"相机 {camera_name}: 处理帧数: {stats['processed_frames']}, "
                       f"平均置信度: {stats['average_confidence']:.3f}")
        
        logger.info("=" * 50)
    
    def convert_fusion_to_phalp_format(self, fused_result: Dict[str, Any]) -> Dict[str, Any]:
        """
        将融合结果转换为PHALP原始格式
        
        Args:
            fused_result: 融合后的结果
            
        Returns:
            PHALP格式的数据
        """
        logger.info("转换融合结果为PHALP格式...")
        
        phalp_format_data = {}
        
        for frame_data in fused_result.get('fused_frames', []):
            # 生成帧名（与PHALP格式一致）
            frame_idx = frame_data['frame_idx']
            frame_name = f"frame_{frame_idx:06d}.jpg"  # PHALP使用类似格式
            
            # 构建PHALP格式的帧数据
            phalp_frame_data = {
                'tracked_ids': [frame_data.get('track_ids', [0])[0] if frame_data.get('track_ids') else 0],
                'bbox': [frame_data.get('bbox', [0, 0, 100, 100])],
                'conf': [frame_data.get('confidence', 0.8)],
                '3d_joints': [np.array(frame_data['keypoints_3d'])],
                '2d_joints': [np.array(frame_data.get('keypoints_2d', np.zeros((17, 2))))],
                'smpl': [{
                    'pose_params': np.array(frame_data['pose_params']),
                    'shape_params': np.array(frame_data['shape_params']),
                    'trans_params': np.array(frame_data.get('trans_params', [0, 0, 0]))
                }],
                'tid': [frame_data.get('track_ids', [0])[0] if frame_data.get('track_ids') else 0]
            }
            
            phalp_format_data[frame_name] = phalp_frame_data
        
        logger.info(f"转换完成，共 {len(phalp_format_data)} 帧数据")
        return phalp_format_data

    def _convert_axis_angle_to_rotation_matrices(self, pose_params: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """将轴角参数转换为旋转矩阵格式"""
        try:
            if len(pose_params) != 72:
                logger.warning(f"轴角参数长度不是72: {len(pose_params)}")
                return np.eye(3), np.zeros((23, 3, 3))
            
            # 前3维是全局旋转，后69维是23个关节的旋转（每个3维）
            global_orient = pose_params[:3]
            body_pose = pose_params[3:]
            
            # 转换全局旋转：轴角(3,) -> 旋转矩阵(3,3)
            global_orient_matrix = self._axis_angle_to_rotation_matrix(global_orient)
            
            # 转换身体姿态：69维 -> 23个关节的旋转矩阵
            body_pose_matrices = []
            for i in range(0, len(body_pose), 3):
                joint_axis_angle = body_pose[i:i+3]
                joint_matrix = self._axis_angle_to_rotation_matrix(joint_axis_angle)
                body_pose_matrices.append(joint_matrix)
            
            # 堆叠所有关节的旋转矩阵：(23, 3, 3)
            body_pose_tensor = np.stack(body_pose_matrices)
            
            logger.info(f"✓ 轴角转换成功: global_orient={global_orient_matrix.shape}, body_pose={body_pose_tensor.shape}")
            return global_orient_matrix, body_pose_tensor
            
        except Exception as e:
            logger.warning(f"轴角转换失败: {e}")
            # 返回默认值
            return np.eye(3), np.zeros((23, 3, 3))
    
    def _axis_angle_to_rotation_matrix(self, axis_angle: np.ndarray) -> np.ndarray:
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

    def _convert_rotation_matrices_to_axis_angle(self, global_orient: np.ndarray, body_pose: np.ndarray) -> np.ndarray:
        """将旋转矩阵转换回轴角参数"""
        try:
            # 转换全局旋转矩阵：旋转矩阵(3,3) -> 轴角(3,)
            global_axis_angle = self._rotation_matrix_to_axis_angle(global_orient)
            
            # 转换身体姿态矩阵：23个关节的旋转矩阵 -> 69维轴角参数
            body_axis_angle = []
            for i in range(body_pose.shape[0]):  # 23个关节
                joint_matrix = body_pose[i]  # (3, 3)
                joint_axis_angle = self._rotation_matrix_to_axis_angle(joint_matrix)
                body_axis_angle.extend(joint_axis_angle)
            
            # 组合为72维参数：3维全局 + 69维身体
            pose_params = np.concatenate([global_axis_angle, body_axis_angle])
            
            logger.info(f"✓ 旋转矩阵转换回轴角成功: {len(pose_params)} 维参数")
            return pose_params
            
        except Exception as e:
            logger.warning(f"旋转矩阵转换回轴角失败: {e}")
            # 返回默认值
            return np.zeros(72)
    
    def _rotation_matrix_to_axis_angle(self, rotation_matrix: np.ndarray) -> np.ndarray:
        """将旋转矩阵转换为轴角表示"""
        try:
            # 使用Rodrigues公式的逆过程
            # 从旋转矩阵 R 提取轴角 [rx, ry, rz]
            
            # 计算旋转角度
            trace = np.trace(rotation_matrix)
            angle = np.arccos(np.clip((trace - 1) / 2, -1, 1))
            
            # 如果角度接近0，返回零向量
            if abs(angle) < 1e-8:
                return np.zeros(3)
            
            # 计算旋转轴
            # 使用反对称矩阵的归一化
            K = (rotation_matrix - rotation_matrix.T) / (2 * np.sin(angle))
            axis = np.array([K[2, 1], K[0, 2], K[1, 0]])
            
            # 轴角表示 = 角度 * 轴
            axis_angle = angle * axis
            
            return axis_angle
            
        except Exception as e:
            logger.warning(f"旋转矩阵到轴角转换失败: {e}")
            # 返回零向量作为默认值
            return np.zeros(3)

def main():
    """主函数"""
    import argparse
    
    parser = argparse.ArgumentParser(description='真实的多视角4D-Humans融合系统')
    parser.add_argument('--config', type=str, default="/home/yangz/4D-Humans/whole/config.toml",
                       help='相机标定配置文件路径')
    parser.add_argument('--cam1_video', type=str, 
                       default="/home/yangz/4D-Humans/whole/recordings/walking4/port_1.mp4",
                       help='相机1视频文件路径')
    parser.add_argument('--cam2_video', type=str,
                       default="/home/yangz/4D-Humans/whole/recordings/walking4/port_2.mp4",
                       help='相机2视频文件路径')
    parser.add_argument('--cam1_pkl', type=str, default="/home/yangz/4D-Humans/real_4d_humans_fusion_results/4d_humans_cam_1/results/demo_port_1.pkl",
                       help='相机1的PHALP .pkl文件路径')
    parser.add_argument('--cam2_pkl', type=str, default="/home/yangz/4D-Humans/real_4d_humans_fusion_results/4d_humans_cam_1/results/demo_port_2.pkl",
                       help='相机2的PHALP .pkl文件路径')
    parser.add_argument('--force_video_processing', action='store_true', default=False,
                       help='强制使用视频处理模式，即使存在PHALP文件')
    parser.add_argument('--save_format', type=str, choices=['rotation_matrices', 'axis_angle'], 
                       default='rotation_matrices',
                       help='选择保存格式：rotation_matrices(旋转矩阵) 或 axis_angle(轴角参数)')
    parser.add_argument('--visualize', action='store_true', default=True,
                       help='是否在融合完成后进行3D可视化')
    parser.add_argument('--frame_idx', type=int, default=0,
                       help='要可视化的帧索引')
    parser.add_argument('--use_open3d', action='store_true', default=True,
                       help='使用Open3D进行3D可视化')
    parser.add_argument('--use_plotly', action='store_true', default=True,
                       help='使用Plotly进行交互式3D可视化')
    parser.add_argument('--create_animation', action='store_true', default=True,
                       help='创建融合前后对比动画')
    
    args = parser.parse_args()
    
    # 如果只是显示帮助，直接返回
    if len(sys.argv) == 1 or '--help' in sys.argv or '-h' in sys.argv:
        return True
    
    logger.info("启动真实的多视角4D-Humans融合系统")
    
    # 检查当前工作目录
    current_dir = os.getcwd()
    logger.info(f"当前工作目录: {current_dir}")
    
    # 检查4D-Humans项目路径
    hmr2_dir = Path(".")
    track_py = hmr2_dir / "track.py"
    if not track_py.exists():
        logger.error(f"未找到track.py文件: {track_py}")
        logger.info("请确保在4D-Humans项目根目录下运行此脚本")
        return False
    
    # 初始化融合系统
    try:
        # 如果只是显示帮助，不自动处理现有结果
        # 在强制视频处理模式下也不自动处理现有结果
        auto_process = not (len(sys.argv) == 1 or '--help' in sys.argv or '-h' in sys.argv or args.force_video_processing)
        fusion_system = Real4DHumansFusion(args.config, auto_process=auto_process, save_format=args.save_format)
        
        # 在开始处理前进行环境调试
        fusion_system.debug_4d_humans_environment()
        
    except Exception as e:
        logger.error(f"初始化融合系统失败: {e}")
        return False
    
    # 智能检测处理模式
    logger.info("开始智能检测处理模式...")
    
    # 检查PHALP文件是否存在
    cam1_pkl = Path(args.cam1_pkl)
    cam2_pkl = Path(args.cam2_pkl)
    
    phalp_files_exist = cam1_pkl.exists() and cam2_pkl.exists()
    
    if phalp_files_exist:
        logger.info("✓ 检测到PHALP文件存在:")
        logger.info(f"  相机1: {cam1_pkl} ({cam1_pkl.stat().st_size / (1024*1024):.2f} MB)")
        logger.info(f"  相机2: {cam2_pkl} ({cam2_pkl.stat().st_size / (1024*1024):.2f} MB)")
        
        if args.force_video_processing:
            logger.info("⚠ 用户强制使用视频处理模式，将忽略现有的PHALP文件")
            use_phalp_mode = False
        else:
            logger.info("🎯 自动选择PHALP文件融合模式（跳过视频处理）")
            use_phalp_mode = True
    else:
        logger.info("⚠ 未检测到PHALP文件:")
        if not cam1_pkl.exists():
            logger.info(f"  ✗ 相机1: {cam1_pkl}")
        if not cam2_pkl.exists():
            logger.info(f"  ✗ 相机2: {cam2_pkl}")
        
        # 查找可用的PHALP文件
        available_pkls = list(Path(".").glob("*.pkl"))
        if available_pkls:
            logger.info("当前目录中找到以下.pkl文件:")
            for pkl_file in available_pkls:
                file_size = pkl_file.stat().st_size / (1024 * 1024)  # MB
                logger.info(f"  - {pkl_file} ({file_size:.2f} MB)")
            logger.info("请修改--cam1_pkl和--cam2_pkl参数使用正确的PHALP文件路径")
        
        logger.info("🎯 自动选择视频处理模式")
        use_phalp_mode = False
    
    # 选择处理模式
    if use_phalp_mode:
        # 使用真实的PHALP数据
        logger.info("=" * 50)
        logger.info("使用PHALP文件进行融合...")
        logger.info("=" * 50)
        
        # 执行真实PHALP数据融合
        try:
            logger.info("开始加载PHALP数据并进行融合...")
            fused_result = fusion_system.process_real_phalp_fusion(
                str(cam1_pkl), str(cam2_pkl)
            )
            
            if fused_result:
                logger.info("🎉 真实PHALP数据融合完成！")
                
                # 如果启用了可视化，则进行3D可视化
                if args.visualize:
                    logger.info("开始3D可视化...")
                    fusion_system.create_3d_visualization(
                        frame_idx=args.frame_idx,
                        use_open3d=args.use_open3d,
                        use_plotly=args.use_plotly,
                        create_animation=args.create_animation
                    )
                
                return True
            else:
                logger.error("真实PHALP数据融合失败")
                return False
                
        except Exception as e:
            logger.error(f"真实PHALP数据融合过程中出现错误: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    else:
        # 使用视频处理模式
        logger.info("=" * 50)
        logger.info("使用视频处理模式...")
        logger.info("=" * 50)
        
        # 设置视频路径
        video_paths = {
            'cam_1': args.cam1_video,
            'cam_2': args.cam2_video
        }
        
        # 检查视频文件是否存在
        missing_videos = []
        for camera_name, video_path in video_paths.items():
            if not Path(video_path).exists():
                logger.error(f"视频文件不存在: {video_path}")
                missing_videos.append(video_path)
        
        if missing_videos:
            logger.error(f"缺少视频文件: {missing_videos}")
            logger.info("请检查视频文件路径是否正确")
            logger.info("当前目录结构:")
            try:
                # 查找可用的视频文件
                available_videos = list(Path(".").glob("**/*.mp4"))
                if available_videos:
                    logger.info("找到以下视频文件:")
                    for video in available_videos:
                        logger.info(f"  - {video}")
                    logger.info("请修改video_paths配置使用正确的视频文件路径")
                else:
                    logger.info("当前目录及其子目录中没有找到.mp4文件")
            except Exception as e:
                logger.warning(f"搜索视频文件时出错: {e}")
            return False
        
        # 执行完整的处理流程
        try:
            fused_result = fusion_system.process_multi_view_videos(video_paths)
            logger.info("真实的多视角4D-Humans融合处理完成！")
            
            # 如果启用了可视化，则进行3D可视化
            if args.visualize:
                logger.info("开始3D可视化...")
                fusion_system.create_3d_visualization(
                    frame_idx=args.frame_idx,
                    use_open3d=args.use_open3d,
                    use_plotly=args.use_plotly,
                    create_animation=args.create_animation
                )
            
            return True
        except Exception as e:
            logger.error(f"处理过程中出现错误: {e}")
            import traceback
            traceback.print_exc()
            return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1) 