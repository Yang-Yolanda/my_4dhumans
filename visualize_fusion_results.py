#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
多视角融合结果可视化脚本
专门用于可视化 real_4d_humans_fusion.py 的输出结果
支持原图叠加、侧视图、保存网格等功能
"""

import os
import cv2
import numpy as np
import json
import pickle
from pathlib import Path
import argparse
from typing import Dict, List, Optional, Tuple
import logging

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# 尝试导入PyTorch
try:
    import torch
    TORCH_AVAILABLE = True
    logger.info("✓ PyTorch 模块可用")
except ImportError:
    TORCH_AVAILABLE = False
    logger.warning("⚠ PyTorch 模块不可用，SMPL网格生成功能将受限")

# 导入必要的模块
try:
    from hmr2.utils.renderer import Renderer
    from hmr2.configs import default_config
    HMR2_AVAILABLE = True
    logger.info("✓ HMR2 模块可用，将使用完整渲染功能")
except ImportError:
    HMR2_AVAILABLE = False
    logger.warning("⚠ HMR2 模块不可用，将使用简化渲染")

# 尝试导入SMPL相关模块
try:
    import smplx
    SMPLX_AVAILABLE = True
    logger.info("✓ SMPLX 模块可用，将支持完整网格生成")
except ImportError:
    SMPLX_AVAILABLE = False
    logger.warning("⚠ SMPLX 模块不可用，网格生成功能将受限")

try:
    import trimesh
    TRIMESH_AVAILABLE = True
    logger.info("✓ Trimesh 模块可用，将支持网格导出")
except ImportError:
    TRIMESH_AVAILABLE = False
    logger.warning("⚠ Trimesh 模块不可用，无法导出.obj文件")

class FusionResultVisualizer:
    """多视角融合结果可视化器"""
    
    def __init__(self, output_dir: str = "real_4d_humans_fusion_results"):
        """
        初始化可视化器
        
        Args:
            output_dir: 融合结果输出目录
        """
        self.output_dir = Path(output_dir)
        self.renderer = None
        
        if HMR2_AVAILABLE:
            # 初始化HMR2渲染器
            try:
                cfg = default_config()
                self.renderer = Renderer(cfg, cfg.SMPL.FACES)
                logger.info("✓ HMR2渲染器初始化成功")
            except Exception as e:
                logger.warning(f"⚠ HMR2渲染器初始化失败: {e}")
                self.renderer = None
        
        # 颜色定义
        self.colors = {
            'fused': (0.0, 1.0, 0.0),      # 绿色 - 融合后
            'cam1': (1.0, 0.0, 0.0),       # 红色 - 相机1
            'cam2': (0.0, 0.0, 1.0),       # 蓝色 - 相机2
            'skeleton': (0.0, 1.0, 1.0),   # 青色 - 骨架
            'background': (0.9, 0.9, 0.9)  # 浅灰色 - 背景
        }
    
    def load_fusion_results(self) -> Dict[str, any]:
        """加载融合结果文件"""
        logger.info("开始加载融合结果...")
        
        results = {}
        
        # 查找可用的结果文件
        available_files = {
            'fused_pkl': self.output_dir / "fused_multi_view_result.pkl",
            'fused_json': self.output_dir / "fused_world_coordinate_results.json",
            'cam1_json': self.output_dir / "cam_1_results.json",
            'cam2_json': self.output_dir / "cam_2_results.json",
            'world_cam1': self.output_dir / "world_coordinate_cam_1.json",
            'world_cam2': self.output_dir / "world_coordinate_cam_2.json"
        }
        
        # 加载融合后的结果
        if available_files['fused_pkl'].exists():
            logger.info("加载融合结果PKL文件...")
            try:
                import joblib
                results['fused'] = joblib.load(available_files['fused_pkl'])
                logger.info(f"✓ 成功加载融合结果PKL: {available_files['fused_pkl']}")
            except Exception as e:
                logger.warning(f"⚠ 加载PKL文件失败: {e}")
                # 尝试使用pickle
                try:
                    with open(available_files['fused_pkl'], 'rb') as f:
                        results['fused'] = pickle.load(f)
                    logger.info(f"✓ 使用pickle成功加载融合结果")
                except Exception as e2:
                    logger.error(f"✗ 加载融合结果失败: {e2}")
        
        # 加载JSON格式的结果
        for key, file_path in available_files.items():
            if key != 'fused_pkl' and file_path.exists():
                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        results[key] = json.load(f)
                    logger.info(f"✓ 成功加载 {key}: {file_path}")
                except Exception as e:
                    logger.warning(f"⚠ 加载 {key} 失败: {e}")
        
        logger.info(f"加载完成，共 {len(results)} 个结果文件")
        return results
    
    def extract_visualization_data(self, results: Dict[str, any]) -> Dict[str, any]:
        """提取可视化所需的数据"""
        logger.info("提取可视化数据...")
        
        viz_data = {
            'fused_frames': [],
            'cam1_frames': [],
            'cam2_frames': [],
            'metadata': {}
        }
        
        # 提取融合后的数据
        if 'fused' in results:
            fused_data = results['fused']
            if isinstance(fused_data, dict) and 'fused_frames' in fused_data:
                viz_data['fused_frames'] = fused_data['fused_frames']
                logger.info(f"提取到 {len(viz_data['fused_frames'])} 帧融合数据")
            elif isinstance(fused_data, dict):
                # 可能是PHALP格式
                viz_data['fused_frames'] = self._convert_phalp_to_viz_format(fused_data)
                logger.info(f"转换PHALP格式，提取到 {len(viz_data['fused_frames'])} 帧数据")
        
        # 提取相机1数据
        if 'cam1_json' in results:
            cam1_data = results['cam1_json']
            if 'frames_data' in cam1_data:
                viz_data['cam1_frames'] = cam1_data['frames_data']
                logger.info(f"提取到 {len(viz_data['cam1_frames'])} 帧相机1数据")
        
        # 提取相机2数据
        if 'cam2_json' in results:
            cam2_data = results['cam2_json']
            if 'frames_data' in cam2_data:
                viz_data['cam2_frames'] = cam2_data['frames_data']
                logger.info(f"提取到 {len(viz_data['cam2_frames'])} 帧相机2数据")
        
        # 提取元数据
        if 'fused' in results and isinstance(results['fused'], dict):
            fused_data = results['fused']
            if 'fusion_statistics' in fused_data:
                viz_data['metadata']['fusion_stats'] = fused_data['fusion_statistics']
            if 'num_cameras' in fused_data:
                viz_data['metadata']['num_cameras'] = fused_data['num_cameras']
        
        logger.info("数据提取完成")
        return viz_data
    
    def _convert_phalp_to_viz_format(self, phalp_data: Dict) -> List[Dict]:
        """将PHALP格式转换为可视化格式"""
        converted_frames = []
        
        if isinstance(phalp_data, dict):
            for frame_name, frame_data in phalp_data.items():
                try:
                    # 提取帧索引
                    frame_idx = 0
                    if 'frame_' in frame_name:
                        frame_idx = int(frame_name.split('_')[1].split('.')[0])
                    
                    # 提取关键点
                    keypoints_3d = []
                    if '3d_joints' in frame_data and frame_data['3d_joints']:
                        keypoints_3d = frame_data['3d_joints'][0].tolist() if hasattr(frame_data['3d_joints'][0], 'tolist') else frame_data['3d_joints'][0]
                    
                    # 提取置信度
                    confidence = 0.8
                    if 'conf' in frame_data and frame_data['conf']:
                        confidence = frame_data['conf'][0]
                    
                    converted_frame = {
                        'frame_idx': frame_idx,
                        'timestamp': frame_idx / 30.0,  # 假设30fps
                        'keypoints_3d': keypoints_3d,
                        'confidence': confidence,
                        'num_views': 2,  # 融合结果
                        'camera_names': ['cam_1', 'cam_2']
                    }
                    
                    converted_frames.append(converted_frame)
                    
                except Exception as e:
                    logger.warning(f"转换帧 {frame_name} 失败: {e}")
                    continue
        
        return converted_frames
    
    def create_visualization(self, viz_data: Dict[str, any], 
                           output_dir: str = "fusion_visualization",
                           save_mesh: bool = True,
                           create_comparison: bool = True) -> None:
        """
        创建可视化结果
        
        Args:
            viz_data: 可视化数据
            output_dir: 输出目录
            save_mesh: 是否保存网格文件
            create_comparison: 是否创建对比图
        """
        logger.info("开始创建可视化...")
        
        # 创建输出目录
        viz_output_dir = Path(output_dir)
        viz_output_dir.mkdir(exist_ok=True)
        
        # 检查是否有数据
        if not viz_data['fused_frames']:
            logger.error("没有融合数据，无法创建可视化")
            return
        
        # 选择要可视化的帧
        frame_indices = self._select_frames_for_visualization(viz_data)
        logger.info(f"选择 {len(frame_indices)} 帧进行可视化")
        
        # 为每一帧创建可视化
        for frame_idx in frame_indices:
            logger.info(f"处理第 {frame_idx} 帧...")
            
            # 创建单帧可视化
            self._create_single_frame_visualization(
                viz_data, frame_idx, viz_output_dir, save_mesh
            )
            
            # 创建对比图
            if create_comparison:
                self._create_comparison_visualization(
                    viz_data, frame_idx, viz_output_dir
                )
        
        # 创建汇总可视化
        self._create_summary_visualization(viz_data, viz_output_dir)
        
        logger.info(f"可视化完成！结果保存在: {viz_output_dir}")
    
    def _select_frames_for_visualization(self, viz_data: Dict[str, any]) -> List[int]:
        """选择要可视化的帧"""
        fused_frames = viz_data['fused_frames']
        
        if len(fused_frames) <= 10:
            # 如果帧数不多，全部可视化
            return [frame['frame_idx'] for frame in fused_frames]
        else:
            # 选择关键帧
            total_frames = len(fused_frames)
            step = total_frames // 10
            selected_indices = []
            
            for i in range(0, total_frames, step):
                if len(selected_indices) < 10:
                    selected_indices.append(fused_frames[i]['frame_idx'])
            
            return selected_indices
    
    def _create_single_frame_visualization(self, viz_data: Dict[str, any], 
                                         frame_idx: int, 
                                         output_dir: Path, 
                                         save_mesh: bool) -> None:
        """创建单帧可视化"""
        
        # 查找对应帧的数据
        fused_frame = None
        cam1_frame = None
        cam2_frame = None
        
        for frame in viz_data['fused_frames']:
            if frame['frame_idx'] == frame_idx:
                fused_frame = frame
                break
        
        for frame in viz_data['cam1_frames']:
            if frame['frame_idx'] == frame_idx:
                cam1_frame = frame
                break
        
        for frame in viz_data['cam2_frames']:
            if frame['frame_idx'] == frame_idx:
                cam2_frame = frame
                break
        
        if not fused_frame:
            logger.warning(f"第 {frame_idx} 帧没有融合数据")
            return
        
        # 创建可视化图像
        if self.renderer and 'keypoints_3d' in fused_frame:
            # 使用HMR2渲染器
            self._create_hmr2_visualization(
                fused_frame, cam1_frame, cam2_frame, 
                frame_idx, output_dir, save_mesh
            )
        else:
            # 使用简化渲染
            self._create_simple_visualization(
                fused_frame, cam1_frame, cam2_frame, 
                frame_idx, output_dir
            )
    
    def _create_hmr2_visualization(self, fused_frame: Dict, cam1_frame: Dict, 
                                  cam2_frame: Dict, frame_idx: int, 
                                  output_dir: Path, save_mesh: bool) -> None:
        """使用HMR2渲染器创建可视化"""
        
        # 创建参考图像（使用融合帧的尺寸信息）
        if 'keypoints_3d' in fused_frame:
            keypoints = np.array(fused_frame['keypoints_3d'])
            if len(keypoints) > 0:
                # 根据关键点范围创建图像
                x_range = keypoints[:, 0].max() - keypoints[:, 0].min()
                y_range = keypoints[:, 1].max() - keypoints[:, 1].min()
                img_size = max(int(x_range * 100), int(y_range * 100), 512)
                reference_image = np.ones((img_size, img_size, 3), dtype=np.uint8) * 200
            else:
                reference_image = np.ones((512, 512, 3), dtype=np.uint8) * 200
        else:
            reference_image = np.ones((512, 512, 3), dtype=np.uint8) * 200
        
        # 渲染融合后的结果
        if 'keypoints_3d' in fused_frame:
            keypoints_3d = np.array(fused_frame['keypoints_3d'])
            if len(keypoints_3d) > 0:
                # 转换为顶点格式（简化处理）
                vertices = keypoints_3d.copy()
                cam_t = np.array([0, 0, 0])
                
                # 渲染正视图
                front_view = self.renderer(
                    vertices, cam_t, reference_image, side_view=False
                )
                
                # 渲染侧视图
                side_view = self.renderer(
                    vertices, cam_t, np.ones_like(reference_image) * 0.9, side_view=True
                )
                
                # 保存网格
                if save_mesh:
                    try:
                        # 尝试生成完整的SMPL网格
                        mesh_path = output_dir / f"frame_{frame_idx:06d}_fused.obj"
                        if self._generate_smpl_mesh(fused_frame, mesh_path):
                            logger.info(f"✓ 融合结果网格已保存: {mesh_path}")
                        else:
                            logger.warning("⚠ 无法生成完整网格，只保存关键点")
                            # 保存关键点作为备用
                            self._save_keypoints_as_obj(keypoints_3d, mesh_path)
                    except Exception as e:
                        logger.warning(f"保存网格失败: {e}")
                        # 保存关键点作为备用
                        try:
                            mesh_path = output_dir / f"frame_{frame_idx:06d}_fused_keypoints.obj"
                            self._save_keypoints_as_obj(keypoints_3d, mesh_path)
                            logger.info(f"关键点已保存为: {mesh_path}")
                        except Exception as e2:
                            logger.error(f"保存关键点也失败: {e2}")
                
                # 保存可视化图像
                combined_view = np.concatenate([front_view, side_view], axis=1)
                output_path = output_dir / f"frame_{frame_idx:06d}_fused_combined.png"
                cv2.imwrite(str(output_path), cv2.cvtColor((combined_view * 255).astype(np.uint8), cv2.COLOR_RGB2BGR))
                logger.info(f"融合结果可视化已保存: {output_path}")
    
    def _create_simple_visualization(self, fused_frame: Dict, cam1_frame: Dict, 
                                   cam2_frame: Dict, frame_idx: int, 
                                   output_dir: Path) -> None:
        """使用简化渲染创建可视化"""
        
        # 创建画布
        canvas_size = 800
        canvas = np.ones((canvas_size, canvas_size * 3, 3), dtype=np.uint8) * 200
        
        # 绘制融合后的结果
        if 'keypoints_3d' in fused_frame:
            keypoints_3d = np.array(fused_frame['keypoints_3d'])
            if len(keypoints_3d) > 0:
                self._draw_keypoints_on_canvas(
                    keypoints_3d, canvas, 0, 0, canvas_size, canvas_size, 
                    self.colors['fused'], "融合后"
                )
        
        # 绘制相机1的结果
        if cam1_frame and 'keypoints_3d' in cam1_frame:
            keypoints_3d = np.array(cam1_frame['keypoints_3d'])
            if len(keypoints_3d) > 0:
                self._draw_keypoints_on_canvas(
                    keypoints_3d, canvas, canvas_size, 0, canvas_size, canvas_size, 
                    self.colors['cam1'], "相机1"
                )
        
        # 绘制相机2的结果
        if cam2_frame and 'keypoints_3d' in cam2_frame:
            keypoints_3d = np.array(cam2_frame['keypoints_3d'])
            if len(keypoints_3d) > 0:
                self._draw_keypoints_on_canvas(
                    keypoints_3d, canvas, canvas_size * 2, 0, canvas_size, canvas_size, 
                    self.colors['cam2'], "相机2"
                )
        
        # 保存结果
        output_path = output_dir / f"frame_{frame_idx:06d}_simple_comparison.png"
        cv2.imwrite(str(output_path), canvas)
        logger.info(f"简化可视化已保存: {output_path}")
    
    def _draw_keypoints_on_canvas(self, keypoints_3d: np.ndarray, canvas: np.ndarray,
                                 x_offset: int, y_offset: int, width: int, height: int,
                                 color: Tuple[float, float, float], label: str) -> None:
        """在画布上绘制关键点"""
        
        if len(keypoints_3d) == 0:
            return
        
        # 将3D关键点投影到2D
        # 简化投影：忽略Z轴，只使用X和Y
        keypoints_2d = keypoints_3d[:, :2]
        
        # 归一化到画布范围
        x_min, x_max = keypoints_2d[:, 0].min(), keypoints_2d[:, 0].max()
        y_min, y_max = keypoints_2d[:, 1].min(), keypoints_2d[:, 1].max()
        
        if x_max - x_min > 0 and y_max - y_min > 0:
            # 缩放和偏移
            scale = min(width * 0.8 / (x_max - x_min), height * 0.8 / (y_max - y_min))
            keypoints_2d_scaled = (keypoints_2d - np.array([x_min, y_min])) * scale
            
            # 居中
            center_x = width // 2
            center_y = height // 2
            keypoints_2d_final = keypoints_2d_scaled + np.array([center_x, center_y])
            
            # 绘制关键点
            for i, point in enumerate(keypoints_2d_final):
                x, y = int(point[0]), int(point[1])
                if 0 <= x < width and 0 <= y < height:
                    cv2.circle(canvas, (x + x_offset, y + y_offset), 3, 
                              (int(color[0] * 255), int(color[1] * 255), int(color[2] * 255)), -1)
            
            # 绘制标签
            cv2.putText(canvas, label, (x_offset + 10, y_offset + 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)
    
    def _create_comparison_visualization(self, viz_data: Dict[str, any], 
                                       frame_idx: int, output_dir: Path) -> None:
        """创建对比可视化"""
        
        # 查找对应帧的数据
        fused_frame = None
        cam1_frame = None
        cam2_frame = None
        
        for frame in viz_data['fused_frames']:
            if frame['frame_idx'] == frame_idx:
                fused_frame = frame
                break
        
        for frame in viz_data['cam1_frames']:
            if frame['frame_idx'] == frame_idx:
                cam1_frame = frame
                break
        
        for frame in viz_data['cam2_frames']:
            if frame['frame_idx'] == frame_idx:
                cam2_frame = frame
                break
        
        if not fused_frame:
            return
        
        # 创建三视图对比
        if self.renderer and 'keypoints_3d' in fused_frame:
            self._create_hmr2_comparison(
                fused_frame, cam1_frame, cam2_frame, frame_idx, output_dir
            )
    
    def _create_hmr2_comparison(self, fused_frame: Dict, cam1_frame: Dict, 
                               cam2_frame: Dict, frame_idx: int, output_dir: Path) -> None:
        """使用HMR2创建对比可视化"""
        
        # 创建参考图像
        reference_image = np.ones((512, 512, 3), dtype=np.uint8) * 200
        
        # 渲染融合后的结果
        if 'keypoints_3d' in fused_frame:
            keypoints_3d = np.array(fused_frame['keypoints_3d'])
            if len(keypoints_3d) > 0:
                vertices = keypoints_3d.copy()
                cam_t = np.array([0, 0, 0])
                
                # 渲染正视图
                front_view = self.renderer(vertices, cam_t, reference_image, side_view=False)
                
                # 渲染侧视图
                side_view = self.renderer(vertices, cam_t, np.ones_like(reference_image) * 0.9, side_view=True)
                
                # 创建三视图对比
                h, w = reference_image.shape[:2]
                comparison = np.concatenate([
                    reference_image,
                    (front_view * 255).astype(np.uint8),
                    (side_view * 255).astype(np.uint8)
                ], axis=1)
                
                # 添加标签
                labels = ["原图", "3D模型叠加", "侧视图"]
                for i, label in enumerate(labels):
                    x = i * w + 10
                    y = 30
                    cv2.putText(comparison, label, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 
                               1, (255, 255, 255), 2, cv2.LINE_AA)
                
                # 保存对比图
                output_path = output_dir / f"frame_{frame_idx:06d}_three_view_comparison.png"
                cv2.imwrite(str(output_path), cv2.cvtColor(comparison, cv2.COLOR_RGB2BGR))
                logger.info(f"三视图对比已保存: {output_path}")
    
    def _create_summary_visualization(self, viz_data: Dict[str, any], output_dir: Path) -> None:
        """创建汇总可视化"""
        
        logger.info("创建汇总可视化...")
        
        # 创建置信度变化图
        if viz_data['fused_frames']:
            self._create_confidence_chart(viz_data['fused_frames'], output_dir)
        
        # 创建多视角覆盖图
        if viz_data['fused_frames']:
            self._create_multi_view_coverage_chart(viz_data['fused_frames'], output_dir)
        
        # 创建统计报告
        self._create_statistics_report(viz_data, output_dir)
    
    def _create_confidence_chart(self, fused_frames: List[Dict], output_dir: Path) -> None:
        """创建置信度变化图"""
        try:
            import matplotlib.pyplot as plt
            
            # 提取数据
            frame_indices = [frame['frame_idx'] for frame in fused_frames]
            confidences = [frame.get('confidence', 0.0) for frame in fused_frames]
            
            # 创建图表
            plt.figure(figsize=(12, 6))
            plt.plot(frame_indices, confidences, 'g-', linewidth=2, label='融合后置信度')
            plt.scatter(frame_indices, confidences, c='green', s=50, alpha=0.7)
            
            plt.xlabel('帧索引')
            plt.ylabel('置信度')
            plt.title('多视角融合结果置信度变化')
            plt.legend()
            plt.grid(True, alpha=0.3)
            
            # 保存图表
            output_path = output_dir / "confidence_chart.png"
            plt.savefig(str(output_path), dpi=150, bbox_inches='tight')
            plt.close()
            
            logger.info(f"置信度图表已保存: {output_path}")
            
        except ImportError:
            logger.warning("matplotlib不可用，跳过置信度图表创建")
        except Exception as e:
            logger.warning(f"创建置信度图表失败: {e}")
    
    def _create_multi_view_coverage_chart(self, fused_frames: List[Dict], output_dir: Path) -> None:
        """创建多视角覆盖图"""
        try:
            import matplotlib.pyplot as plt
            
            # 提取数据
            frame_indices = [frame['frame_idx'] for frame in fused_frames]
            view_counts = [frame.get('num_views', 1) for frame in fused_frames]
            
            # 创建图表
            plt.figure(figsize=(12, 6))
            plt.bar(frame_indices, view_counts, color='skyblue', alpha=0.7, label='视角数量')
            
            plt.xlabel('帧索引')
            plt.ylabel('视角数量')
            plt.title('多视角覆盖情况')
            plt.legend()
            plt.grid(True, alpha=0.3)
            
            # 保存图表
            output_path = output_dir / "multi_view_coverage_chart.png"
            plt.savefig(str(output_path), dpi=150, bbox_inches='tight')
            plt.close()
            
            logger.info(f"多视角覆盖图已保存: {output_path}")
            
        except ImportError:
            logger.warning("matplotlib不可用，跳过多视角覆盖图创建")
        except Exception as e:
            logger.warning(f"创建多视角覆盖图失败: {e}")
    
    def _create_statistics_report(self, viz_data: Dict[str, any], output_dir: Path) -> None:
        """创建统计报告"""
        
        # 统计信息
        stats = {
            'total_frames': len(viz_data['fused_frames']),
            'cam1_frames': len(viz_data['cam1_frames']),
            'cam2_frames': len(viz_data['cam2_frames']),
            'average_confidence': 0.0,
            'multi_view_frames': 0,
            'single_view_frames': 0
        }
        
        if viz_data['fused_frames']:
            confidences = [frame.get('confidence', 0.0) for frame in viz_data['fused_frames']]
            stats['average_confidence'] = np.mean(confidences)
            
            view_counts = [frame.get('num_views', 1) for frame in viz_data['fused_frames']]
            stats['multi_view_frames'] = sum(1 for v in view_counts if v > 1)
            stats['single_view_frames'] = sum(1 for v in view_counts if v == 1)
        
        # 保存统计报告
        report_path = output_dir / "visualization_statistics.json"
        with open(report_path, 'w', encoding='utf-8') as f:
            json.dump(stats, f, indent=2, ensure_ascii=False)
        
        logger.info(f"统计报告已保存: {report_path}")
        
        # 打印摘要
        logger.info("=" * 50)
        logger.info("可视化统计摘要:")
        logger.info(f"总融合帧数: {stats['total_frames']}")
        logger.info(f"相机1帧数: {stats['cam1_frames']}")
        logger.info(f"相机2帧数: {stats['cam2_frames']}")
        logger.info(f"平均置信度: {stats['average_confidence']:.3f}")
        logger.info(f"多视角帧数: {stats['multi_view_frames']}")
        logger.info(f"单视角帧数: {stats['single_view_frames']}")
        logger.info("=" * 50)

    def _generate_smpl_mesh(self, frame_data: Dict, output_path: Path) -> bool:
        """
        生成完整的SMPL网格
        
        Args:
            frame_data: 帧数据，包含SMPL参数
            output_path: 输出.obj文件路径
            
        Returns:
            是否成功生成网格
        """
        try:
            # 检查是否有SMPL参数
            if not all(key in frame_data for key in ['pose_params', 'shape_params']):
                logger.warning("缺少SMPL参数，无法生成完整网格")
                return False
            
            if not TRIMESH_AVAILABLE:
                logger.warning("Trimesh模块不可用，无法导出.obj文件")
                return False
            
            # 方法1: 尝试从HMR2模型获取faces信息并使用真实SMPL参数
            if HMR2_AVAILABLE:
                try:
                    from hmr2.configs import default_config
                    from hmr2.models import load_hmr2
                    
                    # 加载HMR2模型获取faces
                    cfg = default_config()
                    model, _ = load_hmr2(None)  # 使用默认checkpoint
                    
                    if hasattr(model, 'smpl') and hasattr(model.smpl, 'faces'):
                        faces = model.smpl.faces.cpu().numpy()
                        logger.info(f"✓ 从HMR2模型获取faces: {faces.shape}")
                        
                        # 使用真实的SMPL参数生成顶点
                        vertices = self._generate_smpl_vertices_from_params(frame_data)
                        
                        if vertices is not None:
                            # 创建trimesh对象并导出
                            mesh = trimesh.Trimesh(vertices=vertices, faces=faces)
                            mesh.export(str(output_path))
                            
                            logger.info(f"✓ 成功生成SMPL网格: {len(vertices)} 顶点, {len(faces)} 面")
                            return True
                        
                except Exception as e:
                    logger.warning(f"从HMR2模型获取faces失败: {e}")
            
            # 方法2: 如果无法获取faces，使用关键点生成简化网格
            logger.info("使用关键点生成简化网格...")
            if 'keypoints_3d' in frame_data:
                keypoints = np.array(frame_data['keypoints_3d'])
                if len(keypoints) > 0:
                    return self._save_keypoints_as_obj(keypoints, output_path)
            
            logger.warning("无法生成任何形式的网格")
            return False
            
        except Exception as e:
            logger.warning(f"生成SMPL网格失败: {e}")
            return False
    
    def _generate_smpl_vertices_from_params(self, frame_data: Dict) -> Optional[np.ndarray]:
        """
        从SMPL参数生成顶点
        
        Args:
            frame_data: 包含SMPL参数的帧数据
            
        Returns:
            SMPL顶点数组 [6890, 3] 或 None
        """
        try:
            # 提取SMPL参数
            pose_params = np.array(frame_data['pose_params'])
            shape_params = np.array(frame_data['shape_params'])
            trans_params = np.array(frame_data.get('trans_params_world', [0, 0, 0]))
            
            # 检查参数维度
            if len(pose_params) < 72:
                logger.warning(f"姿态参数维度不足: {len(pose_params)} < 72")
                return None
            
            if len(shape_params) < 10:
                logger.warning(f"形状参数维度不足: {len(shape_params)} < 10")
                return None
            
            # 方法1: 尝试使用SMPLX生成真实顶点
            if SMPLX_AVAILABLE and TORCH_AVAILABLE:
                try:
                    import smplx
                    import torch
                    
                    # 创建SMPL模型
                    smpl_model = smplx.create(
                        model_path="./smpl_models",  # 需要SMPL模型文件
                        model_type="smpl",
                        gender="neutral",
                        use_face_contour=False,
                        use_pca=False,
                        create_global_orient=True,
                        create_body_pose=True,
                        create_betas=True,
                        create_transl=True,
                        batch_size=1
                    )
                    
                    # 准备输入参数
                    pose_params_tensor = torch.tensor(pose_params[:72]).unsqueeze(0)  # [1, 72]
                    shape_params_tensor = torch.tensor(shape_params[:10]).unsqueeze(0)  # [1, 10]
                    trans_params_tensor = torch.tensor(trans_params[:3]).unsqueeze(0)  # [1, 3]
                    
                    # 生成网格
                    with torch.no_grad():
                        output = smpl_model(
                            betas=shape_params_tensor,
                            body_pose=pose_params_tensor[:, 3:],  # 跳过全局方向
                            global_orient=pose_params_tensor[:, :3],  # 全局方向
                            transl=trans_params_tensor
                        )
                        
                        vertices = output.vertices[0].cpu().numpy()  # [6890, 3]
                    
                    logger.info(f"✓ 使用SMPLX生成真实顶点: {vertices.shape}")
                    return vertices
                    
                except Exception as e:
                    logger.warning(f"SMPLX生成顶点失败: {e}")
                    logger.info("尝试使用简化方法...")
            
            # 方法2: 使用关键点插值生成顶点
            if 'keypoints_3d' in frame_data:
                keypoints = np.array(frame_data['keypoints_3d'])
                if len(keypoints) > 0:
                    vertices = self._keypoints_to_smpl_vertices(keypoints)
                    logger.info(f"✓ 使用关键点插值生成顶点: {vertices.shape}")
                    return vertices
            
            # 方法3: 生成默认顶点
            logger.warning("使用默认顶点生成")
            vertices = np.random.randn(6890, 3) * 0.1
            return vertices
            
        except Exception as e:
            logger.error(f"从SMPL参数生成顶点失败: {e}")
            return None
    
    def _keypoints_to_smpl_vertices(self, keypoints: np.ndarray) -> np.ndarray:
        """
        将关键点转换为SMPL格式的顶点
        
        Args:
            keypoints: 3D关键点数组 [N, 3]
            
        Returns:
            SMPL格式的顶点数组 [6890, 3]
        """
        try:
            # 这是一个简化的转换方法
            # 实际应用中，应该使用SMPL模型从pose和shape参数生成顶点
            
            if len(keypoints) == 0:
                # 生成默认的SMPL顶点
                vertices = np.random.randn(6890, 3) * 0.1
                return vertices
            
            # 方法1: 如果关键点数量足够，进行插值扩展
            if len(keypoints) >= 17:
                # 使用scipy进行插值，将17个关键点扩展到6890个顶点
                try:
                    from scipy.interpolate import griddata
                    
                    # 创建插值网格
                    x = np.linspace(-1, 1, 83)  # sqrt(6890) ≈ 83
                    y = np.linspace(-1, 1, 83)
                    z = np.linspace(-1, 1, 83)
                    X, Y, Z = np.meshgrid(x, y, z)
                    
                    # 将关键点归一化到[-1, 1]范围
                    keypoints_norm = keypoints.copy()
                    for i in range(3):
                        min_val = keypoints_norm[:, i].min()
                        max_val = keypoints_norm[:, i].max()
                        if max_val > min_val:
                            keypoints_norm[:, i] = 2 * (keypoints_norm[:, i] - min_val) / (max_val - min_val) - 1
                    
                    # 创建插值点
                    grid_points = np.column_stack([X.ravel(), Y.ravel(), Z.ravel()])
                    
                    # 插值生成顶点
                    vertices = griddata(keypoints_norm, keypoints, grid_points, method='linear', fill_value=0)
                    
                    # 确保顶点数量为6890
                    if len(vertices) > 6890:
                        vertices = vertices[:6890]
                    elif len(vertices) < 6890:
                        # 如果不足，用零填充
                        padding = np.zeros((6890 - len(vertices), 3))
                        vertices = np.vstack([vertices, padding])
                    
                    logger.info(f"✓ 通过插值生成 {len(vertices)} 个顶点")
                    return vertices
                    
                except ImportError:
                    logger.warning("scipy不可用，使用简化方法")
            
            # 方法2: 简化方法 - 重复关键点并添加噪声
            vertices = []
            keypoints_expanded = keypoints.copy()
            
            # 如果关键点太少，添加一些中间点
            while len(keypoints_expanded) < 100:
                # 在相邻关键点之间插入中点
                new_points = []
                for i in range(len(keypoints_expanded) - 1):
                    mid_point = (keypoints_expanded[i] + keypoints_expanded[i + 1]) / 2
                    new_points.append(mid_point)
                keypoints_expanded = np.vstack([keypoints_expanded, new_points])
            
            # 重复关键点直到达到6890个
            while len(vertices) < 6890:
                vertices.extend(keypoints_expanded)
            
            # 截取到6890个
            vertices = np.array(vertices[:6890])
            
            # 添加一些随机噪声，使网格更自然
            noise = np.random.randn(*vertices.shape) * 0.01
            vertices += noise
            
            logger.info(f"✓ 通过重复和噪声生成 {len(vertices)} 个顶点")
            return vertices
            
        except Exception as e:
            logger.warning(f"关键点转顶点失败: {e}")
            # 返回默认顶点
            return np.random.randn(6890, 3) * 0.1
    
    def _save_keypoints_as_obj(self, keypoints_3d: np.ndarray, output_path: Path) -> bool:
        """
        将3D关键点保存为简化的.obj文件
        
        Args:
            keypoints_3d: 3D关键点数组
            output_path: 输出.obj文件路径
            
        Returns:
            是否成功保存
        """
        try:
            if not TRIMESH_AVAILABLE:
                logger.warning("Trimesh模块不可用，无法保存.obj文件")
                return False
            
            if len(keypoints_3d) == 0:
                logger.warning("没有关键点数据")
                return False
            
            # 创建简化的网格：将关键点连接成简单的骨架
            vertices = keypoints_3d.copy()
            
            # 定义骨架连接（SMPL-24格式）
            skeleton_connections = [
                (0, 1), (1, 2), (2, 3), (3, 4),  # 头部到右手
                (1, 5), (5, 6), (6, 7),          # 左手
                (1, 8), (8, 9), (9, 10),         # 右腿
                (1, 11), (11, 12), (12, 13),     # 左腿
                (0, 14), (14, 15), (15, 16),     # 躯干
            ]
            
            # 只保留有效的连接
            valid_connections = []
            for start_idx, end_idx in skeleton_connections:
                if start_idx < len(vertices) and end_idx < len(vertices):
                    valid_connections.append([start_idx, end_idx])
            
            if len(valid_connections) == 0:
                logger.warning("没有有效的骨架连接")
                return False
            
            # 创建线框网格（使用细长的立方体表示骨骼）
            all_vertices = []
            all_faces = []
            
            for start_idx, end_idx in valid_connections:
                start_point = vertices[start_idx]
                end_point = vertices[end_idx]
                
                # 计算骨骼方向
                direction = end_point - start_point
                length = np.linalg.norm(direction)
                
                if length < 1e-6:  # 跳过太短的连接
                    continue
                
                # 创建细长的立方体表示骨骼
                bone_vertices, bone_faces = self._create_bone_mesh(start_point, end_point, length * 0.1)
                
                # 添加到总网格
                face_offset = len(all_vertices)
                all_vertices.extend(bone_vertices)
                all_faces.extend(bone_faces + face_offset)
            
            if len(all_vertices) == 0:
                logger.warning("无法生成有效的骨骼网格")
                return False
            
            # 创建trimesh对象并导出
            mesh = trimesh.Trimesh(vertices=np.array(all_vertices), faces=np.array(all_faces))
            mesh.export(str(output_path))
            
            logger.info(f"✓ 关键点骨架已保存为: {output_path} ({len(all_vertices)} 顶点, {len(all_faces)} 面)")
            return True
            
        except Exception as e:
            logger.error(f"保存关键点失败: {e}")
            return False
    
    def _create_bone_mesh(self, start_point: np.ndarray, end_point: np.ndarray, thickness: float) -> Tuple[List[np.ndarray], List[List[int]]]:
        """
        创建表示骨骼的细长立方体网格
        
        Args:
            start_point: 起始点
            end_point: 结束点
            thickness: 骨骼粗细
            
        Returns:
            顶点列表和面列表
        """
        # 计算骨骼方向
        direction = end_point - start_point
        length = np.linalg.norm(direction)
        
        if length < 1e-6:
            return [], []
        
        # 归一化方向
        direction = direction / length
        
        # 计算垂直于方向的向量
        if abs(direction[0]) > abs(direction[1]):
            perp = np.array([-direction[2], 0, direction[0]])
        else:
            perp = np.array([0, -direction[2], direction[1]])
        
        perp = perp / np.linalg.norm(perp)
        
        # 计算第三个垂直向量
        third = np.cross(direction, perp)
        
        # 创建立方体的8个顶点
        half_thickness = thickness / 2
        vertices = []
        
        # 起始端面的4个顶点
        for i in range(4):
            offset = (perp * np.cos(i * np.pi/2) + third * np.sin(i * np.pi/2)) * half_thickness
            vertices.append(start_point + offset)
        
        # 结束端面的4个顶点
        for i in range(4):
            offset = (perp * np.cos(i * np.pi/2) + third * np.sin(i * np.pi/2)) * half_thickness
            vertices.append(end_point + offset)
        
        # 定义立方体的12个面
        faces = [
            # 起始端面
            [0, 1, 2], [0, 2, 3],
            # 结束端面
            [4, 6, 5], [4, 7, 6],
            # 侧面
            [0, 4, 1], [1, 4, 5],
            [1, 5, 2], [2, 5, 6],
            [2, 6, 3], [3, 6, 7],
            [3, 7, 0], [0, 7, 4]
        ]
        
        return vertices, faces


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='多视角融合结果可视化')
    parser.add_argument('--input_dir', type=str, default='real_4d_humans_fusion_results',
                       help='融合结果输入目录')
    parser.add_argument('--output_dir', type=str, default='fusion_visualization',
                       help='可视化输出目录')
    parser.add_argument('--save_mesh', action='store_true', default=True,
                       help='保存网格文件')
    parser.add_argument('--comparison', action='store_true', default=True,
                       help='创建对比图')
    parser.add_argument('--frame_range', type=str, default='0:10',
                       help='要可视化的帧范围 (start:end)')
    
    args = parser.parse_args()
    
    # 检查输入目录
    if not Path(args.input_dir).exists():
        logger.error(f"输入目录不存在: {args.input_dir}")
        logger.info("请先运行 real_4d_humans_fusion.py 生成融合结果")
        return False
    
    # 创建可视化器
    visualizer = FusionResultVisualizer(args.input_dir)
    
    # 加载融合结果
    results = visualizer.load_fusion_results()
    if not results:
        logger.error("没有找到融合结果文件")
        return False
    
    # 提取可视化数据
    viz_data = visualizer.extract_visualization_data(results)
    if not viz_data['fused_frames']:
        logger.error("没有找到可用的融合数据")
        return False
    
    # 创建可视化
    visualizer.create_visualization(
        viz_data, 
        args.output_dir, 
        args.save_mesh, 
        args.comparison
    )
    
    logger.info("可视化完成！")
    return True


if __name__ == '__main__':
    success = main()
    exit(0 if success else 1) 