import numpy as np
import torch
import torch.nn as nn

from phalp.models.backbones import resnet
from phalp.models.heads.apperence_head import TextureHead
from phalp.models.heads.encoding_head import EncodingHead
from phalp.models.heads.smpl_head import SMPLHead
from phalp.utils.smpl_utils import SMPL
from phalp.utils.utils import compute_uvsampler, perspective_projection


class HMAR(nn.Module):
    """
    HMAR 模型用于 3D 人体姿态预测和纹理映射重建。
    主要包括：
      - 使用预训练的 ResNet 作为骨干网络提取图像特征；
      - 利用 TextureHead 和 EncodingHead 生成纹理映射和编码向量；
      - 通过 SMPLHead 预测 SMPL 参数（人体模型参数）以及相机参数；
      - 整体流程将输入图像处理为对应的人体姿态和纹理映射输出。
    """
    
    def __init__(self, cfg):
        super(HMAR, self).__init__()
       
        self.cfg = cfg  # 配置参数
        
        # 定义一些模型常数
        nz_feat, tex_size    = 512, 6
        img_H, img_W         = 256, 256
        
        # 加载存储 SMPL 纹理信息的文件，其中包含面片信息、纹理坐标等
        texture_file         = np.load(self.cfg.SMPL.TEXTURE)
        # 从文件中提取 SMPL 模型的面数据，并转换为 uint32 类型
        self.faces_cpu       = texture_file['smpl_faces'].astype('uint32')
        
        # 加载纹理坐标（vt）和面片纹理索引（ft）
        vt                   = texture_file['vt']
        ft                   = texture_file['ft']
        # 计算 UV 采样器，用于从纹理图中采样颜色信息
        uv_sampler           = compute_uvsampler(vt, ft, tex_size=tex_size)
        uv_sampler           = torch.tensor(uv_sampler, dtype=torch.float)
        uv_sampler           = uv_sampler.unsqueeze(0)  # 添加批次维度

        # 记录 UV 采样器的尺寸信息，并将其重塑为合适的形状
        self.F               = uv_sampler.size(1)   # F: 面数（或相关维度）
        self.T               = uv_sampler.size(2)   # T: 纹理尺寸
        self.uv_sampler      = uv_sampler.view(-1, self.F, self.T*self.T, 2)
        
        # 初始化 ResNet 骨干网络，采用预训练模型来提取图像特征
        self.backbone        = resnet(pretrained=True, num_layers=self.cfg.MODEL.BACKBONE.NUM_LAYERS, cfg=self.cfg)
        # 初始化纹理头部，用于生成纹理映射流（flow）等输出
        self.texture_head    = TextureHead(self.uv_sampler, self.cfg, img_H=img_H, img_W=img_W)
        # 初始化编码头部，用于生成后续处理所需的编码向量
        self.encoding_head   = EncodingHead(cfg=self.cfg, img_H=img_H, img_W=img_W) 

        # 构建 SMPL 模型，参数从配置中读取并转换为小写字典传入
        smpl_cfg             = {k.lower(): v for k,v in dict(cfg.SMPL).items()}
        # import pdb;pdb.set_trace()
        self.smpl            = SMPL(**smpl_cfg)
        
        # 初始化 SMPL 头部，用于预测人体模型参数和相机参数
        self.smpl_head       = SMPLHead(cfg, 
                                        input_dim=cfg.MODEL.SMPL_HEAD.IN_CHANNELS,
                                        pool='pooled')
        
    def load_weights(self, path):
        """
        加载预训练权重，只加载涉及 encoding_head、texture_head、backbone 以及 smplx_head 部分的参数，
        并进行名称替换后加载到模型中。
        """
        checkpoint_file = torch.load(path)
        state_dict_filt = {}
        for k, v in checkpoint_file['model'].items():
            # 过滤掉不需要加载的部分，仅保留部分模块的权重
            if ("encoding_head" in k or "texture_head" in k or "backbone" in k or "smplx_head" in k): 
                state_dict_filt.setdefault(k[5:].replace("smplx", "smpl"), v)
        self.load_state_dict(state_dict_filt, strict=False)

    def forward(self, x):
        """
        前向传播过程：
         1. 利用 ResNet 骨干网络提取特征和跳跃连接（skips）。
         2. 通过纹理头获取流信息，然后将流信息转换为 UV 图像。
         3. 通过全局池化获得姿态编码（pose embedding）。
         4. 使用 SMPL 头部预测人体姿态参数和相机参数。
         5. 返回 UV 图像、处理后的 UV 编码、流信息、姿态编码、SMPL 参数和相机参数。
        """
        # 提取特征和跳跃连接
        feats, skips    = self.backbone(x)
        # 通过纹理头部生成纹理流
        flow            = self.texture_head(skips)
        # 利用生成的流将原始图像转换为对应的 UV 图像（纹理图）
        uv_image        = self.flow_to_texture(flow, x)  #！！！转回来
        
        # 全局池化得到姿态编码
        pose_embeddings = feats.max(3)[0].max(2)[0]
        pose_embeddings = pose_embeddings.view(x.size(0), -1)
        # 使用 SMPL 头部预测人体姿态参数及相机参数，且不反向传播梯度
        with torch.no_grad():
            pred_smpl_params, pred_cam, _ = self.smpl_head(pose_embeddings)

        out = {
            "uv_image"  : uv_image,  # 原始生成的 UV 图像（包含颜色信息）
            "uv_vector" : self.process_uv_image(uv_image),  # 经过预处理，适用于后续自动编码器的 UV 向量
            "flow"      : flow,      # 从纹理头部获得的流信息
            "pose_emb"  : pose_embeddings,  # 图像全局特征编码，表征人体姿态信息
            "pose_smpl" : pred_smpl_params,   # 预测得到的 SMPL 姿态参数
            "pred_cam"  : pred_cam,  # 预测得到的相机参数
        }
        return out    
    
    def process_uv_image(self, uv_image):
        """
        对原始 UV 图像进行处理：
         - 分离出第四个通道作为 UV 掩码信息。
         - 对前三个通道进行缩放归一化处理。
         - 根据掩码将无效区域置 0，并将掩码值归一化为 -1 或 1。
         - 最后将处理后的图像和掩码信息拼接作为 UV 向量返回。
        """
        uv_mask         = uv_image[:, 3:, :, :]        # 提取 UV 掩码部分（通常表示透明度或有效性）
        uv_image        = uv_image[:, :3, :, :]/5.0      # 对颜色通道进行缩放处理
        zeros_          = uv_mask == 0                  # 掩码中无效区域
        ones_           = torch.logical_not(zeros_)      # 掩码中有效区域
        zeros_          = zeros_.repeat(1, 3, 1, 1)       # 将布尔掩码复制到三个通道
        ones_           = ones_.repeat(1, 3, 1, 1)
        uv_image[zeros_] = 0.0                         # 将无效区域颜色置为 0
        uv_mask[zeros_[:, :1, :, :]] = -1.0           # 无效区域标为 -1
        uv_mask[ones_[:, :1, :, :]]  = 1.0             # 有效区域标为 1
        uv_vector       = torch.cat((uv_image, uv_mask), 1)  # 拼接颜色信息和掩码信息作为最终 UV 向量
        
        return uv_vector
    
    def flow_to_texture(self, flow_map, img_x):
        """
        将流信息转换为纹理图像：
         - 调整流数据的通道顺序以匹配 grid_sample 的要求。
         - 利用 grid_sample 从原始图像 img_x 中采样得到对应的纹理图像。
        """
        flow_map   = flow_map.permute(0, 2, 3, 1)   # 将流张量从 (B,C,H,W) 变为 (B,H,W,C)
        uv_images  = torch.nn.functional.grid_sample(img_x, flow_map)
        return uv_images
    
    def autoencoder_hmar(self, x, en=True):
        """
        用于调用编码头部（EncodingHead）：
         - 当 en 为 True 时，根据配置选择不同的编码方式处理图像的前 3 个通道或全部通道。
         - 当 en 为 False 时，直接调用编码头部进行处理。
        """
        if en == True:
            if self.cfg.phalp.encode_type == "3c":
                return self.encoding_head(x[:, :3, :, :], en=en)
            else:
                return self.encoding_head(x, en=en)
        else:
            return self.encoding_head(x, en=en)

    def get_3d_parameters(self, pred_smpl_params, pred_cam, center=np.array([128, 128]), img_size=256, scale=None):
        """
        计算 3D 参数：
         - 根据预测的 SMPL 参数和相机参数，通过 SMPL 模型生成 3D 关节点。
         - 构造相机内参（focal length）并计算相机平移向量 pred_cam_t。
         - 使用透视投影将 3D 关节点投影到图像平面，计算 2D 关键点坐标。
         
        参数说明：
          pred_smpl_params: SMPL 参数，通常为字典格式的各个关键参数。
          pred_cam: 相机参数预测值。
          center: 图像中心坐标，用于对平移结果进行修正，默认值 [128, 128]。
          img_size: 图像尺寸，默认 256。
          scale: 如果提供，则用于调整缩放比例，否则使用默认缩放。

        返回：
          pred_smpl_params, 预测得到的 SMPL 参数；
          pred_keypoints_2d_smpl, 通过透视投影获得的 2D 关键点坐标；
          pred_joints, 生成的 3D 关节点坐标；
          pred_cam_t, 计算得到的相机平移向量。
        """
        
        # 若未提供 scale，则使用默认值
        if scale is not None: 
            pass
        else: 
            scale = np.ones((pred_cam.size(0), 1)) * 256

        batch_size = pred_cam.shape[0]
        dtype      = pred_cam.dtype
        device     = pred_cam.device
        # 根据配置中的焦距参数构造 focal_length 张量
        focal_length = self.cfg.EXTRA.FOCAL_LENGTH * torch.ones(batch_size, 2, device=device, dtype=dtype)
 
        # 使用 SMPL 模型生成 3D 关节点
        smpl_output = self.smpl(**{k: v.float() for k, v in pred_smpl_params.items()}, pose2rot=False)
        pred_joints = smpl_output.joints

        # 计算相机平移向量 pred_cam_t：根据预测的相机参数和缩放比例计算
        pred_cam_t = torch.stack([
            pred_cam[:, 1], 
            pred_cam[:, 2], 
            2 * focal_length[:, 0] / (pred_cam[:, 0] * torch.tensor(scale[:, 0], dtype=dtype, device=device) + 1e-9)
        ], dim=1)
        # 调整平移向量前 2 个分量，使其以图像中心为基准
        pred_cam_t[:, :2] += torch.tensor(center - img_size / 2., dtype=dtype, device=device) * pred_cam_t[:, [2]] / focal_length

        # 在关节点后附加零值（可能用于与其他通道拼接或保证维度一致）
        zeros_ = torch.zeros(batch_size, 1, 3).to(device)
        pred_joints = torch.cat((pred_joints, zeros_), 1)

        # 构造用于透视投影的相机参数：相机中心和归一化的焦距
        camera_center = torch.zeros(batch_size, 2)
        pred_keypoints_2d_smpl = perspective_projection(
            pred_joints,
            rotation=torch.eye(3).unsqueeze(0).expand(batch_size, -1, -1).to(device),
            translation=pred_cam_t.to(device),
            focal_length=focal_length / img_size,
            camera_center=camera_center.to(device)
        )  

        # 将投影后的关键点映射到图像尺度上
        pred_keypoints_2d_smpl = (pred_keypoints_2d_smpl + 0.5) * img_size

        return pred_smpl_params, pred_keypoints_2d_smpl, pred_joints, pred_cam_t