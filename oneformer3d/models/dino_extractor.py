import os
import torch
import torch.nn as nn
from PIL import Image
from torchvision import transforms
import torch.nn.functional as F
import numpy as np
from typing import Tuple, Optional, Union, List
from pathlib import Path
from mmengine.model import BaseModule
import sys
import os
from mmdet3d.registry import MODELS
sys.path.append(os.path.abspath(os.path.dirname(__file__)))



class BiFPN(nn.Module):
    """双向特征金字塔网络，用于特征降维和融合"""
    def __init__(self, in_channels=1024, out_channels=64):
        super().__init__()
        
        # 定义中间通道数
        mid_channels = in_channels // 4  # 256
        
        # 第一层下采样和上采样
        self.down1 = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=1),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True)
        )
        
        self.down2 = nn.Sequential(
            nn.Conv2d(mid_channels, mid_channels//2, kernel_size=1),
            nn.BatchNorm2d(mid_channels//2),
            nn.ReLU(inplace=True)
        )
        
        self.up1 = nn.Sequential(
            nn.Conv2d(mid_channels//2, mid_channels, kernel_size=1),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True)
        )
        
        # 最终融合
        self.fusion_conv = nn.Sequential(
            nn.Conv2d(mid_channels * 2, out_channels, kernel_size=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
        
    def forward(self, x):
        """前向传播
        
        Args:
            x (torch.Tensor): 输入特征图 [B, C, H, W]
            
        Returns:
            torch.Tensor: 输出特征图 [B, out_channels, H, W]
        """
        # 第一次下采样
        level1 = self.down1(x)  # [B, 256, H, W]
        
        # 第二次下采样
        level2 = self.down2(level1)  # [B, 128, H, W]
        
        # 上采样回到level1
        level2_up = self.up1(level2)  # [B, 256, H, W]
        
        # 特征融合
        fused_feat = torch.cat([level1, level2_up], dim=1)  # [B, 512, H, W]
        out = self.fusion_conv(fused_feat)  # [B, 64, H, W]
        
        return out

class FeatureAdapter(nn.Module):
    """特征适配器/投影器，用于在训练时微调特征而无需更新DINOv2模型权重"""
    def __init__(self, in_channels=64, hidden_dim=128, out_channels=64, use_residual=True):
        super().__init__()
        
        self.use_residual = use_residual
        
        # MLP特征投影
        self.mlp = nn.Sequential(
            nn.Conv2d(in_channels, hidden_dim, kernel_size=1),
            nn.BatchNorm2d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_dim, hidden_dim, kernel_size=3, padding=1),
            nn.BatchNorm2d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_dim, out_channels, kernel_size=1),
            nn.BatchNorm2d(out_channels)
        )
        
        # 如果使用残差连接且输入输出通道不同，则添加一个1x1卷积进行调整
        if use_residual and in_channels != out_channels:
            self.shortcut = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        else:
            self.shortcut = nn.Identity()
            
        self.relu = nn.ReLU(inplace=True)
        
    def forward(self, x):
        """前向传播
        
        Args:
            x (torch.Tensor): 输入特征图 [B, C, H, W]
            
        Returns:
            torch.Tensor: 输出特征图 [B, out_channels, H, W]
        """
        identity = self.shortcut(x)
        out = self.mlp(x)
        
        if self.use_residual:
            out = out + identity
            
        out = self.relu(out)
        return out

@MODELS.register_module()
class DINOv2Extractor(BaseModule):
    """DINOv2特征提取器"""
    def __init__(self, 
                 model_path=os.path.abspath(os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 'pretrained/dinov2-large')),
                 device='cuda',
                 adapter_hidden_dim=128,
                 adapter_out_dim=64,
                 use_adapter=True,
                 use_residual=True):
        super().__init__()
        self.device = device
        self.feat_dim = 1024  # DINOv2-large的特征维度
        self.patch_h = 64
        self.patch_w = 96
        self.use_adapter = use_adapter
        
        # 将encoder和decoder分开组织
        self.encoder = None
        self.decoder = None
        
        try:
            from transformers import AutoModel
            self.encoder = AutoModel.from_pretrained(model_path, local_files_only=True)
            self.encoder = self.encoder.to(device)
            self.encoder.eval()
            
            for param in self.encoder.parameters():
                param.requires_grad = False
                
            print("DINOv2 model loaded successfully from local checkpoint")
        except Exception as e:
            print(f"Error loading model: {str(e)}")
            import traceback
            print(traceback.format_exc())
            raise
        
        # 将BiFPN和Adapter组织为decoder
        bifpn_out_channels = 64
        self.decoder = nn.ModuleDict({
            'bifpn': BiFPN(in_channels=self.feat_dim, out_channels=bifpn_out_channels).to(device)
        })
        
        if use_adapter:
            self.decoder['adapter'] = FeatureAdapter(
                in_channels=bifpn_out_channels,
                hidden_dim=adapter_hidden_dim,
                out_channels=adapter_out_dim,
                use_residual=use_residual
            ).to(device)

    def freeze_encoder(self):
        """冻结编码器参数"""
        for param in self.encoder.parameters():
            param.requires_grad = False

    def extract_features(self, image_tensor):
        """提取图像特征
        
        Args:
            image_tensor: tensor
            
        Returns:
            torch.Tensor: 特征图，形状为 [C, H, W]
        """
        # 处理输入
        if isinstance(image_tensor, torch.Tensor):
            img_tensor = image_tensor
            if img_tensor.dim() == 3:
                img_tensor = img_tensor.unsqueeze(0)
            
        
            
            # 图像归一化
            img_tensor = img_tensor / 255.0
            normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                          std=[0.229, 0.224, 0.225])
            img_tensor = normalize(img_tensor.squeeze(0)).unsqueeze(0)
            img_tensor = img_tensor.to(self.device)

        else:
            raise TypeError(f"Unsupported input type: {type(image_tensor)}")
        
        # 提取特征
        with torch.no_grad():
            try:
                outputs = self.encoder(img_tensor, output_hidden_states=True)
                features = outputs.hidden_states[-1]  # [B, N_patches + 1, feat_dim]
                features = features[:, 1:, :]  # [B, N_patches, feat_dim]
                
                # 重塑特征
                B = features.shape[0]
                features = features.permute(0, 2, 1)  # [B, feat_dim, N_patches]
                features = features.reshape(B, -1, self.patch_h, self.patch_w)
                
                # 通过BiFPN处理特征
                features = self.decoder['bifpn'](features)
                
            except Exception as e:
                print(f"Error extracting features: {str(e)}")
                import traceback
                print(traceback.format_exc())
                raise
        
        # 通过特征适配器/投影器处理特征（如果启用）
        if self.use_adapter:
            features = self.decoder['adapter'](features)
        
        features = features.squeeze(0)  # [C, H, W]
        
        # 将特征图上采样到适当分辨率（例如原图像的1/2或1/4）
        # 这里我们使用原图像的一半分辨率作为平衡
        
        return features
