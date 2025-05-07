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



@MODELS.register_module()
class DINOv2Extractor(BaseModule):
    """DINOv2特征提取器"""
    def __init__(self, 
                 model_path=os.path.abspath(os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 'pretrained/dinov2-large')),
                 device='cuda',
                 adapter_hidden_dim=128,
                 adapter_out_dim=64,
                 use_residual=True):
        super().__init__()
        self.device = device
        self.feat_dim = 1024  # DINOv2-large的特征维度
        self.patch_h = 64
        self.patch_w = 96
        
        # 加载DINO编码器
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
        
        # 特征适配器
        self.adapter = FeatureAdapter(
            in_channels=self.feat_dim,  # 1024
            out_channels=adapter_out_dim  # 64
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
                
            except Exception as e:
                print(f"Error extracting features: {str(e)}")
                import traceback
                print(traceback.format_exc())
                raise
        
        # 通过特征适配器处理特征
        features = self.adapter(features)
        features = features.squeeze(0)  # [C, H, W]
        
        return features

class FeatureAdapter(nn.Module):
    def __init__(self, in_channels=1024, out_channels=64):
        super().__init__()
        
        # 使用两层1x1卷积逐步降低通道数
        self.reduce_dim = nn.Sequential(
            # 第一层: 1024 -> 256
            nn.Conv2d(in_channels, 256, kernel_size=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            # 第二层: 256 -> 64
            nn.Conv2d(256, out_channels, kernel_size=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
        
    def forward(self, x):
        # 输入 x: [B, 1024, 64, 96]
        # 输出: [B, 64, 64, 96]
        return self.reduce_dim(x)
