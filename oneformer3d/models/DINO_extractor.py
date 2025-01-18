# DINO_extractor.py

import os
import torch
import torch.nn as nn
from PIL import Image
from torchvision import transforms
import torch.nn.functional as F
import numpy as np
from typing import Tuple, Optional, Union, List
from pathlib import Path

import sys
import os
sys.path.append(os.path.abspath(os.path.dirname(__file__)))

from visualization import visualize_feature_map_multistage


# 这里包含您的 FastNormalizedFusion 和 BiFPN 类定义
class FastNormalizedFusion(nn.Module):
    """快速归一化融合模块"""
    def __init__(self, in_nodes):
        super().__init__()
        self.weights = nn.Parameter(torch.ones(in_nodes, dtype=torch.float32))
        self.eps = 1e-4
        self.relu = nn.ReLU()

    def forward(self, *features):
        # 确保权重为正
        weights = self.relu(self.weights)
        # 归一化权重
        weights = weights / (weights.sum() + self.eps)
        # 加权融合特征
        fused_features = sum(p * w for p, w in zip(features, weights))
        return self.relu(fused_features)

class BiFPN(nn.Module):
    def __init__(self, num_channels=1024):
        super().__init__()
        self.num_channels = num_channels
        
        # 通道降维卷积
        self.reduce_conv = nn.Conv2d(num_channels, 256, 1)
        
        # Level 1: 64x86 -> 128x172
        self.level1_conv = nn.Sequential(
            nn.Conv2d(256, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True)
        )
        
        # Level 2: 128x172 -> 256x344
        self.level2_conv = nn.Sequential(
            nn.Conv2d(256, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True)
        )
        
        # 特征融合
        self.fusion_conv = nn.Sequential(
            nn.Conv2d(512, 256, 1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True)
        )
        
    def forward(self, x):
        # Input: [B, 1024, 64, 86]
        
        # 1. 通道降维
        x = self.reduce_conv(x)  # [B, 256, 64, 86]
        
        # 2. Level 1 处理
        level1_feat = self.level1_conv(x)  # [B, 256, 64, 86]
        level1_up = F.interpolate(level1_feat, size=(128, 172), 
                                mode='bilinear', align_corners=True)
        
        # 3. Level 2 处理
        level2_feat = self.level2_conv(level1_up)  # [B, 256, 128, 172]
        level2_up = F.interpolate(level2_feat, size=(256, 344), 
                                mode='bilinear', align_corners=True)
        
        # 4. 特征融合 (将level2_up下采样回level1_up的尺寸)
        level2_down = F.interpolate(level2_up, size=(128, 172),
                                  mode='bilinear', align_corners=True)
        fused_feat = self.fusion_conv(torch.cat([level1_up, level2_down], dim=1))
        
        return fused_feat  # [B, 256, 128, 172]

class DINOv2Extractor:
    """DINO v2特征提取器"""
    def __init__(
        self,
        image_size: Tuple[int, int] = (896, 1204),  # 高度, 宽度
        device: Optional[str] = None
    ):
        self.image_size = image_size
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {self.device}")
        
        # 初始化模型和转换器
        self._init_model()
        self._init_transform()
        
        # 计算patch相关参数
        # DINOv2-large的patch size是14x14
        self.patch_h = self.image_size[0] // self.patch_size  # 高度方向的patch数量
        self.patch_w = self.image_size[1] // self.patch_size  # 宽度方向的patch数量
        self.feat_dim = 1024  # DINOv2-Large feature dimension
        
        # 初始化BiFPN
        self.bifpn = BiFPN(num_channels=self.feat_dim).to(self.device)
        
    def _init_model(self):
        """Initialize the DINO model."""
        print("Loading DINOv2 model...")
        
        try:
            # Load model directly from local checkpoint
            from transformers import AutoModel
            model_path = 'pretrained/dinov2-large'
            self.model = AutoModel.from_pretrained(model_path)
            self.model = self.model.to(self.device)
            self.model.eval()
            print("DINOv2 model loaded successfully from local checkpoint")
        except Exception as e:
            raise RuntimeError(f"Failed to load model: {str(e)}")
        
    def _init_transform(self):
        """初始化图像转换"""
        # DINOv2-large的patch size是14x14
        self.patch_size = 14  # 直接硬编码patch size
        self.transform = transforms.Compose([
            transforms.Resize(self.image_size),  # 统一将输入图像调整为(798, 1078)
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                              std=[0.229, 0.224, 0.225])
        ])
        
    def extract_features(self, image_path: Union[str, Path]) -> torch.Tensor:
        """
        提取单张图像的特征
        Args:
            image_path: 图像路径
        Returns:
            features: [C, H, W] 的特征张量，最终分辨率为 [256, 128, 172]
        """
        img = Image.open(image_path).convert('RGB')
        img_t = self.transform(img).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            # 使用新版本的transformers接口
            outputs = self.model(img_t, output_hidden_states=True)
            # 获取最后一层的特征
            features = outputs.hidden_states[-1]  # [B, N_patches + 1, feat_dim]
            # 移除CLS token
            features = features[:, 1:, :]  # [B, N_patches, feat_dim]
            
        # 重塑特征为 [B, C, H, W]
        B = features.shape[0]
        features = features.permute(0, 2, 1)  # [B, feat_dim, N_patches]
        features = features.reshape(B, self.feat_dim, self.patch_h, self.patch_w)  # [B, 1024, 64, 86]
        print(f"原始特征形状: {features.shape}")
        
        # 通过BiFPN
        features = self.bifpn(features)  # [B, 256, 128, 172]
        print(f"最终特征形状: {features.shape}")
        
        # 返回特征 [C, H, W]
        return features.squeeze(0)

    def extract_batch_features(self, image_paths: List[str]) -> List[torch.Tensor]:
        """
        批量提取多张图像的特征
        Args:
            image_paths: 图像路径列表
        Returns:
            features_list: 特征张量列表，每个张量形状为[C, H, W]
        """
        features_list = []
        for image_path in image_paths:
            try:
                features = self.extract_features(image_path)
                features_list.append(features)
            except Exception as e:
                print(f"处理图像 {image_path} 时出错: {str(e)}")
                continue
        return features_list

if __name__ == "__main__":
    try:
        # 初始化特征提取器
        extractor = DINOv2Extractor()
        
        # 设置输入和输出目录
        input_dir = './images'
        output_dir = './visualizations'
        
        # 确保输出目录存在
        os.makedirs(output_dir, exist_ok=True)
        
        # 获取所有支持的图像文件
        supported_formats = ('.jpg', '.jpeg', '.png', '.JPG', '.JPEG', '.PNG')
        image_files = [f for f in os.listdir(input_dir) 
                      if f.endswith(supported_formats)]
        
        print(f"找到 {len(image_files)} 个图像文件")
        
        # 处理每个图像
        for i, image_file in enumerate(image_files, 1):

            try:
                # 构建完整的输入输出路径
                input_path = os.path.join(input_dir, image_file)
                output_name = f"{os.path.splitext(image_file)[0]}_features.png"
                output_path = os.path.join(output_dir, output_name)
                
                print(f"\n处理图像 [{i}/{len(image_files)}]: {image_file}")
                
                # 提取特征
                features = extractor.extract_features(input_path)
                print(f"特征形状: {features.shape}")
                
                # 可视化特征
                visualize_feature_map_multistage(
                    features,
                    output_path=output_path
                )
                print(f"特征图已保存至: {output_path}")
                
            except Exception as e:
                print(f"处理 {image_file} 时出错: {str(e)}")
                continue
        
        print("\n所有图像处理完成!")
        
    except Exception as e:
        print(f"发生错误: {str(e)}")
