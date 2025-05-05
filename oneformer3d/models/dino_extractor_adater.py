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
    """DINOv2 feature extractor with configurable output dimensions."""
    
    def __init__(
        self,
        model_path: str,
        device: str = 'cuda',
        feat_dim: int = 1024,
        patch_size: Tuple[int, int] = (64, 96)
    ) -> None:
        super().__init__()
        self.device = device
        self.feat_dim = feat_dim
        self.patch_h, self.patch_w = patch_size
        
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
    
    def _freeze_parameters(self) -> None:
        """Freeze all parameters of the DINO backbone."""
        for param in self.model.parameters():
            param.requires_grad = False
    
    def train(self, mode: bool = True) -> 'DINOv2Extractor':
        """Override train method to keep model in eval mode."""
        super().train(mode)
        self.model.eval()
        return self
    
    def extract_features(
        self, 
        image: torch.Tensor, 
        layer_idx: int
    ) -> torch.Tensor:
        """Extract features from specified layer.
        
        Args:
            image: Input image tensor [B, C, H, W] or [C, H, W]
            layer_idx: Index of the transformer layer to extract features from
            
        Returns:
            Features tensor [B, C, H, W]
        """
        if image.dim() == 3:
            image = image.unsqueeze(0)
            
        # Normalize image
        image = self._normalize_image(image)
        
        with torch.no_grad():
            outputs = self.model(image, output_hidden_states=True)
            features = outputs.hidden_states[layer_idx]
            features = self._reshape_features(features)
            
        return features
    
    def _normalize_image(self, image: torch.Tensor) -> torch.Tensor:
        """Normalize image using ImageNet statistics."""
        image = image / 255.0
        normalize = transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
        return normalize(image.squeeze(0)).unsqueeze(0).to(self.device)
    
    def _reshape_features(self, features: torch.Tensor) -> torch.Tensor:
        """Reshape transformer features to spatial format."""
        B = features.shape[0]
        features = features[:, 1:, :]  # Remove CLS token
        features = features.permute(0, 2, 1)
        return features.reshape(B, -1, self.patch_h, self.patch_w)

@MODELS.register_module()
class DINODPTDecoder(nn.Module):
    """Multi-attention decoder with boundary enhancement for DINOv2 features."""
    
    def __init__(
        self,
        feature_dim: int = 1024,
        hidden_dim: int = 256,
        output_dim: int = 64,
        target_size: Tuple[int, int] = (896, 1344),
        feature_size: Tuple[int, int] = (64, 96),
        device: str = 'cuda'
    ) -> None:
        super().__init__()
        self.hidden_dim = hidden_dim
        self.device = device
        
        # 计算中间尺寸（约原尺寸的3倍）
        mid_h = feature_size[0] * 3
        mid_w = feature_size[1] * 3
        mid_size = (mid_h, mid_w)
        
        # 投影层：将高维特征映射到较低维度
        self.projection = nn.Conv2d(feature_dim, hidden_dim, kernel_size=1)
        
        # 残差块
        self.res_block = nn.Sequential(
            nn.Conv2d(hidden_dim, hidden_dim, kernel_size=3, padding=1),
            nn.BatchNorm2d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_dim, hidden_dim, kernel_size=3, padding=1),
            nn.BatchNorm2d(hidden_dim)
        )
        
        # 通道注意力机制
        self.channel_attention = ChannelAttention(hidden_dim)
        
        # 边界感知注意力 - 替换原来的空间注意力
        self.boundary_attention = BoundaryAwareAttention(hidden_dim)
        
        # 第一层上采样：从特征尺寸到中间尺寸
        self.upsample1 = nn.Sequential(
            nn.Conv2d(hidden_dim, hidden_dim, kernel_size=3, padding=1),
            nn.BatchNorm2d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Upsample(size=mid_size, mode='bilinear', align_corners=True),
            nn.Conv2d(hidden_dim, hidden_dim, kernel_size=3, padding=1),
            nn.BatchNorm2d(hidden_dim),
            nn.ReLU(inplace=True)
        )
        
        # 梯度增强注意力 - 在第二次上采样前使用
        self.gradient_attention = GradientEnhancedAttention()
        
        # 第二层上采样：从中间尺寸到目标尺寸
        self.upsample2 = nn.Sequential(
            nn.Conv2d(hidden_dim, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Upsample(size=target_size, mode='bilinear', align_corners=True),
            nn.Conv2d(128, output_dim, kernel_size=3, padding=1)
        )
        
        # 将所有模块移到指定设备
        self.to(device)
    
    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """
        Args:
            features: 输入特征 [B, C, H, W]
            
        Returns:
            输出特征 [B, output_dim, H*scale, W*scale]
        """
        # 投影特征
        x = self.projection(features)
        
        # 应用残差块
        identity = x
        x = self.res_block(x)
        x = x + identity  # 残差连接
        
        # 应用通道注意力
        x = self.channel_attention(x) * x
        
        # 第一层上采样
        x = self.upsample1(x)
        
        # 应用边界感知注意力
        x = self.boundary_attention(x)
        
        # 应用梯度增强注意力
        x = self.gradient_attention(x)
        
        # 第二层上采样到目标尺寸
        x = self.upsample2(x)
        
        return x


# 通道注意力模块
class ChannelAttention(nn.Module):
    def __init__(self, in_channels, reduction_ratio=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        
        # 共享MLP
        self.fc = nn.Sequential(
            nn.Conv2d(in_channels, in_channels // reduction_ratio, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels // reduction_ratio, in_channels, 1, bias=False)
        )
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        avg_out = self.fc(self.avg_pool(x))
        max_out = self.fc(self.max_pool(x))
        out = avg_out + max_out
        return self.sigmoid(out)


# 空间注意力模块
class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()
        assert kernel_size in (3, 7), "kernel size must be 3 or 7"
        padding = 3 if kernel_size == 7 else 1
        
        self.conv = nn.Conv2d(2, 1, kernel_size=kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        # 生成平均池化和最大池化的特征图（沿通道维度）
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        
        # 拼接池化结果
        out = torch.cat([avg_out, max_out], dim=1)
        
        # 卷积和sigmoid激活
        out = self.conv(out)
        return self.sigmoid(out)


# 边界感知注意力模块
class BoundaryAwareAttention(nn.Module):
    def __init__(self, in_channels, kernel_size=3):
        super(BoundaryAwareAttention, self).__init__()
        self.in_channels = in_channels
        self.kernel_size = kernel_size
        
        # Sobel算子用于边缘检测 - 水平和垂直方向
        self.sobel_x = torch.nn.Parameter(
            torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32).reshape(1, 1, 3, 3),
            requires_grad=False
        )
        self.sobel_y = torch.nn.Parameter(
            torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=torch.float32).reshape(1, 1, 3, 3),
            requires_grad=False
        )
        
        # 自适应边界权重生成器
        self.boundary_weight_generator = nn.Sequential(
            nn.Conv2d(2, 16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 1, kernel_size=1),
            nn.Sigmoid()
        )
        
        # 特征融合
        self.fusion = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True)
        )
    
    def forward(self, x):
        # 计算特征图平均值，用于边缘检测
        avg_x = torch.mean(x, dim=1, keepdim=True)
        
        # 使用Sobel算子提取边缘
        edge_x = F.conv2d(avg_x, self.sobel_x, padding=1)
        edge_y = F.conv2d(avg_x, self.sobel_y, padding=1)
        edge_magnitude = torch.sqrt(edge_x**2 + edge_y**2)
        
        # 归一化边缘强度
        edge_magnitude = edge_magnitude / (torch.max(edge_magnitude) + 1e-8)
        
        # 生成方向信息
        edge_direction = torch.atan2(edge_y, edge_x) / 3.14159
        
        # 边缘特征
        edge_features = torch.cat([edge_magnitude, edge_direction], dim=1)
        
        # 生成边界权重
        boundary_weights = self.boundary_weight_generator(edge_features)
        
        # 增强边界特征 (边界强化策略)
        enhanced_features = x * (1.0 + boundary_weights)
        
        # 特征融合 (残差连接风格)
        output = self.fusion(enhanced_features) + x
        
        return output


# 梯度增强注意力模块
class GradientEnhancedAttention(nn.Module):
    def __init__(self, blur_kernel_size=5, amplify_factor=2.0):
        super(GradientEnhancedAttention, self).__init__()
        self.blur_kernel_size = blur_kernel_size
        self.amplify_factor = amplify_factor
        
        # 高斯滤波器用于平滑特征
        self.gaussian_blur = nn.Conv2d(1, 1, kernel_size=blur_kernel_size, padding=blur_kernel_size//2, bias=False)
        
        # 初始化高斯核
        with torch.no_grad():
            kernel = self._create_gaussian_kernel(blur_kernel_size)
            self.gaussian_blur.weight.copy_(kernel)
            
        # 保持高斯核不可训练
        self.gaussian_blur.weight.requires_grad = False
    
    def _create_gaussian_kernel(self, kernel_size):
        """创建高斯核"""
        x = torch.arange(kernel_size) - (kernel_size - 1) / 2
        x = x.repeat(kernel_size, 1)
        y = x.t()
        
        variance = ((kernel_size - 1) / 6) ** 2  # 标准高斯核的方差设置
        gaussian_kernel = torch.exp(-(x**2 + y**2) / (2 * variance))
        gaussian_kernel = gaussian_kernel / gaussian_kernel.sum()
        
        return gaussian_kernel.view(1, 1, kernel_size, kernel_size)
    
    def forward(self, x):
        # 计算特征图的空间平均
        spatial_mean = torch.mean(x, dim=1, keepdim=True)
        
        # 使用高斯滤波进行平滑
        blurred = self.gaussian_blur(spatial_mean)
        
        # 计算梯度幅值 (原始 - 平滑)
        gradient_magnitude = torch.abs(spatial_mean - blurred)
        
        # 标准化梯度幅值
        gradient_magnitude = gradient_magnitude / (torch.max(gradient_magnitude) + 1e-8)
        
        # 生成增强因子
        enhancement_factor = 1.0 + self.amplify_factor * gradient_magnitude
        
        # 应用梯度增强
        enhanced_features = x * enhancement_factor
        
        return enhanced_features

@MODELS.register_module()
class DINODPTModel(nn.Module):
    """Simplified DINO-DPT model that uses only the final layer features."""
    
    def __init__(
        self,
        model_path: str,
        output_dim: int = 64,
        target_size: Tuple[int, int] = (896, 1344),
        device: str = 'cuda'
    ) -> None:
        super().__init__()
        self.device = device
        self.encoder = DINOv2Extractor(model_path=model_path, device=device)
        self.decoder = DINODPTDecoder(
            feature_dim=self.encoder.feat_dim,
            output_dim=output_dim,
            target_size=target_size,
            device=device
        )
        self.to(device)
        
        # 默认冻结编码器
        self.freeze_encoder()
    
    def freeze_encoder(self):
        """冻结编码器参数，只允许解码器参数更新"""
        for param in self.encoder.parameters():
            param.requires_grad = False
        
        # 验证参数冻结状态
        encoder_params = sum(p.numel() for p in self.encoder.parameters() if p.requires_grad)
        decoder_params = sum(p.numel() for p in self.decoder.parameters() if p.requires_grad)
        total_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        
        print(f"训练参数统计:")
        print(f"  编码器可训练参数: {encoder_params:,d} (应为0)")
        print(f"  解码器可训练参数: {decoder_params:,d}")
        print(f"  总可训练参数: {total_params:,d}")
    
    def unfreeze_encoder(self):
        """解冻编码器参数，允许全模型训练（除非encoder内部有冻结）"""
        # 注意：这只会解冻编码器的外层参数，而不会影响DINOv2内部的冻结设置
        for param in self.encoder.parameters():
            param.requires_grad = True
        
        # 验证参数解冻状态
        encoder_params = sum(p.numel() for p in self.encoder.parameters() if p.requires_grad)
        decoder_params = sum(p.numel() for p in self.decoder.parameters() if p.requires_grad)
        total_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        
        print(f"训练参数统计:")
        print(f"  编码器可训练参数: {encoder_params:,d} (注意：DINOv2内部可能仍有冻结参数)")
        print(f"  解码器可训练参数: {decoder_params:,d}")
        print(f"  总可训练参数: {total_params:,d}")
    
    def forward(self, images: torch.Tensor) -> torch.Tensor:
        # 只提取最后一层特征
        features = self.encoder.extract_features(images, layer_idx=24)
        
        # 确保特征批次维度正确
        if features.dim() == 3:
            features = features.unsqueeze(0)
        
        # 解码
        return self.decoder(features)

def visualize_feature_map_multistage(feature_map, pca_stages=None, output_dir="./feature_visualizations", prefix="view", random_state=42):
    """使用多阶段 PCA 将高维特征图转换为可视化的 RGB 图像"""
    from sklearn.decomposition import PCA
    from PIL import Image
    from datetime import datetime
    import os
    import numpy as np
    
    # 设置默认PCA阶段
    if pca_stages is None:
        pca_stages = [16, 3]
    
    # 确保输入是CPU上的numpy数组
    if torch.is_tensor(feature_map):
        feature_map = feature_map.detach().cpu().numpy()
    
    # 处理批次维度
    if len(feature_map.shape) == 4:  # 如果形状是[B, C, H, W]
        if feature_map.shape[0] == 1:  # 如果批次大小为1
            feature_map = feature_map.squeeze(0)  # 移除批次维度变为[C, H, W]
        else:
            # 如果批次大小>1，只取第一个样本
            print(f"Warning: feature_map has batch size > 1 ({feature_map.shape[0]}). Using only the first sample.")
            feature_map = feature_map[0]
    
    # 此时feature_map应该是[C, H, W]格式
    C, H, W = feature_map.shape
    feature_map_np = feature_map.reshape(C, -1).T  # [H*W, C]
    curr_features = feature_map_np

    # 多阶段 PCA 降维
    for i, n_components in enumerate(pca_stages):
        n_components = min(n_components, curr_features.shape[1])
        pca = PCA(n_components=n_components, random_state=random_state)
        curr_features = pca.fit_transform(curr_features)
        explained_var = sum(pca.explained_variance_ratio_) * 100
        print(f"PCA降维到{n_components}个分量, 解释方差: {explained_var:.2f}%")

    # 归一化到 [0, 1]
    min_vals = curr_features.min(axis=0)
    max_vals = curr_features.max(axis=0)
    curr_features = (curr_features - min_vals) / (max_vals - min_vals + 1e-5)
    
    # 重塑回图像形状 [H, W, 3]
    curr_features = curr_features.reshape(H, W, 3)

    # 转换为RGB图像 (0-255)
    feature_image = (curr_features * 255).astype(np.uint8)
    pil_image = Image.fromarray(feature_image)

    # 确保输出目录存在
    os.makedirs(output_dir, exist_ok=True)
    
    # 简单的日期时间命名
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{prefix}_{timestamp}.png"
    output_path = os.path.join(output_dir, filename)
    
    # 保存图像
    pil_image.save(output_path)
    print(f"可视化结果已保存至: {output_path}")

    return pil_image

def main():
    """Test function to demonstrate the simplified DINO-DPT model functionality."""
    import cv2
    from PIL import Image
    import os.path as osp
    from datetime import datetime
    import numpy as np
    
    # 明确指定使用GPU
    device = 'cuda'
    
    # 配置参数
    image_path = "data/s3dis-urbanbis-yuehai-all-instance/Stanford3dDataset_v1.2_Aligned_Version/Area_5/Tile_132122232000203133/Images/DJI_09222.JPG"
    target_size = (1344, 896)  # (W, H)
    model_path = '/workspace/data/oneformer3d/pretrained/dinov2-large'
    save_dir = 'feature_visualizations'
    
    # 创建保存目录
    os.makedirs(save_dir, exist_ok=True)
    
    # 检查文件是否存在
    if not osp.exists(image_path):
        raise FileNotFoundError(f"Image not found at {image_path}")
    
    try:
        # 加载并预处理图像
        image = Image.open(image_path)
        image = image.resize(target_size, Image.Resampling.LANCZOS)
        image_tensor = transforms.ToTensor()(image).to(device)
        
        # 保存调整大小后的原始图像
        resized_image = transforms.ToPILImage()(image_tensor.cpu())
        resized_image.save(osp.join(save_dir, 'resized_input.jpg'))
        
        # 初始化模型
        model = DINODPTModel(
            model_path=model_path,
            output_dim=64,
            target_size=target_size[::-1],
            device=device
        )
        model.eval()
        
        # 提取特征
        with torch.no_grad():
            # 获取原始DINOv2特征（仅用于可视化）
            dino_features = model.encoder.extract_features(image_tensor, layer_idx=24)
            
            # 获取完整模型输出
            output_features = model(image_tensor)
        
        # 可视化特征
        visualize_feature_map_multistage(
            dino_features.cpu(),
            output_dir=save_dir,
            prefix=f'dino_layer_24',
            pca_stages=[16, 3]
        )
        
        visualize_feature_map_multistage(
            output_features.squeeze(0).cpu(),
            output_dir=save_dir,
            prefix=f'final_output',
            pca_stages=[16, 3]
        )
        
        # 打印特征图尺寸信息
        print("\nFeature shapes:")
        print(f"DINO Layer 24: {dino_features.shape}")
        print(f"Final output: {output_features.shape}")
        
    except Exception as e:
        print(f"Error occurred: {str(e)}")
        import traceback
        print(traceback.format_exc())

if __name__ == '__main__':
    # 设置PYTHONPATH
    import sys
    import os
    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))
    main()