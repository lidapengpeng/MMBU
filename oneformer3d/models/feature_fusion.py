import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Tuple
import numpy as np

from mmdet3d.registry import MODELS
from mmengine.model import BaseModule


@MODELS.register_module()   
class MultiModalFeatureFusion(BaseModule):
    def __init__(self, in_channels=64, chunk_size=128):
        super().__init__()
        self.in_channels = in_channels
        self.chunk_size = chunk_size
        
        # 简单的特征变换网络
        self.feature_transform = nn.Sequential(
            nn.Linear(in_channels, in_channels),
            nn.LayerNorm(in_channels),
            nn.ReLU(inplace=True)
        )
        
        # 简单的注意力网络
        self.attention = nn.Sequential(
            nn.Linear(in_channels * 2, 1),
            nn.Sigmoid()
        )
        
        # 归一化层
        self.norm = nn.LayerNorm(in_channels)
        
    @torch.cuda.amp.autocast()
    def chunked_fusion(self, voxel_feat, image_feat):
        """使用混合精度和更小的分块进行特征融合"""
        N = voxel_feat.size(0)
        device = voxel_feat.device
        fused_features = torch.zeros((N, self.in_channels), device=device, dtype=torch.float32)
        
        for i in range(0, N, self.chunk_size):
            end_idx = min(i + self.chunk_size, N)
            
            # 获取当前块的特征
            voxel_chunk = voxel_feat[i:end_idx]
            image_chunk = image_feat[i:end_idx]
            
            # 计算注意力权重 - 避免大型中间张量
            with torch.cuda.amp.autocast():
                # 分步计算以减少内存使用
                concat_feat = torch.cat([voxel_chunk, image_chunk], dim=-1)
                attention = self.attention(concat_feat)
                
                # 直接计算融合结果
                fused = (1 - attention) * voxel_chunk + attention * image_chunk
            
            # 转回float32并保存结果
            fused_features[i:end_idx] = fused.float()
            
            # 每处理一定数量的块后清理缓存
            if i > 0 and i % (self.chunk_size * 8) == 0:
                torch.cuda.empty_cache()
                
        return fused_features
    
    def forward(self, voxel_features, image_features, valid_mask):
        """内存优化的前向传播函数"""
        # 如果没有有效点，直接返回体素特征
        if not valid_mask.any():
            return voxel_features
        
        # 打印信息以帮助调试
        # valid_count = valid_mask.sum().item()
        # if torch.distributed.get_rank() == 0 and torch.distributed.is_initialized():
        #     print(f"有效点数: {valid_count}, 特征维度: {voxel_features.shape[1]}")
            
        # 提取有效特征 - 减少后续计算的规模
        valid_voxel_feat = voxel_features[valid_mask]
        valid_image_feat = image_features[valid_mask]
        
        # 使用混合精度计算进行特征变换
        with torch.cuda.amp.autocast():
            transformed_voxel = self.feature_transform(valid_voxel_feat)
            transformed_image = self.feature_transform(valid_image_feat)
        
        # 分块融合特征
        fused_features = self.chunked_fusion(transformed_voxel, transformed_image)
        
        # 应用归一化
        with torch.cuda.amp.autocast():
            fused_features = self.norm(fused_features)
        
        # 构建输出特征 - 直接赋值避免额外的克隆操作
        output_features = torch.zeros_like(voxel_features)
        output_features[valid_mask] = fused_features.float()
        
        # 最终清理缓存
        torch.cuda.empty_cache()
        
        return output_features



@MODELS.register_module() 
class CrossViewFeatureFusion(BaseModule):
    """多视角特征融合模块 - 改进版"""
    def __init__(self, in_channels: int = 64, out_channels: int = 64, num_views: int = 6, chunk_size: int = 1024):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_views = num_views
        self.chunk_size = chunk_size
        
        # 全局视角权重（保留但增加灵活性）
        self.global_view_weights = nn.Parameter(torch.ones(self.num_views))
        
        # 局部权重生成网络（为每个点计算视角重要性）
        self.local_weight_net = nn.Sequential(
            nn.Linear(in_channels, in_channels // 2),
            nn.ReLU(inplace=True),
            nn.Linear(in_channels // 2, 1),
            nn.Sigmoid()
        )
        
        # 特征增强转换
        self.feature_transform = nn.Sequential(
            nn.Linear(in_channels, in_channels),
            nn.LayerNorm(in_channels),
            nn.ReLU(inplace=True)
        )
        
        # 最终融合网络（处理加权融合后的特征）
        self.fusion_net = nn.Sequential(
            nn.Linear(in_channels, out_channels),
            nn.BatchNorm1d(out_channels),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1)
        )
    
    def chunked_process_weights(self, features_list, masks_list):
        """分块处理计算局部权重，减少内存使用"""
        # 全局视角权重
        global_weights = torch.sigmoid(self.global_view_weights)
        
        # 计算总有效点数和设备
        device = features_list[0].device
        valid_mask = torch.stack(masks_list).any(dim=0)
        N_total = valid_mask.shape[0]
        valid_indices = torch.where(valid_mask)[0]
        N_valid = valid_indices.shape[0]
        
        # 预分配结果张量
        fused_features = torch.zeros((N_valid, self.in_channels), device=device)
        valid_count = torch.zeros((N_valid, 1), device=device)
        
        # 分块处理每个视角
        for view_idx, (features, mask) in enumerate(zip(features_list, masks_list)):
            if not mask.any():
                continue
                
            # 只处理当前视角有效的点
            view_valid_mask = mask & valid_mask
            view_valid_indices = torch.where(view_valid_mask)[0]
            
            if view_valid_indices.shape[0] == 0:
                continue
                
            # 映射到有效点索引空间
            mapped_indices = torch.zeros_like(valid_mask, dtype=torch.long)
            mapped_indices[valid_indices] = torch.arange(N_valid, device=device)
            target_indices = mapped_indices[view_valid_indices]
            
            # 获取特征并应用变换
            view_features = features
            transformed_features = self.feature_transform(view_features)
            
            # 计算局部权重
            for i in range(0, view_features.shape[0], self.chunk_size):
                end_idx = min(i + self.chunk_size, view_features.shape[0])
                chunk_features = view_features[i:end_idx]
                chunk_indices = target_indices[i:end_idx]
                
                # 计算这些特征的局部重要性权重
                local_weights = self.local_weight_net(chunk_features)
                
                # 应用全局权重和局部权重
                combined_weight = global_weights[view_idx] * local_weights
                weighted_features = transformed_features[i:end_idx] * combined_weight
                
                # 使用累加方式直接聚合到结果张量
                fused_features.index_add_(0, chunk_indices, weighted_features)
                valid_count.index_add_(0, chunk_indices, torch.ones_like(local_weights))
        
        # 平均汇总（避免偏向于被更多视角看到的点）
        valid_count = torch.clamp(valid_count, min=1.0)  # 防止除零
        fused_features = fused_features / valid_count
        
        return fused_features, valid_mask
        
    def forward(
        self,
        view_features: List[torch.Tensor],
        view_masks: List[torch.Tensor]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        # 输入检查
        assert len(view_features) == self.num_views and len(view_masks) == self.num_views, \
            f"Expected {self.num_views} views, got {len(view_features)} features and {len(view_masks)} masks"
        
        N = view_masks[0].shape[0]  # 总体素数量
        device = view_features[0].device
        
        # 设备和维度一致性检查
        for i, (feat, mask) in enumerate(zip(view_features, view_masks)):
            assert feat.device == device and mask.device == device, \
                f"Device mismatch in view {i}"
            assert feat.shape[1] == self.in_channels, \
                f"View {i} features should have {self.in_channels} channels, got {feat.shape[1]}"
            assert mask.shape[0] == N, \
                f"View {i} mask should have length {N}, got {mask.shape[0]}"
            assert mask.sum() == feat.shape[0], \
                f"View {i} has {feat.shape[0]} features but {mask.sum()} valid points"

        # 将所有mask堆叠，找出任一视角可见的点
        stacked_masks = torch.stack(view_masks)
        valid_mask = stacked_masks.any(dim=0)  # [N]
        
        # 如果没有有效体素，返回零特征
        if not valid_mask.any():
            return torch.zeros((N, self.out_channels), device=device), valid_mask
        
        # 使用分块处理计算权重并融合特征
        fused_features, valid_mask_updated = self.chunked_process_weights(view_features, view_masks)
        
        # 应用最终融合网络
        output_features_valid = self.fusion_net(fused_features)
        
        # 构建完整输出张量
        output_features = torch.zeros((N, self.out_channels), device=device)
        output_features[valid_mask] = output_features_valid
        
        return output_features, valid_mask



# @MODELS.register_module() 
# class CrossViewFeatureFusion(BaseModule):
#     """六视角特征融合模块"""
#     def __init__(self, in_channels: int = 64, out_channels: int = 64):
#         super().__init__()
#         self.in_channels = in_channels
#         self.out_channels = out_channels
#         self.num_views = 6
        
#         # 创建6视角融合网络
#         self.fusion_net = nn.Sequential(
#             nn.Linear(in_channels * self.num_views, out_channels),
#             nn.BatchNorm1d(out_channels),
#             nn.ReLU(inplace=True),
#             nn.Dropout(0.1),  # 添加dropout
#             nn.Linear(out_channels, out_channels),  # 添加残差层
#         )
        
#     def forward(
#         self,
#         view_features: List[torch.Tensor],
#         view_masks: List[torch.Tensor]
#     ) -> Tuple[torch.Tensor, torch.Tensor]:
        


#         # 输入检查
#         assert len(view_features) == self.num_views and len(view_masks) == self.num_views, \
#             f"Expected {self.num_views} views, got {len(view_features)} features and {len(view_masks)} masks"
        
#         N = view_masks[0].shape[0]  # 总体素数量
#         device = view_features[0].device
#         # 设备和维度一致性检查
#         for i, (feat, mask) in enumerate(zip(view_features, view_masks)):
#             assert feat.device == device and mask.device == device, \
#                 f"Device mismatch in view {i}"
#             assert feat.shape[1] == self.in_channels, \
#                 f"View {i} features should have {self.in_channels} channels, got {feat.shape[1]}"
#             assert mask.shape[0] == N, \
#                 f"View {i} mask should have length {N}, got {mask.shape[0]}"
#             assert mask.sum() == feat.shape[0], \
#                 f"View {i} has {feat.shape[0]} features but {mask.sum()} valid points"

#         # 将所有mask堆叠 [6, N]
#         stacked_masks = torch.stack(view_masks)
#         valid_mask = stacked_masks.any(dim=0)  # [N]
        
#         # 如果没有有效体素，返回零特征
#         if not valid_mask.any():
#             return torch.zeros((N, self.out_channels), device=device), valid_mask
        
#         # 创建特征存储tensor [N, 6, C]
#         all_features = torch.zeros((N, self.num_views, self.in_channels), device=device)
        
#         # 填充特征
#         for view_idx, (feat, mask) in enumerate(zip(view_features, view_masks)):
#             if mask.any():
#                 view_features_padded = torch.zeros((N, self.in_channels), device=device)
#                 view_features_padded[mask] = feat
#                 all_features[:, view_idx] = view_features_padded
        
#         # 只处理有效点
#         valid_features = all_features[valid_mask]  # [M, 6, C]
        
#         # 重塑并融合特征
#         batch_features = valid_features.reshape(-1, self.num_views * self.in_channels)
#         fused_features = self.fusion_net(batch_features)
        
#         # 构建输出：无效点保持为零特征
#         output_features = torch.zeros((N, self.out_channels), device=device)
#         output_features[valid_mask] = fused_features
        
#         return output_features, valid_mask
    
# class MultiModalFeatureFusion(nn.Module):
#     """多模态特征融合模块"""
#     def __init__(self, in_channels: int = 64):
#         super().__init__()
#         self.in_channels = in_channels
        
#         # 特征融合网络
#         self.fusion_net = nn.Sequential(
#             nn.Linear(in_channels * 2, in_channels),
#             nn.BatchNorm1d(in_channels),
#             nn.ReLU(inplace=True)
#         )
            
#     def forward(
#         self,
#         voxel_features: torch.Tensor,  # [N, C]
#         image_features: torch.Tensor,  # [N, C]
#         valid_mask: torch.Tensor      # [N]
#     ) -> torch.Tensor:
#         """前向传播
        
#         Args:
#             voxel_features: 点云特征 [N, C]
#             image_features: 图像特征 [N, C]
#             valid_mask: 有效点mask [N]
            
#         Returns:
#             fused_features: 融合后的特征 [N, C]
#         """
#         # 输入维度检查
#         assert voxel_features.shape == image_features.shape, \
#             f"Shape mismatch: voxel_features {voxel_features.shape} != image_features {image_features.shape}"
#         assert valid_mask.shape[0] == voxel_features.shape[0], \
#             f"Mask length {valid_mask.shape[0]} doesn't match feature length {voxel_features.shape[0]}"
            
#         # 如果没有有效体素，直接返回几何特征
#         if not valid_mask.any():
#             return voxel_features
        
#         # 提取有效特征
#         valid_voxel_feat = voxel_features[valid_mask]
#         valid_image_feat = image_features[valid_mask]
        
#         # 特征拼接并融合
#         concat_feat = torch.cat([valid_voxel_feat, valid_image_feat], dim=1)
#         fused_feat = self.fusion_net(concat_feat)
        
#         # 构建输出：无效体素保持原始几何特征
#         output_features = voxel_features.clone()
#         output_features[valid_mask] = fused_feat
        
#         return output_features
        """前向传播，处理6个视角的特征融合
        
        Args:
            view_features: 每个视角的特征 [N_i, C]
            view_masks: 每个视角的有效点mask [N]
        
        Returns:
            fused_features: 融合后的特征 [N, C]
            valid_mask: 至少被一个视角看到的点的mask [N]
        Example:
        # 示例数据
            N = 4  # 总点数
            in_channels = 2  # 特征维度
            num_views = 6  # 视角数

            # 输入数据示例
            view_features = [
                torch.tensor([[1.0, 2.0], [3.0, 4.0]]),      # 视角0看到2个点的特征
                torch.tensor([[5.0, 6.0]]),                  # 视角1看到1个点的特征
                torch.empty((0, 2)),                         # 视角2没看到点
                torch.tensor([[7.0, 8.0]]),                  # 视角3看到1个点的特征
                torch.tensor([[9.0, 10.0]]),                 # 视角4看到1个点的特征
                torch.tensor([[11.0, 12.0]])                 # 视角5看到1个点的特征
            ]

            view_masks = [
                torch.tensor([1, 1, 0, 0]),  # 视角0看到点0,1
                torch.tensor([0, 0, 1, 0]),  # 视角1看到点2
                torch.tensor([0, 0, 0, 0]),  # 视角2没看到点
                torch.tensor([0, 0, 0, 1]),  # 视角3看到点3
                torch.tensor([1, 0, 0, 0]),  # 视角4看到点0
                torch.tensor([0, 1, 0, 0])   # 视角5看到点1
            ]
        """ 