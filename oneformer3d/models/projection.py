import torch
from typing import Tuple, Dict, Optional
import numpy as np


class VoxelProjector:
    def __init__(self, voxel_size: float = 0.33):
        """初始化体素投影器"""
        self.voxel_size = voxel_size

    def project_voxels(self, voxel_coords: torch.Tensor, 
                      camera_info: Dict) -> Tuple[torch.Tensor, torch.Tensor, Dict]:
        """体素投影到图像平面
        
        Args:
            voxel_coords: [N, 3] 体素中心点坐标
            camera_info: 相机参数字典，包含投影矩阵和图像大小
            
        Returns:
            proj_points: [N, 2] 投影点坐标
            valid_mask: [N] 有效投影mask
            stats: 投影统计信息
        """
        # 验证相机信息
        if 'image_size' not in camera_info:
            raise ValueError("Camera info must contain 'image_size' field")
        H, W = camera_info['image_size']
        
        # 确保H和W为标量
        if isinstance(H, torch.Tensor):
            H = H.item()
        if isinstance(W, torch.Tensor):
            W = W.item()
        
        # 获取投影矩阵
        proj_mat = camera_info['projection_matrix']
        
        # 将体素坐标转换到世界坐标
        points = voxel_coords * self.voxel_size
        # 添加齐次坐标
        points_homogeneous = torch.ones((points.shape[0], 4), dtype=points.dtype, device=points.device)
        points_homogeneous[:, :3] = points
        
        # 应用投影矩阵 P
        projected = torch.matmul(points_homogeneous, proj_mat.T)
        
        # 深度检查 - 确保点在相机前方
        depth_mask = projected[:, 2] > 0
        
        # 执行NDC转换 (normalized device coordinates)
        ndc = projected.clone()
        ndc = ndc / ndc[:, 2:3]  # 透视除法，将所有坐标除以z
        
        # 转换NDC坐标到像素坐标
        points_2d = torch.zeros((points.shape[0], 2), dtype=points.dtype, device=points.device)
        points_2d[:, 0] = (ndc[:, 0] + 1.0) * 0.5 * W
        points_2d[:, 1] = (1.0 - (ndc[:, 1] + 1.0) * 0.5) * H  # 翻转Y轴，因为图像坐标从上到下增加
        
        # 先将坐标转换为整数
        points_2d = points_2d.round().to(torch.int32)
        # 然后对x坐标进行镜像变换
        points_2d[:, 0] = W - points_2d[:, 0]
        
        # 图像边界检查
        x, y = points_2d[:, 0], points_2d[:, 1]
        boundary_mask = (x >= 0) & (x < W) & (y >= 0) & (y < H)
        
        # 合并所有mask
        valid_mask = depth_mask & boundary_mask
        
        
        return points_2d, valid_mask
    

if __name__ == "__main__":
    # 通过在体素坐标系下选择几个3D点，投影到2D图像，进而验证投影的正确性
    # 创建VoxelProjector实例
    projector = VoxelProjector(voxel_size=0.33)
    
    # 设置测试点坐标
    voxel_points = torch.tensor([
        [323, 164, 58],  # 点1
        [318, 76, 63],   # 点2
        [389, 195, 28],  # 点3
        [342, 157, 106], # 点4
        [343, 81, 102],  # 点5
        [346, 97, 91]    # 点6
    ], dtype=torch.float32, device='cuda:0')
    # 使用上下文中的投影矩阵
    P_4x4 = torch.tensor([          
        [-3.8913e-02,  1.5212e+00, -2.0574e-02, -1.5478e+01],
        [ 1.8245e+00,  5.8655e-02,  8.8614e-01, -2.6710e+02],
        [ 4.3691e-01, -9.8940e-04, -8.9951e-01,  5.0190e+01],
        [ 4.3691e-01, -9.8940e-04, -8.9951e-01,  5.0190e+01]], device='cuda:0')
    # 设置相机参数
    camera_info = {
        'image_size': (torch.tensor(3648., device='cuda:0'), torch.tensor(4864., device='cuda:0')),
        'projection_matrix': P_4x4
    }
    
    # 执行投影
    projected_points, valid_mask, stats = projector.project_voxels(voxel_points, camera_info)
    
    # 打印结果
    print("\n=== 测试VoxelProjector ===")
    print(f"输入体素坐标形状: {voxel_points.shape}")
    print(f"\n投影统计信息:")
    print(f"- 总点数: {stats['total_points']}")
    print(f"- 深度有效点数: {stats['depth_valid']} ({stats['depth_valid']/stats['total_points']*100:.2f}%)")
    print(f"- 图像边界内点数: {stats['boundary_valid']} ({stats['boundary_valid']/stats['total_points']*100:.2f}%)")
    print(f"- 最终有效点数: {stats['final_valid']} ({stats['final_valid']/stats['total_points']*100:.2f}%)")
    
    print("\n有效投影点的对应关系:")
    for i, (valid, point_3d, point_2d) in enumerate(zip(valid_mask, voxel_points, projected_points)):
        if valid:
            print(f"点{i+1} - 3D坐标: {point_3d.tolist()} -> 2D坐标: {point_2d.tolist()}")
        else:
            print(f"点{i+1} - 3D坐标: {point_3d.tolist()} -> 投影无效")
