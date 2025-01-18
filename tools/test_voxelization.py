import os
import numpy as np
import torch
from mmengine.config import Config
import spconv.pytorch as spconv
from oneformer3d.data_processing.s3dis_multiview_dataset import S3DISMultiViewSegDataset

def load_point_cloud(dataset):
    """加载点云数据"""
    print("\n========== Loading Point Cloud ==========")
    
    # 加载第一个样本
    sample = dataset[0]
    points_path = os.path.join(dataset.data_root, sample['lidar_points']['lidar_path'])
    points = np.fromfile(points_path, dtype=np.float32)
    points = points.reshape(-1, sample['lidar_points']['num_pts_feats'])
    
    print(f"\nPoint Cloud Statistics:")
    print(f"- Total points: {len(points)}")
    print(f"- Feature dimensions: {points.shape[1]}")
    print(f"- Memory usage: {points.nbytes / 1024 / 1024:.2f} MB")
    print(f"- Point cloud path: {points_path}")
    
    return points

def test_spconv_basic(points, voxel_size=0.33):
    """测试SpConv的基本功能"""
    print("\n========== Testing SpConv Basic Functions ==========")
    
    # 1. 转换点云数据为张量
    points_tensor = torch.from_numpy(points).float().cuda()
    
    # 2. 创建网格坐标
    coordinates = torch.floor(points_tensor[:, :3] / voxel_size).int()
    
    # 3. 创建SpConv张量
    spatial_shape = coordinates.max(0)[0].cpu().numpy() + 1
    batch_size = 1
    indices = torch.cat([torch.zeros(len(points), 1).cuda(), coordinates], dim=1).int()
    features = points_tensor
    
    sparse_tensor = spconv.SparseConvTensor(
        features=features,
        indices=indices,
        spatial_shape=spatial_shape,
        batch_size=batch_size
    )
    
    print(f"\nSpConv tensor created:")
    print(f"- Features shape: {sparse_tensor.features.shape}")
    print(f"- Indices shape: {sparse_tensor.indices.shape}")
    print(f"- Spatial shape: {sparse_tensor.spatial_shape}")
    print(f"- Batch size: {sparse_tensor.batch_size}")
    
    # 4. 测试不同类型的卷积操作
    print("\nTesting different SpConv operations:")
    
    # SubMConv3d测试
    subm_conv = spconv.SubMConv3d(
        in_channels=6,
        out_channels=32,
        kernel_size=3,
        stride=1,
        padding=1,
        bias=False
    ).cuda()
    
    # SparseConv3d测试
    sparse_conv = spconv.SparseConv3d(
        in_channels=32,
        out_channels=64,
        kernel_size=3,
        stride=2,
        padding=1,
        bias=False
    ).cuda()
    
    try:
        # SubMConv3d前向传播
        out_tensor1 = subm_conv(sparse_tensor)
        print("\nSubMConv3d test:")
        print(f"- Output features shape: {out_tensor1.features.shape}")
        print(f"- Output spatial shape: {out_tensor1.spatial_shape}")
        
        # SparseConv3d前向传播
        out_tensor2 = sparse_conv(out_tensor1)
        print("\nSparseConv3d test:")
        print(f"- Output features shape: {out_tensor2.features.shape}")
        print(f"- Output spatial shape: {out_tensor2.spatial_shape}")
        
        return True
    except Exception as e:
        print(f"\nError during convolution: {str(e)}")
        return False

def main():
    """主函数"""
    # 1. 加载配置
    print("\n========== Loading Configuration ==========")
    config_path = 'configs/oneformer3d_multiview_s3dis.py'
    cfg = Config.fromfile(config_path)
    print(f"- Dataset type: {cfg.dataset_type}")
    print(f"- Data root: {cfg.data_root}")
    
    # 2. 创建数据集实例
    dataset = S3DISMultiViewSegDataset(
        data_root=cfg.data_root,
        ann_file='s3dis_infos_Area_1.pkl'
    )
    
    # 3. 加载点云数据
    points = load_point_cloud(dataset)
    
    # 4. 测试SpConv功能
    success = test_spconv_basic(points)
    if success:
        print("\nSpConv validation completed successfully!")
    else:
        print("\nSpConv validation failed!")

if __name__ == '__main__':
    main() 