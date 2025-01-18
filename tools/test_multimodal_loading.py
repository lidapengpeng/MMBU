import os
import os.path as osp
import numpy as np
import torch
from mmengine.config import Config
import spconv.pytorch as spconv
from oneformer3d.data_processing.s3dis_multiview_dataset import S3DISMultiViewSegDataset
from oneformer3d.models.DINO_extractor import DINOv2Extractor
import cv2

def test_multimodal_loading():
    """测试多模态数据同步加载"""
    print("\n========== Testing Multimodal Data Loading ==========")
    
    # 1. 加载配置
    config_path = 'configs/oneformer3d_multiview_s3dis.py'
    cfg = Config.fromfile(config_path)
    
    # 2. 初始化数据集
    dataset = S3DISMultiViewSegDataset(
        data_root=cfg.data_root,
        ann_file=f's3dis_infos_Area_{cfg.train_area[0]}.pkl',
        test_mode=False
    )
    
    # 3. 获取一个batch的数据
    sample = dataset[0]
    
    # 4. 验证点云数据
    print("\n----- Point Cloud Verification -----")
    try:
        # 加载点云数据
        lidar_points = sample['lidar_points']
        lidar_path = osp.join(dataset.data_root, lidar_points['lidar_path'])
        points = np.fromfile(lidar_path, dtype=np.float32)
        points = points.reshape(-1, lidar_points['num_pts_feats'])
        
        # 加载语义标签
        semantic_path = osp.join(dataset.data_root, sample['pts_semantic_mask_path'])
        semantic_labels = np.fromfile(semantic_path, dtype=np.int64)
        
        # 加载实例标签
        instance_path = osp.join(dataset.data_root, sample['pts_instance_mask_path'])
        instance_labels = np.fromfile(instance_path, dtype=np.int64)
        
        print(f"Point Cloud Statistics:")
        print(f"- Total points: {len(points)}")
        print(f"- Feature dimensions: {points.shape[1]}")
        print(f"- Semantic labels shape: {semantic_labels.shape}")
        print(f"- Instance labels shape: {instance_labels.shape}")
        print(f"- Point cloud value range: [{points.min():.2f}, {points.max():.2f}]")
        
        # 验证数据一致性
        print("\nData Consistency Check:")
        print(f"- Points and semantic labels match: {len(points) == len(semantic_labels)}")
        print(f"- Points and instance labels match: {len(points) == len(instance_labels)}")
        
        # 统计语义标签
        unique_semantic = np.unique(semantic_labels)
        print("\nSemantic Label Statistics:")
        print(f"- Unique semantic labels: {unique_semantic}")
        for label in unique_semantic:
            count = np.sum(semantic_labels == label)
            print(f"  Label {label}: {count} points ({count/len(semantic_labels)*100:.2f}%)")
        
        # 统计实例标签
        unique_instances = np.unique(instance_labels)
        print("\nInstance Label Statistics:")
        print(f"- Number of unique instances: {len(unique_instances)}")
        print(f"- Instance IDs: {unique_instances}")
        
    except Exception as e:
        print(f"Error in point cloud verification: {str(e)}")
    
    # 5. 验证多视图图像数据
    print("\n----- Multi-view Images Verification -----")
    try:
        image_paths = sample['images']['paths']
        image_poses = []
        for img_path in image_paths:
            pose_info = sample['images']['poses'][osp.basename(img_path)]
            pose = np.array(pose_info['rotation_matrix'])
            pose = pose.reshape(3, 3)
            center = np.array([
                pose_info['center']['x'],
                pose_info['center']['y'],
                pose_info['center']['z']
            ])
            pose_matrix = np.eye(4)
            pose_matrix[:3, :3] = pose
            pose_matrix[:3, 3] = center
            image_poses.append(pose_matrix)
        
        print(f"Multi-view Image Statistics:")
        print(f"- Number of views: {len(image_paths)}")
        print(f"- Image paths:")
        for i, path in enumerate(image_paths):
            print(f"  {i+1}: {path}")
            # 检查图像文件
            img_path = osp.join(dataset.data_root, path)
            img = cv2.imread(img_path)
            if img is not None:
                print(f"    Resolution: {img.shape}")
                print(f"    Value range: [{img.min()}, {img.max()}]")
            
        print("\nCamera Pose Statistics:")
        for i, pose in enumerate(image_poses):
            print(f"\nPose Matrix {i+1}:")
            print(f"- Rotation matrix:")
            print(pose[:3, :3])
            print(f"- Translation vector:")
            print(pose[:3, 3])
            # 验证旋转矩阵的正交性
            R = pose[:3, :3]
            orthogonality_error = np.abs(np.dot(R, R.T) - np.eye(3)).max()
            print(f"- Rotation matrix orthogonality error: {orthogonality_error:.6f}")
    
    except Exception as e:
        print(f"Error in multi-view image verification: {str(e)}")
    
    return sample

def test_feature_extraction(sample, dataset):
    """测试特征提取和内存占用"""
    print("\n========== Testing Feature Extraction and Memory Usage ==========")
    
    # 1. 点云体素化和特征提取
    print("\n----- Point Cloud Feature Extraction -----")
    try:
        # 记录初始GPU内存使用
        torch.cuda.empty_cache()
        initial_memory = torch.cuda.memory_allocated() / 1024**2
        print(f"\nInitial GPU memory usage: {initial_memory:.2f} MB")
        
        # 加载点云数据
        lidar_points = sample['lidar_points']
        lidar_path = os.path.join(dataset.data_root, lidar_points['lidar_path'])
        points = np.fromfile(lidar_path, dtype=np.float32)
        points = points.reshape(-1, lidar_points['num_pts_feats'])
        print(f"\n1. 原始点云数据:")
        print(f"- 形状: {points.shape}")
        print(f"- 特征维度: {lidar_points['num_pts_feats']}")
        print(f"- 点数量: {len(points)}")
        print(f"- 数值范围: [{points.min():.2f}, {points.max():.2f}]")
        
        points = torch.from_numpy(points).float().cuda()
        print(f"\nGPU memory after loading points: {torch.cuda.memory_allocated()/1024**2:.2f} MB")
        
        # 体素化
        voxel_size = 0.33
        coordinates = torch.floor(points[:, :3] / voxel_size).int()
        spatial_shape = coordinates.max(0)[0].cpu().numpy() + 1
        batch_size = 1
        indices = torch.cat([torch.zeros(len(points), 1).cuda(), coordinates], dim=1).int()
        
        print(f"\n2. 体素化结果:")
        print(f"- 体素大小: {voxel_size}")
        print(f"- 空间形状: {spatial_shape}")
        print(f"- 坐标范围: x[{coordinates[:, 0].min().item()}, {coordinates[:, 0].max().item()}], "
              f"y[{coordinates[:, 1].min().item()}, {coordinates[:, 1].max().item()}], "
              f"z[{coordinates[:, 2].min().item()}, {coordinates[:, 2].max().item()}]")
        
        # 统计每个体素中点的数量
        unique_coords, counts = torch.unique(coordinates, dim=0, return_counts=True)
        print(f"- 非空体素数量: {len(unique_coords)}")
        print(f"- 平均每个体素点数: {len(points) / len(unique_coords):.2f}")
        print(f"- 最大体素点数: {counts.max().item()}")
        print(f"- 最小体素点数: {counts.min().item()}")
        
        print(f"\nGPU memory after voxelization: {torch.cuda.memory_allocated()/1024**2:.2f} MB")
        
        # 创建SpConv张量
        sparse_tensor = spconv.SparseConvTensor(
            features=points,
            indices=indices,
            spatial_shape=spatial_shape,
            batch_size=batch_size
        )
        
        print(f"\n3. 稀疏张量信息:")
        print(f"- 特征形状: {sparse_tensor.features.shape}")
        print(f"- 空间形状: {sparse_tensor.spatial_shape}")
        print(f"- 索引形状: {sparse_tensor.indices.shape}")
        
        print(f"\nGPU memory after creating sparse tensor: {torch.cuda.memory_allocated()/1024**2:.2f} MB")
        
        # SpConv特征提取
        subm_conv = spconv.SubMConv3d(
            in_channels=6,
            out_channels=32,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=False
        ).cuda()
        
        sparse_conv = spconv.SparseConv3d(
            in_channels=32,
            out_channels=64,
            kernel_size=3,
            stride=2,
            padding=1,
            bias=False
        ).cuda()
        
        # 前向传播
        print(f"\n4. SpConv特征提取:")
        out_tensor1 = subm_conv(sparse_tensor)
        print(f"- SubMConv3d后:")
        print(f"  - 特征形状: {out_tensor1.features.shape}")
        print(f"  - 空间形状: {out_tensor1.spatial_shape}")
        print(f"  - 非零点数: {len(out_tensor1.indices)}")
        print(f"GPU memory after first conv: {torch.cuda.memory_allocated()/1024**2:.2f} MB")
        
        out_tensor2 = sparse_conv(out_tensor1)
        print(f"\n- SparseConv3d后:")
        print(f"  - 特征形状: {out_tensor2.features.shape}")
        print(f"  - 空间形状: {out_tensor2.spatial_shape}")
        print(f"  - 非零点数: {len(out_tensor2.indices)}")
        print(f"GPU memory after second conv: {torch.cuda.memory_allocated()/1024**2:.2f} MB")
        
        print(f"\n5. 特征统计:")
        print(f"- 初始特征: {sparse_tensor.features.shape}")
        print(f"- SubMConv3d后: {out_tensor1.features.shape}")
        print(f"- SparseConv3d后: {out_tensor2.features.shape}")
        print(f"- 最终空间形状: {out_tensor2.spatial_shape}")
        
        # 保存所有中间特征用于后续处理
        point_features = {
            'coordinates': coordinates,  # 原始体素坐标
            'sparse_tensor': sparse_tensor,  # 原始稀疏张量
            'conv1_features': out_tensor1,  # 第一次卷积特征
            'conv2_features': out_tensor2   # 第二次卷积特征
        }
        
    except Exception as e:
        print(f"Error in point cloud feature extraction: {str(e)}")
        point_features = None
    
    # 2. 多视图图像特征提取
    print("\n----- Multi-view Image Feature Extraction -----")
    try:
        extractor = DINOv2Extractor()
        print(f"\nGPU memory before image feature extraction: {torch.cuda.memory_allocated()/1024**2:.2f} MB")
        
        image_paths = []
        for img_path in sample['images']['paths']:
            full_path = osp.join(dataset.data_root, img_path)
            image_paths.append(full_path)
        
        # 提取特征并保持在GPU中
        features = []
        for i, img_path in enumerate(image_paths):
            print(f"\nProcessing image {i+1}/{len(image_paths)}")
            feat = extractor.extract_features(img_path)
            features.append(feat)  # 特征保持在GPU中
            print(f"GPU memory after processing image {i+1}: {torch.cuda.memory_allocated()/1024**2:.2f} MB")
            
            # 检查特征是否在GPU上
            print(f"Feature {i+1} device: {feat.device}")
            print(f"Feature {i+1} memory: {feat.element_size() * feat.nelement() / 1024**2:.2f} MB")
            
            # 验证之前提取的特征是否仍在GPU上
            if i > 0:
                for j in range(i):
                    print(f"Previous feature {j+1} device: {features[j].device}")
                    print(f"Previous feature {j+1} still requires grad: {features[j].requires_grad}")
        
        print(f"\nDINOv2 Feature Statistics:")
        for i, feat in enumerate(features):
            print(f"View {i+1} feature shape: {feat.shape}")
            print(f"View {i+1} device: {feat.device}")
            print(f"View {i+1} memory: {feat.element_size() * feat.nelement() / 1024**2:.2f} MB")
        
        # 测试特征投影
        print("\n----- Testing Feature Projection -----")
        try:
            # 获取第一个视角的相机参数作为示例
            pose_info = sample['images']['poses'][osp.basename(image_paths[0])]
            R = np.array(pose_info['rotation_matrix']).reshape(3, 3)
            t = np.array([
                pose_info['center']['x'],
                pose_info['center']['y'],
                pose_info['center']['z']
            ])
            K = np.array([
                [pose_info['camera_intrinsics']['focal_length'], 0, pose_info['camera_intrinsics']['principal_point']['x']],
                [0, pose_info['camera_intrinsics']['focal_length'], pose_info['camera_intrinsics']['principal_point']['y']],
                [0, 0, 1]
            ])
            
            # 获取体素坐标
            voxel_coords = point_features['coordinates'].float()
            
            # 转换为世界坐标（示例，实际需要根据体素大小调整）
            world_coords = voxel_coords * voxel_size
            
            # 转换为相机坐标
            R_tensor = torch.from_numpy(R).float().cuda()
            t_tensor = torch.from_numpy(t).float().cuda()
            K_tensor = torch.from_numpy(K).float().cuda()
            
            # 计算投影
            cam_coords = torch.matmul(R_tensor, world_coords.t()).t() + t_tensor
            
            # 投影到图像平面
            proj_coords = torch.matmul(K_tensor, cam_coords.t()).t()
            proj_coords = proj_coords[:, :2] / proj_coords[:, 2:3]
            
            print("\nProjection Statistics:")
            print(f"- Number of projected points: {len(proj_coords)}")
            print(f"- Projection coordinate range: [{proj_coords.min().item():.2f}, {proj_coords.max().item():.2f}]")
            print(f"- Points in image bounds: {((proj_coords[:, 0] >= 0) & (proj_coords[:, 0] < 1204) & (proj_coords[:, 1] >= 0) & (proj_coords[:, 1] < 896)).sum().item()}")
            
        except Exception as e:
            print(f"Error in feature projection test: {str(e)}")
            
    except Exception as e:
        print(f"Error in image feature extraction: {str(e)}")
        features = None
    
    # 最终内存使用统计
    print(f"\nFinal GPU memory usage: {torch.cuda.memory_allocated()/1024**2:.2f} MB")
    print(f"Max GPU memory usage: {torch.cuda.max_memory_allocated()/1024**2:.2f} MB")
    
    return point_features, features

def main():
    """主测试函数"""
    try:
        # 1. 加载配置
        config_path = 'configs/oneformer3d_multiview_s3dis.py'
        cfg = Config.fromfile(config_path)
        
        # 2. 初始化数据集
        dataset = S3DISMultiViewSegDataset(
            data_root=cfg.data_root,
            ann_file=f's3dis_infos_Area_{cfg.train_area[0]}.pkl',
            test_mode=False
        )
        
        # 3. 测试数据加载
        sample = test_multimodal_loading()
        
        # 4. 测试特征提取和内存使用
        point_features, image_features = test_feature_extraction(sample, dataset)
        
        print("\n========== All Tests Completed Successfully ==========")
        
    except Exception as e:
        print(f"\nError during testing: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main() 