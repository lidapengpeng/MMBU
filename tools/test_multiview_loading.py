import os.path as osp
import numpy as np
import cv2
from mmengine.config import Config
from oneformer3d.data_processing.s3dis_multiview_dataset import S3DISMultiViewSegDataset

def test_basic_config():
    """测试基础配置加载"""
    print("\n========== Testing Basic Config ==========")
    
    config_path = 'configs/oneformer3d_multiview_s3dis.py'
    cfg = Config.fromfile(config_path)
    
    print("\nBasic Config:")
    print(f"- Dataset type: {cfg.dataset_type}")
    print(f"- Data root: {cfg.data_root}")
    print(f"- Train areas: {cfg.train_area}")
    print(f"- Test area: {cfg.test_area}")
    
    return cfg

def test_dataset_initialization(cfg):
    """测试数据集初始化"""
    print("\n========== Testing Dataset Initialization ==========")
    
    dataset = S3DISMultiViewSegDataset(
        data_root=cfg.data_root,
        ann_file=f's3dis_infos_Area_{cfg.train_area[0]}.pkl',
        test_mode=False
    )
    
    print(f"\nDataset Info:")
    print(f"- Total samples: {len(dataset)}")
    
    sample = dataset[0]
    print(f"\nFirst Sample Keys:")
    for key in sample.keys():
        print(f"- {key}")
        
    return dataset

def test_point_cloud_loading(dataset):
    """测试点云数据加载"""
    print("\n========== Testing Point Cloud Loading ==========")
    
    sample = dataset[0]
    lidar_points = sample['lidar_points']
    
    print("\nPoint Cloud Config:")
    print(f"- Feature dimensions: {lidar_points['num_pts_feats']}")
    
    # 加载点云数据
    lidar_path = osp.join(dataset.data_root, lidar_points['lidar_path'])
    print(f"- Point cloud path: {lidar_path}")
    
    try:
        points = np.fromfile(lidar_path, dtype=np.float32)
        points = points.reshape(-1, lidar_points['num_pts_feats'])
        
        print("\nPoint Cloud Statistics:")
        print(f"- Total points: {len(points)}")
        print(f"- Feature dimensions: {points.shape[1]}")
        print(f"- XYZ bounds:")
        print(f"  Min: {points[:, :3].min(axis=0)}")
        print(f"  Max: {points[:, :3].max(axis=0)}")
        print(f"- RGB bounds:")
        print(f"  Min: {points[:, 3:6].min(axis=0)}")
        print(f"  Max: {points[:, 3:6].max(axis=0)}")
        
        # 数据有效性检查
        print("\nData Validation:")
        print(f"- Contains NaN: {np.isnan(points).any()}")
        print(f"- Contains Inf: {np.isinf(points).any()}")
        
        return points
        
    except Exception as e:
        print(f"Error loading point cloud: {str(e)}")
        return None

def test_semantic_labels(dataset):
    """测试语义标签加载"""
    print("\n========== Testing Semantic Labels ==========")
    
    sample = dataset[0]
    semantic_path = osp.join(dataset.data_root, sample['pts_semantic_mask_path'])
    print(f"\nSemantic Mask Path: {semantic_path}")
    
    try:
        semantic_mask = np.fromfile(semantic_path, dtype=np.int64)
        
        print("\nSemantic Label Statistics:")
        print(f"- Total points: {len(semantic_mask)}")
        unique_labels = np.unique(semantic_mask)
        print(f"- Unique labels: {unique_labels}")
        
        # 统计每个类别的点数
        print("\nPoints per category:")
        for label in unique_labels:
            count = np.sum(semantic_mask == label)
            print(f"  Label {label}: {count} points ({count/len(semantic_mask)*100:.2f}%)")
            
        return semantic_mask
        
    except Exception as e:
        print(f"Error loading semantic labels: {str(e)}")
        return None

def test_instance_labels(dataset):
    """测试实例标签加载"""
    print("\n========== Testing Instance Labels ==========")
    
    sample = dataset[0]
    instance_path = osp.join(dataset.data_root, sample['pts_instance_mask_path'])
    print(f"\nInstance Mask Path: {instance_path}")
    
    try:
        instance_mask = np.fromfile(instance_path, dtype=np.int64)
        
        print("\nInstance Label Statistics:")
        print(f"- Total points: {len(instance_mask)}")
        unique_instances = np.unique(instance_mask)
        print(f"- Number of instances: {len(unique_instances)-1}")  # 减去背景
        
        # 统计每个实例的点数
        print("\nPoints per instance:")
        instance_counts = np.bincount(instance_mask)
        for inst_id, count in enumerate(instance_counts):
            if inst_id == 0:  # 背景
                print(f"  Background: {count} points ({count/len(instance_mask)*100:.2f}%)")
            elif count > 0:
                print(f"  Instance {inst_id}: {count} points ({count/len(instance_mask)*100:.2f}%)")
                
        return instance_mask
        
    except Exception as e:
        print(f"Error loading instance labels: {str(e)}")
        return None

def test_multiview_images(dataset):
    """测试多视角图像加载"""
    print("\n========== Testing Multi-view Images ==========")
    
    sample = dataset[0]
    if 'images' not in sample:
        print("No image data found in sample")
        return None
        
    print(f"\nImage Info:")
    print(f"- Number of images: {len(sample['images']['paths'])}")
    
    images = []
    for i, img_path in enumerate(sample['images']['paths']):
        full_path = osp.join(dataset.data_root, img_path)
        print(f"\nImage {i+1}:")
        print(f"- Path: {img_path}")
        
        try:
            img = cv2.imread(full_path)
            print(f"- Resolution: {img.shape}")
            print(f"- Value range: [{img.min()}, {img.max()}]")
            
            # 获取相机参数
            pose_info = sample['images']['poses'][osp.basename(img_path)]
            print(f"- Camera Parameters:")
            print(f"  - Focal length: {pose_info['camera_intrinsics']['focal_length']:.2f}")
            print(f"  - Principal point: ({pose_info['camera_intrinsics']['principal_point']['x']:.2f}, "
                  f"{pose_info['camera_intrinsics']['principal_point']['y']:.2f})")
            
            images.append(img)
            
        except Exception as e:
            print(f"Error loading image: {str(e)}")
            
    return images if images else None

def test_image_preprocessing(dataset):
    """测试图像预处理"""
    print("\n========== Testing Image Preprocessing ==========")
    
    sample = dataset[0]
    if 'img' not in sample:
        print("No image data found in sample")
        return None
        
    print(f"\nImage Preprocessing Info:")
    print(f"- Number of images: {len(sample['img'])}")
    
    for i, img in enumerate(sample['img']):
        print(f"\nProcessed Image {i+1}:")
        print(f"- Shape: {img.shape}")
        print(f"- Value range: [{img.min():.2f}, {img.max():.2f}]")
        print(f"- Mean: {img.mean():.2f}")
        print(f"- Std: {img.std():.2f}")

def verify_data_consistency(points, semantic_mask, instance_mask):
    """验证数据一致性"""
    print("\n========== Verifying Data Consistency ==========")
    
    if points is None or semantic_mask is None or instance_mask is None:
        print("Missing required data for consistency check")
        return
        
    print("\nPoint Cloud and Label Consistency:")
    print(f"- Point cloud size: {len(points)}")
    print(f"- Semantic mask size: {len(semantic_mask)}")
    print(f"- Instance mask size: {len(instance_mask)}")
    
    if len(points) == len(semantic_mask) == len(instance_mask):
        print("✓ All data sizes match")
    else:
        print("✗ Data size mismatch!")
        
    # 检查标签的有效性
    print("\nLabel Validity:")
    print(f"- Semantic label range: [{semantic_mask.min()}, {semantic_mask.max()}]")
    print(f"- Instance label range: [{instance_mask.min()}, {instance_mask.max()}]")
    print(f"- Number of unique semantic labels: {len(np.unique(semantic_mask))}")
    print(f"- Number of unique instance labels: {len(np.unique(instance_mask))}")

def main():
    """主测试函数"""
    try:
        # 1. 测试配置加载
        cfg = test_basic_config()
        
        # 2. 测试数据集初始化
        dataset = test_dataset_initialization(cfg)
        
        # 3. 测试点云加载
        points = test_point_cloud_loading(dataset)
        
        # 4. 测试语义标签
        semantic_mask = test_semantic_labels(dataset)
        
        # 5. 测试实例标签
        instance_mask = test_instance_labels(dataset)
        
        # 6. 测试多视角图像
        images = test_multiview_images(dataset)
        
        # 7. 测试图像预处理
        test_image_preprocessing(dataset)
        
        # 8. 验证数据一致性
        verify_data_consistency(points, semantic_mask, instance_mask)
        
        print("\n========== All Tests Completed Successfully ==========")
        
    except Exception as e:
        print(f"\nError during testing: {str(e)}")
        import traceback
        print(traceback.format_exc())

if __name__ == '__main__':
    main()