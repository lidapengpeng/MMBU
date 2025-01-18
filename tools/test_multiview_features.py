import os
import os.path as osp
import numpy as np
from mmengine.config import Config
from oneformer3d.data_processing.s3dis_multiview_dataset import S3DISMultiViewSegDataset
from oneformer3d.models.DINO_extractor import DINOv2Extractor
from oneformer3d.models.visualization import visualize_multiview_features, create_multiview_grid
from PIL import Image

def test_feature_extraction():
    """测试多视图特征提取"""
    print("\n========== Testing Multi-view Feature Extraction ==========")
    
    # 1. 加载配置
    config_path = 'configs/oneformer3d_multiview_s3dis.py'
    cfg = Config.fromfile(config_path)
    
    # 2. 初始化数据集
    dataset = S3DISMultiViewSegDataset(
        data_root=cfg.data_root,
        ann_file=f's3dis_infos_Area_{cfg.train_area[0]}.pkl',
        test_mode=False
    )
    
    # 3. 获取第一个样本的图像路径
    sample = dataset[0]
    image_paths = []
    for img_path in sample['images']['paths']:
        full_path = osp.join(dataset.data_root, img_path)
        image_paths.append(full_path)
        # 检查图像尺寸
        img = Image.open(full_path)
        print(f"Image size for {img_path}: {img.size}")
    
    print(f"\nFound {len(image_paths)} images")
    for i, path in enumerate(image_paths):
        print(f"Image {i+1}: {path}")
    
    # 4. 初始化DINO特征提取器
    extractor = DINOv2Extractor()
    
    # 5. 提取特征
    print("\nExtracting features...")
    features = extractor.extract_batch_features(image_paths)
    
    # 6. 可视化特征
    print("\nVisualizing features...")
    output_dir = "visualizations/multiview_features"
    feature_images = visualize_multiview_features(
        features,
        output_dir=output_dir,
        prefix="dino_feature"
    )
    
    # 7. 创建特征网格图
    if feature_images:
        grid_path = osp.join(output_dir, "feature_grid.png")
        grid_image = create_multiview_grid(
            feature_images,
            grid_size=(2, 3),  # 2行3列的网格
            output_path=grid_path
        )
        print(f"\nGrid image saved to: {grid_path}")
    
    return features, feature_images

def main():
    """主测试函数"""
    try:
        features, feature_images = test_feature_extraction()
        print("\n========== All Tests Completed Successfully ==========")
        
    except Exception as e:
        print(f"\nError during testing: {str(e)}")
        raise

if __name__ == "__main__":
    main() 