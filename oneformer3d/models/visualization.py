import os
import numpy as np
import torch
from PIL import Image
from sklearn.decomposition import PCA
import os
import torch
from typing import List, Tuple

def visualize_feature_map_multistage(feature_map, pca_stages=[256, 64, 16, 3], output_path=None):
    """
    使用多阶段 PCA 将高维特征图转换为可视化的 RGB 图像
    
    Args:
        feature_map: [C, H, W] 的特征图张量
        pca_stages: PCA 降维阶段列表
        output_path: 保存路径（可选）
    Returns:
        PIL Image 对象
    """
    # 确保输入是CPU上的numpy数组
    if torch.is_tensor(feature_map):
        feature_map = feature_map.detach().cpu().numpy()
    
    C, H, W = feature_map.shape
    feature_map_np = feature_map.reshape(C, -1).T  # [H*W, C]
    curr_features = feature_map_np

    # 多阶段 PCA 降维
    for n_components in pca_stages:
        n_components = min(n_components, curr_features.shape[1])
        pca = PCA(n_components=n_components)
        curr_features = pca.fit_transform(curr_features)
        explained_var = sum(pca.explained_variance_ratio_)
        print(f"PCA降维到{n_components}个分量, 解释方差: {explained_var:.4f}")

    # 归一化到 [0, 1]
    min_vals = curr_features.min(axis=0)
    max_vals = curr_features.max(axis=0)
    curr_features = (curr_features - min_vals) / (max_vals - min_vals + 1e-5)
    curr_features = curr_features.reshape(H, W, -1)

    # 如果最终维度大于3，取前3个主成分
    if curr_features.shape[2] > 3:
        curr_features = curr_features[:, :, :3]

    # 如果最终维度小于3，进行补充
    if curr_features.shape[2] < 3:
        padding = np.zeros((H, W, 3 - curr_features.shape[2]))
        curr_features = np.concatenate((curr_features, padding), axis=2)

    # 转换为图像
    feature_image = (curr_features * 255).astype(np.uint8)
    pil_image = Image.fromarray(feature_image)

    # 保存图像
    if output_path:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        pil_image.save(output_path)
        print(f"可视化结果已保存至: {output_path}")

    return pil_image

def visualize_multiview_features(features_list, output_dir="visualizations", prefix="feature"):
    """
    可视化多视图特征
    Args:
        features_list: 特征张量列表，每个张量形状为[C, H, W]
        output_dir: 输出目录
        prefix: 输出文件名前缀
    Returns:
        feature_images: 特征图路径列表
    """
    os.makedirs(output_dir, exist_ok=True)
    feature_images = []
    
    for i, features in enumerate(features_list):
        # 确保features是numpy数组
        if torch.is_tensor(features):
            features = features.detach().cpu().numpy()
            
        # 打印特征统计信息
        print(f"\n特征 {i} 统计信息:")
        print(f"形状: {features.shape}")
        print(f"最小值: {features.min():.4f}")
        print(f"最大值: {features.max():.4f}")
        print(f"均值: {features.mean():.4f}")
        print(f"标准差: {features.std():.4f}")
        
        # 使用PCA降维到3通道
        n_components = min(3, features.shape[0])
        features_flat = features.reshape(features.shape[0], -1).T
        pca = PCA(n_components=n_components)
        features_pca = pca.fit_transform(features_flat)
        print(f"PCA解释方差比: {pca.explained_variance_ratio_}")
        
        # 重塑回图像形状
        features_img = features_pca.reshape(features.shape[1], features.shape[2], -1)
        
        # 归一化到[0, 1]范围
        features_img = (features_img - features_img.min()) / (features_img.max() - features_img.min())
        
        # 如果通道数小于3，复制到3通道
        if features_img.shape[-1] < 3:
            features_img = np.repeat(features_img, 3, axis=-1)
        
        # 转换为PIL图像
        img = Image.fromarray((features_img * 255).astype(np.uint8))
        
        # 保存图像
        output_path = os.path.join(output_dir, f"{prefix}_{i}.png")
        img.save(output_path)
        print(f"特征图已保存至: {output_path}")
        feature_images.append(output_path)
    
    return feature_images

def create_multiview_grid(image_paths, grid_size=(2, 3), output_path=None):
    """创建多视图网格图
    Args:
        image_paths: 图像路径列表
        grid_size: 网格大小 (rows, cols)
        output_path: 输出路径
    Returns:
        grid_image: 网格图像
    """
    if not image_paths:
        print("没有可用的特征图")
        return None
        
    # 加载所有图像
    images = []
    for path in image_paths:
        try:
            img = Image.open(path)
            images.append(img)
        except Exception as e:
            print(f"加载图像失败 {path}: {str(e)}")
            
    if not images:
        print("没有成功加载任何图像")
        return None
        
    # 计算网格尺寸
    n_rows, n_cols = grid_size
    cell_width = max(img.width for img in images)
    cell_height = max(img.height for img in images)
    
    # 创建空白网格
    grid_width = n_cols * cell_width
    grid_height = n_rows * cell_height
    grid_image = Image.new('RGB', (grid_width, grid_height), (255, 255, 255))
    
    # 填充网格
    for idx, img in enumerate(images):
        if idx >= n_rows * n_cols:
            break
            
        row = idx // n_cols
        col = idx % n_cols
        x = col * cell_width
        y = row * cell_height
        
        # 调整图像大小以适应网格
        if img.size != (cell_width, cell_height):
            img = img.resize((cell_width, cell_height))
            
        grid_image.paste(img, (x, y))
        
    # 保存网格图像
    if output_path:
        grid_image.save(output_path)
        print(f"网格图像已保存到: {output_path}")
        
    return grid_image
