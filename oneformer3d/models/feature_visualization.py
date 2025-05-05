import os
import numpy as np
import torch
from datetime import datetime
from sklearn.decomposition import PCA
import pickle
from PIL import Image
from sklearn.cluster import KMeans
#######################################################################################################
# ---------------------------------体素特征可视化-----------------------------------------------------
#######################################################################################################

"""
    1、查看点云分支、多视角图像分支、双模态融合是否正确，使用方法
    在fuse_features函数中，在推理模式下，添加如下代码：
        # if not self.training:  # 仅在推理模式下进行可视化
        #     from .feature_visualization import visualize_features
        #     # 创建输出目录
        #     os.makedirs('feature_visualizations', exist_ok=True)
        #     viz_results = visualize_features(
        #         fused_image_features,
        #         voxel_coords,
        #         output_path='feature_visualizations'
        #     )
        #     viz_results = visualize_features(
        #         point_features,
        #         voxel_coords,
        #         output_path='feature_visualizations'
        #     )
        #     viz_results = visualize_features(
        #         fused_features,
        #         voxel_coords,
        #         output_path='feature_visualizations'
        #     )
"""
def visualize_features(features, voxel_coords, output_path, prefix='', random_state=42):
    """
    Visualize 64-dimensional features by reducing to RGB colors using optimized two-step PCA.
    
    Args:
        features (torch.Tensor or np.ndarray): Feature tensor of shape [N, 64]
        voxel_coords (torch.Tensor or np.ndarray): Voxel coordinates of shape [N, 3]
        output_path (str): Directory to save visualization files
        prefix (str): Optional prefix for the output filename
        random_state (int): Random seed for reproducibility
        valid_mask (torch.Tensor or np.ndarray, optional): Boolean mask of shape [N] indicating valid features
    
    Returns:
        str: Path to the saved visualization file
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_path, exist_ok=True)
    
    # Convert tensors to numpy arrays if needed
    features_np = features.detach().cpu().numpy() if isinstance(features, torch.Tensor) else features
    voxel_coords_np = voxel_coords.detach().cpu().numpy() if isinstance(voxel_coords, torch.Tensor) else voxel_coords
    
    # 特征维度降维前先进行对比度增强
    # 分开处理非零特征和零特征
    non_zero_mask = ~np.all(features_np == 0, axis=1)
    
    if np.any(non_zero_mask):
        # 只对非零特征进行增强和PCA
        non_zero_features = features_np[non_zero_mask]
        
        # 计算非零特征的均值和标准差
        mean = np.mean(non_zero_features, axis=0, keepdims=True)
        std = np.std(non_zero_features, axis=0, keepdims=True)
        
        # Z-score标准化后拉伸对比度
        normalized_features = (non_zero_features - mean) / (std + 1e-8)
        enhanced_features = np.tanh(normalized_features * 2)  # 非线性增强对比度
        
        # Two-step PCA optimized for non-zero features
        print("Applying two-step PCA reduction...")
        
        # Step 1: Reduce from original dim to 16 dimensions
        n_components = min(16, enhanced_features.shape[0], enhanced_features.shape[1])
        pca1 = PCA(n_components=n_components, random_state=random_state)
        features_mid = pca1.fit_transform(enhanced_features)
        explained_var1 = sum(pca1.explained_variance_ratio_) * 100
        print(f"Step 1: {enhanced_features.shape[1]} → 16 dimensions, explained variance: {explained_var1:.2f}%")
        
        # Step 2: Reduce from 16 to 3 dimensions
        pca2 = PCA(n_components=3, random_state=random_state)
        features_rgb = pca2.fit_transform(features_mid)
        explained_var2 = sum(pca2.explained_variance_ratio_) * 100
        print(f"Step 2: 16 → 3 dimensions, explained variance: {explained_var2:.2f}%")
        
        # 归一化到 [0, 255] 范围
        features_min = np.min(features_rgb, axis=0, keepdims=True)
        features_max = np.max(features_rgb, axis=0, keepdims=True)
        features_rgb = ((features_rgb - features_min) / (features_max - features_min + 1e-10)) * 255
        
        # 创建完整的RGB特征数组（包含零特征）
        all_features_rgb = np.zeros((features_np.shape[0], 3))
        all_features_rgb[non_zero_mask] = features_rgb
        
        features_rgb = all_features_rgb
    else:
        print("No non-zero features found!")
        features_rgb = np.zeros((features_np.shape[0], 3))
    
    # Calculate approximate total explained variance
    total_explained = (explained_var1/100) * (explained_var2/100) * 100
    print(f"Approximate total explained variance: {total_explained:.2f}%")
    
    # Create output file name with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    feature_type = prefix if prefix else "feature"
    filename = f"{feature_type}_pca_{timestamp}.txt"
    file_path = os.path.join(output_path, filename)
    
    # Combine coordinates and RGB values
    output_data = np.hstack((voxel_coords_np, features_rgb))
    
    # Save to file
    print(f"Saving visualization to {file_path}...")
    np.savetxt(file_path, output_data, fmt='%.6f', delimiter=' ')
    
    return file_path



"""
    2、查看投影是否正确，使用方法
    在extract_image_features函数中，在推理模式下，添加如下代码：
        # if  len(view_masks) > 0:
        #     os.makedirs('visualization_data', exist_ok=True)
        #     save_path = os.path.join('visualization_data', f'voxel_viz_data_{torch.rand(1).item():.4f}.pkl')
        #     save_voxel_visualization_data(
                voxel_coords=voxel_coords, 
                view_features=view_features,
                view_masks=view_masks,
                proj_matrices=proj_matrices,
                save_path=save_path
            )
"""
def save_voxel_visualization_data(voxel_coords, view_features, view_masks, proj_matrices, 
                                 voxel_size=0.333, save_path="voxel_viz_data.pkl"):
    """
    Guarda datos de vóxeles y sus proyecciones para visualización en Open3D.
    
    Args:
        voxel_coords (torch.Tensor): Coordenadas de vóxeles [N, 3]
        view_features (list): Lista de características por vista
        view_masks (list): Lista de máscaras de proyección válidas por vista
        proj_matrices (list): Lista de matrices de proyección por vista
        voxel_size (float): Tamaño de cada vóxel (por defecto 0.333)
        save_path (str): Ruta donde guardar el archivo
    """
    try:
        print(f"\n=== Guardando datos de visualización de vóxeles ===")
        print(f"Ruta de guardado: {os.path.abspath(save_path)}")
        
        # Verificar que los datos de entrada sean válidos
        print(f"Verificando datos de entrada:")
        print(f"- Tipo de voxel_coords: {type(voxel_coords)}")
        print(f"- Número de vistas en view_masks: {len(view_masks)}")
        print(f"- Número de vistas en view_features: {len(view_features)}")
        print(f"- Número de vistas en proj_matrices: {len(proj_matrices)}")
        
        # Convertir tensores a numpy para facilitar la serialización
        # Usar detach() para tensores con requires_grad=True
        voxel_coords_np = voxel_coords.detach().cpu().numpy()
        view_masks_np = [mask.detach().cpu().numpy() for mask in view_masks]
        view_features_np = [feat.detach().cpu().numpy() for feat in view_features]
        proj_matrices_np = [proj.detach().cpu().numpy() for proj in proj_matrices]
        
        # Generar colores distintos para cada vista
        view_colors = [
            [1.0, 0.0, 0.0],  # Rojo para vista 0
            [0.0, 1.0, 0.0],  # Verde para vista 1
            [0.0, 0.0, 1.0],  # Azul para vista 2
            [1.0, 1.0, 0.0],  # Amarillo para vista 3
            [0.0, 1.0, 1.0],  # Cian para vista 4
            [1.0, 0.0, 1.0],  # Magenta para vista 5
        ]
        
        # Limitar a 6 vistas como máximo
        max_views = min(len(view_masks_np), 6)
        print(f"Limitando a {max_views} vistas")
        
        view_masks_np = view_masks_np[:max_views]
        view_features_np = view_features_np[:max_views]
        proj_matrices_np = proj_matrices_np[:max_views]
        
        # Guardar los datos necesarios para la visualización
        visualization_data = {
            'voxel_coords': voxel_coords_np,
            'voxel_size': voxel_size,
            'view_masks': view_masks_np,
            'view_features': view_features_np,
            'proj_matrices': proj_matrices_np,
            'view_colors': view_colors[:max_views],
            'num_views': max_views
        }
        
        # Verificar tamaño de los datos antes de guardar
        print(f"Tamaño aproximado de datos:")
        print(f"- voxel_coords: {voxel_coords_np.nbytes / (1024*1024):.2f} MB")
        for i in range(max_views):
            print(f"- view_features[{i}]: {view_features_np[i].nbytes / (1024*1024):.2f} MB")
        
        # Asegurarse de que el directorio existe
        save_dir = os.path.dirname(os.path.abspath(save_path))
        if save_dir and not os.path.exists(save_dir):
            os.makedirs(save_dir)
            print(f"Creado directorio: {save_dir}")
        
        # Guardar el archivo
        print(f"Intentando guardar datos en {save_path}...")
        with open(save_path, 'wb') as f:
            pickle.dump(visualization_data, f)
        
        # Verificar que el archivo se haya creado
        if os.path.exists(save_path):
            file_size = os.path.getsize(save_path) / (1024*1024)
            print(f"Archivo guardado correctamente: {save_path} ({file_size:.2f} MB)")
        else:
            print(f"¡ADVERTENCIA! El archivo no existe después de intentar guardarlo")
        
        print(f"Número total de vóxeles: {len(voxel_coords_np)}")
        print(f"Número de vistas: {max_views}")
        for i in range(max_views):
            print(f"Vista {i}: {np.sum(view_masks_np[i])} vóxeles visibles ({np.sum(view_masks_np[i])/len(voxel_coords_np)*100:.2f}%)")
            
        return True
        
    except Exception as e:
        print(f"ERROR al guardar datos de visualización: {str(e)}")
        import traceback
        traceback.print_exc()
        return False
    

#######################################################################################################
# ---------------------------------图像特征可视化-----------------------------------------------------
#######################################################################################################
def visualize_feature_map_multistage(feature_map, pca_stages=None, output_dir="./feature_visualizations", prefix="view_{view_idx}", random_state=42):
    """
    使用多阶段 PCA 将高维特征图转换为可视化的 RGB 图像
    
    Args:
        feature_map: [C, H, W] 的特征图张量 (C=64, H=64, W=96)
        pca_stages: PCA 降维阶段列表，默认为[256, 64, 16, 3]
        output_path: 保存路径（可选）
        random_state: 随机种子，确保结果可重现
    Returns:
        PIL Image 对象
    """
  # 设置默认PCA阶段
    if pca_stages is None:
        pca_stages = [16, 3]
    
    # 确保输入是CPU上的numpy数组
    if torch.is_tensor(feature_map):
        feature_map = feature_map.detach().cpu().numpy()
    
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


