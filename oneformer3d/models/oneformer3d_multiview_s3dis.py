import torch
import torch.nn as nn
import torch.nn.functional as F
import spconv.pytorch as spconv
from mmdet3d.registry import MODELS
from mmdet3d.models import Base3DDetector

from torch_scatter import scatter_mean

from mmdet3d.structures import PointData
from mmdet3d.models import Base3DDetector

import MinkowskiEngine as ME
from .oneformer3d import ScanNetOneFormer3DMixin
from .dino_extractor_adater import DINODPTModel
from .projection import VoxelProjector
from .feature_fusion import MultiModalFeatureFusion, CrossViewFeatureFusion
from ..postprocessing.mask_matrix_nms import mask_matrix_nms
import traceback
import numpy as np
from .feature_visualization import visualize_feature_map_multistage



@MODELS.register_module()
class MultiViewS3DISOneFormer3D(Base3DDetector):
    r"""OneFormer3D for S3DIS dataset.

    Args:
        in_channels (int): Number of input channels.
        num_channels (int): NUmber of output channels.
        voxel_size (float): Voxel size.
        num_classes (int): Number of classes.
        min_spatial_shape (int): Minimal shape for spconv tensor.
        backbone (ConfigDict): Config dict of the backbone.
        decoder (ConfigDict): Config dict of the decoder.
        criterion (ConfigDict): Config dict of the criterion.
        train_cfg (dict, optional): Config dict of training hyper-parameters.
            Defaults to None.
        test_cfg (dict, optional): Config dict of test hyper-parameters.
            Defaults to None.
        data_preprocessor (dict or ConfigDict, optional): The pre-process
            config of :class:`BaseDataPreprocessor`.  it usually includes,
                ``pad_size_divisor``, ``pad_value``, ``mean`` and ``std``.
        init_cfg (dict or ConfigDict, optional): the config to control the
            initialization. Defaults to None.
        use_multiview (bool): Whether to use multiview features.
    """

    def __init__(self,
                 in_channels,
                 num_channels,
                 voxel_size,
                 num_classes,
                 min_spatial_shape,
                 backbone=None,
                 decoder=None,
                 criterion=None,
                 train_cfg=None,
                 test_cfg=None,
                 data_preprocessor=None,
                 init_cfg=None,
                 use_multiview=False):
        super().__init__(
            data_preprocessor=data_preprocessor, init_cfg=init_cfg)
        self.unet = MODELS.build(backbone)
        self.decoder = MODELS.build(decoder)
        self.criterion = MODELS.build(criterion)
        self.voxel_size = voxel_size
        self.num_classes = num_classes
        self.min_spatial_shape = min_spatial_shape
        self.train_cfg = train_cfg
        self.test_cfg = test_cfg
        self.use_multiview = use_multiview
        
        # 初始化多视角相关组件
        if self.use_multiview:
            self.image_encoder = DINODPTModel(
                model_path='/workspace/data/oneformer3d/pretrained/dinov2-large',
                output_dim=64,
                target_size=(896, 1344),
                device='cuda'
            )
            
            self.projector = VoxelProjector(voxel_size=0.33)
            self.cross_view_fusion = CrossViewFeatureFusion(
                in_channels=64,
                out_channels=64
            )
            self.multimodal_fusion = MultiModalFeatureFusion(
                in_channels=64
            )

        self._init_layers(in_channels, num_channels)
        self._freeze_pretrained_weights()      
        self.multiview_data = None
        self.num_channels = num_channels

    def _init_layers(self, in_channels, num_channels):
        self.input_conv = spconv.SparseSequential(
            spconv.SubMConv3d(
                in_channels,
                num_channels,
                kernel_size=3,
                padding=1,
                bias=False,
                indice_key='subm1'))
        self.output_layer = spconv.SparseSequential(
            torch.nn.BatchNorm1d(num_channels, eps=1e-4, momentum=0.1),
            torch.nn.ReLU(inplace=True))
        
    def _freeze_pretrained_weights(self):
        """冻结预训练的权重"""
        if self.use_multiview:
            # 冻结DINOv2 Encoder
            self.image_encoder.freeze_encoder()
            
            # 确保decoder是可训练的
            for param in self.image_encoder.decoder.parameters():
                param.requires_grad = True
            
            # 验证参数状态
            self._verify_parameter_states()
        
        # 冻结点云骨干网络， False表示冻结
        for param in self.unet.parameters():
            param.requires_grad = True
        print("Complete SpConvUNet backbone has been frozen")

    def _verify_parameter_states(self):
        """验证参数的冻结状态"""
        if self.use_multiview:
            # 检查编码器参数
            encoder_params = sum(p.numel() for p in self.image_encoder.encoder.parameters() 
                               if p.requires_grad)
            # 检查解码器参数
            decoder_params = sum(p.numel() for p in self.image_encoder.decoder.parameters() 
                               if p.requires_grad)
            
            # 检查其他模块参数 - 需要检查对象是否有parameters方法
            projector_params = 0  # VoxelProjector不是nn.Module
            
            # 检查融合模块是否有parameters方法
            fusion_params = 0
            if hasattr(self.multimodal_fusion, 'parameters'):
                fusion_params += sum(p.numel() for p in self.multimodal_fusion.parameters() 
                                  if p.requires_grad)
            if hasattr(self.cross_view_fusion, 'parameters'):
                fusion_params += sum(p.numel() for p in self.cross_view_fusion.parameters() 
                                   if p.requires_grad)
            
            print("\n参数冻结状态验证:")
            print(f"  DINOv2编码器可训练参数: {encoder_params:,d} (应为0)")
            print(f"  图像解码器可训练参数: {decoder_params:,d}")
            print(f"  投影器可训练参数: {projector_params:,d}")
            print(f"  特征融合模块可训练参数: {fusion_params:,d}")

    def extract_image_features(self, images, proj_matrices,  img_shapes, voxel_coords):
        """Extract and project features from multi-view images."""
        # print("\n=== Image Feature Extraction Debug ===")
        # print(f"Input images shape: {images.shape}")
        # print(f"Projection matrices shape: {proj_matrices.shape}")
        # print(f"Voxel coordinates shape: {voxel_coords.shape}")
        
        view_features = []
        view_masks = []
        
        for view_idx, (image, proj_mat, img_shape) in enumerate(zip(images, proj_matrices, img_shapes)):
            # 处理的是每一个视角           
            # 使用DINODPTModel提取特征
            with torch.no_grad():  # 确保编码器部分不计算梯度
                # 提取DINOv2特征
                dino_features = self.image_encoder.encoder.extract_features(image, layer_idx=24)
            
            # 通过可训练的decoder处理特征
            image_features = self.image_encoder.decoder(dino_features)
            image_features = image_features.squeeze(0)
            # print(f"image_features.shape: {image_features.shape}")
            # -----------------------------------------------------------------# 
            # 可视化图像特征
            # visualize_feature_map_multistage(image_features, output_dir="./feature_visualizations", prefix=f'view_{view_idx}')
            # -----------------------------------------------------------------# 
            try:
                # 构建完整的相机信息字典
                camera_info = {
                    'projection_matrix': proj_mat,
                    'image_size': (img_shape[0], img_shape[1])
                }
                # Print camera info details
                # print("\n=== Camera Info Details ===")
                # print(f"Image size: {camera_info['image_size']}")
                # print("\nProjection matrix:")
                # print(camera_info['projection_matrix'])
                # 将场景中所有的体素块，投影到当前视角图像上,得到的proj_points, valid_mask是场景总的体素数量
                with torch.no_grad():
                    proj_points, valid_mask = self.projector.project_voxels(
                        voxel_coords, 
                        camera_info                   
                    )
                # print(f"voxel_coords.shape: {voxel_coords.shape}")
                # print(f"proj_points.shape: {proj_points.shape}")
                # print(f"valid_mask.shape: {valid_mask.shape}")
                ##################################################################################
                # Save colored point cloud to txt file
                # os.makedirs('point_cloud_vis', exist_ok=True)
                
                # # Convert voxel coordinates to numpy array
                # coords_np = voxel_coords.cpu().numpy()
                # valid_mask_np = valid_mask.cpu().numpy()
                
                # # Create color array (red for valid points, blue for invalid)
                # colors = np.zeros((len(coords_np), 3), dtype=np.uint8)
                # colors[valid_mask_np] = [255, 0, 0]  # Red for valid points
                # colors[~valid_mask_np] = [0, 0, 255]  # Blue for invalid points
                
                # # Combine coordinates and colors
                # point_cloud_data = np.hstack((coords_np, colors))
                
                # # Save to txt file
                # save_path = f'point_cloud_vis/colored_points_view_{view_idx}.txt'
                # np.savetxt(save_path, point_cloud_data, fmt='%d %d %d %d %d %d', 
                #             header='x y z r g b', comments='')
                ##################################################################################
                if valid_mask.sum() > 0:
                    valid_points = proj_points[valid_mask]
                    # 计算原始图像坐标到特征图坐标的缩放比例
                    orig_H, orig_W = img_shape[0], img_shape[1]
                    feature_H, feature_W = image_features.shape[1], image_features.shape[2]
                    
                    # 计算缩放因子
                    scale_w = feature_W / orig_W
                    scale_h = feature_H / orig_H
                    
                    # 应用缩放
                    valid_points[:, 0] = valid_points[:, 0] * scale_w
                    valid_points[:, 1] = valid_points[:, 1] * scale_h

                    # 向下取整并确保在特征图范围内
                    valid_points_discrete = valid_points.clone().floor().long()
                    valid_points_discrete[:, 0].clamp_(0, feature_W - 1)
                    valid_points_discrete[:, 1].clamp_(0, feature_H - 1)

                    # 使用向量化操作获取特征
                    sampled_features = image_features[:, valid_points_discrete[:, 1], valid_points_discrete[:, 0]]
                    sampled_features = sampled_features.permute(1, 0)  # 调整维度顺序为 [N, C]
                    
                    # 检查特征维度是否匹配 num_channels，不匹配则进行调整
                    if sampled_features.shape[1] != self.num_channels:
                        # 使用简单的线性投影将特征维度调整为 num_channels
                        if not hasattr(self, 'feature_adaptor'):
                            self.feature_adaptor = nn.Linear(
                                sampled_features.shape[1], 
                                self.num_channels
                            ).to(sampled_features.device)
                        sampled_features = self.feature_adaptor(sampled_features)
                    
                    # 对特征进行归一化处理，提高数值稳定性
                    if sampled_features.shape[0] > 0:
                        # 减均值除标准差
                        mean = sampled_features.mean(dim=0, keepdim=True)
                        std = sampled_features.std(dim=0, keepdim=True) + 1e-8
                        sampled_features = (sampled_features - mean) / std
                    
                    view_features.append(sampled_features)
                    view_masks.append(valid_mask)
                    
                    # 特征可视化代码
                    # if not self.training:
                    #     from .feature_visualization import visualize_features
                    #     # 创建输出目录
                    #     os.makedirs('feature_visualizations', exist_ok=True)
                        
                    #     # 为所有体素创建特征张量，初始化为0
                    #     all_features = torch.zeros((voxel_coords.shape[0], sampled_features.shape[1]), 
                    #                             dtype=torch.float32,
                    #                             device=sampled_features.device)
                        
                    #     # 只更新能投影到图像平面的体素的特征
                    #     all_features[valid_mask] = sampled_features
                        
                    #     # 可视化所有体素的特征
                    #     visualize_features(
                    #         all_features,  # [N, C] 形状的特征，包含所有体素
                    #         voxel_coords,  # 所有体素的坐标
                    #         output_path='feature_visualizations',
                    #         prefix=f'view_{view_idx}_all_voxels_features'
                    #     )
                        
                    #     # 同时也保存只有投影体素的可视化结果，用于对比
                    #     visualize_features(
                    #         sampled_features,  # [N, C] 形状的特征
                    #         voxel_coords[valid_mask],  # 只取有效体素的坐标
                    #         output_path='feature_visualizations',
                    #         prefix=f'view_{view_idx}_projected_voxels_features'
                    #     )
                    # if not self.training:
                    #     # Create output directory if it doesn't exist
                    #     os.makedirs('feature_visualizations', exist_ok=True)
                        
                    #     # Get full RGB features for all voxels, 初始化为灰色 (128,128,128)
                    #     full_features = torch.full((voxel_coords.shape[0], 3), 
                    #                              128.0,  # 使用浮点数
                    #                              dtype=torch.float32,  # 明确指定float32类型
                    #                              device=sampled_features.device)
                        
                    #     # 只更新能投影到图像平面的点的颜色
                    #     full_features[valid_mask] = sampled_features[:, :3]  # Only take first 3 channels as RGB
                        
                    #     # Combine coordinates and RGB features
                    #     coords_and_colors = torch.cat([voxel_coords.float(), full_features], dim=1)
                        
                    #     # Save to txt file
                    #     save_path = f'feature_visualizations/point_cloud_view_{view_idx}.txt'
                    #     np.savetxt(save_path, 
                    #              coords_and_colors.cpu().numpy(),
                    #              fmt='%d %d %d %.1f %.1f %.1f',  # 前3个%d是xyz坐标，后3个%.1f是RGB浮点值
                    #              comments='')
                
                else:
                    print("No valid projections for this view")
                    view_features.append(torch.zeros((0, self.num_channels), 
                                                  device=image.device))
                    view_masks.append(valid_mask)
                
            except Exception as e:
                print(f"Error in view {view_idx} processing: {str(e)}")
                print(f"Stack trace: {traceback.format_exc()}")  # 添加堆栈跟踪
                view_features.append(torch.zeros((0, self.num_channels), 
                                              device=image.device))
                view_masks.append(torch.zeros(len(voxel_coords), dtype=torch.bool,
                                           device=voxel_coords.device))
        
        return {
            'features': view_features,
            'masks': view_masks
        }

    def fuse_features(self, point_features, image_features, voxel_coords):
        """Fuse point cloud and image features.
        
        Args:
            point_features (Tensor): Features from point cloud [N, C]
            image_features (dict): Features from images
                features (List[Tensor]): Features per view
                masks (List[Tensor]): Valid masks per view
                
        Returns:
            Tensor: Fused features [N, C]
        """
        # print("\n=== Feature Fusion Debug ===")
        # print(f"Point features shape: {point_features.shape}")

        # First fuse multi-view image features
        fused_image_features, valid_mask = self.cross_view_fusion(
            image_features['features'],
            image_features['masks']
        )
        # Then fuse with point features
        fused_features = self.multimodal_fusion(
            point_features, 
            fused_image_features,
            valid_mask
        )
        # -----------------------------------------------------------------# 
        # if not self.training:  # 仅在推理模式下进行可视化
        #     from .feature_visualization import visualize_features
        #     # 创建输出目录
        #     os.makedirs('feature_visualizations', exist_ok=True)
        #     visualize_features(fused_image_features, 
        #                        voxel_coords, output_path='feature_visualizations', prefix='fused_image_features')
        #     visualize_features(point_features,
        #                        voxel_coords, output_path='feature_visualizations', prefix='point_features')
        #     visualize_features(
        #         fused_features,
        #         voxel_coords,
        #         output_path='feature_visualizations',
        #         prefix='fused_features'
        #     )
        return fused_features

    def extract_feat(self, x):
        """Extract features with multiview fusion if enabled."""
        # print("\n=== Feature Extraction Debug ===")
        # print(f"Input sparse tensor shape: {x.features.shape}")
        # print(f"use_multiview: {self.use_multiview}")
        
        # 在网络处理前保存多视角数据
        if self.use_multiview:
            self.multiview_data = {
                'images': getattr(x, 'images', None),
                'proj_matrices': getattr(x, 'proj_matrices', None),
                'img_shapes': getattr(x, 'img_shapes', None)
            }

        
        # Extract features from sparse tensor.
        x = self.input_conv(x)
        x, _ = self.unet(x)
        x = self.output_layer(x)
        
        out = []
        for i in x.indices[:, 0].unique():
            batch_features = x.features[x.indices[:, 0] == i]
            # print(f"\nProcessing batch {i}")
            # print(f"Batch features shape: {batch_features.shape}")
            
            if self.use_multiview and self.multiview_data is not None and \
               self.multiview_data['images'] is not None:
                try:
                    # 选择加载哪个场景，从第0个场景开始
                    batch_idx = i.item()
                    # batch_mask, 得到True/False, 此处x.indices是一整个batch
                    # x.indices[:, 0]，所有的点，但是第0列索引是batch_idx
                    # 每行对应一个体素，每个体素对应一个batch_idx
                    batch_mask = x.indices[:, 0] == i
                    # 得到batch_mask为True的行，即属于当前batch（场景0）的体素坐标
                    voxel_coords = x.indices[batch_mask, 1:]
                    # 使用保存的多视角数据
                    image_features = self.extract_image_features(
                        self.multiview_data['images'][batch_idx],
                        self.multiview_data['proj_matrices'][batch_idx],                   
                        self.multiview_data['img_shapes'][batch_idx],
                        voxel_coords
                    )

                    # 特征融合
                    batch_features = self.fuse_features(batch_features, image_features, voxel_coords)
                    
                except Exception as e:
                    print(f"\nError in multiview processing:")
                    print(f"Error type: {type(e)}")
                    print(f"Error message: {str(e)}")
                    import traceback
                    print(f"Full traceback:\n{traceback.format_exc()}")
            
            out.append(batch_features)
        
        # 清理多视角数据
        self.multiview_data = None
        return out

    def collate(self, points, elastic_points=None):
        """Collate batch of points to sparse tensor.
        
        Args:
            points (List[Tensor]): Batch of points.
            elastic_points (List[Tensor], optional): Elastic coordinates.
            
        Returns:
            Tuple: coordinates, features, inverse_mapping, spatial_shape 
        """
        if elastic_points is None:
            coordinates, features = ME.utils.batch_sparse_collate(
                [((p[:, :3] - p[:, :3].min(0)[0]) / self.voxel_size,
                  torch.hstack((p[:, 3:], p[:, :3] - p[:, :3].mean(0))))
                 for p in points])
        else:
            coordinates, features = ME.utils.batch_sparse_collate(
                [((el_p - el_p.min(0)[0]),
                  torch.hstack((p[:, 3:], p[:, :3] - p[:, :3].mean(0))))
                 for el_p, p in zip(elastic_points, points)])

        spatial_shape = torch.clip(
            coordinates.max(0)[0][1:] + 1, self.min_spatial_shape)
        field = ME.TensorField(features=features, coordinates=coordinates)
        tensor = field.sparse()
        coordinates = tensor.coordinates
        features = tensor.features
        inverse_mapping = field.inverse_mapping(tensor.coordinate_map_key)

        return coordinates, features, inverse_mapping, spatial_shape

    def _forward(*args, **kwargs):
        """Implement abstract method of Base3DDetector."""
        pass

    def loss(self, batch_inputs_dict, batch_data_samples, **kwargs):
        """Calculate losses with multiview support."""
        
        # 处理多视角输入
        if self.use_multiview and 'imgs' in batch_inputs_dict:
            images = batch_inputs_dict['imgs']
            batch_inputs_dict['img'] = images
        
        # 准备点云数据
        coordinates, features, inverse_mapping, spatial_shape = self.collate(
            batch_inputs_dict['points'],
            batch_inputs_dict.get('elastic_coords', None))

        x = spconv.SparseConvTensor(
            features, coordinates, spatial_shape, len(batch_data_samples))
        
        # 设置多视角数据
        if self.use_multiview and 'img' in batch_inputs_dict:
            x.images = batch_inputs_dict['img']
            
            # 保存原始图像尺寸信息
            img_shapes = []
            for sample in batch_data_samples:
                view_shapes = [shape[:2] for shape in sample.metainfo['img_shape']]
                img_shapes.append(view_shapes)
            x.img_shapes = torch.tensor(img_shapes, dtype=torch.float32, device=x.images.device)
                
            # 处理投影矩阵
            proj_matrices = []
            for sample in batch_data_samples:
                sample_matrices = [torch.tensor(mat, dtype=torch.float32, device=x.images.device) 
                                 for mat in sample.metainfo['world2img']]
                sample_matrices = torch.stack(sample_matrices)
                proj_matrices.append(sample_matrices)          
            x.proj_matrices = torch.stack(proj_matrices)
       
        # 特征提取和解码
        x = self.extract_feat(x)
        x = self.decoder(x)

        sp_gt_instances = []
        for i in range(len(batch_data_samples)):
            voxel_superpoints = inverse_mapping[coordinates[:, 0][inverse_mapping] == i]
            voxel_superpoints = torch.unique(voxel_superpoints, return_inverse=True)[1]
            inst_mask = batch_data_samples[i].gt_pts_seg.pts_instance_mask
            sem_mask = batch_data_samples[i].gt_pts_seg.pts_semantic_mask
            assert voxel_superpoints.shape == inst_mask.shape

            batch_data_samples[i].gt_instances_3d.sp_sem_masks = \
                                self.get_gt_semantic_masks(sem_mask,
                                                            voxel_superpoints,
                                                            self.num_classes)
            batch_data_samples[i].gt_instances_3d.sp_inst_masks = \
                                self.get_gt_inst_masks(inst_mask,
                                                       voxel_superpoints)
            sp_gt_instances.append(batch_data_samples[i].gt_instances_3d)

        loss = self.criterion(x, sp_gt_instances)
        
        return loss

    def predict(self, batch_inputs_dict, batch_data_samples, **kwargs):
        """Predict results from a batch of inputs and data samples with post-processing."""
        # 1. 处理多视角图像输入
        if self.use_multiview:
            # 检查图像输入
            img_key = 'img' if 'img' in batch_inputs_dict else 'imgs'
            if img_key in batch_inputs_dict:
                images = batch_inputs_dict[img_key]
                
                # 处理多视角图像格式
                if isinstance(images, list):
                    images = torch.stack([img.permute(2,0,1) for img in images])
                elif images.dim() == 5 and images.shape[2] == 3:
                    pass
                elif images.dim() == 4 and images.shape[1] == 3:
                    images = images.unsqueeze(1)  # 添加视角维度
                else:
                    raise ValueError(f"Unexpected image format: {images.shape}")
                
                batch_inputs_dict['img'] = images  # 统一使用'img'键

        # 2. 准备点云数据
        coordinates, features, inverse_mapping, spatial_shape = self.collate(
            batch_inputs_dict['points'],
            batch_inputs_dict.get('elastic_coords', None))

        x = spconv.SparseConvTensor(
            features, coordinates, spatial_shape, len(batch_data_samples))

        # 3. 添加多视角数据到sparse tensor
        if self.use_multiview and 'img' in batch_inputs_dict:
            x.images = batch_inputs_dict['img']  # [B, V, C, H, W]

            # 保存原始图像尺寸信息
            img_shapes = []
            for sample in batch_data_samples:
                view_shapes = [shape[:2] for shape in sample.metainfo['img_shape']]
                img_shapes.append(view_shapes)
            x.img_shapes = torch.tensor(img_shapes, dtype=torch.float32, device=x.images.device)
            
            # 处理投影矩阵
            proj_matrices = []
            for sample in batch_data_samples:
                sample_matrices = [torch.tensor(mat, dtype=torch.float32, device=x.images.device) 
                                 for mat in sample.metainfo['world2img']]
                sample_matrices = torch.stack(sample_matrices)  # [V, 4, 4]
                proj_matrices.append(sample_matrices)
            
            x.proj_matrices = torch.stack(proj_matrices)  # [B, V, 4, 4]

        # 4. 特征提取和预测
        x = self.extract_feat(x)
        x = self.decoder(x)
        results_list = self.predict_by_feat(x, inverse_mapping)

        # 5. 保存结果
        for i, data_sample in enumerate(batch_data_samples):
            data_sample.pred_pts_seg = results_list[i]
        return batch_data_samples

    def predict_by_feat(self, out, superpoints):
        """Predict instance, semantic, and panoptic masks for a single scene.

        Args:
            out (Dict): Decoder output, each value is List of len 1. Keys:
                `cls_preds` of shape (n_queries, n_instance_classes + 1),
                `masks` of shape (n_queries, n_points),
                `scores` of shape (n_queris, 1) or None.
            superpoints (Tensor): of shape (n_raw_points,).

        Returns:
            List[PointData]: of len 1 with `pts_semantic_mask`,
                `pts_instance_mask`, `instance_labels`, `instance_scores`.
        """
        pred_labels = out['cls_preds'][0]
        pred_masks = out['masks'][0]
        pred_scores = out['scores'][0]

        inst_res = self.pred_inst(pred_masks[:-self.test_cfg.num_sem_cls, :],
                                  pred_scores[:-self.test_cfg.num_sem_cls, :],
                                  pred_labels[:-self.test_cfg.num_sem_cls, :],
                                  superpoints, self.test_cfg.inst_score_thr)
        sem_res = self.pred_sem(pred_masks[-self.test_cfg.num_sem_cls:, :],
                                superpoints)
        pan_res = self.pred_pan(pred_masks, pred_scores, pred_labels,
                                superpoints)

        pts_semantic_mask = [sem_res.cpu().numpy(), pan_res[0].cpu().numpy()]
        pts_instance_mask = [inst_res[0].cpu().bool().numpy(),
                             pan_res[1].cpu().numpy()]

        return [
            PointData(
                pts_semantic_mask=pts_semantic_mask,
                pts_instance_mask=pts_instance_mask,
                instance_labels=inst_res[1].cpu().numpy(),
                instance_scores=inst_res[2].cpu().numpy())]

    def pred_inst(self, pred_masks, pred_scores, pred_labels,
                  superpoints, score_threshold):
        """Predict instance masks for a single scene.

        Args:
            pred_masks (Tensor): of shape (n_queries, n_points).
            pred_scores (Tensor): of shape (n_queris, 1).
            pred_labels (Tensor): of shape (n_queries, n_instance_classes + 1).
            superpoints (Tensor): of shape (n_raw_points,).
            score_threshold (float): minimal score for predicted object.

        Returns:
            Tuple:
                Tensor: mask_preds of shape (n_preds, n_raw_points),
                Tensor: labels of shape (n_preds,),
                Tensor: scors of shape (n_preds,).
        """
        scores = F.softmax(pred_labels, dim=-1)[:, :-1]
        scores *= pred_scores

        labels = torch.arange(
            self.num_classes,
            device=scores.device).unsqueeze(0).repeat(
                self.decoder.num_queries - self.test_cfg.num_sem_cls,
                1).flatten(0, 1)
        
        scores, topk_idx = scores.flatten(0, 1).topk(
            self.test_cfg.topk_insts, sorted=False)
        labels = labels[topk_idx]

        topk_idx = torch.div(topk_idx, self.num_classes, rounding_mode='floor')
        mask_pred = pred_masks
        mask_pred = mask_pred[topk_idx]
        mask_pred_sigmoid = mask_pred.sigmoid()
        if self.test_cfg.get('obj_normalization', None):
            mask_pred_thr = mask_pred_sigmoid > \
                self.test_cfg.obj_normalization_thr
            mask_scores = (mask_pred_sigmoid * mask_pred_thr).sum(1) / \
                (mask_pred_thr.sum(1) + 1e-6)
            scores = scores * mask_scores

        if self.test_cfg.get('nms', None):
            kernel = self.test_cfg.matrix_nms_kernel
            scores, labels, mask_pred_sigmoid, _ = mask_matrix_nms(
                mask_pred_sigmoid, labels, scores, kernel=kernel)

        mask_pred = mask_pred_sigmoid > self.test_cfg.sp_score_thr
        mask_pred = mask_pred[:, superpoints]
        # score_thr
        score_mask = scores > score_threshold
        scores = scores[score_mask]
        labels = labels[score_mask]
        mask_pred = mask_pred[score_mask]

        # npoint_thr
        mask_pointnum = mask_pred.sum(1)
        npoint_mask = mask_pointnum > self.test_cfg.npoint_thr
        scores = scores[npoint_mask]
        labels = labels[npoint_mask]
        mask_pred = mask_pred[npoint_mask]

        return mask_pred, labels, scores
   
    def pred_sem(self, pred_masks, superpoints):
        """Predict semantic masks for a single scene.

        Args:
            pred_masks (Tensor): of shape (n_points, n_semantic_classes).
            superpoints (Tensor): of shape (n_raw_points,).        

        Returns:
            Tensor: semantic preds of shape
                (n_raw_points, 1).
        """
        mask_pred = pred_masks.sigmoid()
        mask_pred = mask_pred[:, superpoints]
        seg_map = mask_pred.argmax(0)
        return seg_map

    def pred_pan(self, pred_masks, pred_scores, pred_labels,
                 superpoints):
        """Predict panoptic masks for a single scene.
        
        Args:
            pred_masks (Tensor): of shape (n_queries, n_points).
            pred_scores (Tensor): of shape (n_queris, 1).
            pred_labels (Tensor): of shape (n_queries, n_instance_classes + 1).
            superpoints (Tensor): of shape (n_raw_points,).
        
        Returns:
            Tuple:
                Tensor: semantic mask of shape (n_raw_points,),
                Tensor: instance mask of shape (n_raw_points,).
        """
        stuff_cls = pred_masks.new_tensor(self.test_cfg.stuff_cls).long()
        sem_map = self.pred_sem(
            pred_masks[-self.test_cfg.num_sem_cls + stuff_cls, :], superpoints)
        sem_map_src_mapping = stuff_cls[sem_map]

        n_cls = self.test_cfg.num_sem_cls
        thr = self.test_cfg.pan_score_thr
        mask_pred, labels, scores = self.pred_inst(
            pred_masks[:-n_cls, :], pred_scores[:-n_cls, :],
            pred_labels[:-n_cls, :], superpoints, thr)
        
        thing_idxs = torch.zeros_like(labels)
        for thing_cls in self.test_cfg.thing_cls:
            thing_idxs = thing_idxs.logical_or(labels == thing_cls)
        
        mask_pred = mask_pred[thing_idxs]
        scores = scores[thing_idxs]
        labels = labels[thing_idxs]

        if mask_pred.shape[0] == 0:
            return sem_map_src_mapping, sem_map

        scores, idxs = scores.sort()
        labels = labels[idxs]
        mask_pred = mask_pred[idxs]

        inst_idxs = torch.arange(
            0, mask_pred.shape[0], device=mask_pred.device).view(-1, 1)
        insts = inst_idxs * mask_pred
        things_inst_mask, idxs = insts.max(axis=0)
        things_sem_mask = labels[idxs]

        inst_idxs, num_pts = things_inst_mask.unique(return_counts=True)
        for inst, pts in zip(inst_idxs, num_pts):
            if pts <= self.test_cfg.npoint_thr and inst != 0:
                things_inst_mask[things_inst_mask == inst] = 0

        things_inst_mask = torch.unique(
            things_inst_mask, return_inverse=True)[1]
        things_inst_mask[things_inst_mask != 0] += len(stuff_cls) - 1
        things_sem_mask[things_inst_mask == 0] = 0
      
        sem_map_src_mapping[things_inst_mask != 0] = 0
        sem_map[things_inst_mask != 0] = 0
        sem_map += things_inst_mask
        sem_map_src_mapping += things_sem_mask
        return sem_map_src_mapping, sem_map

    @staticmethod
    def get_gt_semantic_masks(mask_src, sp_pts_mask, num_classes):    
        """Create ground truth semantic masks.
        
        Args:
            mask_src (Tensor): of shape (n_raw_points, 1).
            sp_pts_mask (Tensor): of shape (n_raw_points, 1).
            num_classes (Int): number of classes.
        
        Returns:
            sp_masks (Tensor): semantic mask of shape (n_points, num_classes).
        """

        mask = torch.nn.functional.one_hot(
            mask_src, num_classes=num_classes + 1)

        mask = mask.T
        sp_masks = scatter_mean(mask.float(), sp_pts_mask, dim=-1)
        sp_masks = sp_masks > 0.5
        sp_masks[-1, sp_masks.sum(axis=0) == 0] = True
        assert sp_masks.sum(axis=0).max().item() == 1

        return sp_masks

    @staticmethod
    def get_gt_inst_masks(mask_src, sp_pts_mask):
        """Create ground truth instance masks.
        
        Args:
            mask_src (Tensor): of shape (n_raw_points, 1).
            sp_pts_mask (Tensor): of shape (n_raw_points, 1).
        
        Returns:
            sp_masks (Tensor): semantic mask of shape (n_points, num_inst_obj).
        """
        mask = mask_src.clone()
        if torch.sum(mask == -1) != 0:
            mask[mask == -1] = torch.max(mask) + 1
            mask = torch.nn.functional.one_hot(mask)[:, :-1]
        else:
            mask = torch.nn.functional.one_hot(mask)

        mask = mask.T
        sp_masks = scatter_mean(mask, sp_pts_mask, dim=-1)
        sp_masks = sp_masks > 0.5

        return sp_masks
