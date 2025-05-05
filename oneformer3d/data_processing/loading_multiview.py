# oneformer3d/data_processing/loading_multiview.py

from typing import Dict, List, Optional, Sequence, Tuple, Union
import os.path as osp

import mmengine
import numpy as np
import mmcv
from mmcv.transforms import LoadImageFromFile
from mmdet3d.registry import TRANSFORMS
from mmdet3d.structures import (
    CameraInstance3DBoxes,
    Det3DDataSample,
    PointData
)

# 大鹏
@TRANSFORMS.register_module()
class LoadMultiViewImageFromFiles_(LoadImageFromFile):
    """Load multi-view image from files.

    Args:
        to_float32 (bool): Whether to convert the loaded image to float32.
            Defaults to False.
        color_type (str): The color type of the loaded image.
            Defaults to 'color'.
        imdecode_backend (str): The image decoding backend type. Options are
            'cv2', 'pillow', 'turbojpeg', 'tifffile', 'none'.
            Defaults to 'cv2'.
        num_views (int, optional): Number of view images. Defaults to 6.
        backend_args (dict, optional): Arguments to instantiate the
            corresponding backend. Defaults to None.
    """

    def __init__(self,
                 to_float32: bool = False,
                 color_type: str = 'color',
                 imdecode_backend: str = 'cv2',
                 num_views: int = 6,
                 backend_args: Optional[dict] = None) -> None:
        super().__init__(
            to_float32=to_float32,
            color_type=color_type,
            imdecode_backend=imdecode_backend,
            backend_args=backend_args)
        self.num_views = num_views

    def load_image(self, img_path: str) -> np.ndarray:
        """Load an image from file.

        Args:
            img_path (str): Path to the image file.

        Returns:
            np.ndarray: The loaded image.
        """
        img_bytes = mmengine.fileio.get(img_path, backend_args=self.backend_args)
        img = mmcv.imfrombytes(
            img_bytes, flag=self.color_type, backend=self.imdecode_backend)
        return img

    def transform(self, results: dict) -> dict:
        """Transform function to load multi-view image from files.

        Args:
            results (dict): Result dict containing multi-view image filenames.

        Returns:
            dict: The result dict containing the multi-view image data.
        """
        # 获取图像路径和相机参数
        image_paths = results['images']['paths']
        poses = results['images']['poses']
        
        # 验证视角数量
        assert len(image_paths) == self.num_views, \
            f'Expected {self.num_views} views, got {len(image_paths)}'

        # 加载图像
        filenames = []
        imgs = []
        img_shapes = []
        
        # print("\n=== Image Loading Debug " + "="*50)
        for img_path in image_paths:
            # 构建完整的图像路径
            if 'data_root' in results:
                # 由于图像路径已经包含了 'images/' 前缀，所以直接使用 data_root
                full_path = osp.join(results['data_root'], img_path)
            else:
                full_path = img_path
            # 使用父类的load_image方法加载图像
            try:
                img = self.load_image(full_path)
                if img is None:
                    raise ValueError(f'Failed to load image: {full_path}')
            except Exception as e:
                print(f'Error loading image {full_path}: {str(e)}')
                raise
            
            if self.to_float32:
                img = img.astype(np.float32)
                
            filenames.append(full_path)
            imgs.append(img)
            img_shapes.append(img.shape)
            # print(f"Loaded image {osp.basename(img_path)} with shape: {img.shape}")

        # print("=== End Image Loading Debug " + "="*50 + "\n")

        # print(f"Total loaded images: {len(imgs)}")  # Verify image count
        # for i, img in enumerate(imgs):
        #     print(f"Image {i} shape: {img.shape}")

        # 注释掉或删除堆叠操作
        # imgs = np.stack(imgs, axis=0)  # 不再预先堆叠
        
        # 处理相机参数
        cam_params = {}
        for idx, img_path in enumerate(image_paths):
            img_name = osp.basename(img_path)
            pose_info = poses[img_name]
            
            # 提取相机参数
            R = np.array(pose_info['rotation_matrix'], dtype=np.float32)
            C = np.array([
                pose_info['center']['x'],
                pose_info['center']['y'],
                pose_info['center']['z']
            ], dtype=np.float32).reshape(3, 1)
            
            # 计算真正的平移向量 T = -RC
            T = -R @ C
            
            cam_params[idx] = {
                'rotation_matrix': R,
                'camera_center': C,
                'translation_vector': T,
                'intrinsic_matrix': np.array(
                    pose_info['camera_intrinsics']['intrinsic_matrix'], 
                    dtype=np.float32),
                'distortion': np.array([
                    pose_info['camera_intrinsics']['distortion']['k1'],
                    pose_info['camera_intrinsics']['distortion']['k2'],
                    pose_info['camera_intrinsics']['distortion']['k3'],
                    pose_info['camera_intrinsics']['distortion']['p1'],
                    pose_info['camera_intrinsics']['distortion']['p2']
                ], dtype=np.float32),
                'image_size': np.array([
                    pose_info['camera_intrinsics']['image_size']['width'],
                    pose_info['camera_intrinsics']['image_size']['height']
                ], dtype=np.float32)
            }

        results['filename'] = filenames
        results['img'] = imgs  # 直接传递图像列表
        results['img_shape'] = img_shapes
        results['ori_shape'] = img_shapes
        results['cam_params'] = cam_params
        results['num_views'] = self.num_views
        results['world2img'] = self._compute_world2img(cam_params)
        
        # print(f"Final 'img' field type: {type(results['img'])}")  # 应该显示 list
        # print(f"Number of images in list: {len(results['img'])}")
        
        return results

    def _compute_world2img(self, cam_params: Dict) -> List[np.ndarray]:
        """Compute world to image transformation matrices for each camera.

        Args:
            cam_params (dict): Camera parameters for each view.

        Returns:
            List[np.ndarray]: World to image transformation matrices.
        """
        world2img_mats = []
        for idx in range(self.num_views):
            cam_param = cam_params[idx]
            rotation = cam_param['rotation_matrix']
            # print("\n=== rotation矩阵 ===")
            # print(rotation)
            # 先经过R矩阵转置，然后第二列第二列取反，
            rotation = np.array([
                [-rotation[0,0], -rotation[1,0], rotation[2,0]],
                [-rotation[0,1], -rotation[1,1], rotation[2,1]],
                [-rotation[0,2], -rotation[1,2], rotation[2,2]]
            ])
            # print("\n=== rotation矩阵 ===")
            # print(rotation)
            center = cam_param['camera_center']
            
            # 设置视野参数
            intrinsic = cam_param['intrinsic_matrix']
            img_size = cam_param['image_size']
            
            # 从内参矩阵计算FOV
            fx = intrinsic[0, 0]
            fy = intrinsic[1, 1]
            width = img_size[0]
            height = img_size[1]
            
            # 计算tan(FOV/2)值
            tan_half_fov_x = width / (2.0 * fx)
            tan_half_fov_y = height / (2.0 * fy)
            
            fov_y = np.degrees(2.0 * np.arctan(tan_half_fov_y))
            aspect_ratio = tan_half_fov_x / tan_half_fov_y
            
            # 计算相机到世界的变换矩阵
            T_world_to_camera = np.eye(4)
            T_world_to_camera[:3, :3] = rotation.T  # 转置旋转矩阵以获得世界到相机的旋转
            T_world_to_camera[:3, 3] = (-rotation.T @ center).flatten()  # 相机中心的平移，将(3,1)转换为(3,)
            
            # 计算透视投影参数
            fov_y_rad = np.radians(fov_y)
            tan_half_fov_y = np.tan(fov_y_rad / 2.0)
            tan_half_fov_x = tan_half_fov_y * aspect_ratio
            
            # 创建透视投影矩阵
            perspective = np.zeros((4, 4))
            perspective[0, 0] = 1.0 / tan_half_fov_x
            perspective[1, 1] = 1.0 / tan_half_fov_y
            perspective[2, 2] = 1.0  # far / (far - near), 简化为1
            perspective[2, 3] = 0.0  # -far * near / (far - near), 简化为0
            perspective[3, 2] = 1.0
            
            # 完整的投影矩阵
            world2img_mat = perspective @ T_world_to_camera
            #print(world2img_mat)
            world2img_mats.append(world2img_mat)

        return world2img_mats