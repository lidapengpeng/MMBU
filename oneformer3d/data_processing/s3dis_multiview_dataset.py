import os.path as osp
import mmengine
import numpy as np
import cv2
from mmdet3d.registry import DATASETS

@DATASETS.register_module(force=True)
class S3DISMultiViewSegDataset:
    """最简化版本的数据集类"""
    def __init__(self, 
                 data_root, 
                 ann_file, 
                 pipeline=None,
                 test_mode=False):
        """初始化
        Args:
            data_root (str): 数据根目录
            ann_file (str): 标注文件名
            pipeline (list[dict]): 数据处理流水线
            test_mode (bool): 是否为测试模式
        """
        self.data_root = data_root
        self.ann_file = ann_file
        self.test_mode = test_mode
        self.pipeline = pipeline
        
        # 加载数据
        self._load_data()
        
    def _load_data(self):
        """加载并检查数据"""
        # 1. 加载标注文件
        ann_path = osp.join(self.data_root, self.ann_file)
        print(f"\nLoading annotation file: {ann_path}")
        try:
            data = mmengine.load(ann_path)
        except Exception as e:
            raise Exception(f"Error loading {ann_path}: {str(e)}")
            
        # 2. 检查数据格式
        if not isinstance(data, dict) or 'data_list' not in data:
            raise ValueError(f"Invalid data format in {ann_path}")
            
        self.data_infos = data['data_list']
        print(f"Successfully loaded {len(self.data_infos)} samples")
        
        # 3. 打印第一个样本的结构
        if self.data_infos:
            print("\nFirst sample structure:")
            self._print_data_structure(self.data_infos[0])
            
    def _print_data_structure(self, data_info, indent=0):
        """递归打印数据结构"""
        for key, value in data_info.items():
            prefix = ' ' * (indent * 2)
            if isinstance(value, dict):
                print(f"{prefix}{key}:")
                self._print_data_structure(value, indent + 1)
            else:
                print(f"{prefix}{key}: {type(value)}")

    def _load_images(self, data_info):
        """加载多视角图像
        Args:
            data_info (dict): 单个样本的信息字典
        Returns:
            list[np.ndarray]: 图像列表
        """
        if 'images' not in data_info:
            return None
            
        images = []
        for img_path in data_info['images']['paths']:
            full_path = osp.join(self.data_root, img_path)
            try:
                img = cv2.imread(full_path)
                if img is None:
                    print(f"Warning: Failed to load image {full_path}")
                    continue
                images.append(img)
            except Exception as e:
                print(f"Error loading image {full_path}: {str(e)}")
                continue
                
        return images if images else None

    def prepare_data(self, idx):
        """准备单个样本的数据
        Args:
            idx (int): 样本索引
        Returns:
            dict: 处理后的数据字典
        """
        data_info = self.data_infos[idx]
        
        # 1. 加载图像
        images = self._load_images(data_info)
        if images is not None:
            data_info['img'] = images
            
        # 2. 应用数据处理流水线
        if self.pipeline is not None:
            data_info = self.pipeline(data_info)
            
        return data_info

    def __len__(self):
        """返回数据集大小"""
        return len(self.data_infos)

    def __getitem__(self, idx):
        """获取单个样本"""
        print(f"\nGetting sample {idx}")
        return self.prepare_data(idx)