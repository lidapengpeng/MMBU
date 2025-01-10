import torch
import torch.distributed as dist
import numpy as np
from mmdet3d.registry import DATASETS
from mmdet3d.datasets import S3DISSegDataset_
from mmdet3d.structures import PointData

@DATASETS.register_module()
class S3DISMultiViewSegDataset(S3DISSegDataset_):
    """Support multi-view and multi-GPU S3DIS dataset.
    
    This dataset class supports:
    1. Multiple GPUs with configurable batch size
    2. Balanced loading of point clouds and images across GPUs
    3. Flexible number of views per scene
    """
    
    def __init__(self,
                 data_root,
                 ann_file,
                 num_views=6,
                 test_mode=False,
                 *args,
                 **kwargs):
        super().__init__(data_root=data_root, 
                        ann_file=ann_file,
                        test_mode=test_mode,
                        *args, 
                        **kwargs)
        self.num_views = num_views
        # Initialize distributed info
        self.dist_info = self._get_dist_info()
        
    def _get_dist_info(self):
        """Get distributed training info."""
        if dist.is_available() and dist.is_initialized():
            rank = dist.get_rank()
            world_size = dist.get_world_size()
        else:
            rank = 0
            world_size = 1
            
        return {'rank': rank, 'world_size': world_size}
    
    def _distribute_data(self, idx):
        """Distribute data across GPUs in a balanced way.
        
        For each scene:
        - First GPU handles point cloud and first N/world_size images
        - Other GPUs handle remaining images evenly
        """
        rank = self.dist_info['rank']
        world_size = self.dist_info['world_size']
        
        info = self.data_infos[idx]
        data = {}
        
        # Calculate image distribution
        imgs_per_gpu = self.num_views // world_size
        extra_imgs = self.num_views % world_size
        
        start_img = rank * imgs_per_gpu + min(rank, extra_imgs)
        end_img = start_img + imgs_per_gpu + (1 if rank < extra_imgs else 0)
        
        # GPU 0 loads point cloud + its share of images
        if rank == 0:
            # Load point cloud
            points = self._load_points(info['points'])
            data['points'] = points
            
            # Load point cloud annotations if in training mode
            if not self.test_mode:
                annos = self.get_ann_info(idx)
                data.update(annos)
        
        # All GPUs load their share of images
        image_info = info['images']
        img_paths = image_info['paths'][start_img:end_img]
        img_poses = [image_info['poses'][p] for p in img_paths]
        
        data.update({
            'img_paths': img_paths,
            'img_poses': img_poses,
            'scene_idx': idx,
            'total_views': self.num_views,
            'view_indices': (start_img, end_img)
        })
        
        return data

    def __getitem__(self, idx):
        """Get item with balanced distribution across GPUs."""
        data = self._distribute_data(idx)
        return self.pipeline(data)
        
    def evaluate(self, results, logger=None, **kwargs):
        """Evaluate with support for distributed evaluation."""
        # Gather results from all GPUs
        if dist.is_available() and dist.is_initialized():
            results = self._gather_results(results)
            
        # Only rank 0 performs evaluation
        if self.dist_info['rank'] == 0:
            return super().evaluate(results, logger, **kwargs)
        return None
        
    def _gather_results(self, results):
        """Gather results from all GPUs."""
        # Implementation of results gathering
        world_size = self.dist_info['world_size']
        
        if world_size == 1:
            return results
            
        # Gather all results
        all_results = [None for _ in range(world_size)]
        dist.all_gather_object(all_results, results)
        
        if self.dist_info['rank'] == 0:
            # Merge results
            merged_results = []
            for scene_results in zip(*all_results):
                scene_data = PointData()
                for gpu_result in scene_results:
                    if gpu_result is not None:
                        for k, v in gpu_result.items():
                            if k not in scene_data:
                                scene_data[k] = v
                merged_results.append(scene_data)
            return merged_results
            
        return None 