# tools/test_multiview_loading.py

import os
import torch
import torch.distributed as dist
from mmengine import Config 
from mmdet3d.datasets import build_dataset
from mmdet3d.utils import get_root_logger
import argparse
from tqdm import tqdm
import numpy as np

def parse_args():
    parser = argparse.ArgumentParser(description='Test multi-view data loading')
    parser.add_argument('config', help='config file path')
    parser.add_argument('--local_rank', type=int, default=0)
    parser.add_argument('--num-samples', type=int, default=10,
                       help='number of samples to test')
    parser.add_argument('--seed', type=int, default=42, 
                       help='random seed')
    return parser.parse_args()

def init_dist():
    """Initialize distributed environment."""
    if torch.cuda.device_count() > 1:
        torch.cuda.set_device(int(os.environ['LOCAL_RANK']))
        dist.init_process_group(backend='nccl')
    return dist.get_rank() if dist.is_initialized() else 0

def set_random_seed(seed):
    """Set random seed for reproducibility."""
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def validate_point_cloud(data, logger):
    """Validate point cloud data structure and content."""
    points = data['points']
    
    # Check basic properties
    logger.info(f"Point cloud shape: {points.shape}")
    logger.info(f"Point cloud type: {points.dtype}")
    logger.info(f"Point cloud range: [{points.min():.2f}, {points.max():.2f}]")
    
    # Validate dimensions
    assert points.dim() == 3, "Points should be 3D tensor (batch, num_points, channels)"
    assert points.size(-1) == 6, "Points should have 6 channels (x,y,z,r,g,b)"
    
    # Check for NaN or Inf values
    if torch.isnan(points).any():
        logger.warning("Warning: NaN values found in point cloud")
    if torch.isinf(points).any():
        logger.warning("Warning: Inf values found in point cloud")
        
    # Memory usage
    mem_mb = points.element_size() * points.nelement() / (1024 * 1024)
    logger.info(f"Point cloud memory usage: {mem_mb:.2f} MB")
    
    return True

def validate_images(data, cfg, world_size, rank, logger):
    """Validate multi-view image data."""
    num_images = len(data['img_paths'])
    logger.info(f"Number of images loaded: {num_images}")
    
    # Check paths
    for path in data['img_paths']:
        assert os.path.exists(path), f"Image not found: {path}"
        
    # Validate image distribution
    expected_imgs = cfg.data.train.num_views // world_size
    if rank < cfg.data.train.num_views % world_size:
        expected_imgs += 1
    assert num_images == expected_imgs, \
        f"Expected {expected_imgs} images, got {num_images}"
        
    # Check camera poses
    assert len(data['img_poses']) == num_images, \
        "Number of poses must match number of images"
    
    for pose in data['img_poses']:
        assert 'rotation_matrix' in pose, "Missing rotation matrix"
        assert 'translation' in pose, "Missing translation vector"
        
    # Log view distribution info
    start_idx, end_idx = data['view_indices']
    logger.info(f"View indices: [{start_idx}, {end_idx})")
    logger.info(f"Images per GPU: {num_images}")
    
    return True

def test_batch_loading(dataset, cfg, logger, num_samples=10):
    """Test batch data loading with detailed validation."""
    dataloader = build_dataloader(
        dataset,
        samples_per_gpu=cfg.data.samples_per_gpu,
        workers_per_gpu=cfg.data.workers_per_gpu,
        dist=dist.is_initialized(),
        shuffle=False)
    
    rank = dist.get_rank() if dist.is_initialized() else 0
    world_size = dist.get_world_size() if dist.is_initialized() else 1
    
    # Track memory usage
    torch.cuda.reset_peak_memory_stats()
    initial_mem = torch.cuda.memory_allocated()
    
    # Test multiple batches
    for i, data_batch in enumerate(tqdm(dataloader)):
        if i >= num_samples:
            break
            
        logger.info(f"\n{'='*20} Batch {i} {'='*20}")
        logger.info(f"Rank {rank}/{world_size}")
        
        try:
            # Validate point cloud data (GPU 0)
            if rank == 0 and 'points' in data_batch:
                valid_points = validate_point_cloud(data_batch, logger)
                if not valid_points:
                    logger.error("Point cloud validation failed!")
                    continue
                    
                if not dataset.test_mode:
                    assert 'pts_semantic_mask' in data_batch, "Missing semantic mask"
                    mask = data_batch['pts_semantic_mask']
                    logger.info(f"Semantic mask shape: {mask.shape}")
            
            # Validate image data (all GPUs)
            if 'img_paths' in data_batch:
                valid_images = validate_images(data_batch, cfg, world_size, rank, logger)
                if not valid_images:
                    logger.error("Image validation failed!")
                    continue
            
            # Log memory usage
            current_mem = torch.cuda.memory_allocated()
            peak_mem = torch.cuda.max_memory_allocated()
            logger.info(f"\nMemory Statistics:")
            logger.info(f"Current GPU memory: {current_mem/1024/1024:.2f} MB")
            logger.info(f"Peak GPU memory: {peak_mem/1024/1024:.2f} MB")
            logger.info(f"Memory increase: {(current_mem-initial_mem)/1024/1024:.2f} MB")
            
        except Exception as e:
            logger.error(f"Error processing batch {i}: {str(e)}")
            raise e

def build_dataloader(dataset, samples_per_gpu, workers_per_gpu, dist=True, shuffle=True):
    """Build PyTorch dataloader with distributed support."""
    if dist:
        sampler = torch.utils.data.DistributedSampler(dataset, shuffle=shuffle)
        shuffle = False
        batch_size = samples_per_gpu
        num_workers = workers_per_gpu
    else:
        sampler = None
        batch_size = samples_per_gpu
        num_workers = workers_per_gpu

    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        sampler=sampler,
        num_workers=num_workers,
        shuffle=shuffle,
        pin_memory=True,
        collate_fn=dataset.collate_fn if hasattr(dataset, 'collate_fn') else None)

    return dataloader

def main():
    # Parse arguments
    args = parse_args()
    
    # Set random seed
    set_random_seed(args.seed)
    
    # Initialize distributed environment
    rank = init_dist()
    
    # Load config
    cfg = Config.fromfile(args.config)
    
    # Initialize logger
    logger = get_root_logger()
    logger.info(f"Testing with config:\n{cfg.pretty_text}")
    
    # Build dataset
    dataset = build_dataset(cfg.data.train)
    logger.info(f"Dataset size: {len(dataset)}")
    
    # Test data loading
    test_batch_loading(dataset, cfg, logger, args.num_samples)
    
    if rank == 0:
        logger.info("\nValidation completed successfully!")

if __name__ == '__main__':
    main()