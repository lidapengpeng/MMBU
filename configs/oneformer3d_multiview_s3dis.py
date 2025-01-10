# configs/oneformer3d_multiview_s3dis.py

_base_ = ['oneformer3d_1xb2_s3dis-area-5.py']

# Modify dataset settings
train_pipeline = [
    dict(
        type='LoadPointsFromFile',
        coord_type='LIDAR',
        load_dim=6,
        use_dim=[0, 1, 2, 3, 4, 5]),
    dict(
        type='LoadMultiViewImageFromFile',
        to_float32=True),
    dict(
        type='LoadAnnotations3D',
        with_bbox_3d=False,
        with_label_3d=False,
        with_mask_3d=True,
        with_seg_3d=True),
    # Add other transformations as needed
]

test_pipeline = [
    dict(
        type='LoadPointsFromFile',
        coord_type='LIDAR',
        load_dim=6,
        use_dim=[0, 1, 2, 3, 4, 5]),
    dict(
        type='LoadMultiViewImageFromFile',
        to_float32=True),
    # Add other test transformations
]

dataset_type = 'S3DISMultiViewSegDataset'
data_root = 'data/s3dis/'

# Example configurations for different GPU counts
num_gpus_configs = {
    4: dict(batch_size=2, num_views=6),  # 4 GPUs
    6: dict(batch_size=3, num_views=6),  # 6 GPUs
    8: dict(batch_size=4, num_views=6),  # 8 GPUs
    10: dict(batch_size=5, num_views=6), # 10 GPUs
}

# Get number of GPUs and corresponding config
import torch
num_gpus = torch.cuda.device_count()
gpu_config = num_gpus_configs.get(num_gpus, dict(batch_size=1, num_views=6))

data = dict(
    samples_per_gpu=gpu_config['batch_size'],
    workers_per_gpu=4,
    train=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file='s3dis_infos_Area_1.pkl',
        num_views=gpu_config['num_views'],
        pipeline=train_pipeline,
        modality=dict(
            use_lidar=True,
            use_camera=True),
        test_mode=False),
    val=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file='s3dis_infos_Area_5.pkl',
        num_views=gpu_config['num_views'],
        pipeline=test_pipeline,
        modality=dict(
            use_lidar=True,
            use_camera=True),
        test_mode=True),
    test=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file='s3dis_infos_Area_5.pkl',
        num_views=gpu_config['num_views'],
        pipeline=test_pipeline,
        modality=dict(
            use_lidar=True,
            use_camera=True),
        test_mode=True)
)

# Training settings for distributed training
dist_params = dict(backend='nccl')
find_unused_parameters = True

# Adjust learning rate according to batch size
optimizer = dict(
    lr=0.001 * gpu_config['batch_size']  # Scale learning rate with batch size
)