# configs/oneformer3d_multiview_s3dis.py

# Dataset settings 
dataset_type = 'S3DISMultiViewSegDataset'
data_root = 'data/s3dis/'

# Area settings
train_area = [1, 2, 3, 4, 6]
test_area = 5

# 图像预处理参数
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53],
    std=[58.395, 57.12, 57.375],
    to_rgb=True
)

# 最基础的数据加载pipeline
train_pipeline = [
    # 1. 加载多视角图像
    dict(
        type='LoadMultiViewImageFromFile',
        to_float32=True,
        color_type='color',
        backend_args=dict(backend='local'),
    ),
    
    # 2. 图像预处理
    dict(
        type='MultiViewImageResize',
        img_scale=(896, 1204),  # 调整到DINO输入大小
        keep_ratio=True
    ),
    dict(
        type='MultiViewImageNormalize',
        **img_norm_cfg
    ),
    
    # 3. 加载点云
    dict(
        type='LoadPointsFromFile_',
        coord_type='DEPTH',
        load_dim=6,
        use_dim=[0, 1, 2, 3, 4, 5],
    ),
    
    # 4. 加载标注
    dict(
        type='LoadAnnotations3D_',
        with_bbox_3d=False,
        with_label_3d=False,
        with_mask_3d=True,
        with_seg_3d=True,
    ),
    
    # 5. 基础数据打包
    dict(
        type='Pack3DDetInputs_',
        keys=[
            'points',
            'pts_semantic_mask', 
            'pts_instance_mask',
            'img'
        ],
        meta_keys=[
            'img_paths',
            'img_shape',
            'pts_filename',
        ]
    )
]

# 使用相同的pipeline进行测试
test_pipeline = train_pipeline

train_dataloader = dict(
    batch_size=1,  # 先用单个样本测试
    num_workers=0,  # 单进程便于调试
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file=[f's3dis_infos_Area_{i}.pkl' for i in train_area],
        pipeline=train_pipeline,
        modality=dict(
            use_lidar=True,
            use_camera=True
        ),
        test_mode=False,
        data_prefix=dict(
            pts='points',
            pts_instance_mask='instance_mask',
            pts_semantic_mask='semantic_mask',
            img='images'
        )
    )
)

val_dataloader = dict(
    batch_size=1,
    num_workers=0,
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file=f's3dis_infos_Area_{test_area}.pkl',
        pipeline=test_pipeline,
        modality=dict(
            use_lidar=True,
            use_camera=True
        ),
        test_mode=True,
        data_prefix=dict(
            pts='points',
            pts_instance_mask='instance_mask',
            pts_semantic_mask='semantic_mask',
            img='images'
        )
    )
)

test_dataloader = val_dataloader

# 最基础的运行设置
default_scope = 'mmdet3d'