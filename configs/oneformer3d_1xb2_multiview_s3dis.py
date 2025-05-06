
# 1、基础配置
_base_ = [
    'mmdet3d::_base_/default_runtime.py',
]
custom_imports = dict(
    imports=['oneformer3d', 'oneformer3d.metrics.visualization_evaluator'],
    allow_failed_imports=False)
fp16 = dict(loss_scale='dynamic')
# 2、模型配置
num_instance_classes = 7 
num_semantic_classes = 7
num_channels = 64


model = dict(
    type='MultiViewS3DISOneFormer3D',
    data_preprocessor=dict(type='Det3DDataPreprocessor'),
    in_channels=6,
    num_channels=num_channels,
    voxel_size=0.33,
    num_classes=num_instance_classes,
    min_spatial_shape=128,
    use_multiview=True,
    backbone=dict(
        type='SpConvUNet',
        num_planes=[num_channels * (i + 1) for i in range(5)],
        return_blocks=True),
    decoder=dict(
        type='QueryDecoder',
        num_layers=3,
        num_classes=num_instance_classes,
        num_instance_queries=100,
        num_semantic_queries=num_semantic_classes,
        num_instance_classes=num_instance_classes,
        in_channels=num_channels,
        d_model=256,
        num_heads=8,
        hidden_dim=1024,
        dropout=0.0,
        activation_fn='gelu',
        iter_pred=True,
        attn_mask=True,
        fix_attention=True,
        objectness_flag=True),
    criterion=dict(
        type='S3DISUnifiedCriterion',
        num_semantic_classes=num_semantic_classes,
        sem_criterion=dict(
            type='S3DISSemanticCriterion',
            loss_weight=2.0),
        inst_criterion=dict(
            type='InstanceCriterion',
            matcher=dict(
                type='HungarianMatcher',
                costs=[
                    dict(type='QueryClassificationCost', weight=0.5),
                    dict(type='MaskBCECost', weight=1.0),
                    dict(type='MaskDiceCost', weight=1.0)]),
            loss_weight=[0.5, 1.0, 1.0, 0.5],
            num_classes=num_instance_classes,
            non_object_weight=0.1,
            fix_dice_loss_weight=True,
            iter_matcher=True,
            fix_mean_loss=True)),
    train_cfg=dict(),
    test_cfg=dict(
        topk_insts=60,
        inst_score_thr=0.0,
        pan_score_thr=0.1,
        npoint_thr=100,
        obj_normalization=True,
        obj_normalization_thr=0.01,
        sp_score_thr=0.10,
        nms=True,
        matrix_nms_kernel='linear',
        num_sem_cls=num_semantic_classes,
        stuff_cls=[0, 1, 2, 3, 4, 5],
        thing_cls=[6]))

# 3、数据集配置
dataset_type = 'URBANBISDataset'  # 使用多视角数据集_用于分割
data_root = 'data/s3dis-urbanbis-yuehai-all-instance' 
data_prefix = dict(
    pts='',  # 点云数据目录
    pts_instance_mask='',  # 实例掩码目录
    pts_semantic_mask='',  # 语义掩码目录
    images=''  # 图像目录
)

train_area = [1, 2, 3, 4, 6]
test_area = 5

train_pipeline = [
    dict(
        type='LoadPointsFromFile',
        coord_type='LIDAR',
        shift_height=False,
        use_color=True,
        load_dim=6,
        use_dim=[0, 1, 2, 3, 4, 5]),
    dict(
        type='LoadMultiViewImageFromFiles_', 
        to_float32=True,
        color_type='color',
        imdecode_backend='cv2',
        num_views=6,
        backend_args=dict(backend='local')
    ),
    dict(
        type='LoadAnnotations3D',
        with_bbox_3d=False,
        with_label_3d=False,
        with_mask_3d=True,
        with_seg_3d=True),
    dict(
        type='PointSample_',
        num_points=180000),
    dict(
        type='PointInstClassMapping_',
        num_classes=num_instance_classes),
    # dict(
    #     type='RandomFlip3D',
    #     sync_2d=False,
    #     flip_ratio_bev_horizontal=0.5,
    #     flip_ratio_bev_vertical=0.5),
    # dict(
    #     type='GlobalRotScaleTrans',
    #     rot_range=[0.0, 0.0],
    #     scale_ratio_range=[0.9, 1.1],
    #     translation_std=[.1, .1, .1],
    #     shift_height=False),
    dict(
        type='NormalizePointsColor_',
        color_mean=[127.5, 127.5, 127.5]
        ),

    dict(
        type='Pack3DDetInputs_',
        keys=[
            'points', 'img', 'gt_labels_3d',
            'pts_semantic_mask', 'pts_instance_mask'
        ])
]

test_pipeline = [
    dict(
        type='LoadPointsFromFile',
        coord_type='LIDAR',
        shift_height=False,
        use_color=True,
        load_dim=6,
        use_dim=[0, 1, 2, 3, 4, 5]),
    dict(
        type='LoadMultiViewImageFromFiles_', 
        to_float32=True,
        color_type='color',
        imdecode_backend='cv2',
        num_views=6,
        backend_args=dict(backend='local')
    ),
    dict(
        type='LoadAnnotations3D',
        with_bbox_3d=False,
        with_label_3d=False,
        with_mask_3d=True,
        with_seg_3d=True),
    # dict(
    #     type='RandomFlip3D',
    #     sync_2d=False,
    #     flip_ratio_bev_horizontal=0.5),
    # dict(
    #     type='GlobalRotScaleTrans',
    #     rot_range=[0.0, 0.0],
    #     scale_ratio_range=[0.9, 1.1],
    #     translation_std=[.1, .1, .1],
    #     shift_height=False),
    dict(
        type='NormalizePointsColor_',
        color_mean=[127.5, 127.5, 127.5]
        ),

    dict(
        type='Pack3DDetInputs_', 
        keys=['points', 'img']
    )
]

# 4、训练配置
# run settings
train_dataloader = dict(
    batch_size=1,
    num_workers=2,
    persistent_workers=True,
    pin_memory=True,
    sampler=dict(type='DefaultSampler', shuffle=True),
    dataset=dict(
        type='ConcatDataset',
        datasets=[
            dict(
                type=dataset_type,
                data_root=data_root,
                ann_file=f's3dis_infos_Area_{i}.pkl',
                pipeline=train_pipeline,
                filter_empty_gt=True,
                data_prefix=data_prefix,
                box_type_3d='LiDAR',
                backend_args=None
            ) for i in train_area
        ]))

val_dataloader = dict(
    batch_size=1,
    num_workers=2,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file=f's3dis_infos_Area_{test_area}.pkl',
        pipeline=test_pipeline,  # 使用test_pipeline
        test_mode=True,
        data_prefix=data_prefix,
        box_type_3d='LiDAR',
        backend_args=None))
test_dataloader = val_dataloader


class_names = ['terrain', 'vegetation', 'water', 'bridge', 'vehicle', 'boat', 'building', 'unlabeled']
label2cat = {i: name for i, name in enumerate(class_names)}
metric_meta = dict(
    label2cat=label2cat,
    ignore_index=[num_semantic_classes],
    classes=class_names,
    dataset_name='UrbanBIS')
    
sem_mapping = [0, 1, 2, 3, 4, 5, 6]

val_evaluator = dict(
    type='UnifiedSegMetric',
    # stuff_class_inds=[0, 1, 2, 3, 4, 5, 6, 12],
    # thing_class_inds=[7, 8, 9, 10, 11],
    stuff_class_inds=[0, 1, 2, 3, 4, 5],
    thing_class_inds=[6],
    min_num_points=50,
    id_offset=2**16,
    sem_mapping=sem_mapping,
    inst_mapping=sem_mapping,
    submission_prefix_semantic=None,
    submission_prefix_instance=None,
    metric_meta=metric_meta)
# test_evaluator = val_evaluator
test_evaluator = [
    dict(
        type='UnifiedSegMetric',
        stuff_class_inds=[0, 1, 2, 3, 4, 5],
        thing_class_inds=[6],
        min_num_points=50,
        id_offset=2**16,
        sem_mapping=sem_mapping,
        inst_mapping=sem_mapping,
        submission_prefix_semantic=None,
        submission_prefix_instance=None,
        metric_meta=metric_meta,
        prefix='seg_'
    ),
    dict(
        type='VisualizationEvaluator',
        stuff_class_inds=[0, 1, 2, 3, 4, 5],
        thing_class_inds=[6],
        min_num_points=50,
        id_offset=2**16,
        sem_mapping=sem_mapping,
        inst_mapping=sem_mapping,
        metric_meta=metric_meta,
        prefix='vis_'
    )
]

optim_wrapper = dict(
    type='OptimWrapper',
    optimizer=dict(
        type='AdamW', 
        lr=0.0001 * 10,  # 根据GPU数量调整
        weight_decay=0.05,
        eps=1e-8,
        betas=(0.9, 0.999)
    ),
    clip_grad=dict(max_norm=10, norm_type=2))

param_scheduler = dict(type='PolyLR', begin=0, end=500, power=0.9)

custom_hooks = [dict(type='EmptyCacheHook', after_iter=True)]
default_hooks = dict(
    checkpoint=dict(
        interval=16,
        max_keep_ckpts=1,
        save_best=['all_ap_50%', 'miou', 'building_ap50%'],
        rule='greater'))

# load_from = 'work_dirs/tmp/instance-only-oneformer3d_1xb2_scannet-and-structured3d.pth'
# load_from = None
load_from = 'work_dirs/tmp/best_all_ap_50%_epoch_280_fixed.pth'


train_cfg = dict(
    type='EpochBasedTrainLoop',
    max_epochs=400,
    val_interval=1)  # 每16个epoch验证一次
val_cfg = dict(type='ValLoop')
test_cfg = dict(type='TestLoop')

find_unused_parameters = True

env_cfg = dict(
    cudnn_benchmark=False,  # 默认关闭
    mp_cfg=dict(
        mp_start_method='fork',
        opencv_num_threads=20),
    dist_cfg=dict(backend='nccl'))

# 可视化配置
vis_backends = [
    dict(type='TensorboardVisBackend', 
         save_dir='tensorboard_logs'),  # 指定TensorBoard日志保存路径
    dict(type='LocalVisBackend')  # 本地可视化
]
visualizer = dict(
    type='Det3DLocalVisualizer', 
    vis_backends=vis_backends,
    name='visualizer')

# 工作目录配置
work_dir = './work_dirs/oneformer3d_1xb2_multiview_s3dis'

default_scope = 'mmdet3d'  # 设置默认域