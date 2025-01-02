# 训练

PYTHON=./ CUDA_VISIBLE_DEVICES=1,2,3 bash tools/dist_train.sh configs/oneformer3d_1xb2_s3dis-area-5.py 3

# 测试

# 第1步:修复权重格式
python tools/fix_spconv_checkpoint.py --in-path work_dirs/oneformer3d_1xb2_s3dis-area-5/best_all_ap_50%_epoch_422.pth \
    --out-path work_dirs/oneformer3d_1xb2_s3dis-area-5/best_all_ap_50%_epoch_422.pth

# 第2步:使用修复后的权重进行测试
PYTHONPATH=./ python tools/test.py configs/oneformer3d_1xb2_s3dis-area-5.py work_dirs/oneformer3d_1xb2_s3dis-area-5/best_all_ap_50%_epoch_422.pth

# 文件夹层级结构
oneformer3d
├── configs
│   ├── instance-only-oneformer3d_1xb2_scannet-and-structured3d.py
│   ├── oneformer3d_1xb2_s3dis-area-5.py
│   ├── oneformer3d_1xb4_scannet.py
│   └── oneformer3d_1xb4_scannet200.py
├── data
│   ├── 3sdis
│   ├── s3dis-origin
│   ├── scannet
│   └── structured3d
├── oneformer3d
│   ├── __pycache__
├── data_processing
│   ├── __pycache__
│   ├── data_preprocessor.py
│   ├── formatting.py
│   ├── loading.py
│   ├── s3dis_dataset.py
│   ├── scannet_dataset.py
│   ├── structured3d_dataset.py
│   ├── structures.py
│   └── transforms_3d.py
├── loss
│   ├── __pycache__
│   ├── instance_criterion.py
│   ├── semantic_criterion.py
│   └── unified_criterion.py
├── metrics
│   ├── __pycache__
│   ├── evaluate_semantic_instance.py
│   ├── instance_seg_eval.py
│   ├── instance_seg_metric.py
│   ├── unified_metric.py
│   └── visualization_evaluator.py
├── models
│   ├── __pycache__
│   ├── mink_unet.py
│   ├── oneformer3d.py
│   ├── query_decoder.py
│   └── spconv_unet.py
├── postprocessing
│   ├── __pycache__
│   ├── mask_matrix_nms.py
│   └── __init__.py
├── tools
├── work_dirs
├── .gitignore
└── README.md
