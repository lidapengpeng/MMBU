# OneFormer3D 🚀

一个基于深度学习的3D场景理解框架。

## 📁 项目结构

```
oneformer3d
├── 📂 configs                    # 配置文件
│   ├── instance-only-oneformer3d_1xb2_scannet-and-structured3d.py
│   ├── oneformer3d_1xb2_s3dis-area-5.py
│   ├── oneformer3d_multiview_s3dis.py
├── 📂 data                       # 数据集
│   ├── 3sdis
│   ├── s3dis-origin
├── 📂 oneformer3d                # 核心代码
├── 📂 data_processing            # 数据处理
│   ├── data_preprocessor.py
│   ├── formatting.py
│   ├── loading.py
│   ├── s3dis_dataset.py
│   ├── s3dis_multiview_dataset.py
│   ├── scannet_dataset.py
│   ├── structured3d_dataset.py
│   ├── structures.py
│   └── transforms_3d.py
├── 📂 loss                       # 损失函数
│   ├── instance_criterion.py
│   ├── semantic_criterion.py
│   └── unified_criterion.py
├── 📂 metrics                    # 评估指标
│   ├── evaluate_semantic_instance.py
│   ├── instance_seg_eval.py
│   ├── instance_seg_metric.py
│   ├── unified_metric.py
│   └── visualization_evaluator.py
├── 📂 models                     # 模型定义
│   ├── mink_unet.py
│   ├── oneformer3d.py
│   ├── query_decoder.py
│   └── spconv_unet.py
├── 📂 postprocessing             # 后处理
│   ├── mask_matrix_nms.py
│   └── __init__.py
├── 📂 tools                      # 工具脚本
│   ├── test_multiview_loading.py
├── 📂 work_dirs                  # 工作目录
├── .gitignore
└── README.md
```

## 🚀 使用说明

### 训练模型

使用以下命令开始训练：


```bash


PYTHONPATH=./ CUDA_VISIBLE_DEVICES=2,3,4 bash tools/dist_train.sh configs/oneformer3d_1xb2_s3dis-area-5.py 3
```

### 测试模型

测试分为两个步骤：

1. 修复权重格式：
```bash
python tools/fix_spconv_checkpoint.py \
    --in-path work_dirs/oneformer3d_1xb2_s3dis-area-5/best_all_ap_50%_epoch_422.pth \
    --out-path work_dirs/oneformer3d_1xb2_s3dis-area-5/best_all_ap_50%_epoch_422.pth
```

2. 使用修复后的权重进行测试：
```bash
PYTHONPATH=./ python tools/test.py \
    configs/oneformer3d_1xb2_s3dis-area-5.py \
    work_dirs/oneformer3d_1xb2_s3dis-area-5/best_all_ap_50%_epoch_422.pth
```

