# OneFormer3D 🚀

一个基于深度学习的3D场景理解框架。

## 📋 目录

- [环境要求](#环境要求)
- [安装步骤](#安装步骤)
- [项目结构](#项目结构)
- [使用说明](#使用说明)
  - [训练模型](#训练模型)
  - [测试模型](#测试模型)
- [引用](#引用)

## 🔧 环境要求

- Python 3.7+
- CUDA 11.0+
- PyTorch 1.7+
- MMDetection3D

## 📥 安装步骤

```bash
# 克隆仓库
git clone https://github.com/your-username/oneformer3d.git
cd oneformer3d

# 安装依赖
pip install -r requirements.txt
```

## 📁 项目结构

```
oneformer3d
├── 📂 configs                    # 配置文件
│   ├── instance-only-oneformer3d_1xb2_scannet-and-structured3d.py
│   ├── oneformer3d_1xb2_s3dis-area-5.py
│   ├── oneformer3d_1xb4_scannet.py
│   └── oneformer3d_1xb4_scannet200.py
├── 📂 data                       # 数据集
│   ├── 3sdis
│   ├── s3dis-origin
│   ├── scannet
│   └── structured3d
├── 📂 oneformer3d                # 核心代码
├── 📂 data_processing            # 数据处理
│   ├── data_preprocessor.py
│   ├── formatting.py
│   ├── loading.py
│   ├── s3dis_dataset.py
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

## 📚 引用

如果您在研究中使用了本项目，请引用以下论文：

```bibtex
@article{oneformer3d2023,
    title={OneFormer3D: One Framework for 3D Scene Understanding},
    author={Author1 and Author2},
    journal={arXiv preprint},
    year={2023}
}
```

## 📝 许可证

本项目采用 [MIT 许可证](LICENSE)。

## 🤝 贡献

欢迎提交 issue 和 pull request！


- `points/xxxxx.bin`：提取的点云数据。
- `instance_mask/xxxxx.bin`：每个点云的实例标签，取值范围为 \[0, ${实例个数}\]，其中 0 代表未标注的点。
- `semantic_mask/xxxxx.bin`：每个点云的语义标签，取值范围为 \[0, 12\]。
- `s3dis_infos_Area_1.pkl`：区域 1 的数据信息，每个房间的详细信息如下：
  - info\['point_cloud'\]: {'num_features': 6, 'lidar_idx': sample_idx}.
  - info\['pts_path'\]: `points/xxxxx.bin` 点云的路径。
  - info\['pts_instance_mask_path'\]: `instance_mask/xxxxx.bin` 实例标签的路径。
  - info\['pts_semantic_mask_path'\]: `semantic_mask/xxxxx.bin` 语义标签的路径。
