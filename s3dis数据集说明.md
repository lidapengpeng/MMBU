# S3DIS 数据集说明文档

## 目录结构
```
s3dis
├── meta_data
│   ├── class_names.txt    # 类别名称列表
│   └── anno_paths.txt     # 标注文件路径列表
├── points                 # 点云数据
│   ├── Area_1_Tile_132122232000203113.bin
│   ├── Area_1_Tile_132122232000203131.bin
│   ├── Area_1_Tile_132122232000212003.bin
│   ├── Area_2_Tile_132122232000203311.bin
│   ├── Area_2_Tile_132122232000203313.bin
│   ├── Area_3_Tile_132122232000203332.bin
│   ├── Area_3_Tile_132122232000203333.bin
│   ├── Area_4_Tile_132122232000212001.bin
│   ├── Area_5_Tile_132122232000212002.bin
│   └── Area_6_Tile_132122232000212003.bin
├── instance_mask         # 实例分割掩码
│   └── [与points目录结构相同]
├── semantic_mask        # 语义分割掩码
│   └── [与points目录结构相同]
├── images              # 航拍图像
│   └── DJI_*.JPG      # 53张航拍图像
├── s3dis_infos_Area_1.pkl  # Area_1数据信息（6.6KB）
├── s3dis_infos_Area_2.pkl  # Area_2数据信息（4.4KB）
├── s3dis_infos_Area_3.pkl  # Area_3数据信息（4.5KB）
├── s3dis_infos_Area_4.pkl  # Area_4数据信息（2.3KB）
├── s3dis_infos_Area_5.pkl  # Area_5数据信息（2.4KB）
└── s3dis_infos_Area_6.pkl  # Area_6数据信息（2.4KB）
```

## 数据格式说明

### 1. 点云数据 (`points/*.bin`)
- 格式：二进制文件，每个点包含6个float32类型的值
- 数据列：[x, y, z, r, g, b]
  - x, y, z：点的3D坐标（单位：米）
  - r, g, b：点的RGB颜色值（范围：0-255，已归一化）

### 2. 实例掩码 (`instance_mask/*.bin`)
- 格式：二进制文件，每个点对应一个int64类型的值
- 取值范围：[0, N]，其中：
  - 0：表示非建筑物点或背景
  - 1~N：表示不同建筑物实例的ID，每个建筑物实例有唯一的ID
- 与点云数据一一对应
- 生成规则：
  - 非建筑物点的实例ID统一设为0

### 3. 语义掩码 (`semantic_mask/*.bin`)
- 格式：二进制文件，每个点对应一个int64类型的值
- 取值范围：[0, 6]，其中：
  - 0：terrain（地形）
  - 1：vegetation（植被）
  - 2：water（水体）
  - 3：bridge（桥梁）
  - 4：vehicle（车辆）
  - 5：boat（船只）
  - 6：building（建筑物）
- 与点云数据一一对应

### 4. 图像数据 (`images/DJI_*.JPG`)
- 格式：JPG图像文件
- 数量：53张航拍图像
- 分辨率：5472 x 3648 像素
- 文件大小：7.1MB-9.2MB不等
- 相机参数：
  - 焦距：8.66753243842032
  - 主点：(2730.69084363013, 1817.24966201053)
  - 畸变参数：
    - k1: -0.0140670805317081
    - k2: 0.00324263777777604
    - k3: 0.00619586248431039
    - p1: -0.000924850738300265
    - p2: -0.0012529168987956

### 5. 区域信息文件 (`s3dis_infos_Area_{1-6}.pkl`)
每个pkl文件包含以下结构：
```python
{
    "metainfo": {
        "categories": {
            "building": 0  # 建筑物类别标签
        },
        "dataset": "s3dis",
        "info_version": "1.1"
    },
    "data_list": [  # 场景列表
        {
            "sample_idx": int,  # 场景索引
            "lidar_points": {
                "num_pts_feats": 6,  # 点特征维度
                "lidar_path": "points/Area_X_Tile_*.bin"  # 点云文件路径
            },
            "instances": [  # 建筑物实例列表
                {
                    "bbox_3d": [x, y, z, dx, dy, dz],  # 3D边界框参数
                    "bbox_label_3d": 0  # 类别标签（建筑物）
                }
            ],
            "pts_semantic_mask_path": "semantic_mask/Area_X_Tile_*.bin",
            "pts_instance_mask_path": "instance_mask/Area_X_Tile_*.bin",
            "images": {  # 图像信息
                "num_images": int,
                "paths": ["images/DJI_*.JPG", ...],
                "poses": {  # 每张图像的位姿信息
                    "DJI_*.JPG": {
                        "rotation_matrix": [[r11, r12, r13], 
                                          [r21, r22, r23], 
                                          [r31, r32, r33]],
                        "center": {
                            "x": float,
                            "y": float,
                            "z": float
                        },
                        "camera_intrinsics": {
                            "focal_length": float,
                            "principal_point": {"x": float, "y": float},
                            "distortion": {
                                "k1": float, "k2": float, "k3": float,
                                "p1": float, "p2": float
                            }
                        }
                    }
                }
            }
        }
    ]
}
```

## 数据集统计
- 总场景数：10个
- 总图像数：53张
- 区域分布：
  - Area_1：3个场景
  - Area_2：2个场景
  - Area_3：2个场景
  - Area_4：1个场景
  - Area_5：1个场景
  - Area_6：1个场景
- 类别数：7个（terrain, vegetation, water, bridge, vehicle, boat, building）

## 数据处理流程
1. 原始数据处理：
   - 从原始txt文件读取点云数据（N×6矩阵）
   - RGB值归一化到[0, 255]范围
   - 根据文件名前缀确定语义类别
   - 从building文件名中提取实例ID

2. 数据转换：
   - 点云数据：float32类型，保存为bin文件
   - 语义掩码：int64类型，保存为bin文件
   - 实例掩码：int64类型，保存为bin文件
   - 图像数据：复制到指定目录
   - 位姿信息：从json文件读取并保存到pkl文件

## 注意事项
1. 所有点云数据和掩码文件中的点数量必须一致
2. 实例掩码和语义掩码使用int64类型存储，以确保足够的数值范围
3. 建筑物实例的bbox_3d格式为[x, y, z, dx, dy, dz]，其中：
   - (x, y, z)：边界框中心点坐标（单位：米）
   - (dx, dy, dz)：边界框在三个轴向上的尺寸（单位：米）
4. 每个Area的数据都是独立的，可以单独加载和处理
