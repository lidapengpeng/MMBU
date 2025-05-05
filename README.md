pip install transformers==4.39.3
pip install tensorboard==2.11.2

# 启动TensorBoard服务
tensorboard --logdir=tensorboard_logs --port=6006 --host=172.31.224.6 --bind_all


```bash

PYTHONPATH=./ CUDA_VISIBLE_DEVICES=1,2,3,4 bash tools/dist_train.sh configs/oneformer3d_1xb2_multiview_s3dis.py 4


PYTHONPATH=./ CUDA_VISIBLE_DEVICES=2,3,4 bash tools/dist_train.sh configs/oneformer3d_1xb2_s3dis-area-5.py 3


PYTHONPATH=./ CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7,8,9 bash tools/dist_train.sh configs/oneformer3d_1xb2_multiview_s3dis.py 10
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


 python tools/fix_spconv_checkpoint.py --in-path work_dirs/oneformer3d_1xb2_multiview_s3dis/20250311_132423/best_all_ap_50%_epoch_12.pth --out-path work_dirs/oneformer3d_1xb2_multiview_s3dis/20250311_132423/best_all_ap_50%_epoch_12_fix.pth

 PYTHONPATH=./ python tools/test.py configs/oneformer3d_1xb2_multiview_s3dis.py work_dirs/oneformer3d_1xb2_multiview_s3dis/20250311_132423/best_all_ap_50%_epoch_12_fix.pth


