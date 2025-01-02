# 训练

PYTHON=./ CUDA_VISIBLE_DEVICES=1,2,3 bash tools/dist_train.sh configs/oneformer3d_1xb2_s3dis-area-5.py 3

# 测试

# 第1步:修复权重格式
python tools/fix_spconv_checkpoint.py --in-path work_dirs/oneformer3d_1xb2_s3dis-area-5/best_all_ap_50%_epoch_422.pth \
    --out-path work_dirs/oneformer3d_1xb2_s3dis-area-5/best_all_ap_50%_epoch_422.pth

# 第2步:使用修复后的权重进行测试
PYTHONPATH=./ python tools/test.py configs/oneformer3d_1xb2_s3dis-area-5.py work_dirs/oneformer3d_1xb2_s3dis-area-5/best_all_ap_50%_epoch_422.pth