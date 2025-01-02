from mmengine.registry import METRICS
from ..metrics.unified_metric import UnifiedSegMetric
import numpy as np
import os
import os.path as osp
import torch

@METRICS.register_module() 
class VisualizationEvaluator(UnifiedSegMetric):
    """导出语义分割和实例分割的预测结果"""
    
    def compute_metrics(self, results):
        # 获取原始评估结果
        metrics = super().compute_metrics(results)
        
        # 遍历每个场景结果
        for i, (eval_ann, pred_results) in enumerate(results):
            # 获取预测结果 
            instance_mask = pred_results['pts_instance_mask']
            semantic_mask = pred_results['pts_semantic_mask']
            
            # 确保是单个mask而不是列表
            if isinstance(instance_mask, list):
                instance_mask = instance_mask[0]
            if isinstance(semantic_mask, list):
                semantic_mask = semantic_mask[0]
                
            # 添加维度检查
            print(f"Scene {i} - Raw points in annotation:", 
                  len(eval_ann.get('pts_semantic_mask', [])))
            print(f"Scene {i} - Points in prediction:", 
                  semantic_mask.shape[0])
            
            # 保存结果
            save_path = osp.join('visualization', f'scene_{i:04d}.txt')
            self.export_masks(
                instance_preds=instance_mask,
                semantic_preds=semantic_mask,
                save_path=save_path
            )
            
        return metrics
        
    @staticmethod
    def export_masks(instance_preds, semantic_preds, save_path):
        """导出分割mask为txt格式
        
        Args:
            instance_preds: 实例分割预测结果 (54, n_points)
            semantic_preds: 语义分割预测结果 (n_points,)
            save_path: 保存路径
        """
        # 转换为numpy格式
        if isinstance(instance_preds, torch.Tensor):
            instance_preds = instance_preds.cpu().numpy()
        if isinstance(semantic_preds, torch.Tensor):
            semantic_preds = semantic_preds.cpu().numpy()
            
        # 调试信息
        print("Instance predictions shape:", instance_preds.shape)
        print("Semantic predictions shape:", semantic_preds.shape)
        print("Expected raw points:", len(semantic_preds))
        
        # 确保维度匹配
        if instance_preds.shape[1] != len(semantic_preds):
            raise ValueError(
                f"Dimension mismatch: instance_preds has {instance_preds.shape[1]} points "
                f"but semantic_preds has {len(semantic_preds)} points. "
                "Please ensure the predictions are mapped back to raw point cloud."
            )
        
        # 将54个实例预测合并为单个实例ID
        instance_preds = instance_preds.T  # 转置为 (n_points, 54)
        instance_ids = np.argmax(instance_preds, axis=1)  # (n_points,)
        
        # 组合数据
        combined_data = np.column_stack([
            semantic_preds,    # 语义分割ID (n_points,)
            instance_ids,      # 实例分割ID (n_points,)
        ])
        
        # 确保输出目录存在
        os.makedirs(osp.dirname(save_path), exist_ok=True)
        
        # 保存为txt,只有两列
        np.savetxt(
            save_path,
            combined_data,
            fmt='%d %d', # 整数格式
            header='semantic_id instance_id'
        )