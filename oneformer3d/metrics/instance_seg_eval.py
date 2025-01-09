# Copied from mmdet3d/evaluation/functional/instance_seg_eval.py
# We fix instance seg metric to accept boolean instance seg mask of
# shape (n_points, n_instances) instead of integer mask of shape
# (n_points, ).
import numpy as np
from mmengine.logging import print_log
from terminaltables import AsciiTable
import matplotlib.pyplot as plt
from pathlib import Path

from .evaluate_semantic_instance import scannet_eval


# 1) We fix this line: info[file_name]['mask'] = mask[i].
# 2) mask.max() + 1 in for is always equal to 2.
#    We have changed it to mask.shape[0] for iterating over all masks.
def aggregate_predictions(masks, labels, scores, valid_class_ids):
    """Maps predictions to ScanNet evaluator format.

    Args:
        masks (list[torch.Tensor]): Per scene predicted instance masks.
        labels (list[torch.Tensor]): Per scene predicted instance labels.
        scores (list[torch.Tensor]): Per scene predicted instance scores.
        valid_class_ids (tuple[int]): Ids of valid categories.

    Returns:
        list[dict]: Per scene aggregated predictions.
    """
    infos = []
    for id, (mask, label, score) in enumerate(zip(masks, labels, scores)):
        mask = mask.numpy()
        label = label.numpy()
        score = score.numpy()
        info = dict()
        for i in range(mask.shape[0]):
            # match pred_instance['filename'] from assign_instances_for_scan
            file_name = f'{id}_{i}'
            info[file_name] = dict()
            info[file_name]['mask'] = mask[i]
            info[file_name]['label_id'] = valid_class_ids[label[i]]
            info[file_name]['conf'] = score[i]
        infos.append(info)
    return infos


# For some reason the inputs are not torch.Tensor but np.ndarray.
# We just remove torch -> numpy conversion here.
def check_data_consistency(gt_semantic_masks, gt_instance_masks, valid_class_ids, save_dir=None):
    """检查数据一致性，包括语义标签和实例标签的对应关系。

    Args:
        gt_semantic_masks (list[np.ndarray]): 语义分割标签
        gt_instance_masks (list[np.ndarray]): 实例分割标签
        valid_class_ids (tuple[int]): 有效的类别ID
        save_dir (str, optional): 保存统计信息的目录
    """
    print("\n=== 开始数据一致性检查 ===")
    
    for scene_idx, (semantic_mask, instance_mask) in enumerate(zip(gt_semantic_masks, gt_instance_masks)):
        print(f"\n检查场景 {scene_idx}:")
        print(f"语义标签形状: {semantic_mask.shape}")
        print(f"实例标签形状: {instance_mask.shape}")
        
        # 检查标签值范围
        semantic_unique = np.unique(semantic_mask)
        instance_unique = np.unique(instance_mask)
        print(f"唯一语义标签: {semantic_unique}")
        print(f"实例数量: {len(instance_unique)-1}")  # 减去背景
        
        # 统计每个实例的信息
        instance_stats = []
        for inst_id in instance_unique:
            if inst_id == 0:  # 跳过背景
                continue
                
            inst_mask = instance_mask == inst_id
            inst_semantic = semantic_mask[inst_mask]
            unique_labels = np.unique(inst_semantic)
            
            # 计算每个语义标签的点数和百分比
            label_stats = []
            total_points = len(inst_semantic)
            for label in unique_labels:
                count = np.sum(inst_semantic == label)
                percentage = (count / total_points) * 100
                label_stats.append(f"标签{label}: {count}点 ({percentage:.2f}%)")
            
            instance_stats.append({
                'instance_id': inst_id,
                'total_points': total_points,
                'unique_labels': unique_labels,
                'label_stats': label_stats,
                'is_valid': len(unique_labels) == 1 and unique_labels[0] in valid_class_ids
            })
        
        # 输出详细统计信息
        print("\n实例统计信息:")
        for stat in instance_stats:
            print(f"\n实例 {stat['instance_id']}:")
            print(f"总点数: {stat['total_points']}")
            print(f"包含的语义标签: {stat['unique_labels']}")
            print("标签统计:")
            for label_stat in stat['label_stats']:
                print(f"  {label_stat}")
            print(f"是否有效: {'是' if stat['is_valid'] else '否'}")
        
        # 如果需要保存统计信息
        if save_dir:
            save_dir = Path(save_dir)
            save_dir.mkdir(parents=True, exist_ok=True)
            
            # 保存统计信息到文本文件
            with open(save_dir / f'scene_{scene_idx}_stats.txt', 'w') as f:
                f.write(f"场景 {scene_idx} 统计信息\n")
                f.write(f"语义标签形状: {semantic_mask.shape}\n")
                f.write(f"实例标签形状: {instance_mask.shape}\n")
                f.write(f"唯一语义标签: {semantic_unique}\n")
                f.write(f"实例数量: {len(instance_unique)-1}\n\n")
                
                for stat in instance_stats:
                    f.write(f"\n实例 {stat['instance_id']}:\n")
                    f.write(f"总点数: {stat['total_points']}\n")
                    f.write(f"包含的语义标签: {stat['unique_labels']}\n")
                    f.write("标签统计:\n")
                    for label_stat in stat['label_stats']:
                        f.write(f"  {label_stat}\n")
                    f.write(f"是否有效: {'是' if stat['is_valid'] else '否'}\n")
    
    print("\n=== 数据一致性检查完成 ===")

def rename_gt(gt_semantic_masks, gt_instance_masks, valid_class_ids):
    """Rename instance masks with semantic labels.

    Args:
        gt_semantic_masks (list[np.ndarray]): Per scene gt semantic masks.
        gt_instance_masks (list[np.ndarray]): Per scene gt instance masks.
        valid_class_ids (tuple[int]): Ids of valid categories.

    Returns:
        list[np.array]: Per scene instance masks.
    """
    # 首先进行数据一致性检查
    check_data_consistency(gt_semantic_masks, gt_instance_masks, valid_class_ids)
    
    renamed_instance_masks = []
    for scene_idx, (semantic_mask, instance_mask) in enumerate(zip(gt_semantic_masks,
                                            gt_instance_masks)):
        print(f"\nProcessing scene {scene_idx}")
        print(f"Scene semantic mask shape: {semantic_mask.shape}")
        print(f"Scene instance mask shape: {instance_mask.shape}")
        
        unique = np.unique(instance_mask)
        print(f"Number of unique instances in scene: {len(unique)}")
        assert len(unique) < 1000
        
        for i in unique:
            if i == 0:  # 通常0表示背景，跳过
                continue
                
            semantic_instance = semantic_mask[instance_mask == i]
            semantic_unique = np.unique(semantic_instance)
            
            if len(semantic_unique) > 1:
                print(f"\nWarning: Instance {i} in scene {scene_idx} has multiple semantic labels!")
                print(f"Semantic labels found: {semantic_unique}")
                print(f"Label counts:")
                for label in semantic_unique:
                    count = np.sum(semantic_instance == label)
                    percentage = (count / len(semantic_instance)) * 100
                    print(f"Label {label}: {count} points ({percentage:.2f}%)")
            
            try:
                assert len(semantic_unique) == 1
                if semantic_unique[0] in valid_class_ids:
                    instance_mask[instance_mask == i] = 1000 * semantic_unique[0] + i
            except AssertionError:
                print(f"\nAssertion failed for instance {i} in scene {scene_idx}")
                print(f"Valid class IDs: {valid_class_ids}")
                continue  # 跳过这个实例，继续处理其他实例
                
        renamed_instance_masks.append(instance_mask)
    return renamed_instance_masks

def instance_seg_eval(gt_semantic_masks,
                      gt_instance_masks,
                      pred_instance_masks,
                      pred_instance_labels,
                      pred_instance_scores,
                      valid_class_ids,
                      class_labels,
                      options=None,
                      logger=None):
    """Instance Segmentation Evaluation.

    Evaluate the result of the instance segmentation.

    Args:
        gt_semantic_masks (list[torch.Tensor]): Ground truth semantic masks.
        gt_instance_masks (list[torch.Tensor]): Ground truth instance masks.
        pred_instance_masks (list[torch.Tensor]): Predicted instance masks.
        pred_instance_labels (list[torch.Tensor]): Predicted instance labels.
        pred_instance_scores (list[torch.Tensor]): Predicted instance labels.
        valid_class_ids (tuple[int]): Ids of valid categories.
        class_labels (tuple[str]): Names of valid categories.
        options (dict, optional): Additional options. Keys may contain:
            `overlaps`, `min_region_sizes`, `distance_threshes`,
            `distance_confs`. Default: None.
        logger (logging.Logger | str, optional): The way to print the mAP
            summary. See `mmdet.utils.print_log()` for details. Default: None.

    Returns:
        dict[str, float]: Dict of results.
    """
    assert len(valid_class_ids) == len(class_labels)
    id_to_label = {
        valid_class_ids[i]: class_labels[i]
        for i in range(len(valid_class_ids))
    }
    preds = aggregate_predictions(
        masks=pred_instance_masks,
        labels=pred_instance_labels,
        scores=pred_instance_scores,
        valid_class_ids=valid_class_ids)
    gts = rename_gt(gt_semantic_masks, gt_instance_masks, valid_class_ids)
    metrics = scannet_eval(
        preds=preds,
        gts=gts,
        options=options,
        valid_class_ids=valid_class_ids,
        class_labels=class_labels,
        id_to_label=id_to_label)
    header = ['classes', 'AP_0.25', 'AP_0.50', 'AP', 'Prec_0.50', 'Rec_0.50']
    rows = []
    for label, data in metrics['classes'].items():
        aps = [data['ap25%'], data['ap50%'], data['ap'], data['prec50%'], data['rec50%']]
        rows.append([label] + [f'{ap:.4f}' for ap in aps])
    aps = metrics['all_ap_25%'], metrics['all_ap_50%'], metrics['all_ap'], metrics['all_prec_50%'], metrics['all_rec_50%']
    footer = ['Overall'] + [f'{ap:.4f}' for ap in aps]
    table = AsciiTable([header] + rows + [footer])
    table.inner_footing_row_border = True
    print_log('\n' + table.table, logger=logger)
    return metrics
