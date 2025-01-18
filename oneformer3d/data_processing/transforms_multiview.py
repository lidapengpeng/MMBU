import numpy as np
import cv2
from mmcv.transforms import BaseTransform
from mmengine.registry import TRANSFORMS

@TRANSFORMS.register_module()
class MultiViewImageResize(BaseTransform):
    """多视角图像缩放"""
    
    def __init__(self, img_scale=(896, 1204), keep_ratio=True):
        self.img_scale = img_scale
        self.keep_ratio = keep_ratio
        
    def transform(self, results):
        """应用转换
        Args:
            results (dict): 包含多视角图像的结果字典
        Returns:
            dict: 更新后的结果字典
        """
        if 'img' not in results:
            return results
            
        imgs = results['img']
        if not isinstance(imgs, list):
            return results
            
        resized_imgs = []
        for img in imgs:
            if self.keep_ratio:
                # 保持长宽比的缩放
                h, w = img.shape[:2]
                scale = min(self.img_scale[0] / h, self.img_scale[1] / w)
                new_h, new_w = int(h * scale), int(w * scale)
                resized_img = cv2.resize(img, (new_w, new_h))
                
                # 填充到目标大小
                pad_h = self.img_scale[0] - new_h
                pad_w = self.img_scale[1] - new_w
                top = pad_h // 2
                left = pad_w // 2
                
                resized_img = cv2.copyMakeBorder(
                    resized_img, top, pad_h - top, left, pad_w - left,
                    cv2.BORDER_CONSTANT, value=0)
            else:
                # 直接缩放到目标大小
                resized_img = cv2.resize(img, self.img_scale[::-1])
                
            resized_imgs.append(resized_img)
            
        results['img'] = resized_imgs
        results['img_shape'] = [img.shape for img in resized_imgs]
        
        return results

@TRANSFORMS.register_module()
class MultiViewImageNormalize(BaseTransform):
    """多视角图像标准化"""
    
    def __init__(self, mean, std, to_rgb=True):
        self.mean = np.array(mean, dtype=np.float32)
        self.std = np.array(std, dtype=np.float32)
        self.to_rgb = to_rgb
        
    def transform(self, results):
        """应用转换
        Args:
            results (dict): 包含多视角图像的结果字典
        Returns:
            dict: 更新后的结果字典
        """
        if 'img' not in results:
            return results
            
        imgs = results['img']
        if not isinstance(imgs, list):
            return results
            
        normalized_imgs = []
        for img in imgs:
            img = img.astype(np.float32)
            
            # BGR to RGB
            if self.to_rgb:
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                
            # 标准化
            img = (img - self.mean) / self.std
            normalized_imgs.append(img)
            
        results['img'] = normalized_imgs
        return results
