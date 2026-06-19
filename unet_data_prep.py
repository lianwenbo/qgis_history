"""
UNet 数据准备模块
LabelMe JSON → 语义分割图 (0=背景, 1=经线, 2=纬线)
"""
import json
import numpy as np
import cv2
from pathlib import Path
from tqdm import tqdm


class UNetDataPreparer:
    """UNet 数据准备器"""
    
    def __init__(self, raw_data_dir, output_dir):
        self.raw_data_dir = Path(raw_data_dir)
        self.output_dir = Path(output_dir)
        self.images_dir = self.output_dir / "images"
        self.masks_dir = self.output_dir / "masks"
        self.images_dir.mkdir(parents=True, exist_ok=True)
        self.masks_dir.mkdir(parents=True, exist_ok=True)
    
    def _create_mask(self, shapes, img_shape):
        """从标注创建语义分割图 (0=背景, 1=经线, 2=纬线)"""
        h, w = img_shape[:2]
        mask = np.zeros((h, w), dtype=np.uint8)
        
        for shape in shapes:
            label = shape.get('label', '')
            points = shape.get('points', [])
            shape_type = shape.get('shape_type', 'line')
            
            if len(points) < 2:
                continue
            
            points_array = np.array(points, dtype=np.int32)
            
            # 根据标签决定像素值
            if label == 'vertical_line':
                pixel_value = 1
            elif label == 'horizontal_arc':
                pixel_value = 2
            else:
                continue
            
            # 绘制线条 (宽度5像素，确保线条足够宽以便学习)
            if shape_type == 'line' and len(points_array) == 2:
                cv2.line(mask, 
                        tuple(points_array[0]), 
                        tuple(points_array[1]), 
                        pixel_value, 
                        thickness=3)
            elif shape_type in ['polygon', 'linestrip'] and len(points_array) > 2:
                # 折线/多边形：分段画线
                for i in range(len(points_array) - 1):
                    cv2.line(mask,
                            tuple(points_array[i]),
                            tuple(points_array[i + 1]),
                            pixel_value,
                            thickness=3)
                # 闭合多边形
                if shape_type == 'polygon':
                    cv2.line(mask,
                            tuple(points_array[-1]),
                            tuple(points_array[0]),
                            pixel_value,
                            thickness=3)
            else:
                # 多段线
                for i in range(len(points_array) - 1):
                    cv2.line(mask,
                            tuple(points_array[i]),
                            tuple(points_array[i + 1]),
                            pixel_value,
                            thickness=3)
        
        return mask
    
    def prepare(self):
        """准备所有数据"""
        json_files = sorted(self.raw_data_dir.glob("*.json"))
        print(f"找到 {len(json_files)} 个标注文件")
        
        for json_file in tqdm(json_files, desc="转换标注"):
            # 读取JSON
            with open(json_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            # 获取图像路径
            image_filename = data.get('imagePath', '')
            image_path = self.raw_data_dir / image_filename
            if not image_path.exists():
                # 尝试同名不同扩展名
                for ext in ['.jpg', '.jpeg', '.png', '.JPG']:
                    test_path = json_file.with_suffix(ext)
                    if test_path.exists():
                        image_path = test_path
                        break
            
            if not image_path.exists():
                print(f"  跳过 {json_file.name}: 找不到图像 {image_path}")
                continue
            
            # 读取图像
            img = cv2.imread(str(image_path))
            if img is None:
                print(f"  跳过 {json_file.name}: 无法读取图像")
                continue
            
            h, w = img.shape[:2]
            
            # 创建语义分割mask
            shapes = data.get('shapes', [])
            mask = self._create_mask(shapes, (h, w, 3))
            
            # 保存 (用PNG以避免压缩)
            output_name = json_file.stem
            cv2.imwrite(str(self.images_dir / f"{output_name}.png"), img)
            cv2.imwrite(str(self.masks_dir / f"{output_name}.png"), mask)
        
        print(f"\n完成！图像: {len(list(self.images_dir.glob('*.png')))} 张, "
              f"Mask: {len(list(self.masks_dir.glob('*.png')))} 张")
        print(f"保存到: {self.output_dir}")


if __name__ == '__main__':
    preparer = UNetDataPreparer(
        raw_data_dir='map_line_dataset/raw_data',
        output_dir='map_line_dataset/unet_format'
    )
    preparer.prepare()
