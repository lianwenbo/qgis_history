import json
import os
import cv2
import numpy as np
from pathlib import Path
from tqdm import tqdm


class LabelmeToYOLOConverter:
    def __init__(self, raw_data_dir, output_dir):
        self.raw_data_dir = Path(raw_data_dir)
        self.output_dir = Path(output_dir)
        self.images_dir = self.output_dir / 'images'
        self.labels_dir = self.output_dir / 'labels'
        
        # 类别映射
        self.class_map = {
            'vertical_line': 0,
            'horizontal_arc': 1
        }
        
        self._create_dirs()
    
    def _create_dirs(self):
        self.images_dir.mkdir(parents=True, exist_ok=True)
        self.labels_dir.mkdir(parents=True, exist_ok=True)
    
    def _create_mask_from_polygon(self, shape, img_shape):
        """从多边形创建mask"""
        mask = np.zeros(img_shape[:2], dtype=np.uint8)
        points = np.array(shape['points'], dtype=np.int32)
        
        if shape['shape_type'] == 'line':
            # 直线：扩展成多边形
            x1, y1 = points[0]
            x2, y2 = points[1]
            
            # 计算直线方向和垂直方向
            dx = x2 - x1
            dy = y2 - y1
            length = np.sqrt(dx**2 + dy**2)
            
            if length > 0:
                # 归一化垂直向量
                nx = -dy / length
                ny = dx / length
                
                # 线宽（像素）
                line_width = 3
                
                # 计算四个角点
                p1 = [int(x1 + nx * line_width), int(y1 + ny * line_width)]
                p2 = [int(x2 + nx * line_width), int(y2 + ny * line_width)]
                p3 = [int(x2 - nx * line_width), int(y2 - ny * line_width)]
                p4 = [int(x1 - nx * line_width), int(y1 - ny * line_width)]
                
                polygon = np.array([p1, p2, p3, p4], dtype=np.int32)
                cv2.fillPoly(mask, [polygon], 255)
        else:
            # 折线/曲线：扩展成多边形
            if len(points) >= 2:
                # 计算每个线段的垂直向量
                expanded_points_left = []
                expanded_points_right = []
                
                for i in range(len(points)):
                    if i < len(points) - 1:
                        x1, y1 = points[i]
                        x2, y2 = points[i + 1]
                        dx = x2 - x1
                        dy = y2 - y1
                    else:
                        x1, y1 = points[i - 1]
                        x2, y2 = points[i]
                        dx = x2 - x1
                        dy = y2 - y1
                    
                    length = np.sqrt(dx**2 + dy**2)
                    if length > 0:
                        nx = -dy / length
                        ny = dx / length
                    else:
                        nx, ny = 0, 1
                    
                    line_width = 3
                    expanded_points_left.append([
                        int(points[i][0] + nx * line_width),
                        int(points[i][1] + ny * line_width)
                    ])
                    expanded_points_right.append([
                        int(points[i][0] - nx * line_width),
                        int(points[i][1] - ny * line_width)
                    ])
                
                # 组合成闭合多边形
                polygon = np.array(
                    expanded_points_left + expanded_points_right[::-1],
                    dtype=np.int32
                )
                cv2.fillPoly(mask, [polygon], 255)
        
        return mask
    
    def _mask_to_yolo_polygon(self, mask):
        """将mask转换为YOLO格式的多边形坐标"""
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        polygons = []
        for contour in contours:
            if len(contour) >= 3:
                # 简化轮廓
                epsilon = 0.001 * cv2.arcLength(contour, True)
                approx = cv2.approxPolyDP(contour, epsilon, True)
                
                if len(approx) >= 3:
                    polygon = approx.squeeze().tolist()
                    polygons.append(polygon)
        
        return polygons
    
    def convert(self):
        """转换所有LabelMe标注到YOLO格式"""
        json_files = list(self.raw_data_dir.glob('*.json'))
        
        print(f"找到 {len(json_files)} 个标注文件")
        
        for json_file in tqdm(json_files, desc="转换标注"):
            with open(json_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            # 读取图像
            img_path = self.raw_data_dir / data['imagePath']
            if not img_path.exists():
                print(f"警告：图像文件不存在 {img_path}")
                continue
            
            img = cv2.imread(str(img_path))
            if img is None:
                print(f"警告：无法读取图像 {img_path}")
                continue
            
            img_height, img_width = img.shape[:2]
            
            # 复制图像到输出目录
            output_img_path = self.images_dir / data['imagePath']
            cv2.imwrite(str(output_img_path), img)
            
            # 生成YOLO标注
            label_lines = []
            for shape in data['shapes']:
                label = shape['label']
                if label not in self.class_map:
                    continue
                
                class_id = self.class_map[label]
                
                # 创建mask
                mask = self._create_mask_from_polygon(shape, img.shape)
                
                # 转换为YOLO多边形
                polygons = self._mask_to_yolo_polygon(mask)
                
                for polygon in polygons:
                    if len(polygon) < 3:
                        continue
                    
                    # 归一化坐标
                    normalized = []
                    for x, y in polygon:
                        nx = max(0, min(1, x / img_width))
                        ny = max(0, min(1, y / img_height))
                        normalized.extend([nx, ny])
                    
                    line = f"{class_id} {' '.join(map(str, normalized))}"
                    label_lines.append(line)
            
            # 保存标注文件
            label_file = self.labels_dir / (json_file.stem + '.txt')
            with open(label_file, 'w', encoding='utf-8') as f:
                f.write('\n'.join(label_lines))
        
        print(f"转换完成！输出到 {self.output_dir}")
        return self.output_dir


def create_yolo_dataset_yaml(output_dir, class_names):
    """创建YOLO数据集配置文件"""
    # 使用绝对路径
    output_dir_abs = Path(output_dir).resolve()
    yaml_content = f"""path: {output_dir_abs}
train: images
val: images

names:
"""
    for i, name in enumerate(class_names):
        yaml_content += f"  {i}: {name}\n"
    
    yaml_path = output_dir_abs / 'dataset.yaml'
    with open(yaml_path, 'w', encoding='utf-8') as f:
        f.write(yaml_content)
    
    return yaml_path


if __name__ == '__main__':
    # 转换数据
    converter = LabelmeToYOLOConverter(
        raw_data_dir='map_line_dataset/raw_data',
        output_dir='map_line_dataset/yolo_format'
    )
    output_dir = converter.convert()
    
    # 创建数据集配置
    create_yolo_dataset_yaml(
        output_dir=output_dir,
        class_names=['vertical_line', 'horizontal_arc']
    )
