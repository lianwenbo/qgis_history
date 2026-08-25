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
        """从标注创建语义分割图 (0=背景, 1=经线, 2=纬线, 3=分隔线)"""
        h, w = img_shape[:2]
        mask = np.zeros((h, w), dtype=np.uint8)
        
        label_map = {
            'vertical_line': 1,
            'horizontal_arc': 2,
            'splitter': 3
        }
        
        for shape in shapes:
            label = shape.get('label', '')
            points = shape.get('points', [])
            shape_type = shape.get('shape_type', 'line')
            
            if len(points) < 2:
                continue
            
            if label not in label_map:
                continue
            
            pixel_value = label_map[label]
            points_array = np.array(points, dtype=np.int32)
            
            thickness = 9 if label == 'splitter' else 3
            
            if shape_type == 'line' and len(points_array) == 2:
                cv2.line(mask, 
                        tuple(points_array[0]), 
                        tuple(points_array[1]), 
                        pixel_value, 
                        thickness=thickness)
            elif shape_type in ['polygon', 'linestrip'] and len(points_array) > 2:
                for i in range(len(points_array) - 1):
                    cv2.line(mask,
                            tuple(points_array[i]),
                            tuple(points_array[i + 1]),
                            pixel_value,
                            thickness=thickness)
                if shape_type == 'polygon':
                    cv2.line(mask,
                            tuple(points_array[-1]),
                            tuple(points_array[0]),
                            pixel_value,
                            thickness=thickness)
            else:
                for i in range(len(points_array) - 1):
                    cv2.line(mask,
                            tuple(points_array[i]),
                            tuple(points_array[i + 1]),
                            pixel_value,
                            thickness=thickness)
        
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


def _clip_line_segment(p1, p2, rect_x1, rect_y1, rect_x2, rect_y2):
    """Liang-Barsky 线段裁剪算法：计算线段与矩形的交点
    
    Args:
        p1, p2: 线段两个端点 [x, y]
        rect_x1, rect_y1, rect_x2, rect_y2: 矩形边界
    
    Returns:
        list: 裁剪后的线段点列表（0个/1个/2个点）
              2个点表示线段穿过矩形（部分或全部在内部）
              1个点表示端点刚好在边界上
              0个点表示线段完全在矩形外
    """
    x1, y1 = p1
    x2, y2 = p2
    dx = x2 - x1
    dy = y2 - y1
    
    p = [-dx, dx, -dy, dy]
    q = [x1 - rect_x1, rect_x2 - x1, y1 - rect_y1, rect_y2 - y1]
    
    u1 = 0.0
    u2 = 1.0
    
    for i in range(4):
        if p[i] == 0:
            if q[i] < 0:
                return []
        else:
            t = q[i] / p[i]
            if p[i] < 0:
                if t > u2:
                    return []
                if t > u1:
                    u1 = t
            else:
                if t < u1:
                    return []
                if t < u2:
                    u2 = t
    
    result = []
    if u1 <= u2:
        if u1 > 0:
            result.append([x1 + u1 * dx, y1 + u1 * dy])
        if u2 < 1:
            result.append([x1 + u2 * dx, y1 + u2 * dy])
        if u1 == 0 and u2 == 1:
            result = [list(p1), list(p2)]
        elif u1 == 0:
            result.insert(0, list(p1))
        elif u2 == 1:
            result.append(list(p2))
    
    return result


def _clip_polyline_to_rect(points, rect_x1, rect_y1, rect_x2, rect_y2):
    """将折线裁剪到矩形内，返回裁剪后的多段折线（可能被切成多段）
    
    Args:
        points: 折线点列表 [[x, y], ...]
        rect_x1, rect_y1, rect_x2, rect_y2: 矩形边界
    
    Returns:
        list: 裁剪后的折线列表，每条折线是点列表 [[x, y], ...]
    """
    if len(points) < 2:
        return []
    
    segments = []
    current_segment = []
    
    for i in range(len(points) - 1):
        p1 = points[i]
        p2 = points[i + 1]
        
        clipped = _clip_line_segment(p1, p2, rect_x1, rect_y1, rect_x2, rect_y2)
        
        if len(clipped) == 2:
            if len(current_segment) == 0:
                current_segment = [clipped[0], clipped[1]]
            else:
                if abs(current_segment[-1][0] - clipped[0][0]) < 1e-6 and abs(current_segment[-1][1] - clipped[0][1]) < 1e-6:
                    current_segment.append(clipped[1])
                else:
                    if len(current_segment) >= 2:
                        segments.append(current_segment)
                    current_segment = [clipped[0], clipped[1]]
        elif len(clipped) == 1:
            if len(current_segment) > 0:
                if abs(current_segment[-1][0] - clipped[0][0]) < 1e-6 and abs(current_segment[-1][1] - clipped[0][1]) < 1e-6:
                    pass
                else:
                    if len(current_segment) >= 2:
                        segments.append(current_segment)
                    current_segment = []
        else:
            if len(current_segment) >= 2:
                segments.append(current_segment)
            current_segment = []
    
    if len(current_segment) >= 2:
        segments.append(current_segment)
    
    return segments


def split_into_patches(raw_data_dir, output_dir, patch_size=512, overlap=0.5):
    """将原始大图切分为 512×512 的 patch，生成新的 labelme 格式数据
    
    使用线段-矩形求交算法，正确处理：
    - 穿过 patch 边界的线（在边界处生成新交点）
    - 经线/竖直线（两个端点都在 patch 外但线穿过 patch）
    
    Args:
        raw_data_dir: 原始数据目录
        output_dir: 输出目录（与 raw_data 同一层级）
        patch_size: patch 大小
        overlap: 重叠率 (0.0~1.0)
    """
    raw_data_dir = Path(raw_data_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    json_files = sorted(raw_data_dir.glob("*.json"))
    print(f"找到 {len(json_files)} 个标注文件")
    
    step = int(patch_size * (1 - overlap))
    
    for json_file in tqdm(json_files, desc="切分图像"):
        with open(json_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        image_filename = data.get('imagePath', '')
        image_path = raw_data_dir / image_filename
        if not image_path.exists():
            for ext in ['.jpg', '.jpeg', '.png', '.JPG']:
                test_path = json_file.with_suffix(ext)
                if test_path.exists():
                    image_path = test_path
                    break
        
        if not image_path.exists():
            print(f"  跳过 {json_file.name}: 找不到图像")
            continue
        
        img = cv2.imread(str(image_path))
        if img is None:
            print(f"  跳过 {json_file.name}: 无法读取图像")
            continue
        
        h, w = img.shape[:2]
        base_name = json_file.stem
        
        rows = ((h - patch_size + step - 1) // step) + 1
        cols = ((w - patch_size + step - 1) // step) + 1
        
        padded_h = (rows - 1) * step + patch_size
        padded_w = (cols - 1) * step + patch_size
        
        bg_color = [245, 235, 210]
        padded_img = np.full((padded_h, padded_w, 3), bg_color, dtype=np.uint8)
        padded_img[:h, :w] = img
        
        all_shapes = []
        for shape in data.get('shapes', []):
            label = shape.get('label', '')
            points = shape.get('points', [])
            shape_type = shape.get('shape_type', 'line')
            
            if len(points) < 2:
                continue
            
            global_points = [[px, py] for (px, py) in points]
            
            all_shapes.append({
                'label': label,
                'points': global_points,
                'shape_type': shape_type,
                'flags': shape.get('flags', {}),
                'group_id': shape.get('group_id', None)
            })
        
        total_patches = 0
        
        for row in range(rows):
            for col in range(cols):
                y1_rect = row * step
                y2_rect = y1_rect + patch_size
                x1_rect = col * step
                x2_rect = x1_rect + patch_size
                
                patch_img = padded_img[y1_rect:y2_rect, x1_rect:x2_rect]
                
                patch_shapes = []
                
                for shape in all_shapes:
                    label = shape['label']
                    global_points = shape['points']
                    shape_type = shape['shape_type']
                    
                    clipped_segments = _clip_polyline_to_rect(
                        global_points, x1_rect, y1_rect, x2_rect, y2_rect
                    )
                    
                    for seg_points in clipped_segments:
                        if len(seg_points) < 2:
                            continue
                        
                        local_points = [
                            [p[0] - x1_rect, p[1] - y1_rect]
                            for p in seg_points
                        ]
                        
                        patch_shapes.append({
                            'label': label,
                            'points': local_points,
                            'shape_type': shape_type if shape_type != 'polygon' else 'linestrip',
                            'flags': shape.get('flags', {}),
                            'group_id': shape.get('group_id', None)
                        })
                
                patch_name = f"{base_name}_p{row:02d}_{col:02d}"
                patch_img_path = f"{patch_name}.jpg"
                patch_json_path = f"{patch_name}.json"
                
                cv2.imwrite(str(output_dir / patch_img_path), patch_img)
                
                patch_json = {
                    'version': data.get('version', '4.5.7'),
                    'flags': data.get('flags', {}),
                    'shapes': patch_shapes,
                    'imagePath': patch_img_path,
                    'imageData': None,
                    'imageHeight': patch_size,
                    'imageWidth': patch_size
                }
                
                with open(output_dir / patch_json_path, 'w', encoding='utf-8') as f:
                    json.dump(patch_json, f, ensure_ascii=False, indent=2)
                
                total_patches += 1
        
        print(f"  {base_name}: {rows}×{cols} = {total_patches} 个 patch")
    
    total_images = len(list(output_dir.glob('*.jpg')))
    total_jsons = len(list(output_dir.glob('*.json')))
    print(f"\n完成！共 {total_images} 张图像, {total_jsons} 个标注文件")
    print(f"保存到: {output_dir}")


if __name__ == '__main__':
    preparer = UNetDataPreparer(
        raw_data_dir='map_line_dataset/raw_data',
        output_dir='map_line_dataset/unet_format'
    )
    preparer.prepare()
