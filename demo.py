#!/usr/bin/env python3
"""
传统算法检测地图经纬线 - 改进版
使用 Sobel 边缘检测而不是简单的颜色阈值
"""
import cv2
import numpy as np
from skimage.measure import approximate_polygon
from pathlib import Path


def traditional_map_line_detection(image_path):
    """改进的传统算法检测经纬线"""
    # 读入图像
    img = cv2.imread(image_path)
    if img is None:
        print(f"无法读取图像: {image_path}")
        return None, None
    
    h, w = img.shape[:2]
    
    # 预处理
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (3, 3), 0)
    
    # --------------------------
    # 检测经线（垂直线）
    # --------------------------
    sobel_x = cv2.Sobel(blurred, cv2.CV_64F, 1, 0, ksize=3)
    sobel_x = np.uint8(np.absolute(sobel_x))
    _, vertical_mask = cv2.threshold(sobel_x, 30, 255, cv2.THRESH_BINARY)
    
    # 形态学操作连接断开的线
    vertical_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 15))
    vertical_mask = cv2.morphologyEx(vertical_mask, cv2.MORPH_CLOSE, vertical_kernel)
    
    # Hough变换检测直线
    vertical_lines = []
    lines = cv2.HoughLinesP(
        vertical_mask, 1, np.pi/180, threshold=80,
        minLineLength=int(h * 0.5), maxLineGap=30
    )
    
    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0]
            angle = abs(np.arctan2(y2 - y1, x2 - x1) * 180 / np.pi)
            if 75 < angle < 105:
                vertical_lines.append([(int(x1), int(y1)), (int(x2), int(y2))])
    
    # --------------------------
    # 检测纬线（水平弧线）
    # --------------------------
    sobel_y = cv2.Sobel(blurred, cv2.CV_64F, 0, 1, ksize=3)
    sobel_y = np.uint8(np.absolute(sobel_y))
    _, horizontal_mask = cv2.threshold(sobel_y, 30, 255, cv2.THRESH_BINARY)
    
    # 形态学操作连接断开的线
    horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 1))
    horizontal_mask = cv2.morphologyEx(horizontal_mask, cv2.MORPH_CLOSE, horizontal_kernel)
    
    # 找轮廓
    contours, _ = cv2.findContours(horizontal_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    
    horizontal_arcs = []
    for cnt in contours:
        if cv2.arcLength(cnt, closed=False) > w * 0.3:
            points = approximate_polygon(cnt.squeeze(), tolerance=2)
            horizontal_arcs.append(points.tolist())
    
    return vertical_lines, horizontal_arcs


def visualize_results(img, vertical_lines, horizontal_arcs, output_path):
    """可视化结果"""
    vis_img = img.copy()
    
    # 绘制经线（红色）
    for (x1, y1), (x2, y2) in vertical_lines:
        cv2.line(vis_img, (x1, y1), (x2, y2), (0, 0, 255), 2)
    
    # 绘制纬线（绿色）
    for arc in horizontal_arcs:
        if len(arc) >= 2:
            points = np.array(arc, dtype=np.int32)
            cv2.polylines(vis_img, [points], False, (0, 255, 0), 2)
    
    cv2.imwrite(output_path, vis_img)
    return vis_img


if __name__ == '__main__':
    # 测试
    raw_data_dir = Path('map_line_dataset/raw_data')
    image_files = list(raw_data_dir.glob('*.jpg'))
    
    if image_files:
        test_image = str(image_files[0])
        print(f"处理图像: {test_image}")
        
        vertical_lines, horizontal_arcs = traditional_map_line_detection(test_image)
        
        if vertical_lines is not None:
            print(f"检测到 {len(vertical_lines)} 条经线")
            print(f"检测到 {len(horizontal_arcs)} 条纬线")
            
            img = cv2.imread(test_image)
            output_dir = Path('output')
            output_dir.mkdir(exist_ok=True)
            output_path = output_dir / f'demo_{Path(test_image).name}'
            visualize_results(img, vertical_lines, horizontal_arcs, str(output_path))
            print(f"结果已保存到: {output_path}")
