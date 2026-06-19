import cv2
import numpy as np
from skimage.measure import approximate_polygon
from collections import defaultdict


class CenterlineExtractor:
    """中心线提取器"""
    
    @staticmethod
    def skeletonize(mask):
        """细化算法提取骨架"""
        # 使用Zhang-Suen细化算法
        size = np.size(mask)
        skel = np.zeros(mask.shape, np.uint8)
        
        img = mask.copy()
        img = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)[1]
        
        element = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))
        done = False
        
        while not done:
            eroded = cv2.erode(img, element)
            temp = cv2.dilate(eroded, element)
            temp = cv2.subtract(img, temp)
            skel = cv2.bitwise_or(skel, temp)
            img = eroded.copy()
            
            zeros = size - cv2.countNonZero(img)
            if zeros == size:
                done = True
        
        return skel
    
    @staticmethod
    def skeleton_to_points(skeleton):
        """将骨架转换为有序点集"""
        # 查找轮廓
        contours, _ = cv2.findContours(skeleton, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        
        if not contours:
            return None
        
        # 取最长的轮廓
        contour = max(contours, key=lambda c: cv2.arcLength(c, False))
        points = contour.squeeze()
        
        return points


class LineDetector:
    """直线检测器"""
    
    @staticmethod
    def is_vertical_line(points, angle_threshold=15):
        """判断是否为垂直线"""
        if len(points) < 2:
            return False
        
        # 计算整体方向
        x_coords = points[:, 0]
        y_coords = points[:, 1]
        
        # 计算线性回归
        n = len(points)
        x_mean = np.mean(x_coords)
        y_mean = np.mean(y_coords)
        
        numerator = np.sum((x_coords - x_mean) * (y_coords - y_mean))
        denominator = np.sum((x_coords - x_mean) ** 2)
        
        if abs(denominator) < 1e-6:
            return True
        
        slope = numerator / denominator
        angle = np.abs(np.arctan(slope) * 180 / np.pi)
        
        # 垂直线应该接近90度（即与x轴夹角接近90度）
        return (90 - angle_threshold) <= angle <= (90 + angle_threshold)
    
    @staticmethod
    def fit_line(points):
        """拟合直线"""
        if len(points) < 2:
            return None
        
        # 使用最小二乘法拟合直线
        [vx, vy, x, y] = cv2.fitLine(points, cv2.DIST_L2, 0, 0.01, 0.01)
        
        # 计算直线端点（延长到图像边界）
        height, width = 2048, 2892  # 默认尺寸
        
        # 找到直线上的两个点
        lefty = int((-x * vy / vx) + y) if abs(vx) > 1e-6 else 0
        righty = int(((width - x) * vy / vx) + y) if abs(vx) > 1e-6 else 0
        
        # 或者用另一种方式找端点
        p1 = np.array([x - vx * 1000, y - vy * 1000]).flatten()
        p2 = np.array([x + vx * 1000, y + vy * 1000]).flatten()
        
        # 裁剪到图像范围内
        def clip_point(pt, w, h):
            return np.clip(pt[0], 0, w), np.clip(pt[1], 0, h)
        
        p1 = clip_point(p1, width, height)
        p2 = clip_point(p2, width, height)
        
        return np.array([p1, p2])


class ArcFilter:
    """弧线过滤器"""
    
    @staticmethod
    def is_horizontal_arc(points, angle_threshold=45):
        """判断是否为水平弧线"""
        if len(points) < 3:
            return False
        
        # 计算整体方向
        x_coords = points[:, 0]
        y_coords = points[:, 1]
        
        # 计算线性回归
        n = len(points)
        x_mean = np.mean(x_coords)
        y_mean = np.mean(y_coords)
        
        numerator = np.sum((x_coords - x_mean) * (y_coords - y_mean))
        denominator = np.sum((x_coords - x_mean) ** 2)
        
        if abs(denominator) < 1e-6:
            return False
        
        slope = numerator / denominator
        angle = np.abs(np.arctan(slope) * 180 / np.pi)
        
        # 水平线应该接近0度
        return angle <= angle_threshold or angle >= (180 - angle_threshold)
    
    @staticmethod
    def simplify_arc(points, tolerance=2.0):
        """简化弧线（抽稀）"""
        if len(points) < 3:
            return points
        
        return approximate_polygon(points, tolerance=tolerance)


class LineMerger:
    """线段合并器"""
    
    @staticmethod
    def distance_between_lines(line1, line2):
        """计算两条直线之间的距离"""
        # 简单计算端点之间的最小距离
        p1, p2 = line1
        p3, p4 = line2
        
        dists = [
            np.linalg.norm(p1 - p3),
            np.linalg.norm(p1 - p4),
            np.linalg.norm(p2 - p3),
            np.linalg.norm(p2 - p4)
        ]
        
        return min(dists)
    
    @staticmethod
    def merge_lines(lines, distance_threshold=50):
        """合并相近的线段"""
        if len(lines) <= 1:
            return lines
        
        # 使用简单的聚类方法
        merged = []
        used = [False] * len(lines)
        
        for i in range(len(lines)):
            if used[i]:
                continue
            
            group = [lines[i]]
            used[i] = True
            
            for j in range(i + 1, len(lines)):
                if used[j]:
                    continue
                
                dist = LineMerger.distance_between_lines(lines[i], lines[j])
                if dist < distance_threshold:
                    group.append(lines[j])
                    used[j] = True
            
            # 合并组内的线段（这里简单取第一个）
            if group:
                merged.append(group[0])
        
        return merged


class PostProcessor:
    """后处理流水线"""
    
    def __init__(self):
        self.centerline_extractor = CenterlineExtractor()
        self.line_detector = LineDetector()
        self.arc_filter = ArcFilter()
        self.line_merger = LineMerger()
    
    def process_mask(self, mask, class_id):
        """
        处理单个mask
        
        Args:
            mask: 二值mask
            class_id: 类别ID (0: vertical_line, 1: horizontal_arc)
        
        Returns:
            处理后的线段或弧线
        """
        # 1. 提取中心线
        skeleton = self.centerline_extractor.skeletonize(mask)
        points = self.centerline_extractor.skeleton_to_points(skeleton)
        
        if points is None or len(points) < 2:
            return None
        
        result = None
        
        if class_id == 0:  # vertical_line
            # 2. 判断是否为垂直线
            if self.line_detector.is_vertical_line(points):
                # 3. 拟合直线
                line = self.line_detector.fit_line(points)
                if line is not None:
                    result = {'type': 'vertical_line', 'points': line}
        
        elif class_id == 1:  # horizontal_arc
            # 2. 判断是否为水平弧线
            if self.arc_filter.is_horizontal_arc(points):
                # 3. 简化弧线
                simplified = self.arc_filter.simplify_arc(points, tolerance=2.0)
                result = {'type': 'horizontal_arc', 'points': simplified}
        
        return result
    
    def process_results(self, masks, class_ids, confidences, confidence_threshold=0.5):
        """
        处理所有模型输出结果
        
        Args:
            masks: 模型输出的masks
            class_ids: 类别ID列表
            confidences: 置信度列表
            confidence_threshold: 置信度阈值
        
        Returns:
            处理后的结果字典
        """
        results = {
            'vertical_lines': [],
            'horizontal_arcs': []
        }
        
        for mask, class_id, conf in zip(masks, class_ids, confidences):
            if conf < confidence_threshold:
                continue
            
            processed = self.process_mask(mask, class_id)
            if processed is not None:
                if processed['type'] == 'vertical_line':
                    results['vertical_lines'].append(processed['points'])
                elif processed['type'] == 'horizontal_arc':
                    results['horizontal_arcs'].append(processed['points'])
        
        # 合并垂直线
        if results['vertical_lines']:
            results['vertical_lines'] = self.line_merger.merge_lines(
                results['vertical_lines'],
                distance_threshold=50
            )
        
        return results


def visualize_results(img, results, output_path=None):
    """可视化结果"""
    vis_img = img.copy()
    
    # 绘制垂直线（红色）
    for line in results.get('vertical_lines', []):
        if len(line) >= 2:
            p1 = tuple(map(int, line[0]))
            p2 = tuple(map(int, line[1]))
            cv2.line(vis_img, p1, p2, (0, 0, 255), 2)
    
    # 绘制水平弧线（绿色）
    for arc in results.get('horizontal_arcs', []):
        if len(arc) >= 2:
            points = np.array(arc, dtype=np.int32)
            cv2.polylines(vis_img, [points], False, (0, 255, 0), 2)
    
    if output_path:
        cv2.imwrite(output_path, vis_img)
    
    return vis_img
