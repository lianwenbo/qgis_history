import cv2
import numpy as np
import torch
from pathlib import Path
from functools import partial
from ultralytics import YOLO
from post_processing import PostProcessor


# Monkey patch: 修复 PyTorch 2.6+ 的安全加载问题
original_torch_load = torch.load
torch.load = partial(original_torch_load, weights_only=False)


class HybridMapLineDetector:
    """混合方案的地图经纬线检测器"""
    
    def __init__(self, model_path=None):
        """
        初始化检测器
        
        Args:
            model_path: 训练好的模型路径，如果为None则使用传统算法
        """
        self.model = None
        self.use_deep_learning = False
        self.post_processor = PostProcessor()
        
        if model_path and Path(model_path).exists():
            try:
                self.model = YOLO(model_path)
                self.use_deep_learning = True
                print(f"加载深度学习模型成功: {model_path}")
            except Exception as e:
                print(f"加载模型失败: {e}，将使用传统算法")
    
    def _traditional_detect(self, img):
        """
        改进的传统算法检测
        
        Args:
            img: 输入图像
        
        Returns:
            检测结果
        """
        h, w = img.shape[:2]
        
        # --------------------------
        # 步骤1: 预处理
        # --------------------------
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (3, 3), 0)
        
        # --------------------------
        # 步骤2: 检测经线（垂直线）
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
                    vertical_lines.append(np.array([[x1, y1], [x2, y2]]))
        
        # --------------------------
        # 步骤3: 检测纬线（水平弧线）
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
        from skimage.measure import approximate_polygon
        for cnt in contours:
            if cv2.arcLength(cnt, closed=False) > w * 0.3:
                points = approximate_polygon(cnt.squeeze(), tolerance=2)
                horizontal_arcs.append(points)
        
        return {
            'vertical_lines': vertical_lines,
            'horizontal_arcs': horizontal_arcs
        }
    
    def _deep_learning_detect(self, img):
        """
        深度学习检测
        
        Args:
            img: 输入图像
        
        Returns:
            检测结果
        """
        # 运行模型推理
        results = self.model(img, conf=0.3, iou=0.5)
        
        if not results or len(results) == 0:
            return {'vertical_lines': [], 'horizontal_arcs': []}
        
        result = results[0]
        
        # 获取masks、类别和置信度
        masks = []
        class_ids = []
        confidences = []
        
        if result.masks is not None:
            for i, mask_data in enumerate(result.masks.data):
                # mask_data是tensor，转换为numpy
                mask = mask_data.cpu().numpy()
                
                # 调整到原图大小
                mask = cv2.resize(mask, (img.shape[1], img.shape[0]))
                mask = (mask > 0.5).astype(np.uint8) * 255
                
                masks.append(mask)
                class_ids.append(int(result.boxes.cls[i]))
                confidences.append(float(result.boxes.conf[i]))
        
        # 后处理
        processed_results = self.post_processor.process_results(
            masks, class_ids, confidences,
            confidence_threshold=0.3
        )
        
        return processed_results
    
    def detect(self, img):
        """
        检测经纬线
        
        Args:
            img: 输入图像（BGR格式）
        
        Returns:
            检测结果字典
        """
        if self.use_deep_learning and self.model is not None:
            return self._deep_learning_detect(img)
        else:
            return self._traditional_detect(img)
    
    def detect_image(self, image_path, output_dir='output'):
        """
        检测单张图像
        
        Args:
            image_path: 图像路径
            output_dir: 输出目录
        
        Returns:
            检测结果
        """
        # 读取图像
        img = cv2.imread(image_path)
        if img is None:
            print(f"无法读取图像: {image_path}")
            return None
        
        # 检测
        results = self.detect(img)
        
        # 创建输出目录
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # 可视化结果
        img_name = Path(image_path).name
        vis_path = output_path / f'result_{img_name}'
        visualize_results(img, results, str(vis_path))
        
        # 保存坐标结果
        coord_path = output_path / f'result_{img_name}.txt'
        with open(coord_path, 'w', encoding='utf-8') as f:
            f.write("=== Vertical Lines (经线) ===\n")
            for i, line in enumerate(results['vertical_lines']):
                f.write(f"Line {i}: {line.tolist() if hasattr(line, 'tolist') else line}\n")
            
            f.write("\n=== Horizontal Arcs (纬线) ===\n")
            for i, arc in enumerate(results['horizontal_arcs']):
                f.write(f"Arc {i}: {arc.tolist() if hasattr(arc, 'tolist') else arc}\n")
        
        print(f"检测完成！结果已保存到: {output_dir}")
        print(f"  - 可视化: {vis_path}")
        print(f"  - 坐标: {coord_path}")
        print(f"  - 经线数量: {len(results['vertical_lines'])}")
        print(f"  - 纬线数量: {len(results['horizontal_arcs'])}")
        
        return results


def visualize_results(img, results, output_path=None):
    """可视化结果"""
    vis_img = img.copy()
    
    # 绘制垂直线（红色）
    for line in results.get('vertical_lines', []):
        if len(line) >= 2:
            # 处理不同的数据格式
            if hasattr(line[0], '__iter__'):
                p1 = tuple(map(int, line[0]))
                p2 = tuple(map(int, line[1]))
            else:
                # 兼容可能的不同格式
                p1 = tuple(map(int, line[:2]))
                p2 = tuple(map(int, line[2:])) if len(line) > 2 else p1
            
            cv2.line(vis_img, p1, p2, (0, 0, 255), 2)
    
    # 绘制水平弧线（绿色）
    for arc in results.get('horizontal_arcs', []):
        if len(arc) >= 2:
            points = np.array(arc, dtype=np.int32)
            cv2.polylines(vis_img, [points], False, (0, 255, 0), 2)
    
    if output_path:
        cv2.imwrite(output_path, vis_img)
    
    return vis_img


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='混合方案地图经纬线检测')
    parser.add_argument('--image', type=str, required=True, help='输入图像路径')
    parser.add_argument('--model', type=str, default=None, help='训练好的模型路径')
    parser.add_argument('--output', type=str, default='output', help='输出目录')
    
    args = parser.parse_args()
    
    # 创建检测器
    detector = HybridMapLineDetector(model_path=args.model)
    
    # 检测图像
    detector.detect_image(args.image, output_dir=args.output)
