"""
交互式标注工具
功能：
  1. 加载地图图像和模型预测的mask（作为草稿）
  2. 用鼠标在地图上画经线/纬线
  3. 删除不需要的线条
  4. 保存为训练数据

操作：
  左键拖动：画经线（红色）
  右键拖动：画纬线（绿色）
  中键拖动：橡皮擦
  'v'：切换为经线模式
  'h'：切换为纬线模式
  'e'：切换为擦除模式
  '+'：增加画笔粗细
  '-'：减小画笔粗细
  'r'：重置（重新加载原始预测
  'c'：清除所有标注
  's'：保存
  'q'：退出
  'z'：撤销
"""
import numpy as np
import cv2
from pathlib import Path
import torch
import copy
import sys

from unet_model import UNet
from unet_post_processing import post_process


class InteractiveLabeler:
    def __init__(self, image_path, model_path=None, output_dir='map_line_dataset/raw_data'):
        self.image_path = Path(image_path)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # 读取图像
        img = cv2.imread(str(image_path))
        if img is None:
            raise FileNotFoundError(f"找不到图像: {image_path}")
        
        self.original_image = img.copy()
        self.image_for_display = img.copy()
        
        # mask: 0=背景, 1=经线, 2=纬线
        h, w = img.shape[:2]
        self.mask = np.zeros((h, w), dtype=np.uint8)
        
        # 撤销栈
        self.undo_stack = []
        
        # 当前模式: 'vertical'=经线, 'horizontal'=纬线, 'erase'=擦除
        self.mode = 'vertical'
        self.brush_size = 3
        self.drawing = False
        self.last_point = None
        
        # 如果有模型，先跑推理作为草稿
        if model_path and Path(model_path).exists():
            print(f"加载模型: {model_path}")
            self._run_inference(model_path)
        
        # 更新显示
        self._update_display()
    
    def _run_inference(self, model_path):
        """用模型生成草稿"""
        device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
        model = UNet(n_channels=3, n_classes=3).to(device)
        checkpoint = torch.load(str(model_path), map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        model.eval()
        
        img_rgb = cv2.cvtColor(self.original_image, cv2.COLOR_BGR2RGB)
        h, w = self.original_image.shape[:2]
        resized = cv2.resize(img_rgb, (512, 512), interpolation=cv2.INTER_LINEAR)
        normalized = resized.astype(np.float32) / 255.0
        x = torch.from_numpy(normalized).permute(2, 0, 1).unsqueeze(0).to(device)
        
        with torch.no_grad():
            output = model(x)
            output = torch.softmax(output, dim=1)
            pred = torch.argmax(output, dim=1)
        
        pred_mask = pred.squeeze().cpu().numpy().astype(np.uint8)
        pred_mask_full = cv2.resize(pred_mask, (w, h), interpolation=cv2.INTER_NEAREST)
        
        # 后处理
        self.mask = pred_mask_full.copy()
        print("模型推理完成，作为编辑的草稿")
    
    def _update_display(self):
        """更新显示"""
        self.image_for_display = self.original_image.copy()
        
        # 画mask到显示图
        # 经线：红色 (0, 0, 255)
        # 纬线：绿色 (0, 255, 0)
        self.image_for_display[self.mask == 1] = [0, 0, 255]
        self.image_for_display[self.mask == 2] = [0, 255, 0]
        
        # 显示当前模式
        mode_text = {
            'vertical': '经线 (V)',
            'horizontal': '纬线 (H)',
            'erase': '擦除 (E)'
        }
        text = f"{mode_text[self.mode]} | 画笔: {self.brush_size}"
        cv2.putText(self.image_for_display, text, (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    
    def mouse_callback(self, event, x, y, flags, param):
        """鼠标事件"""
        if event == cv2.EVENT_LBUTTONDOWN:
            self.drawing = True
            self.last_point = (x, y)
            self._save_undo_state()
        
        elif event == cv2.EVENT_MOUSEMOVE:
            if self.drawing:
                self._draw(x, y)
        
        elif event == cv2.EVENT_LBUTTONUP:
            self.drawing = False
            self.last_point = None
        
        self._update_display()
    
    def _draw(self, x, y):
        """在当前位置画"""
        if self.last_point is None:
            self.last_point = (x, y)
        
        color = 0 if self.mode == 'erase' else (1 if self.mode == 'vertical' else 2)
        thickness = self.brush_size * 2 + 1
        
        # 用圆填充
        if self.mode == 'erase':
            cv2.circle(self.mask, (x, y), thickness, 0, -1)
        else:
            cv2.line(self.mask, self.last_point, (x, y), color, thickness)
        
        self.last_point = (x, y)
    
    def _save_undo_state(self):
        """保存撤销状态"""
        self.undo_stack.append(self.mask.copy())
        if len(self.undo_stack) > 20:
            self.undo_stack.pop(0)
    
    def undo(self):
        """撤销"""
        if self.undo_stack:
            self.mask = self.undo_stack.pop()
    
    def save(self):
        """保存标注为训练数据"""
        name = self.image_path.stem
        
        # 保存mask
        mask_output = self.output_dir / f"{name}_mask.png"
        cv2.imwrite(str(mask_output), self.mask)
        
        # 保存原始图像（如果还没有）
        img_output = self.output_dir / f"{name}.jpg"
        if not img_output.exists():
            cv2.imwrite(str(img_output), self.original_image)
        
        # 生成 LabelMe 格式的简化版 JSON
        import json
        shapes = []
        h, w = self.original_image.shape[:2]
        
        # 分别提取经线和纬线的轮廓
        for label, label_name in [(1, 'vertical_line'), (2, 'horizontal_arc')]:
            binary_mask = (self.mask == label).astype(np.uint8) * 255
            contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            for contour in contours:
                if len(contour) < 10:
                    continue
                # 简化轮廓
                contour = cv2.approxPolyDP(contour, 1, True)
                
                points = contour.reshape(-1, 2).tolist()
                
                shapes.append({
                    'label': label_name,
                    'points': points,
                    'group_id': None,
                    'shape_type': 'polygon',
                    'flags': {}
                })
        
        # 保存LabelMe JSON
        json_data = {
            'version': '4.5.6',
            'flags': {},
            'shapes': shapes,
            'imagePath': f"{name}.jpg",
            'imageData': None,
            'imageHeight': h,
            'imageWidth': w
        }
        
        json_path = self.output_dir / f"{name}.json"
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(json_data, f, ensure_ascii=False, indent=2)
        
        print(f"\n✅ 已保存:")
        print(f"  图像: {img_output}")
        print(f"  Mask: {mask_output}")
        print(f"  JSON: {json_path}")
        print(f"  标注数量: {len(shapes)}")
        
        return str(json_path)
    
    def run(self):
        """主循环"""
        window_name = f"标注工具 - 按s保存, q退出"
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(window_name, min(1200, self.original_image.shape[1]), min(800, self.original_image.shape[0]))
        cv2.setMouseCallback(window_name, self.mouse_callback)
        
        print("\n=== 交互式标注启动 ===\n")
        print("操作说明:")
        print("  左键拖动: 画当前模式的线")
        print("  v: 切换到经线模式")
        print("  h: 切换到纬线模式")
        print("  e: 切换到擦除模式")
        print("  +/-: 调整画笔大小")
        print("  z: 撤销上一步")
        print("  r: 重置为模型预测结果")
        print("  c: 清除所有标注")
        print("  s: 保存")
        print("  q: 退出\n")
        
        while True:
            cv2.imshow(window_name, self.image_for_display)
            key = cv2.waitKey(1) & 0xFF
            
            if key == ord('q'):
                break
            elif key == ord('s'):
                self.save()
            elif key == ord('v'):
                self.mode = 'vertical'
                print("→ 经线模式")
            elif key == ord('h'):
                self.mode = 'horizontal'
                print("→ 纬线模式")
            elif key == ord('e'):
                self.mode = 'erase'
                print("→ 擦除模式")
            elif key == ord('+') or key == ord('='):
                self.brush_size = min(self.brush_size + 1, 20)
                print(f"→ 画笔大小: {self.brush_size}")
            elif key == ord('-'):
                self.brush_size = max(self.brush_size - 1, 1)
                print(f"→ 画笔大小: {self.brush_size}")
            elif key == ord('z'):
                self.undo()
                print("→ 撤销")
            elif key == ord('r'):
                print("→ 重置")
                self._save_undo_state()
                self._run_inference('models/unet_map_lines.pth') if Path('models/unet_map_lines.pth').exists() else None
            elif key == ord('c'):
                print("→ 清除所有标注")
                self._save_undo_state()
                self.mask = np.zeros(self.mask.shape, dtype=np.uint8)
            
            self._update_display()
        
        cv2.destroyAllWindows()
        print("\n标注结束")


def main():
    if len(sys.argv) < 2:
        print("用法: python interactive_labeling.py <地图路径> [模型路径]")
        print("示例: python interactive_labeling.py map_line_dataset/raw_data/08-52新疆.jpg models/unet_map_lines.pth")
        sys.exit(1)
    
    image_path = sys.argv[1]
    model_path = sys.argv[2] if len(sys.argv) > 2 else 'models/unet_map_lines.pth'
    
    labeler = InteractiveLabeler(image_path, model_path)
    labeler.run()


if __name__ == '__main__':
    main()
