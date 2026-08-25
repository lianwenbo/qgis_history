"""
增量训练平台 - 一体化工具
功能：
  1. 收集所有已标注的数据 (LabelMe JSON + mask图像)
  2. 转换为UNet训练格式
  3. 增量训练模型
  4. 保存最佳模型

流程：
  python incremental_pipeline.py train     # 训练
  python incremental_pipeline.py predict <image_path>  # 推理
"""
import json
import numpy as np
import cv2
import torch
import torch.nn as nn
from pathlib import Path
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import sys

from unet_model import UNet
from unet_post_processing import post_process


class MaskToDataset:
    """从已保存的mask图像转换为UNet训练数据"""
    
    def __init__(self, raw_data_dir, output_dir):
        self.raw_data_dir = Path(raw_data_dir)
        self.output_dir = Path(output_dir)
        self.images_dir = self.output_dir / "images"
        self.masks_dir = self.output_dir / "masks"
        self.images_dir.mkdir(parents=True, exist_ok=True)
        self.masks_dir.mkdir(parents=True, exist_ok=True)
    
    def convert_from_mask(self, image_path, mask_path):
        """从mask文件转换为训练数据"""
        img = cv2.imread(str(image_path))
        mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
        
        if img is None or mask is None:
            print(f"  跳过 {image_path.name}: 无法读取图像或mask")
            return False
        
        # 确保尺寸一致
        h, w = img.shape[:2]
        if mask.shape[:2] != (h, w):
            mask = cv2.resize(mask, (w, h), interpolation=cv2.INTER_NEAREST)
        
        # 保存
        name = Path(image_path).stem
        cv2.imwrite(str(self.images_dir / f"{name}.png"), img)
        cv2.imwrite(str(self.masks_dir / f"{name}.png"), mask)
        return True
    
    def convert_from_json(self, json_path):
        """从LabelMe JSON文件转换为训练数据"""
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        image_path = Path(json_path).parent / data.get('imagePath', '')
        
        if not image_path.exists():
            # 尝试同名不同扩展名
            for ext in ['.jpg', '.jpeg', '.png']:
                test = json_path.with_suffix(ext)
                if test.exists():
                    image_path = test
                    break
        
        if not image_path.exists():
            print(f"  跳过 {json_path.name}: 找不到图像")
            return False
        
        img = cv2.imread(str(image_path))
        if img is None:
            print(f"  跳过 {json_path.name}: 无法读取图像")
            return False
        
        h, w = img.shape[:2]
        
        # 创建语义分割mask
        mask = np.zeros((h, w), dtype=np.uint8)
        
        for shape in data.get('shapes', []):
            label = shape.get('label', '')
            points = shape.get('points', [])
            
            if len(points) < 2:
                continue
            
            if label not in ['vertical_line', 'horizontal_arc']:
                continue
            
            label_value = 1 if label == 'vertical_line' else 2
            pts = np.array(points, dtype=np.int32)
            
            if len(pts) == 2:
                cv2.line(mask, tuple(pts[0]), tuple(pts[1]), label_value, thickness=3)
            else:
                for i in range(len(pts) - 1):
                    cv2.line(mask, tuple(pts[i]), tuple(pts[i+1]), label_value, thickness=3)
                if shape.get('shape_type') == 'polygon':
                    cv2.line(mask, tuple(pts[-1]), tuple(pts[0]), label_value, thickness=3)
        
        # 保存
        name = Path(image_path).stem
        cv2.imwrite(str(self.images_dir / f"{name}.png"), img)
        cv2.imwrite(str(self.masks_dir / f"{name}.png"), mask)
        return True
    
    def process_all(self):
        """处理所有标注数据"""
        print(f"扫描目录: {self.raw_data_dir}")
        
        count = 0
        
        # 优先处理 JSON 文件 (LabelMe格式)
        json_files = sorted(self.raw_data_dir.glob("*.json"))
        print(f"找到 {len(json_files)} 个JSON标注文件")
        
        for json_file in json_files:
            if self.convert_from_json(json_file):
                count += 1
        
        # 处理 _mask.png 文件 (交互式标注输出)
        mask_files = sorted(self.raw_data_dir.glob("*_mask.png"))
        # 跳过已经通过JSON处理的
        for mask_file in mask_files:
            base_name = mask_file.stem.replace('_mask', '')
            
            # 检查是否有对应的JSON文件 (有JSON的话优先用JSON)
            if (self.raw_data_dir / f"{base_name}.json").exists():
                continue
            
            # 找对应的图像
            image_path = None
            for ext in ['.jpg', '.jpeg', '.png']:
                test = self.raw_data_dir / f"{base_name}{ext}"
                if test.exists() and test != mask_file:
                    image_path = test
                    break
            
            if image_path and self.convert_from_mask(image_path, mask_file):
                count += 1
        
        print(f"\n✅ 成功转换 {count} 张地图")
        print(f"   训练数据: {self.output_dir}")
        return count


class UNetTrainer:
    """UNet训练器 - 支持增量训练"""
    
    def __init__(self, data_dir, model_path='models/unet_map_lines.pth',
                 epochs=100, img_size=512, batch_size=2, lr=1e-4):
        self.data_dir = Path(data_dir)
        self.model_path = Path(model_path)
        self.model_path.parent.mkdir(parents=True, exist_ok=True)
        self.epochs = epochs
        self.img_size = img_size
        self.batch_size = batch_size
        self.lr = lr
    
    def train(self, resume=True):
        """训练模型
        
        Args:
            resume: 是否从已有模型继续训练 (增量训练)
        """
        # 设备
        device = torch.device('mps' if torch.backends.mps.is_available() else 
                              'cuda' if torch.cuda.is_available() else 'cpu')
        print(f"使用设备: {device}")
        
        # 数据集
        images_dir = self.data_dir / "images"
        masks_dir = self.data_dir / "masks"
        
        image_files = sorted(images_dir.glob("*.png")) + sorted(images_dir.glob("*.jpg"))
        if not image_files:
            print(f"没有找到训练图像: {images_dir}")
            return False
        
        print(f"数据集大小: {len(image_files)} 张地图")
        
        # 创建自定义数据集 (直接从文件列表加载)
        class MapDataset(Dataset):
            def __init__(self, image_files, img_size, masks_dir):
                self.image_files = image_files
                self.img_size = img_size
                self.masks_dir = masks_dir
            
            def __len__(self):
                return len(self.image_files)
            
            def __getitem__(self, idx):
                img_path = self.image_files[idx]
                mask_path = self.masks_dir / f"{img_path.stem}.png"
                
                img = cv2.imread(str(img_path))
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                img = cv2.resize(img, (self.img_size, self.img_size), interpolation=cv2.INTER_LINEAR)
                img = img.astype(np.float32) / 255.0
                
                mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
                mask = cv2.resize(mask, (self.img_size, self.img_size), interpolation=cv2.INTER_NEAREST)
                mask = mask.astype(np.int64)
                
                img_tensor = torch.from_numpy(img).permute(2, 0, 1)
                mask_tensor = torch.from_numpy(mask)
                
                return img_tensor, mask_tensor
        
        dataset = MapDataset(image_files, self.img_size, masks_dir)
        dataloader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True, num_workers=0)
        
        # 模型
        model = UNet(n_channels=3, n_classes=3).to(device)
        
        # 如果有已有模型且resume=True，加载它
        start_epoch = 0
        if resume and self.model_path.exists():
            print(f"加载已有模型继续训练: {self.model_path}")
            checkpoint = torch.load(str(self.model_path), map_location=device)
            model.load_state_dict(checkpoint['model_state_dict'])
            if 'epoch' in checkpoint:
                start_epoch = checkpoint.get('epoch', 0)
            print(f"  从epoch {start_epoch} 继续")
        else:
            print("从零开始训练")
        
        # 损失 + 优化器
        class_weights = torch.tensor([1.0, 50.0, 50.0]).to(device)
        criterion = nn.CrossEntropyLoss(weight=class_weights)
        optimizer = torch.optim.Adam(model.parameters(), lr=self.lr)
        
        # 恢复优化器状态 (仅在resume且有optimizer state)
        if resume and self.model_path.exists():
            checkpoint = torch.load(str(self.model_path), map_location=device)
            if 'optimizer_state_dict' in checkpoint:
                optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        # 训练
        best_loss = float('inf')
        for epoch in range(start_epoch, start_epoch + self.epochs):
            model.train()
            total_loss = 0
            num_batches = 0
            
            pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{start_epoch + self.epochs}")
            for imgs, masks in pbar:
                imgs = imgs.to(device)
                masks = masks.to(device)
                
                outputs = model(imgs)
                loss = criterion(outputs, masks)
                
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                total_loss += loss.item()
                num_batches += 1
                pbar.set_postfix(loss=f"{loss.item():.4f}")
            
            avg_loss = total_loss / max(num_batches, 1)
            
            if (epoch + 1) % 10 == 0 or epoch == start_epoch:
                print(f"  Epoch {epoch+1}/{start_epoch + self.epochs}, Loss={avg_loss:.4f}")
            
            if avg_loss < best_loss:
                best_loss = avg_loss
                torch.save({
                    'epoch': epoch + 1,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'loss': best_loss,
                }, str(self.model_path))
        
        print(f"\n✅ 训练完成!")
        print(f"   最佳Loss: {best_loss:.4f}")
        print(f"   模型保存: {self.model_path}")
        return True


def train_command():
    """训练命令"""
    print("\n=== 增量训练平台 ===\n")
    
    # 步骤1: 数据转换
    print("步骤1: 转换标注数据...")
    converter = MaskToDataset(
        raw_data_dir='map_line_dataset/raw_data',
        output_dir='map_line_dataset/unet_format'
    )
    converter.process_all()
    
    # 步骤2: 训练
    print("\n步骤2: 训练模型...")
    trainer = UNetTrainer(
        data_dir='map_line_dataset/unet_format',
        model_path='models/unet_map_lines.pth',
        epochs=100,
        img_size=512,
        batch_size=2,
        lr=1e-4
    )
    trainer.train(resume=True)


def predict_command(image_path):
    """推理命令 - 对单张图像预测"""
    print(f"\n=== 对图像推理: {image_path} ===\n")
    
    device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
    
    # 加载模型
    model_path = 'models/unet_map_lines.pth'
    if not Path(model_path).exists():
        print(f"❌ 找不到模型: {model_path}")
        print("请先运行: python incremental_pipeline.py train")
        return
    
    print(f"加载模型: {model_path}")
    model = UNet(n_channels=3, n_classes=3).to(device)
    checkpoint = torch.load(str(model_path), map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    # 读取图像
    img = cv2.imread(str(image_path))
    if img is None:
        print(f"❌ 无法读取图像: {image_path}")
        return
    
    h, w = img.shape[:2]
    print(f"图像大小: {w}x{h}")
    
    # 预处理
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    resized = cv2.resize(img_rgb, (512, 512), interpolation=cv2.INTER_LINEAR)
    normalized = resized.astype(np.float32) / 255.0
    x = torch.from_numpy(normalized).permute(2, 0, 1).unsqueeze(0).to(device)
    
    # 推理
    print("推理中...")
    with torch.no_grad():
        output = model(x)
        output = torch.softmax(output, dim=1)
        pred = torch.argmax(output, dim=1)
    
    pred_mask = pred.squeeze().cpu().numpy().astype(np.uint8)
    pred_mask_full = cv2.resize(pred_mask, (w, h), interpolation=cv2.INTER_NEAREST)
    
    n_v = int((pred_mask_full == 1).sum())
    n_h = int((pred_mask_full == 2).sum())
    print(f"检测到 - 经线像素: {n_v}, 纬线像素: {n_h}")
    
    # 后处理
    print("后处理中...")
    processed_mask = post_process(pred_mask_full, (h, w))
    
    n_v2 = int((processed_mask == 1).sum())
    n_h2 = int((processed_mask == 2).sum())
    print(f"后处理后 - 经线像素: {n_v2}, 纬线像素: {n_h2}")
    
    # 保存结果
    Path('output').mkdir(exist_ok=True)
    name = Path(image_path).stem
    
    # 结果1: 模型直接预测
    result1 = img.copy()
    result1[pred_mask_full == 1] = [0, 0, 255]
    result1[pred_mask_full == 2] = [0, 255, 0]
    cv2.imwrite(f'output/{name}_raw.jpg', result1)
    
    # 结果2: 后处理后的
    result2 = img.copy()
    result2[processed_mask == 1] = [0, 0, 255]
    result2[processed_mask == 2] = [0, 255, 0]
    cv2.imwrite(f'output/{name}_processed.jpg', result2)
    
    # 保存mask (用于交互式编辑)
    cv2.imwrite(f'output/{name}_mask.png', processed_mask)
    
    print(f"\n✅ 推理完成!")
    print(f"   原始结果: output/{name}_raw.jpg")
    print(f"   后处理结果: output/{name}_processed.jpg")
    print(f"   编辑用mask: output/{name}_mask.png")
    print(f"\n提示: 如果结果不理想，可以运行:")
    print(f"  python interactive_labeling.py {image_path}")
    print(f"来手动修正标注，然后重新训练。")


def main():
    if len(sys.argv) < 2:
        print("用法:")
        print("  python incremental_pipeline.py train           # 训练/增量训练模型")
        print("  python incremental_pipeline.py predict <图>     # 对一张图推理")
        print("\n示例:")
        print("  python incremental_pipeline.py train")
        print("  python incremental_pipeline.py predict map_line_dataset/raw_data/08-52新疆.jpg")
        sys.exit(1)
    
    command = sys.argv[1]
    
    if command == 'train':
        train_command()
    
    elif command == 'predict':
        if len(sys.argv) < 3:
            print("请指定图像路径")
            print("  python incremental_pipeline.py predict <图>")
            sys.exit(1)
        predict_command(sys.argv[2])
    
    else:
        print(f"未知命令: {command}")
        print("可用命令: train, predict")


if __name__ ==