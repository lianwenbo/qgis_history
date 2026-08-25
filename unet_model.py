"""
UNet 模型定义 + 训练 + 推理
专为经纬线检测设计的语义分割模型
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import numpy as np
import cv2
import json
from pathlib import Path
from tqdm import tqdm
import os
from functools import partial


# Monkey patch for PyTorch 2.6+ compatibility
_torch_load = torch.load
torch.load = partial(_torch_load, weights_only=False) if hasattr(torch, '__version__') else _torch_load


# ============================================================
# UNet 模型
# ============================================================

class DoubleConv(nn.Module):
    """(卷积 → BN → ReLU) × 2"""
    
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
    
    def forward(self, x):
        return self.double_conv(x)


class Down(nn.Module):
    """下采样：MaxPool → DoubleConv"""
    
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels)
        )
    
    def forward(self, x):
        return self.maxpool_conv(x)


class Up(nn.Module):
    """上采样：UpConv → 拼接 → DoubleConv"""
    
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
        self.conv = DoubleConv(in_channels, out_channels)
    
    def forward(self, x1, x2):
        x1 = self.up(x1)
        # 对齐尺寸
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]
        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                       diffY // 2, diffY - diffY // 2])
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class UNet(nn.Module):
    """UNet 语义分割模型
    
    Args:
        n_channels: 输入通道数 (3=RGB)
        n_classes: 输出类别数 (4=背景/经线/纬线/分隔线)
    """
    
    def __init__(self, n_channels=3, n_classes=4):
        super().__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        
        # 编码器 (下采样)
        self.inc = DoubleConv(n_channels, 64)
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)
        self.down4 = Down(512, 1024)
        
        # 解码器 (上采样 + 跳跃连接)
        self.up1 = Up(1024, 512)
        self.up2 = Up(512, 256)
        self.up3 = Up(256, 128)
        self.up4 = Up(128, 64)
        
        # 输出层
        self.outc = nn.Conv2d(64, n_classes, kernel_size=1)
    
    def forward(self, x):
        x1 = self.inc(x)      # 原始尺寸, 64通道
        x2 = self.down1(x1)   # 1/2尺寸, 128通道
        x3 = self.down2(x2)   # 1/4尺寸, 256通道
        x4 = self.down3(x3)   # 1/8尺寸, 512通道
        x5 = self.down4(x4)   # 1/16尺寸, 1024通道
        
        x = self.up1(x5, x4)  # 1/8尺寸, 512通道
        x = self.up2(x, x3)   # 1/4尺寸, 256通道
        x = self.up3(x, x2)   # 1/2尺寸, 128通道
        x = self.up4(x, x1)   # 原始尺寸, 64通道
        
        logits = self.outc(x) # 原始尺寸, n_classes通道
        return logits


# ============================================================
# 数据集
# ============================================================

class MapLineDataset(Dataset):
    """地图经纬线数据集（支持三种格式）
    
    mode='patch_labelme': 从 patch 的 labelme JSON 读取（实时生成mask）
    mode='patch_cache': 从 patch 图像 + 预生成的 mask 缓存读取（最快）
    mode='mask': 从 images/ + masks/ 读取（unet_format 格式）
    
    Args:
        data_dir: 数据目录（patch模式）或图像目录（mask模式）
        masks_dir: mask目录（仅mask模式）
        img_size: 训练时的图像尺寸
        transform: 是否启用数据增强
        mode: 'patch_labelme' / 'patch_cache' / 'mask'
        line_thickness: 线宽（仅 patch_labelme 模式使用）
        label_map: 标签映射字典
    """
    
    def __init__(self, data_dir, masks_dir=None, img_size=512, transform=True, 
                 mode='patch_cache', line_thickness=3, label_map=None):
        self.data_dir = Path(data_dir)
        self.img_size = img_size
        self.transform = transform
        self.mode = mode
        self.line_thickness = line_thickness
        
        if label_map is None:
            self.label_map = {
                'vertical_line': 1,
                'horizontal_arc': 2,
                'splitter': 3
            }
        else:
            self.label_map = label_map
        
        if mode == 'patch_cache':
            self.images = sorted(self.data_dir.glob("*.jpg"))
            self.masks_dir = Path(masks_dir) if masks_dir else self.data_dir
            print(f"加载 {len(self.images)} 张 patch 图像 (cache模式)")
        elif mode == 'patch_labelme':
            self.json_files = sorted(self.data_dir.glob("*.json"))
            print(f"加载 {len(self.json_files)} 个 patch labelme 标注")
        else:
            self.images_dir = self.data_dir
            self.masks_dir = Path(masks_dir)
            self.images = sorted(self.images_dir.glob("*.png"))
            print(f"加载 {len(self.images)} 张图像")
    
    def __len__(self):
        if self.mode == 'patch_labelme':
            return len(self.json_files)
        else:
            return len(self.images)
    
    def _shapes_to_mask(self, shapes, img_h, img_w):
        """从 labelme shapes 生成 mask"""
        mask = np.zeros((img_h, img_w), dtype=np.uint8)
        for shape in shapes:
            label = shape.get('label', '')
            points = shape.get('points', [])
            shape_type = shape.get('shape_type', 'line')
            
            if label not in self.label_map:
                continue
            if len(points) < 2:
                continue
            
            pixel_value = self.label_map[label]
            pts = np.array(points, dtype=np.int32)
            
            thickness = self.line_thickness
            if label == 'splitter':
                thickness = max(thickness, 9)
            
            if shape_type == 'line' and len(pts) == 2:
                cv2.line(mask, tuple(pts[0]), tuple(pts[1]), pixel_value, thickness=thickness)
            elif shape_type in ['polygon', 'linestrip'] and len(pts) > 2:
                for i in range(len(pts) - 1):
                    cv2.line(mask, tuple(pts[i]), tuple(pts[i+1]), pixel_value, thickness=thickness)
                if shape_type == 'polygon':
                    cv2.line(mask, tuple(pts[-1]), tuple(pts[0]), pixel_value, thickness=thickness)
            else:
                for i in range(len(pts) - 1):
                    cv2.line(mask, tuple(pts[i]), tuple(pts[i+1]), pixel_value, thickness=thickness)
        
        return mask
    
    def __getitem__(self, idx):
        if self.mode == 'patch_labelme':
            json_path = self.json_files[idx]
            with open(json_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            img_path = self.data_dir / data.get('imagePath', '')
            if not img_path.exists():
                img_path = json_path.with_suffix('.jpg')
            
            img = cv2.imread(str(img_path))
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            h, w = img.shape[:2]
            
            shapes = data.get('shapes', [])
            mask = self._shapes_to_mask(shapes, h, w)
        elif self.mode == 'patch_cache':
            img_path = self.images[idx]
            mask_path = self.masks_dir / (img_path.stem + '_mask.png')
            img = cv2.imread(str(img_path))
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
            h, w = img.shape[:2]
        else:
            img_path = self.images[idx]
            mask_path = self.masks_dir / img_path.name
            img = cv2.imread(str(img_path))
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
            h, w = img.shape[:2]
        
        if h != self.img_size or w != self.img_size:
            img = cv2.resize(img, (self.img_size, self.img_size), interpolation=cv2.INTER_LINEAR)
            mask = cv2.resize(mask, (self.img_size, self.img_size), interpolation=cv2.INTER_NEAREST)
        
        if self.transform and np.random.random() > 0.5:
            factor = np.random.uniform(0.8, 1.2)
            img = np.clip(img * factor, 0, 255).astype(np.uint8)
        
        img = img.astype(np.float32) / 255.0
        mask = mask.astype(np.int64)
        
        img_tensor = torch.from_numpy(img).permute(2, 0, 1)
        mask_tensor = torch.from_numpy(mask)
        
        return img_tensor, mask_tensor


# ============================================================
# 训练函数
# ============================================================

def train_unet(data_dir, model_path='models/unet_map_lines.pth', 
               epochs=30, img_size=512, batch_size=8, lr=1e-4, 
               mode='patch_cache', masks_dir=None, num_workers=2):
    """训练 UNet 模型
    
    Args:
        data_dir: 数据目录
        model_path: 模型保存路径
        epochs: 训练轮数
        img_size: 输入图像尺寸
        batch_size: 批大小
        lr: 学习率
        mode: 'patch_labelme' / 'patch_cache' / 'mask'
        masks_dir: mask缓存目录（mode='patch_cache'时使用）
        num_workers: DataLoader的worker数
    """
    device = torch.device('mps' if torch.backends.mps.is_available() else 
                          'cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")
    
    Path(model_path).parent.mkdir(parents=True, exist_ok=True)
    
    dataset = MapLineDataset(data_dir, masks_dir=masks_dir, img_size=img_size, mode=mode)
    
    use_pin_memory = (device.type == 'cuda')
    dataloader = DataLoader(
        dataset, 
        batch_size=batch_size, 
        shuffle=True, 
        num_workers=num_workers,
        pin_memory=use_pin_memory,
        persistent_workers=(num_workers > 0)
    )
    print(f"训练样本数: {len(dataset)}, batch_size: {batch_size}, workers: {num_workers}")
    
    model = UNet(n_channels=3, n_classes=4).to(device)
    
    class_weights = torch.tensor([1.0, 50.0, 50.0, 80.0]).to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    
    scaler = None
    use_amp = (device.type == 'cuda')
    if use_amp:
        scaler = torch.amp.GradScaler('cuda')
        print("启用混合精度训练 (AMP)")
    
    best_loss = float('inf')
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        num_batches = 0
        
        pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{epochs}")
        for imgs, masks in pbar:
            imgs = imgs.to(device)
            masks = masks.to(device)
            
            optimizer.zero_grad()
            
            if use_amp and scaler:
                with torch.amp.autocast('cuda'):
                    outputs = model(imgs)
                    loss = criterion(outputs, masks)
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                outputs = model(imgs)
                loss = criterion(outputs, masks)
                loss.backward()
                optimizer.step()
            
            total_loss += loss.item()
            num_batches += 1
            
            pbar.set_postfix(loss=f"{loss.item():.4f}")
        
        scheduler.step()
        avg_loss = total_loss / max(num_batches, 1)
        
        if (epoch + 1) % 5 == 0 or epoch == 0:
            print(f"  Epoch {epoch+1}/{epochs}, Loss={avg_loss:.4f}, LR={scheduler.get_last_lr()[0]:.2e}")
        
        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': best_loss,
            }, model_path)
    
    print(f"\n训练完成！最佳 Loss: {best_loss:.4f}")
    print(f"模型保存: {model_path}")
    return model


# ============================================================
# 推理函数
# ============================================================

def predict_unet(model_path, image_path, output_dir='output', img_size=512):
    """用训练好的 UNet 预测经纬线和分隔线
    
    Args:
        model_path: 模型权重路径
        image_path: 输入图像路径
        output_dir: 输出目录
        img_size: 推理时的图像尺寸 (用于模型输入)
    """
    device = torch.device('mps' if torch.backends.mps.is_available() else 
                          'cuda' if torch.cuda.is_available() else 'cpu')
    
    # 加载模型
    model = UNet(n_channels=3, n_classes=4).to(device)
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    # 读取图像
    img = cv2.imread(str(image_path))
    if img is None:
        print(f"无法读取图像: {image_path}")
        return None
    
    original_h, original_w = img.shape[:2]
    
    # 预处理
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_resized = cv2.resize(img_rgb, (img_size, img_size), interpolation=cv2.INTER_LINEAR)
    img_normalized = img_resized.astype(np.float32) / 255.0
    
    # 转换为 Tensor
    x = torch.from_numpy(img_normalized).permute(2, 0, 1).unsqueeze(0).to(device)
    
    # 推理
    with torch.no_grad():
        output = model(x)
        output = torch.softmax(output, dim=1)  # [1, 3, H, W]
        pred = torch.argmax(output, dim=1)    # [1, H, W]
    
    # 转换回 numpy
    pred_mask = pred.squeeze().cpu().numpy().astype(np.uint8)
    
    # 恢复原始尺寸
    pred_mask_full = cv2.resize(pred_mask, (original_w, original_h), 
                                 interpolation=cv2.INTER_NEAREST)
    
    # 创建可视化结果
    result = img.copy()
    
    # 经线 (红色)
    vertical_mask = (pred_mask_full == 1)
    result[vertical_mask] = [0, 0, 255]  # BGR: 红色
    
    # 纬线 (绿色)
    horizontal_mask = (pred_mask_full == 2)
    result[horizontal_mask] = [0, 255, 0]  # BGR: 绿色
    
    # 分隔线 (橙色)
    splitter_mask = (pred_mask_full == 3)
    result[splitter_mask] = [0, 165, 255]  # BGR: 橙色
    
    # 保存
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    output_name = Path(image_path).stem
    cv2.imwrite(str(Path(output_dir) / f"unet_{output_name}.jpg"), result)
    
    # 保存纯 mask (用于后处理)
    cv2.imwrite(str(Path(output_dir) / f"unet_{output_name}_mask.png"), pred_mask_full)
    
    # 打印统计
    n_vertical = int(vertical_mask.sum())
    n_horizontal = int(horizontal_mask.sum())
    n_splitter = int(splitter_mask.sum())
    print(f"经线像素: {n_vertical}, 纬线像素: {n_horizontal}, 分隔线像素: {n_splitter}")
    print(f"结果保存: {output_dir}/unet_{output_name}.jpg")
    
    return result


def predict_unet_sliding_window(model_path, image_path, output_dir='output',
                                 patch_size=512, overlap=0.5, fill_color=(245, 235, 210)):
    """滑窗推理：大图切成 patch 逐个预测，加权融合拼接成完整结果

    Args:
        model_path: 模型权重路径
        image_path: 输入图像路径
        output_dir: 输出目录
        patch_size: patch 大小
        overlap: 重叠率 (0~1)
        fill_color: 边缘填充颜色 (RGB)
    """
    device = torch.device('mps' if torch.backends.mps.is_available() else
                          'cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")

    # 加载模型
    model = UNet(n_channels=3, n_classes=4).to(device)
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    print(f"模型加载完成: {model_path}")

    # 读取图像
    img = cv2.imread(str(image_path))
    if img is None:
        print(f"无法读取图像: {image_path}")
        return None
    original_h, original_w = img.shape[:2]
    print(f"原图尺寸: {original_w}x{original_h}")

    # 转换为 RGB
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # 计算步长
    stride = int(patch_size * (1 - overlap))

    # 计算需要多少个 patch（含边缘填充）
    n_rows = int(np.ceil((original_h - patch_size) / stride)) + 1
    n_cols = int(np.ceil((original_w - patch_size) / stride)) + 1

    # 计算填充后尺寸
    pad_h = (n_rows - 1) * stride + patch_size
    pad_w = (n_cols - 1) * stride + patch_size
    print(f"填充后尺寸: {pad_w}x{pad_h}, 网格: {n_rows}行 x {n_cols}列")

    # 边缘填充（纯色）
    img_padded = np.full((pad_h, pad_w, 3), fill_color, dtype=np.uint8)
    img_padded[:original_h, :original_w] = img_rgb

    # 构建融合权重图
    weight_map = _build_weight_map(patch_size)

    # 初始化累加器
    prob_accum = np.zeros((pad_h, pad_w, 4), dtype=np.float32)
    weight_accum = np.zeros((pad_h, pad_w), dtype=np.float32)

    # 逐个 patch 推理
    total_patches = n_rows * n_cols
    print(f"开始滑窗推理，共 {total_patches} 个 patch...")

    with torch.no_grad():
        for i in range(n_rows):
            for j in range(n_cols):
                y = i * stride
                x = j * stride

                # 裁剪 patch
                patch = img_padded[y:y+patch_size, x:x+patch_size]
                patch_norm = patch.astype(np.float32) / 255.0

                # 转 Tensor
                x_tensor = torch.from_numpy(patch_norm).permute(2, 0, 1).unsqueeze(0).to(device)

                # 推理
                output = model(x_tensor)
                prob = torch.softmax(output, dim=1).squeeze().cpu().numpy()
                prob = np.transpose(prob, (1, 2, 0))  # [H, W, C]

                # 加权累加
                w = weight_map[:, :, np.newaxis]
                prob_accum[y:y+patch_size, x:x+patch_size] += prob * w
                weight_accum[y:y+patch_size, x:x+patch_size] += weight_map

    # 归一化（加权平均）
    weight_accum_safe = np.maximum(weight_accum, 1e-6)
    prob_avg = prob_accum / weight_accum_safe[:, :, np.newaxis]

    # 取 argmax 得到最终 mask
    pred_mask_padded = np.argmax(prob_avg, axis=2).astype(np.uint8)

    # 裁剪回原图尺寸
    pred_mask_full = pred_mask_padded[:original_h, :original_w]

    # 创建可视化结果
    result = img.copy()

    # 经线 (红色)
    vertical_mask = (pred_mask_full == 1)
    result[vertical_mask] = [0, 0, 255]  # BGR: 红色

    # 纬线 (绿色)
    horizontal_mask = (pred_mask_full == 2)
    result[horizontal_mask] = [0, 255, 0]  # BGR: 绿色

    # 分隔线 (橙色)
    splitter_mask = (pred_mask_full == 3)
    result[splitter_mask] = [0, 165, 255]  # BGR: 橙色

    # 保存
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    output_name = Path(image_path).stem
    cv2.imwrite(str(Path(output_dir) / f"unet_{output_name}.jpg"), result)
    cv2.imwrite(str(Path(output_dir) / f"unet_{output_name}_mask.png"), pred_mask_full)

    # 打印统计
    n_vertical = int(vertical_mask.sum())
    n_horizontal = int(horizontal_mask.sum())
    n_splitter = int(splitter_mask.sum())
    print(f"\n结果统计:")
    print(f"  经线像素: {n_vertical:,}")
    print(f"  纬线像素: {n_horizontal:,}")
    print(f"  分隔线像素: {n_splitter:,}")
    print(f"  结果保存: {output_dir}/unet_{output_name}.jpg")

    return result


def _build_weight_map(patch_size):
    """构建中心权重高、边缘权重低的融合权重图（线性衰减）"""
    w = np.ones((patch_size, patch_size), dtype=np.float32)
    border = patch_size // 4
    for i in range(border):
        weight = (i + 1) / (border + 1)
        w[i, :] = weight
        w[-(i+1), :] = weight
        w[:, i] = weight
        w[:, -(i+1)] = weight
    return w


def prepare_mask_cache(data_dir, output_dir, line_thickness=3, label_map=None):
    """预先生成所有 patch 的 mask 缓存，训练时直接读取，节省 CPU 时间
    
    Args:
        data_dir: patch 的 labelme 数据目录
        output_dir: mask 缓存输出目录
        line_thickness: 线宽
        label_map: 标签映射
    """
    if label_map is None:
        label_map = {
            'vertical_line': 1,
            'horizontal_arc': 2,
            'splitter': 3
        }
    
    data_dir = Path(data_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    json_files = sorted(data_dir.glob("*.json"))
    print(f"生成 mask 缓存: {len(json_files)} 个 patch")
    
    for json_path in tqdm(json_files, desc="生成mask"):
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        img_h = data.get('imageHeight', 512)
        img_w = data.get('imageWidth', 512)
        shapes = data.get('shapes', [])
        
        mask = np.zeros((img_h, img_w), dtype=np.uint8)
        for shape in shapes:
            label = shape.get('label', '')
            points = shape.get('points', [])
            shape_type = shape.get('shape_type', 'line')
            
            if label not in label_map:
                continue
            if len(points) < 2:
                continue
            
            pixel_value = label_map[label]
            pts = np.array(points, dtype=np.int32)
            
            thickness = line_thickness
            if label == 'splitter':
                thickness = max(thickness, 9)
            
            if shape_type == 'line' and len(pts) == 2:
                cv2.line(mask, tuple(pts[0]), tuple(pts[1]), pixel_value, thickness=thickness)
            elif shape_type in ['polygon', 'linestrip'] and len(pts) > 2:
                for i in range(len(pts) - 1):
                    cv2.line(mask, tuple(pts[i]), tuple(pts[i+1]), pixel_value, thickness=thickness)
                if shape_type == 'polygon':
                    cv2.line(mask, tuple(pts[-1]), tuple(pts[0]), pixel_value, thickness=thickness)
            else:
                for i in range(len(pts) - 1):
                    cv2.line(mask, tuple(pts[i]), tuple(pts[i+1]), pixel_value, thickness=thickness)
        
        mask_path = output_dir / (json_path.stem + '_mask.png')
        cv2.imwrite(str(mask_path), mask)
    
    print(f"mask 缓存已保存到: {output_dir}")


def predict_unet_sliding_window(model_path, image_path, output_dir='output',
                                 patch_size=512, overlap=0.5):
    """滑窗推理 + 加权融合，输出与原图同尺寸的分割结果
    
    Args:
        model_path: 模型权重路径
        image_path: 输入图像路径
        output_dir: 输出目录
        patch_size: 滑窗大小
        overlap: 重叠率
    """
    device = torch.device('mps' if torch.backends.mps.is_available() else 
                          'cuda' if torch.cuda.is_available() else 'cpu')
    
    model = UNet(n_channels=3, n_classes=4).to(device)
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    img = cv2.imread(str(image_path))
    if img is None:
        print(f"无法读取图像: {image_path}")
        return None
    
    original_h, original_w = img.shape[:2]
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    step = int(patch_size * (1 - overlap))
    
    cols = int(np.ceil((original_w - patch_size) / step)) + 1
    rows = int(np.ceil((original_h - patch_size) / step)) + 1
    
    bg_color = np.array([245, 235, 210], dtype=np.float32)
    padded_w = (cols - 1) * step + patch_size
    padded_h = (rows - 1) * step + patch_size
    
    padded_img = np.full((padded_h, padded_w, 3), bg_color, dtype=np.float32)
    padded_img[:original_h, :original_w] = img_rgb.astype(np.float32) / 255.0
    
    weight_map = _build_weight_map(patch_size)
    
    prob_sum = np.zeros((padded_h, padded_w, 4), dtype=np.float32)
    weight_sum = np.zeros((padded_h, padded_w), dtype=np.float32)
    
    total_patches = rows * cols
    print(f"滑窗推理: {rows}×{cols} = {total_patches} 个 patch")
    
    patch_count = 0
    with torch.no_grad():
        for r in range(rows):
            for c in range(cols):
                y = r * step
                x = c * step
                
                patch = padded_img[y:y+patch_size, x:x+patch_size]
                x_tensor = torch.from_numpy(patch).permute(2, 0, 1).unsqueeze(0).to(device)
                
                output = model(x_tensor)
                output = torch.softmax(output, dim=1)
                prob = output.squeeze(0).permute(1, 2, 0).cpu().numpy()
                
                prob_sum[y:y+patch_size, x:x+patch_size] += prob * weight_map[:, :, np.newaxis]
                weight_sum[y:y+patch_size, x:x+patch_size] += weight_map
                
                patch_count += 1
                if patch_count % 10 == 0 or patch_count == total_patches:
                    print(f"  进度: {patch_count}/{total_patches}")
    
    weight_sum = np.maximum(weight_sum, 1e-6)
    fused_prob = prob_sum / weight_sum[:, :, np.newaxis]
    
    fused_prob_crop = fused_prob[:original_h, :original_w]
    pred_mask = np.argmax(fused_prob_crop, axis=2).astype(np.uint8)
    
    result = img.copy()
    
    vertical_mask = (pred_mask == 1)
    result[vertical_mask] = [0, 0, 255]
    
    horizontal_mask = (pred_mask == 2)
    result[horizontal_mask] = [0, 255, 0]
    
    splitter_mask = (pred_mask == 3)
    result[splitter_mask] = [0, 165, 255]
    
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    output_name = Path(image_path).stem
    cv2.imwrite(str(Path(output_dir) / f"unet_{output_name}.jpg"), result)
    cv2.imwrite(str(Path(output_dir) / f"unet_{output_name}_mask.png"), pred_mask)
    
    n_vertical = int(vertical_mask.sum())
    n_horizontal = int(horizontal_mask.sum())
    n_splitter = int(splitter_mask.sum())
    print(f"经线像素: {n_vertical}, 纬线像素: {n_horizontal}, 分隔线像素: {n_splitter}")
    print(f"结果保存: {output_dir}/unet_{output_name}.jpg")
    
    return result


if __name__ == '__main__':
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == 'predict':
        # 预测模式
        if len(sys.argv) > 2:
            image_path = sys.argv[2]
        else:
            image_path = 'map_line_dataset/raw_data/08-52新疆.jpg'
        predict_unet('models/unet_map_lines.pth', image_path)
    else:
        # 训练模式
        print("准备 UNet 数据...")
        from unet_data_prep import UNetDataPreparer
        preparer = UNetDataPreparer(
            raw_data_dir='map_line_dataset/raw_data',
            output_dir='map_line_dataset/unet_format'
        )
        preparer.prepare()
        
        print("\n开始训练 UNet...")
        train_unet(
            data_dir='map_line_dataset/unet_format',
            model_path='models/unet_map_lines.pth',
            epochs=200,
            img_size=512,
            batch_size=2,
            lr=1e-4
        )
