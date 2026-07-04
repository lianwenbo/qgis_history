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
    """地图经纬线数据集
    
    Args:
        images_dir: 图像目录
        masks_dir: mask目录
        img_size: 训练时的图像尺寸 (int, 例如512)
        transform: 是否启用数据增强
    """
    
    def __init__(self, images_dir, masks_dir, img_size=512, transform=True):
        self.images_dir = Path(images_dir)
        self.masks_dir = Path(masks_dir)
        self.img_size = img_size
        self.transform = transform
        
        self.images = sorted(self.images_dir.glob("*.png"))
        print(f"加载 {len(self.images)} 张图像")
    
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        img_path = self.images[idx]
        mask_path = self.masks_dir / img_path.name
        
        # 读取图像和mask
        img = cv2.imread(str(img_path))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
        
        h, w = img.shape[:2]
        
        # 调整大小
        img = cv2.resize(img, (self.img_size, self.img_size), interpolation=cv2.INTER_LINEAR)
        mask = cv2.resize(mask, (self.img_size, self.img_size), interpolation=cv2.INTER_NEAREST)
        
        # 数据增强 (简单的颜色增强)
        if self.transform and np.random.random() > 0.5:
            # 随机亮度调整
            factor = np.random.uniform(0.8, 1.2)
            img = np.clip(img * factor, 0, 255).astype(np.uint8)
        
        # 归一化
        img = img.astype(np.float32) / 255.0
        mask = mask.astype(np.int64)
        
        # 转换为 Tensor (HWC → CHW)
        img_tensor = torch.from_numpy(img).permute(2, 0, 1)
        mask_tensor = torch.from_numpy(mask)
        
        return img_tensor, mask_tensor


# ============================================================
# 训练函数
# ============================================================

def train_unet(data_dir, model_path='models/unet_map_lines.pth', 
               epochs=200, img_size=512, batch_size=2, lr=1e-4):
    """训练 UNet 模型
    
    Args:
        data_dir: 数据目录 (包含 images/ 和 masks/)
        model_path: 模型保存路径
        epochs: 训练轮数
        img_size: 输入图像尺寸
        batch_size: 批大小
        lr: 学习率
    """
    # 设备
    device = torch.device('mps' if torch.backends.mps.is_available() else 
                          'cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")
    
    # 创建目录
    Path(model_path).parent.mkdir(parents=True, exist_ok=True)
    
    # 数据集
    images_dir = Path(data_dir) / "images"
    masks_dir = Path(data_dir) / "masks"
    
    dataset = MapLineDataset(images_dir, masks_dir, img_size=img_size)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    
    # 模型
    model = UNet(n_channels=3, n_classes=4).to(device)
    
    # 损失函数 + 优化器
    # 使用加权交叉熵：经线/纬线/分隔线的像素远少于背景
    class_weights = torch.tensor([1.0, 50.0, 50.0, 80.0]).to(device)  # 背景/经线/纬线/分隔线
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    
    # 训练
    best_loss = float('inf')
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        num_batches = 0
        
        pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{epochs}")
        for imgs, masks in pbar:
            imgs = imgs.to(device)
            masks = masks.to(device)
            
            # 前向传播
            outputs = model(imgs)
            
            # 计算损失
            loss = criterion(outputs, masks)
            
            # 反向传播
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            num_batches += 1
            
            pbar.set_postfix(loss=f"{loss.item():.4f}")
        
        avg_loss = total_loss / max(num_batches, 1)
        
        # 打印
        if (epoch + 1) % 5 == 0 or epoch == 0:
            print(f"  Epoch {epoch+1}/{epochs}, Loss={avg_loss:.4f}")
        
        # 保存最佳模型
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
