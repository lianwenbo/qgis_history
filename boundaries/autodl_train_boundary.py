"""
AutoDL 边界检测 UNet 训练脚本（二级边界）
上传到 AutoDL 后直接运行: bash run_on_autodl.sh

分类：
    0 - 背景
    1 - 一级边界（省界/外边界，红色粗线）
    2 - 二级边界（府界/内部边界，绿色细线）

数据结构（上传后解压到 ~/boundary_train/）:
    boundary_train/
    ├── autodl_train_boundary.py   (本文件)
    ├── run_on_autodl.sh           (一键安装+训练脚本)
    └── labelme/                   (4张清朝省图 + LabelMe JSON)
        ├── 08-7直隶.jpg
        ├── 08-7直隶.json
        ├── 08-24河南.jpg
        ├── 08-24河南.json
        ├── 08-31浙江.jpg
        ├── 08-31浙江.json
        ├── 08-46广西.jpg
        └── 08-46广西.json
"""

import time
import argparse
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import numpy as np
import cv2
import json
from tqdm import tqdm

from PIL import Image


def imread_unicode(path):
    """兼容中文路径的图片读取（Pillow 优先，读取失败立即抛异常，不静默跳过）"""
    p = str(path)
    pil_img = Image.open(p).convert('RGB')
    arr = np.array(pil_img)
    return cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)


# ============================================================
# UNet 模型（自包含，不依赖外部模块）
# ============================================================

class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)


class Down(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(nn.MaxPool2d(2), DoubleConv(in_channels, out_channels))

    def forward(self, x):
        return self.maxpool_conv(x)


class Up(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
        self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]
        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2, diffY // 2, diffY - diffY // 2])
        return self.conv(torch.cat([x2, x1], dim=1))


class UNet(nn.Module):
    def __init__(self, n_channels=3, n_classes=2):
        super().__init__()
        self.inc = DoubleConv(n_channels, 64)
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)
        self.down4 = Down(512, 1024)
        self.up1 = Up(1024, 512)
        self.up2 = Up(512, 256)
        self.up3 = Up(256, 128)
        self.up4 = Up(128, 64)
        self.outc = nn.Conv2d(64, n_classes, kernel_size=1)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        return self.outc(x)


# ============================================================
# 数据准备
# ============================================================

LABEL_MAP = {
    "boundary_1": 1,  # 一级边界（粗）
    "boundary_2": 2,  # 二级边界（细）
    "boundary": 1,    # 兼容旧标注，视为一级
}


def labelme_to_mask(json_path, line_thickness_1=8, line_thickness_2=3):
    """LabelMe JSON → 多类 mask (0=背景, 1=一级边界, 2=二级边界)"""
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    img_h = data['imageHeight']
    img_w = data['imageWidth']
    mask = np.zeros((img_h, img_w), dtype=np.uint8)
    for shape in data.get('shapes', []):
        label = shape.get('label', 'boundary')
        class_id = LABEL_MAP.get(label, 1)
        thickness = line_thickness_1 if class_id == 1 else line_thickness_2
        pts = np.array(shape.get('points', []), dtype=np.int32)
        if len(pts) < 2:
            continue
        for i in range(len(pts) - 1):
            cv2.line(mask, tuple(pts[i]), tuple(pts[i + 1]), class_id, thickness=thickness)
    return mask


class BoundaryPatchDataset(Dataset):
    """确定性滑窗数据集（参照经纬线 unet_data_prep.split_into_patches 的方式）

    - 按固定 stride 在全图上切分 patch_size×patch_size 的窗口（overlap 控制重叠）
    - 保留所有含边界像素的 patch，按 bg_ratio 采样纯背景 patch 降低假阳性
    - 每个 patch 每次取用时做随机增强（翻转/90°旋转/亮度抖动）
    """

    def __init__(self, labelme_dir, patch_size=512, overlap=0.5,
                 line_thickness_1=5, line_thickness_2=2,
                 bg_ratio=0.2, min_boundary_px=20, augment=True):
        self.patch_size = patch_size
        self.augment = augment

        labelme_dir = Path(labelme_dir)
        self.samples = []
        for json_path in sorted(labelme_dir.glob("*.json")):
            if json_path.name.startswith("._"):
                continue
            img_path = labelme_dir / f"{json_path.stem}.jpg"
            if not img_path.exists() or img_path.name.startswith("._"):
                continue
            try:
                with open(str(json_path), 'r', encoding='utf-8') as _f:
                    json.load(_f)
            except Exception:
                continue
            self.samples.append((str(img_path), str(json_path)))

        # 预加载图像和 mask，并 pad 到滑窗整网格
        self.images = []
        self.masks = []
        for img_path, json_path in self.samples:
            img = imread_unicode(img_path)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            mask = labelme_to_mask(json_path, line_thickness_1, line_thickness_2)
            print(f"  加载: {Path(img_path).stem} ({img.shape[1]}x{img.shape[0]}), boundary={mask.sum()} px")
            img, mask = self._pad_to_grid(img, mask, patch_size, overlap)
            self.images.append(img)
            self.masks.append(mask)

        # 生成确定性滑窗位置
        stride = int(patch_size * (1 - overlap))
        self.positions = []
        bg_positions = []

        for idx, (img, mask) in enumerate(zip(self.images, self.masks)):
            h, w = img.shape[:2]
            rows = int(np.ceil((h - patch_size) / stride)) + 1
            cols = int(np.ceil((w - patch_size) / stride)) + 1
            n_boundary = 0
            for r in range(rows):
                for c in range(cols):
                    y = min(r * stride, h - patch_size)
                    x = min(c * stride, w - patch_size)
                    patch_mask = mask[y:y+patch_size, x:x+patch_size]
                    boundary_px = int((patch_mask > 0).sum())
                    if boundary_px >= min_boundary_px:
                        self.positions.append((idx, y, x))
                        n_boundary += 1
                    elif boundary_px == 0:
                        bg_positions.append((idx, y, x))
            stem = Path(self.samples[idx][0]).stem
            print(f"    {stem}: 网格 {rows}x{cols}={rows*cols}, 含边界patch={n_boundary}, 纯背景候选={len(bg_positions)}")

        # 采样背景 patch
        n_bg = int(len(self.positions) * bg_ratio)
        if n_bg > 0 and bg_positions:
            rng = np.random.default_rng(42)
            chosen = rng.choice(len(bg_positions), size=min(n_bg, len(bg_positions)), replace=False)
            self.positions.extend(bg_positions[i] for i in chosen)

        print(f"  总 patch 数: {len(self.positions)} "
              f"(含边界 {len(self.positions)-min(n_bg, len(bg_positions))} + 背景 {min(n_bg, len(bg_positions))})")

    @staticmethod
    def _pad_to_grid(img, mask, patch_size, overlap):
        stride = int(patch_size * (1 - overlap))
        h, w = img.shape[:2]
        rows = int(np.ceil((h - patch_size) / stride)) + 1
        cols = int(np.ceil((w - patch_size) / stride)) + 1
        pad_h = (rows - 1) * stride + patch_size
        pad_w = (cols - 1) * stride + patch_size
        if pad_h == h and pad_w == w:
            return img, mask
        padded_img = np.zeros((pad_h, pad_w, 3), dtype=np.uint8)
        padded_img[:] = np.array([245, 235, 210], dtype=np.uint8)
        padded_img[:h, :w] = img
        padded_mask = np.zeros((pad_h, pad_w), dtype=np.uint8)
        padded_mask[:h, :w] = mask
        return padded_img, padded_mask

    def __len__(self):
        return len(self.positions)

    def __getitem__(self, idx):
        img_idx, y, x = self.positions[idx]
        img = self.images[img_idx]
        mask = self.masks[img_idx]
        ps = self.patch_size

        patch_img = img[y:y+ps, x:x+ps].copy()
        patch_mask = mask[y:y+ps, x:x+ps].copy()

        if self.augment:
            if np.random.random() > 0.5:
                patch_img = np.flip(patch_img, axis=1).copy()
                patch_mask = np.flip(patch_mask, axis=1).copy()
            if np.random.random() > 0.5:
                patch_img = np.flip(patch_img, axis=0).copy()
                patch_mask = np.flip(patch_mask, axis=0).copy()
            k = np.random.randint(0, 4)
            if k > 0:
                patch_img = np.rot90(patch_img, k).copy()
                patch_mask = np.rot90(patch_mask, k).copy()
            if np.random.random() > 0.5:
                factor = np.random.uniform(0.8, 1.2)
                patch_img = np.clip(patch_img.astype(np.float32) * factor, 0, 255).astype(np.uint8)

        patch_img = patch_img.astype(np.float32) / 255.0
        patch_mask = patch_mask.astype(np.int64)

        return torch.from_numpy(patch_img).permute(2, 0, 1), torch.from_numpy(patch_mask)


# ============================================================
# 训练
# ============================================================

def main():
    parser = argparse.ArgumentParser(description='AutoDL 边界 UNet 训练')
    parser.add_argument('--labelme_dir', type=str, default='labelme')
    parser.add_argument('--model_path', type=str, default='models/unet_boundary.pth')
    parser.add_argument('--epochs', type=int, default=30)
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--lr', type=float, default=2e-4)
    parser.add_argument('--patch_size', type=int, default=512)
    parser.add_argument('--overlap', type=float, default=0.5)
    parser.add_argument('--bg_ratio', type=float, default=0.2)
    parser.add_argument('--line_thickness_1', type=int, default=5)
    parser.add_argument('--line_thickness_2', type=int, default=2)
    parser.add_argument('--num_workers', type=int, default=4)
    args = parser.parse_args()

    # 环境检查
    print("=" * 60)
    print("AutoDL 边界检测 UNet 训练")
    print("=" * 60)
    print(f"PyTorch: {torch.__version__}")

    if torch.cuda.is_available():
        device = torch.device('cuda')
        gpu_name = torch.cuda.get_device_name(0)
        gpu_mem = torch.cuda.get_device_properties(0).total_memory / 1024**3
        print(f"GPU: {gpu_name} ({gpu_mem:.1f} GB)")
    else:
        device = torch.device('cpu')
        print("WARNING: 未检测到 GPU")
    print(f"设备: {device}")
    print("=" * 60)

    # 数据
    print(f"\n加载数据: {args.labelme_dir}")
    dataset = BoundaryPatchDataset(
        args.labelme_dir,
        patch_size=args.patch_size,
        overlap=args.overlap,
        bg_ratio=args.bg_ratio,
        line_thickness_1=args.line_thickness_1,
        line_thickness_2=args.line_thickness_2,
    )

    dataloader = DataLoader(
        dataset, batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers, pin_memory=True, persistent_workers=True
    )
    print(f"样本数: {len(dataset)}, batch_size: {args.batch_size}, batches/epoch: {len(dataloader)}")

    # 模型
    model = UNet(n_channels=3, n_classes=3).to(device)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"模型参数: {total_params / 1e6:.1f} M")

    Path(args.model_path).parent.mkdir(parents=True, exist_ok=True)

    class_weights = torch.tensor([1.0, 30.0, 30.0]).to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    use_amp = (device.type == 'cuda')
    scaler = torch.amp.GradScaler('cuda') if use_amp else None
    if use_amp:
        print("启用混合精度 (AMP)")

    # 训练循环
    print(f"\n开始训练: {args.epochs} epochs")
    print("-" * 60)

    best_loss = float('inf')
    start_time = time.time()

    for epoch in range(args.epochs):
        model.train()
        total_loss = 0
        num_batches = 0
        epoch_start = time.time()

        pbar = tqdm(dataloader, desc=f"Epoch {epoch + 1}/{args.epochs}", ncols=100)
        for imgs, masks in pbar:
            imgs = imgs.to(device, non_blocking=True)
            masks = masks.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)

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
        epoch_time = time.time() - epoch_start

        if (epoch + 1) % 5 == 0 or epoch == 0:
            elapsed = time.time() - start_time
            print(f"  Epoch {epoch + 1}/{args.epochs} | Loss={avg_loss:.4f} | "
                  f"LR={scheduler.get_last_lr()[0]:.2e} | "
                  f"Time={epoch_time:.1f}s | Total={elapsed:.0f}s")

        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': best_loss,
                'n_classes': 3,
                'patch_size': args.patch_size,
                'line_thickness_1': args.line_thickness_1,
                'line_thickness_2': args.line_thickness_2,
            }, args.model_path)

    total_time = time.time() - start_time
    print(f"\n{'=' * 60}")
    print(f"训练完成! 耗时: {total_time:.0f}s ({total_time/60:.1f}min)")
    print(f"最佳 Loss: {best_loss:.4f}")
    print(f"模型保存: {args.model_path}")
    print(f"{'=' * 60}")


if __name__ == '__main__':
    main()
