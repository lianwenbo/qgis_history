"""
行政边界 UNet 训练 Pipeline
数据来源：boundaries/labelme/ 下的 3 张标注图
流程：LabelMe JSON → mask 生成 → 随机裁切 patch → UNet 训练
"""

import sys
sys.path.insert(0, str(__import__('pathlib').Path(__file__).parent.parent))

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import numpy as np
import cv2
import json
from pathlib import Path
from tqdm import tqdm

from unet_model import UNet


# ============================================================
# 数据准备：LabelMe → Mask
# ============================================================

def labelme_to_mask(json_path, line_thickness=3):
    """从 LabelMe JSON 生成二值 mask（boundary=1, background=0）"""
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    img_h = data['imageHeight']
    img_w = data['imageWidth']
    shapes = data.get('shapes', [])

    mask = np.zeros((img_h, img_w), dtype=np.uint8)
    for shape in shapes:
        points = shape.get('points', [])
        if len(points) < 2:
            continue
        pts = np.array(points, dtype=np.int32)
        for i in range(len(pts) - 1):
            cv2.line(mask, tuple(pts[i]), tuple(pts[i + 1]), 1, thickness=line_thickness)

    return mask


def generate_masks(labelme_dir, output_dir, line_thickness=3):
    """批量生成 mask 并保存"""
    labelme_dir = Path(labelme_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    json_files = sorted(labelme_dir.glob("*.json"))
    print(f"生成 mask: {len(json_files)} 张")

    for json_path in json_files:
        mask = labelme_to_mask(json_path, line_thickness)
        mask_path = output_dir / f"{json_path.stem}_mask.png"
        cv2.imwrite(str(mask_path), mask * 255)
        print(f"  {json_path.stem}: boundary={mask.sum()} px")

    return len(json_files)


# ============================================================
# Dataset：随机裁切 patch
# ============================================================

class BoundaryPatchDataset(Dataset):
    """从原图+mask中随机裁切 patch 进行训练

    策略：每张图按 patches_per_image 次随机裁切，优先在有边界的区域采样
    """

    def __init__(self, labelme_dir, masks_dir, patch_size=512,
                 patches_per_image=50, line_thickness=3, augment=True):
        self.patch_size = patch_size
        self.augment = augment
        self.patches_per_image = patches_per_image

        labelme_dir = Path(labelme_dir)
        masks_dir = Path(masks_dir)

        self.samples = []
        json_files = sorted(labelme_dir.glob("*.json"))
        for json_path in json_files:
            img_path = labelme_dir / f"{json_path.stem}.jpg"
            mask_path = masks_dir / f"{json_path.stem}_mask.png"
            if img_path.exists() and mask_path.exists():
                self.samples.append((str(img_path), str(mask_path)))

        self.total = len(self.samples) * patches_per_image
        print(f"训练集: {len(self.samples)} 张图 × {patches_per_image} patch = {self.total} 样本")

    def __len__(self):
        return self.total

    def __getitem__(self, idx):
        img_idx = idx // self.patches_per_image
        img_path, mask_path = self.samples[img_idx]

        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        mask = (mask > 0).astype(np.uint8)

        h, w = img.shape[:2]
        ps = self.patch_size

        # 50% 概率在边界区域采样，50% 随机
        if np.random.random() < 0.5:
            coords = np.column_stack(np.where(mask > 0))
            if len(coords) > 0:
                pt = coords[np.random.randint(len(coords))]
                cy, cx = pt[0], pt[1]
                y = max(0, min(cy - ps // 2, h - ps))
                x = max(0, min(cx - ps // 2, w - ps))
            else:
                y = np.random.randint(0, max(1, h - ps))
                x = np.random.randint(0, max(1, w - ps))
        else:
            y = np.random.randint(0, max(1, h - ps))
            x = np.random.randint(0, max(1, w - ps))

        patch_img = img[y:y + ps, x:x + ps]
        patch_mask = mask[y:y + ps, x:x + ps]

        # 如果裁到边缘不够大，pad
        if patch_img.shape[0] < ps or patch_img.shape[1] < ps:
            pad_img = np.zeros((ps, ps, 3), dtype=np.uint8)
            pad_mask = np.zeros((ps, ps), dtype=np.uint8)
            pad_img[:patch_img.shape[0], :patch_img.shape[1]] = patch_img
            pad_mask[:patch_mask.shape[0], :patch_mask.shape[1]] = patch_mask
            patch_img = pad_img
            patch_mask = pad_mask

        # 数据增强
        if self.augment:
            # 随机翻转
            if np.random.random() > 0.5:
                patch_img = np.flip(patch_img, axis=1).copy()
                patch_mask = np.flip(patch_mask, axis=1).copy()
            if np.random.random() > 0.5:
                patch_img = np.flip(patch_img, axis=0).copy()
                patch_mask = np.flip(patch_mask, axis=0).copy()
            # 随机旋转 90°
            k = np.random.randint(0, 4)
            if k > 0:
                patch_img = np.rot90(patch_img, k).copy()
                patch_mask = np.rot90(patch_mask, k).copy()
            # 颜色抖动
            if np.random.random() > 0.5:
                factor = np.random.uniform(0.8, 1.2)
                patch_img = np.clip(patch_img.astype(np.float32) * factor, 0, 255).astype(np.uint8)

        # 归一化
        patch_img = patch_img.astype(np.float32) / 255.0
        patch_mask = patch_mask.astype(np.int64)

        img_tensor = torch.from_numpy(patch_img).permute(2, 0, 1)
        mask_tensor = torch.from_numpy(patch_mask)

        return img_tensor, mask_tensor


# ============================================================
# 训练
# ============================================================

def train(labelme_dir='boundaries/labelme',
          masks_dir='boundaries/masks',
          model_path='boundaries/models/unet_boundary.pth',
          epochs=60, patch_size=512, batch_size=4, lr=2e-4,
          patches_per_image=80, line_thickness=3):
    """训练边界检测 UNet"""

    labelme_dir = Path(labelme_dir)
    masks_dir = Path(masks_dir)

    # Step 1: 生成 mask
    print("=" * 50)
    print("Step 1: 生成 mask")
    print("=" * 50)
    generate_masks(labelme_dir, masks_dir, line_thickness)

    # Step 2: 构建数据集
    print("\n" + "=" * 50)
    print("Step 2: 构建数据集")
    print("=" * 50)
    dataset = BoundaryPatchDataset(
        labelme_dir, masks_dir,
        patch_size=patch_size,
        patches_per_image=patches_per_image,
        line_thickness=line_thickness
    )

    dataloader = DataLoader(
        dataset, batch_size=batch_size, shuffle=True,
        num_workers=2, pin_memory=False, persistent_workers=True
    )

    # Step 3: 模型
    print("\n" + "=" * 50)
    print("Step 3: 训练模型")
    print("=" * 50)
    device = torch.device('mps' if torch.backends.mps.is_available() else
                          'cuda' if torch.cuda.is_available() else 'cpu')
    print(f"设备: {device}")

    model = UNet(n_channels=3, n_classes=2).to(device)
    Path(model_path).parent.mkdir(parents=True, exist_ok=True)

    # 边界像素远少于背景，加大权重
    class_weights = torch.tensor([1.0, 30.0]).to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    best_loss = float('inf')
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        num_batches = 0

        pbar = tqdm(dataloader, desc=f"Epoch {epoch + 1}/{epochs}")
        for imgs, masks in pbar:
            imgs = imgs.to(device)
            masks = masks.to(device)

            optimizer.zero_grad()
            outputs = model(imgs)
            loss = criterion(outputs, masks)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            num_batches += 1
            pbar.set_postfix(loss=f"{loss.item():.4f}")

        scheduler.step()
        avg_loss = total_loss / max(num_batches, 1)

        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(f"  Epoch {epoch + 1}/{epochs}, Loss={avg_loss:.4f}, LR={scheduler.get_last_lr()[0]:.2e}")

        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': best_loss,
                'n_classes': 2,
                'patch_size': patch_size,
            }, model_path)

    print(f"\n训练完成！最佳 Loss: {best_loss:.4f}")
    print(f"模型保存: {model_path}")


# ============================================================
# 推理
# ============================================================

def predict(model_path, image_path, output_dir='boundaries/output',
            patch_size=512, overlap=0.25):
    """滑窗推理"""
    device = torch.device('mps' if torch.backends.mps.is_available() else
                          'cuda' if torch.cuda.is_available() else 'cpu')

    model = UNet(n_channels=3, n_classes=2).to(device)
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    img = cv2.imread(str(image_path))
    if img is None:
        print(f"无法读取: {image_path}")
        return None

    original_h, original_w = img.shape[:2]
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0

    stride = int(patch_size * (1 - overlap))
    n_rows = max(1, int(np.ceil((original_h - patch_size) / stride)) + 1)
    n_cols = max(1, int(np.ceil((original_w - patch_size) / stride)) + 1)

    pad_h = (n_rows - 1) * stride + patch_size
    pad_w = (n_cols - 1) * stride + patch_size

    padded = np.zeros((pad_h, pad_w, 3), dtype=np.float32)
    padded[:original_h, :original_w] = img_rgb

    # 权重图
    weight_1d = np.ones(patch_size, dtype=np.float32)
    border = patch_size // 4
    for i in range(border):
        weight_1d[i] = (i + 1) / (border + 1)
        weight_1d[-(i + 1)] = (i + 1) / (border + 1)
    weight_map = np.outer(weight_1d, weight_1d)

    prob_sum = np.zeros((pad_h, pad_w), dtype=np.float32)
    weight_sum = np.zeros((pad_h, pad_w), dtype=np.float32)

    with torch.no_grad():
        for r in range(n_rows):
            for c in range(n_cols):
                y, x = r * stride, c * stride
                patch = padded[y:y + patch_size, x:x + patch_size]
                t = torch.from_numpy(patch).permute(2, 0, 1).unsqueeze(0).to(device)
                out = model(t)
                prob = torch.softmax(out, dim=1)[0, 1].cpu().numpy()
                prob_sum[y:y + patch_size, x:x + patch_size] += prob * weight_map
                weight_sum[y:y + patch_size, x:x + patch_size] += weight_map

    weight_sum = np.maximum(weight_sum, 1e-6)
    prob_final = (prob_sum / weight_sum)[:original_h, :original_w]
    pred_mask = (prob_final > 0.5).astype(np.uint8)

    # 保存
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    stem = Path(image_path).stem
    cv2.imwrite(str(Path(output_dir) / f"{stem}_pred_mask.png"), pred_mask * 255)

    vis = img.copy()
    vis[pred_mask > 0] = (0, 255, 255)
    cv2.imwrite(str(Path(output_dir) / f"{stem}_pred_vis.png"), vis)

    print(f"  边界像素: {pred_mask.sum()}, 保存: {output_dir}/{stem}_pred_vis.png")
    return pred_mask


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('mode', choices=['train', 'predict'], default='train', nargs='?')
    parser.add_argument('--image', type=str, default=None)
    parser.add_argument('--model', type=str, default='boundaries/models/unet_boundary.pth')
    parser.add_argument('--epochs', type=int, default=60)
    parser.add_argument('--batch-size', type=int, default=4)
    parser.add_argument('--patch-size', type=int, default=512)
    parser.add_argument('--lr', type=float, default=2e-4)
    parser.add_argument('--patches-per-image', type=int, default=80)
    parser.add_argument('--line-thickness', type=int, default=3)
    args = parser.parse_args()

    if args.mode == 'train':
        train(
            labelme_dir='boundaries/labelme',
            masks_dir='boundaries/masks',
            model_path=args.model,
            epochs=args.epochs,
            patch_size=args.patch_size,
            batch_size=args.batch_size,
            lr=args.lr,
            patches_per_image=args.patches_per_image,
            line_thickness=args.line_thickness,
        )
    elif args.mode == 'predict':
        if args.image:
            predict(args.model, args.image)
        else:
            test_dir = Path('boundaries/test_data')
            for img_path in sorted(test_dir.glob("*.jpg")):
                print(f"推理: {img_path.stem}")
                predict(args.model, str(img_path))
