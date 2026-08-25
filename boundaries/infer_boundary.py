"""
边界检测推理脚本
输入：训练好的 3 类 UNet 模型 + 原图
输出：
  - <vis_dir>/ : 推理中间结果（mask、概率图、可视化叠加）
  - <out_dir>/ : 生成的 LabelMe 格式标注

管线：滑窗推理 → argmax mask → 每类分别骨架化 → 弥合 → 合并 → 平滑 → 简化 → 输出 LabelMe
"""

import cv2
import numpy as np
import json
import shutil
import sys
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path
from tqdm import tqdm
from skimage.morphology import skeletonize
from PIL import Image

sys.path.insert(0, str(Path(__file__).parent))
from generate_labelme import bridge_gaps, trace_skeleton_lines, merge_lines, simplify_line
from generate_labelme_v2 import smooth_line_coords, remove_short_lines


# ============================================================
# 与训练一致的 UNet 定义
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
    def __init__(self, n_channels=3, n_classes=3):
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


def imread_unicode(path):
    """Pillow 读取，兼容中文路径"""
    pil_img = Image.open(str(path)).convert('RGB')
    arr = np.array(pil_img)
    return cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)


def _build_weight_map(patch_size):
    """中心权重高、边缘低的高斯权重图"""
    sigma = patch_size / 6.0
    yy, xx = np.mgrid[0:patch_size, 0:patch_size]
    center = (patch_size - 1) / 2.0
    w = np.exp(-((xx - center) ** 2 + (yy - center) ** 2) / (2 * sigma ** 2))
    return w.astype(np.float32)


def sliding_predict(model, img_bgr, device, patch_size=512, overlap=0.5, fill_color=(245, 235, 210)):
    """滑窗推理 3 类分割，返回 (pred_mask, prob_map)
    - pred_mask: (H, W) uint8, 0/1/2
    - prob_map:  (H, W, 3) float32, softmax概率
    """
    original_h, original_w = img_bgr.shape[:2]
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

    stride = int(patch_size * (1 - overlap))
    n_rows = int(np.ceil((original_h - patch_size) / stride)) + 1
    n_cols = int(np.ceil((original_w - patch_size) / stride)) + 1
    pad_h = (n_rows - 1) * stride + patch_size
    pad_w = (n_cols - 1) * stride + patch_size

    img_padded = np.full((pad_h, pad_w, 3), fill_color, dtype=np.uint8)
    img_padded[:original_h, :original_w] = img_rgb

    weight_map = _build_weight_map(patch_size)
    n_classes = 3
    prob_accum = np.zeros((pad_h, pad_w, n_classes), dtype=np.float32)
    weight_accum = np.zeros((pad_h, pad_w), dtype=np.float32)

    total = n_rows * n_cols
    pbar = tqdm(total=total, desc="滑窗推理", ncols=100)
    with torch.no_grad():
        for i in range(n_rows):
            for j in range(n_cols):
                y = i * stride
                x = j * stride
                patch = img_padded[y:y + patch_size, x:x + patch_size]
                patch_norm = patch.astype(np.float32) / 255.0
                t = torch.from_numpy(patch_norm).permute(2, 0, 1).unsqueeze(0).to(device)
                output = model(t)
                prob = torch.softmax(output, dim=1).squeeze().cpu().numpy()
                prob = np.transpose(prob, (1, 2, 0))
                w = weight_map[:, :, np.newaxis]
                prob_accum[y:y + patch_size, x:x + patch_size] += prob * w
                weight_accum[y:y + patch_size, x:x + patch_size] += weight_map
                pbar.update(1)
    pbar.close()

    weight_accum_safe = np.maximum(weight_accum, 1e-6)
    prob_avg = prob_accum / weight_accum_safe[:, :, np.newaxis]
    pred_mask = np.argmax(prob_avg, axis=2).astype(np.uint8)
    return pred_mask[:original_h, :original_w], prob_avg[:original_h, :original_w].astype(np.float32)


def mask_to_lines_per_class(pred_mask, verbose=True):
    """
    对每类 mask 分别：骨架化 → 弥合 → 追踪线段 → 合并 → 平滑 → 简化 → 去碎片
    返回 (thick_lines, thin_lines)
    """
    thick_mask = (pred_mask == 1).astype(np.uint8) * 255
    thin_mask = (pred_mask == 2).astype(np.uint8) * 255

    def _process_one(mask, class_name):
        t0 = time.perf_counter()
        if np.count_nonzero(mask) == 0:
            if verbose:
                print(f"    [{class_name}] 空 mask, 跳过")
            return []
        t1 = time.perf_counter()
        skel = skeletonize((mask > 0).astype(np.uint8)).astype(np.uint8) * 255
        t2 = time.perf_counter()
        skel = bridge_gaps(skel, max_gap=15)
        t3 = time.perf_counter()
        skel = skeletonize((skel > 0).astype(np.uint8)).astype(np.uint8) * 255
        t4 = time.perf_counter()
        lines = trace_skeleton_lines(skel, min_length=15)
        n_trace = len(lines)
        t5 = time.perf_counter()
        lines = merge_lines(lines, dist_thresh=15.0, angle_thresh_deg=120)
        n_merge1 = len(lines)
        t6 = time.perf_counter()
        lines = [smooth_line_coords(l, window=7) for l in lines]
        t7 = time.perf_counter()
        lines = [simplify_line(l, epsilon=2.0) for l in lines]
        t8 = time.perf_counter()
        lines = remove_short_lines(lines, min_points=3)
        t9 = time.perf_counter()
        lines = merge_lines(lines, dist_thresh=10.0, angle_thresh_deg=120)
        n_merge2 = len(lines)
        t10 = time.perf_counter()
        if verbose:
            print(
                f"    [{class_name}] 追踪={n_trace}→合并1={n_merge1}→合并2={n_merge2}→输出={len(lines)} | "
                f"骨架={t2-t1:.1f}s 弥合={t3-t2:.1f}s 追踪={t5-t4:.1f}s 合并1={t6-t5:.1f}s "
                f"平滑7={t7-t6:.1f}s 简化2={t8-t7:.1f}s 去碎片={t9-t8:.1f}s 合并2={t10-t9:.1f}s "
                f"总计={t10-t0:.1f}s"
            )
        return lines

    thick_lines = _process_one(thick_mask, "boundary_1")
    thin_lines = _process_one(thin_mask, "boundary_2")
    return thick_lines, thin_lines


def lines_to_labelme_json(thick_lines, thin_lines, image_name, img_h, img_w):
    shapes = []
    for pts in thick_lines:
        if len(pts) < 2:
            continue
        shapes.append({"label": "boundary_1", "points": pts, "group_id": None,
                       "shape_type": "linestrip", "flags": {}})
    for pts in thin_lines:
        if len(pts) < 2:
            continue
        shapes.append({"label": "boundary_2", "points": pts, "group_id": None,
                       "shape_type": "linestrip", "flags": {}})
    return {"version": "5.3.1", "flags": {}, "shapes": shapes,
            "imagePath": image_name, "imageData": None,
            "imageHeight": img_h, "imageWidth": img_w}


def visualize_lines(img_bgr, thick_lines, thin_lines, output_path):
    vis = img_bgr.copy()
    for pts in thick_lines:
        a = np.array(pts, dtype=np.int32)
        for i in range(len(a) - 1):
            cv2.line(vis, tuple(a[i]), tuple(a[i + 1]), (0, 0, 255), 3)
    for pts in thin_lines:
        a = np.array(pts, dtype=np.int32)
        for i in range(len(a) - 1):
            cv2.line(vis, tuple(a[i]), tuple(a[i + 1]), (0, 255, 0), 2)
    cv2.imwrite(str(output_path), vis)


def visualize_mask_overlay(img_bgr, pred_mask, output_path):
    overlay = img_bgr.copy()
    overlay[pred_mask == 1] = (0, 0, 255)
    overlay[pred_mask == 2] = (0, 255, 0)
    cv2.imwrite(str(output_path), overlay)


def visualize_mask_color(pred_mask, output_path):
    """将 0/1/2 的灰度 mask 映射为彩色（红蓝绿）便于肉眼查看"""
    h, w = pred_mask.shape
    vis = np.zeros((h, w, 3), dtype=np.uint8)
    vis[pred_mask == 0] = (0, 0, 0)
    vis[pred_mask == 1] = (0, 0, 255)
    vis[pred_mask == 2] = (0, 255, 0)
    cv2.imwrite(str(output_path), vis)


def main():
    import argparse
    parser = argparse.ArgumentParser(description="边界检测推理")
    parser.add_argument('--model', type=str, default=str(Path.home() / "Downloads" / "unet_boundary_autodl_20260823_080318.pth"))
    parser.add_argument('--image_dir', type=str, default=None, help="图片目录，默认 map_line_dataset/gcp/")
    parser.add_argument('--out_dir', type=str, default=None, help="labelme输出目录，默认 boundaries/chgis_labelme/")
    parser.add_argument('--vis_dir', type=str, default=None, help="中间结果目录，默认 boundaries/output/")
    parser.add_argument('--targets', type=str, nargs='+', default=["08-35湖北"], help="要处理的图名（不带后缀）")
    args = parser.parse_args()

    project_root = Path(__file__).parent
    repo_root = project_root.parent
    model_path = Path(args.model)
    image_dir = Path(args.image_dir) if args.image_dir else (repo_root / "map_line_dataset" / "gcp")
    output_dir = Path(args.vis_dir) if args.vis_dir else (project_root / "output")
    labelme_out_dir = Path(args.out_dir) if args.out_dir else (project_root / "chgis_labelme")
    output_dir.mkdir(parents=True, exist_ok=True)
    labelme_out_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if torch.backends.mps.is_available():
        device = torch.device('mps')
    print(f"设备: {device}")

    # 加载模型（兼容 2 类或 3 类 checkpoint）
    checkpoint = torch.load(str(model_path), map_location=device)
    n_classes = checkpoint.get('n_classes', 3)
    print(f"加载模型: {model_path} (n_classes={n_classes})")
    model = UNet(n_channels=3, n_classes=n_classes).to(device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    targets = args.targets

    for stem in targets:
        t_total0 = time.perf_counter()
        print(f"\n{'=' * 60}")
        print(f"处理: {stem}")
        print('=' * 60)
        img_path = image_dir / f"{stem}.jpg"
        if not img_path.exists():
            print(f"  跳过（不存在）: {img_path}")
            continue

        t0 = time.perf_counter()
        img_bgr = imread_unicode(str(img_path))
        h, w = img_bgr.shape[:2]
        print(f"  尺寸: {w}x{h}  [读图 {time.perf_counter()-t0:.1f}s]")

        t1 = time.perf_counter()
        pred_mask, _ = sliding_predict(model, img_bgr, device)
        cls_counts = np.bincount(pred_mask.flatten(), minlength=3)
        print(f"  [1/3 滑窗推理 {time.perf_counter()-t1:.1f}s] "
              f"像素分布: 背景={cls_counts[0]:,}, boundary_1={cls_counts[1]:,}, boundary_2={cls_counts[2]:,}")

        cv2.imwrite(str(output_dir / f"{stem}_pred_mask.png"), pred_mask)
        visualize_mask_color(pred_mask, output_dir / f"{stem}_pred_mask_color.jpg")
        visualize_mask_overlay(img_bgr, pred_mask, output_dir / f"{stem}_pred_overlay.jpg")

        t2 = time.perf_counter()
        thick_lines, thin_lines = mask_to_lines_per_class(pred_mask)
        print(f"  [2/3 骨架→线段 {time.perf_counter()-t2:.1f}s] "
              f"一级边界: {len(thick_lines)} 条, 二级边界: {len(thin_lines)} 条")

        t3 = time.perf_counter()
        visualize_lines(img_bgr, thick_lines, thin_lines, output_dir / f"{stem}_pred_lines.jpg")
        json_data = lines_to_labelme_json(thick_lines, thin_lines, f"{stem}.jpg", h, w)
        dst_jpg = labelme_out_dir / f"{stem}.jpg"
        if Path(img_path).resolve() != dst_jpg.resolve():
            shutil.copy2(str(img_path), str(dst_jpg))
        with open(labelme_out_dir / f"{stem}.json", "w", encoding="utf-8") as f:
            json.dump(json_data, f, ensure_ascii=False, indent=2)
        print(f"  [3/3 可视化+输出 {time.perf_counter()-t3:.1f}s] -> {labelme_out_dir.name}/{stem}.json")
        print(f"  单图总耗时: {time.perf_counter()-t_total0:.1f}s")

    print(f"\n{'=' * 60}")
    print("全部完成")
    print(f"中间结果: {output_dir}/")
    print(f"LabelMe 标注: {labelme_out_dir}/")
    print('=' * 60)


if __name__ == "__main__":
    main()
