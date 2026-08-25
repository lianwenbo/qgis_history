"""
推理 + 校准工作流：
1. 从源目录随机抽取 10 张图片放入 test_data
2. 使用指定模型推理，输出到 verify_data（图片 + 推理可视化 + Labelme JSON）
3. 用 Labelme 打开 verify_data 进行人工校准
4. 校准后将 verify_data 中的图片和 JSON 移入 raw_data 重新训练

用法:
    python generate_test_comparison.py [--model models/xxx.pth] [--num 10]
"""
import sys
import re
import random
import shutil
import json
import argparse
from pathlib import Path

import cv2
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent))
from unet_model import UNet, _build_weight_map
from generate_verify_labels import (
    mask_to_skeleton_lines, merge_vertical_lines, merge_horizontal_lines
)

# ============================================================
# 配置
# ============================================================
SOURCE_DIR = Path.home() / "Work/historical_map/古代中国地图"
RAW_DATA_DIR = Path(__file__).parent / "map_line_dataset/raw_data"
TEST_DATA_DIR = Path(__file__).parent / "map_line_dataset/test_data"
VERIFY_DATA_DIR = Path(__file__).parent / "map_line_dataset/verify_data"

PATCH_SIZE = 512
OVERLAP = 0.5
FILL_COLOR = (245, 235, 210)


# ============================================================
# 推理
# ============================================================
def run_inference_with_model(model, image_path, device):
    """对单张图片进行滑窗推理"""
    img = cv2.imread(str(image_path))
    original_h, original_w = img.shape[:2]
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    stride = int(PATCH_SIZE * (1 - OVERLAP))
    n_rows = int(np.ceil((original_h - PATCH_SIZE) / stride)) + 1
    n_cols = int(np.ceil((original_w - PATCH_SIZE) / stride)) + 1
    pad_h = (n_rows - 1) * stride + PATCH_SIZE
    pad_w = (n_cols - 1) * stride + PATCH_SIZE

    img_padded = np.full((pad_h, pad_w, 3), FILL_COLOR, dtype=np.uint8)
    img_padded[:original_h, :original_w] = img_rgb

    weight_map = _build_weight_map(PATCH_SIZE)
    prob_accum = np.zeros((pad_h, pad_w, 4), dtype=np.float32)
    weight_accum = np.zeros((pad_h, pad_w), dtype=np.float32)

    with torch.no_grad():
        for i in range(n_rows):
            for j in range(n_cols):
                y = i * stride
                x = j * stride
                patch = img_padded[y:y+PATCH_SIZE, x:x+PATCH_SIZE]
                patch_norm = patch.astype(np.float32) / 255.0
                x_tensor = torch.from_numpy(patch_norm).permute(2, 0, 1).unsqueeze(0).to(device)
                output = model(x_tensor)
                prob = torch.softmax(output, dim=1).squeeze().cpu().numpy()
                prob = np.transpose(prob, (1, 2, 0))
                w = weight_map[:, :, np.newaxis]
                prob_accum[y:y+PATCH_SIZE, x:x+PATCH_SIZE] += prob * w
                weight_accum[y:y+PATCH_SIZE, x:x+PATCH_SIZE] += weight_map

    weight_accum_safe = np.maximum(weight_accum, 1e-6)
    prob_avg = prob_accum / weight_accum_safe[:, :, np.newaxis]
    pred_mask = np.argmax(prob_avg, axis=2).astype(np.uint8)
    return pred_mask[:original_h, :original_w]


# ============================================================
# Mask → Labelme JSON
# ============================================================
def generate_labelme_json(image_path, pred_mask, output_path):
    img = cv2.imread(str(image_path))
    h, w = img.shape[:2]
    shapes = []

    v_lines = mask_to_skeleton_lines(pred_mask, 1)
    v_merged = merge_vertical_lines(v_lines, img_height=h)
    for pts in v_merged:
        shapes.append({
            "label": "vertical_line", "points": pts,
            "group_id": None, "description": "",
            "shape_type": "line", "flags": {}, "mask": None
        })

    h_lines = mask_to_skeleton_lines(pred_mask, 2)
    h_merged = merge_horizontal_lines(h_lines)
    for pts in h_merged:
        shapes.append({
            "label": "horizontal_arc", "points": pts,
            "group_id": None, "description": "",
            "shape_type": "linestrip", "flags": {}, "mask": None
        })

    s_lines = mask_to_skeleton_lines(pred_mask, 3)
    s_merged = merge_horizontal_lines(s_lines)
    for pts in s_merged:
        shapes.append({
            "label": "splitter", "points": pts,
            "group_id": None, "description": "",
            "shape_type": "linestrip", "flags": {}, "mask": None
        })

    labelme_data = {
        "version": "5.4.1", "flags": {},
        "shapes": shapes,
        "imagePath": image_path.name,
        "imageData": None,
        "imageHeight": h, "imageWidth": w
    }

    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(labelme_data, f, ensure_ascii=False, indent=2)

    return len(v_merged), len(h_merged), len(s_merged)


# ============================================================
# Main
# ============================================================
def main():
    parser = argparse.ArgumentParser(description='推理 + 校准工作流')
    parser.add_argument('--model', type=str, required=True, help='模型路径')
    parser.add_argument('--num', type=int, default=10, help='抽取图片数量')
    parser.add_argument('--seed', type=int, default=None, help='随机种子（不指定则随机）')
    args = parser.parse_args()

    if args.seed is not None:
        random.seed(args.seed)

    prefix_pattern = re.compile(r'^(\d+-\d+)')

    # Collect used prefixes (raw_data + existing test_data)
    used = set()
    for d in [RAW_DATA_DIR, TEST_DATA_DIR]:
        if d.exists():
            for f in d.iterdir():
                m = prefix_pattern.match(f.name)
                if m:
                    used.add(m.group(1))

    # Scan source
    all_images = []
    for ext in ('*.jpg', '*.png', '*.jpeg', '*.tif'):
        all_images.extend(SOURCE_DIR.rglob(ext))

    prefix_map = {}
    for img_path in all_images:
        m = prefix_pattern.match(img_path.name)
        if not m:
            continue
        prefix = m.group(1)
        vol = int(prefix.split('-')[0])
        if vol <= 3:
            continue
        if '全图' in img_path.name or '总图' in img_path.name:
            continue
        if prefix in used:
            continue
        if prefix not in prefix_map:
            prefix_map[prefix] = []
        prefix_map[prefix].append(img_path)

    num = min(args.num, len(prefix_map))
    candidates = [(p, random.choice(paths)) for p, paths in prefix_map.items()]
    selected = random.sample(candidates, num)

    # Prepare directories
    TEST_DATA_DIR.mkdir(parents=True, exist_ok=True)
    VERIFY_DATA_DIR.mkdir(parents=True, exist_ok=True)

    # Clean verify_data
    for f in VERIFY_DATA_DIR.iterdir():
        f.unlink()

    # Copy images to test_data
    print("=" * 60)
    print(f"Step 1: 抽取 {num} 张图片 → test_data")
    print("=" * 60)
    for prefix, src_path in sorted(selected):
        dst = TEST_DATA_DIR / src_path.name
        shutil.copy2(src_path, dst)
        print(f"  [{prefix}] {src_path.name}")

    # Load model
    device = torch.device('mps' if torch.backends.mps.is_available() else
                          'cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\n设备: {device}")
    print(f"模型: {args.model}")

    model = UNet(n_channels=3, n_classes=4).to(device)
    checkpoint = torch.load(args.model, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    print(f"  epoch: {checkpoint.get('epoch', '?')}, loss: {checkpoint.get('loss', '?'):.4f}")

    # Inference → verify_data
    print(f"\n{'=' * 60}")
    print(f"Step 2: 推理 → verify_data")
    print("=" * 60)

    for prefix, src_path in sorted(selected):
        img_path = TEST_DATA_DIR / src_path.name
        print(f"\n  [{prefix}] {src_path.name}")

        pred_mask = run_inference_with_model(model, img_path, device)

        # Copy image to verify_data
        shutil.copy2(img_path, VERIFY_DATA_DIR / src_path.name)

        # Save inference visualization
        img = cv2.imread(str(img_path))
        overlay = img.copy()
        overlay[pred_mask == 1] = [0, 0, 255]
        overlay[pred_mask == 2] = [0, 255, 0]
        overlay[pred_mask == 3] = [0, 165, 255]
        blended = cv2.addWeighted(img, 0.5, overlay, 0.5, 0)
        cv2.imwrite(str(VERIFY_DATA_DIR / f"{src_path.stem}_infer.jpg"), blended)

        # Generate Labelme JSON
        json_path = VERIFY_DATA_DIR / f"{src_path.stem}.json"
        nv, nh, ns = generate_labelme_json(VERIFY_DATA_DIR / src_path.name, pred_mask, json_path)
        print(f"    标注: 经线={nv} 纬线={nh} 分隔线={ns}")

    print(f"\n{'=' * 60}")
    print("完成!")
    print(f"  测试图片: {TEST_DATA_DIR}")
    print(f"  校准目录: {VERIFY_DATA_DIR}")
    print(f"\n下一步:")
    print(f"  1. labelme {VERIFY_DATA_DIR}")
    print(f"  2. 校准标注后，将 .jpg + .json 移入 raw_data 重新训练")
    print("=" * 60)


if __name__ == "__main__":
    main()
