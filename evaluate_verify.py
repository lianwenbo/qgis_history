"""
验证集评估脚本：
对 verify_data 中的标注数据，逐张推理并与人工标注对比，
输出每个类别的 IoU / Dice / Precision / Recall。
"""
import sys
import json
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from generate_verify_labels import run_inference

# ============================================================
# 配置
# ============================================================
VERIFY_DATA_DIR = Path(__file__).parent / "map_line_dataset/verify_data"
LINE_THICKNESS = 3
SPLITTER_THICKNESS = 9

LABEL_MAP = {
    'vertical_line': 1,
    'horizontal_arc': 2,
    'splitter': 3,
}
CLASS_NAMES = {0: '背景', 1: '经线', 2: '纬线', 3: '分隔线'}


# ============================================================
# 标注 JSON → Mask
# ============================================================
def labelme_json_to_mask(json_path):
    """从 Labelme JSON 渲染 ground truth mask"""
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    h = data['imageHeight']
    w = data['imageWidth']
    mask = np.zeros((h, w), dtype=np.uint8)

    for shape in data.get('shapes', []):
        label = shape.get('label', '')
        points = shape.get('points', [])
        shape_type = shape.get('shape_type', 'line')

        if label not in LABEL_MAP or len(points) < 2:
            continue

        pixel_value = LABEL_MAP[label]
        pts = np.array(points, dtype=np.int32)
        thickness = SPLITTER_THICKNESS if label == 'splitter' else LINE_THICKNESS

        if shape_type == 'line' and len(pts) == 2:
            cv2.line(mask, tuple(pts[0]), tuple(pts[1]), pixel_value, thickness=thickness)
        else:
            for i in range(len(pts) - 1):
                cv2.line(mask, tuple(pts[i]), tuple(pts[i + 1]), pixel_value, thickness=thickness)

    return mask


# ============================================================
# 指标计算
# ============================================================
def compute_metrics(gt_mask, pred_mask, num_classes=4):
    """逐类别计算 IoU, Dice, Precision, Recall"""
    results = {}
    for c in range(num_classes):
        gt_c = (gt_mask == c)
        pred_c = (pred_mask == c)

        tp = int(np.logical_and(gt_c, pred_c).sum())
        fp = int(np.logical_and(~gt_c, pred_c).sum())
        fn = int(np.logical_and(gt_c, ~pred_c).sum())

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        iou = tp / (tp + fp + fn) if (tp + fp + fn) > 0 else 0.0
        dice = 2 * tp / (2 * tp + fp + fn) if (2 * tp + fp + fn) > 0 else 0.0

        results[c] = {
            'precision': precision,
            'recall': recall,
            'iou': iou,
            'dice': dice,
            'tp': tp, 'fp': fp, 'fn': fn
        }
    return results


# ============================================================
# Main
# ============================================================
def main():
    json_files = sorted(VERIFY_DATA_DIR.glob("*.json"))
    if not json_files:
        print("verify_data 中没有找到 JSON 标注文件")
        return

    print(f"验证集: {len(json_files)} 张图片")
    print(f"目录: {VERIFY_DATA_DIR}")
    print("=" * 70)

    # 累计 TP/FP/FN 用于全局指标
    global_stats = {c: {'tp': 0, 'fp': 0, 'fn': 0} for c in range(4)}

    for json_path in json_files:
        img_name = json_path.stem
        img_path = json_path.with_suffix('.jpg')
        if not img_path.exists():
            img_path = json_path.with_suffix('.png')
        if not img_path.exists():
            print(f"  跳过 {img_name}: 找不到图片")
            continue

        print(f"\n[{img_name}]")

        # Ground truth mask
        gt_mask = labelme_json_to_mask(json_path)

        # 推理
        pred_mask = run_inference(img_path)

        # 确保尺寸一致
        if gt_mask.shape != pred_mask.shape:
            pred_mask = cv2.resize(pred_mask, (gt_mask.shape[1], gt_mask.shape[0]),
                                   interpolation=cv2.INTER_NEAREST)

        # 计算指标
        metrics = compute_metrics(gt_mask, pred_mask)

        # 打印单图结果（仅非背景类）
        print(f"  {'类别':<6} {'IoU':>6} {'Dice':>6} {'Prec':>6} {'Recall':>6} {'TP':>8} {'FP':>8} {'FN':>8}")
        print(f"  {'-'*62}")
        for c in range(1, 4):
            m = metrics[c]
            if m['tp'] + m['fp'] + m['fn'] == 0:
                continue
            print(f"  {CLASS_NAMES[c]:<6} {m['iou']:>6.3f} {m['dice']:>6.3f} "
                  f"{m['precision']:>6.3f} {m['recall']:>6.3f} "
                  f"{m['tp']:>8,} {m['fp']:>8,} {m['fn']:>8,}")

        # 累计
        for c in range(4):
            global_stats[c]['tp'] += metrics[c]['tp']
            global_stats[c]['fp'] += metrics[c]['fp']
            global_stats[c]['fn'] += metrics[c]['fn']

    # 全局汇总
    print("\n" + "=" * 70)
    print("全局汇总（所有验证图片）")
    print(f"  {'类别':<6} {'IoU':>6} {'Dice':>6} {'Prec':>6} {'Recall':>6} {'TP':>8} {'FP':>8} {'FN':>8}")
    print(f"  {'-'*62}")

    mean_iou_fg = []
    for c in range(1, 4):
        s = global_stats[c]
        tp, fp, fn = s['tp'], s['fp'], s['fn']
        if tp + fp + fn == 0:
            continue
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        iou = tp / (tp + fp + fn) if (tp + fp + fn) > 0 else 0.0
        dice = 2 * tp / (2 * tp + fp + fn) if (2 * tp + fp + fn) > 0 else 0.0
        mean_iou_fg.append(iou)
        print(f"  {CLASS_NAMES[c]:<6} {iou:>6.3f} {dice:>6.3f} "
              f"{precision:>6.3f} {recall:>6.3f} "
              f"{tp:>8,} {fp:>8,} {fn:>8,}")

    if mean_iou_fg:
        print(f"\n  前景 mIoU: {np.mean(mean_iou_fg):.4f}")


if __name__ == "__main__":
    main()
