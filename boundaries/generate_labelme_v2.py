"""
generate_labelme_2class 的后处理版本（v2）:
  1. 先用 generate_labelme_2class 的逻辑生成基础标注
  2. 后处理：
     a. 点吸附 —— 每个节点向 ROI 内最近的暗线像素偏移（不超出 ROI 范围）
     b. 去碎片 —— 删除点数 <= 2 的线段
     c. 同类合并 —— 同 label 的线段首尾距离近则连接

输出到 boundaries/labelmev2/
"""

import cv2
import numpy as np
import json
import shutil
from pathlib import Path
from skimage.morphology import skeletonize
from extract_boundaries import extract_band_roi
from generate_labelme import (
    bridge_gaps, trace_skeleton_lines, merge_lines, simplify_line
)
from generate_labelme_2class import (
    auto_width_threshold, extract_two_level_boundaries,
    to_labelme_json_2class, visualize_two_levels
)


def extract_dark_line_mask(img_bgr, roi_mask):
    """提取 ROI 内的暗线像素"""
    lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2Lab)
    l_ch = lab[:, :, 0].astype(float)

    roi_L = l_ch[roi_mask > 0].astype(np.uint8)
    otsu_val, _ = cv2.threshold(roi_L, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    dark_thresh = otsu_val * 0.7

    dark_mask = ((l_ch < dark_thresh) & (roi_mask > 0)).astype(np.uint8) * 255

    # 去噪
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2, 2))
    dark_mask = cv2.morphologyEx(dark_mask, cv2.MORPH_OPEN, kernel)

    return dark_mask


def snap_points_to_darkline(lines, dark_mask, roi_mask, max_snap_dist=10):
    """
    将线段节点吸附到最近的暗线像素。
    约束：吸附后的点必须在 ROI 范围内，且偏移距离不超过 max_snap_dist。
    """
    if np.count_nonzero(dark_mask) == 0:
        return lines

    # 对暗线 mask 计算最近点映射（距离变换 + 最近标签）
    # 用 distanceTransform 获取距离，用 connectedComponents 反查最近点
    # 更高效：直接用暗线的坐标集做 KDTree
    dark_pts = np.column_stack(np.where(dark_mask > 0))  # (y, x)
    if len(dark_pts) == 0:
        return lines

    from scipy.spatial import cKDTree
    tree = cKDTree(dark_pts)  # tree of (y, x)

    snapped_lines = []
    for line_pts in lines:
        new_pts = []
        for x, y in line_pts:
            # 查询最近暗线像素
            dist, idx = tree.query([y, x])
            if dist <= max_snap_dist:
                ny, nx = dark_pts[idx]
                # 确认吸附后仍在 ROI 内
                if roi_mask[ny, nx] > 0:
                    new_pts.append([int(nx), int(ny)])
                else:
                    new_pts.append([x, y])
            else:
                new_pts.append([x, y])
        snapped_lines.append(new_pts)

    return snapped_lines


def smooth_line_coords(points, window=5):
    """对线段坐标做滑动平均平滑，消除吸附后的跳跃"""
    if len(points) <= window:
        return points
    pts = np.array(points, dtype=np.float64)
    smoothed = pts.copy()
    half = window // 2
    for i in range(half, len(pts) - half):
        smoothed[i] = pts[i - half:i + half + 1].mean(axis=0)
    return smoothed.astype(np.int32).tolist()


def remove_short_lines(lines, min_points=3):
    """删除点数过少的碎片线段"""
    return [l for l in lines if len(l) >= min_points]


def merge_same_class_lines(lines, dist_thresh=15.0, angle_thresh_deg=120):
    """同类型线段首尾合并（复用已有 merge_lines）"""
    return merge_lines(lines, dist_thresh=dist_thresh, angle_thresh_deg=angle_thresh_deg)


def process_one_v2(input_path, labelme_dir_v2, vis_dir):
    """处理单张地图：生成 + 后处理"""
    stem = input_path.stem
    print(f"处理: {stem}")

    img = cv2.imread(str(input_path))
    h, w = img.shape[:2]
    print(f"  尺寸: {w}x{h}")

    # Step 1: 基础提取（同 v1）
    roi_mask = extract_band_roi(img)
    thick_skel, thin_skel, threshold = extract_two_level_boundaries(img)

    print(f"  自动阈值 (Otsu): {threshold:.1f}")

    thick_lines = trace_skeleton_lines(thick_skel, min_length=15)
    thick_lines = merge_lines(thick_lines, dist_thresh=15.0, angle_thresh_deg=120)

    thin_lines = trace_skeleton_lines(thin_skel, min_length=15)
    thin_lines = merge_lines(thin_lines, dist_thresh=15.0, angle_thresh_deg=120)

    print(f"  基础: 一级={len(thick_lines)}, 二级={len(thin_lines)}")

    # Step 2: 提取暗线 mask（吸附目标）
    dark_mask = extract_dark_line_mask(img, roi_mask)
    dark_px = np.count_nonzero(dark_mask)
    print(f"  暗线像素: {dark_px} ({dark_px / max(1, np.count_nonzero(roi_mask)) * 100:.1f}% of ROI)")

    # Step 3: 后处理（纯几何，不再做暗线吸附，避免错吸）
    thick_lines = [smooth_line_coords(l, window=5) for l in thick_lines]
    thin_lines = [smooth_line_coords(l, window=5) for l in thin_lines]
    thick_lines = [simplify_line(l, epsilon=3.0) for l in thick_lines]
    thin_lines = [simplify_line(l, epsilon=3.0) for l in thin_lines]

    # 3b. 去碎片（点数 <= 2）
    thick_lines = remove_short_lines(thick_lines, min_points=3)
    thin_lines = remove_short_lines(thin_lines, min_points=3)

    # 3c. 同类再次合并（吸附后端点可能更近了）
    thick_lines = merge_same_class_lines(thick_lines)
    thin_lines = merge_same_class_lines(thin_lines)

    print(f"  后处理: 一级={len(thick_lines)}, 二级={len(thin_lines)}")

    # 生成 LabelMe JSON
    labelme_data = to_labelme_json_2class(thick_lines, thin_lines, f"{stem}.jpg", h, w)
    print(f"  总 shapes: {len(labelme_data['shapes'])}")

    # 复制原图
    dst_img = labelme_dir_v2 / f"{stem}.jpg"
    if not dst_img.exists():
        shutil.copy2(str(input_path), str(dst_img))

    # 保存 JSON
    json_path = labelme_dir_v2 / f"{stem}.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(labelme_data, f, ensure_ascii=False, indent=2)

    # 可视化
    vis_path = vis_dir / f"{stem}_v2_vis.jpg"
    visualize_two_levels(img, thick_lines, thin_lines, vis_path)
    print(f"  -> {json_path.name}, {vis_path.name}\n")


def main():
    input_dir = Path("/Users/bytedance/Work/qgis_only/map_line_dataset/gcp")
    labelme_dir_v2 = Path("/Users/bytedance/Work/qgis_only/boundaries/labelmev2")
    vis_dir = Path("/Users/bytedance/Work/qgis_only/boundaries/output")
    labelme_dir_v2.mkdir(parents=True, exist_ok=True)
    vis_dir.mkdir(parents=True, exist_ok=True)

    targets = ["05-54淮南道", "07-11岭北行省北部", "08-48云南"]

    for stem in targets:
        input_path = input_dir / f"{stem}.jpg"
        if not input_path.exists():
            print(f"跳过: {input_path}")
            continue
        process_one_v2(input_path, labelme_dir_v2, vis_dir)

    print("全部完成")


if __name__ == "__main__":
    main()
