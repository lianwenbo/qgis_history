"""
从原始地图图像直接提取两级边界线，生成 LabelMe JSON:
  - boundary_1: 一级边界（粗线，省/道级）
  - boundary_2: 二级边界（细线，州/府级）

方法：
  1. 提取洋红/紫色带状 ROI
  2. 对 ROI 做距离变换，得到每个像素到 ROI 边缘的距离（≈ 线条半宽）
  3. 骨架化后，用 Otsu 自动找到粗/细分界阈值
  4. 两类骨架分别追踪线段，同级别积极合并首尾相近的线段
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


def auto_width_threshold(dist_map, skeleton):
    """
    用 Otsu 方法在骨架像素的宽度分布上自动找到粗/细分界阈值。
    """
    widths = dist_map[skeleton > 0]
    if len(widths) == 0:
        return 6.0

    # 将宽度值缩放到 0-255 供 Otsu 使用（精度 0.1px）
    w_scaled = np.clip(widths * 10, 0, 255).astype(np.uint8)
    otsu_val, _ = cv2.threshold(w_scaled, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    threshold = otsu_val / 10.0

    return threshold


def extract_two_level_boundaries(img_bgr):
    """
    从原始图像直接提取两级边界线骨架，自动判断粗细阈值。

    返回: (thick_skeleton, thin_skeleton, threshold)
    """
    roi_mask = extract_band_roi(img_bgr)

    # 距离变换：每个 ROI 内像素到边缘的距离 ≈ 线条半宽
    dist_map = cv2.distanceTransform(roi_mask, cv2.DIST_L2, 5)

    # 骨架化 + 断裂弥合
    skeleton = skeletonize((roi_mask > 0).astype(np.uint8)).astype(np.uint8) * 255
    bridged = bridge_gaps(skeleton, max_gap=20)
    bridged = skeletonize((bridged > 0).astype(np.uint8)).astype(np.uint8) * 255

    # 自动确定粗细分界阈值
    threshold = auto_width_threshold(dist_map, bridged)

    # 按阈值在骨架像素级分类
    thick_skel = ((bridged > 0) & (dist_map >= threshold)).astype(np.uint8) * 255
    thin_skel = ((bridged > 0) & (dist_map < threshold)).astype(np.uint8) * 255

    return thick_skel, thin_skel, threshold


def to_labelme_json_2class(thick_lines, thin_lines, image_path, img_height, img_width):
    """生成包含两类标签的 LabelMe JSON"""
    shapes = []

    for line_pts in thick_lines:
        simplified = simplify_line(line_pts, epsilon=2.0)
        if len(simplified) < 2:
            continue
        shapes.append({
            "label": "boundary_1",
            "points": simplified,
            "group_id": None,
            "shape_type": "linestrip",
            "flags": {}
        })

    for line_pts in thin_lines:
        simplified = simplify_line(line_pts, epsilon=2.0)
        if len(simplified) < 2:
            continue
        shapes.append({
            "label": "boundary_2",
            "points": simplified,
            "group_id": None,
            "shape_type": "linestrip",
            "flags": {}
        })

    return {
        "version": "5.3.1",
        "flags": {},
        "shapes": shapes,
        "imagePath": image_path,
        "imageData": None,
        "imageHeight": img_height,
        "imageWidth": img_width
    }


def visualize_two_levels(img, thick_lines, thin_lines, output_path):
    """可视化：红色粗线=一级，绿色细线=二级"""
    vis = img.copy()
    for line_pts in thick_lines:
        pts = np.array(line_pts, dtype=np.int32)
        for i in range(len(pts) - 1):
            cv2.line(vis, tuple(pts[i]), tuple(pts[i + 1]), (0, 0, 255), 3)
    for line_pts in thin_lines:
        pts = np.array(line_pts, dtype=np.int32)
        for i in range(len(pts) - 1):
            cv2.line(vis, tuple(pts[i]), tuple(pts[i + 1]), (0, 255, 0), 2)
    cv2.imwrite(str(output_path), vis)


def process_one(input_path, labelme_dir, vis_dir):
    """处理单张地图：从原图直接提取两级边界"""
    stem = input_path.stem
    print(f"处理: {stem}")

    img = cv2.imread(str(input_path))
    h, w = img.shape[:2]
    print(f"  尺寸: {w}x{h}")

    # 核心：直接从原图像素提取两级边界（自动阈值）
    thick_skel, thin_skel, threshold = extract_two_level_boundaries(img)

    thick_px = np.count_nonzero(thick_skel)
    thin_px = np.count_nonzero(thin_skel)
    print(f"  自动阈值 (Otsu): {threshold:.1f} (半宽)")
    print(f"  粗线骨架: {thick_px} px, 细线骨架: {thin_px} px")

    # 分别追踪线段，同级别内积极合并（dist_thresh=15, angle=120°）
    thick_lines = trace_skeleton_lines(thick_skel, min_length=15)
    thick_lines = merge_lines(thick_lines, dist_thresh=15.0, angle_thresh_deg=120)

    thin_lines = trace_skeleton_lines(thin_skel, min_length=15)
    thin_lines = merge_lines(thin_lines, dist_thresh=15.0, angle_thresh_deg=120)

    print(f"  一级边界线段: {len(thick_lines)}")
    print(f"  二级边界线段: {len(thin_lines)}")

    # 生成 LabelMe JSON
    labelme_data = to_labelme_json_2class(thick_lines, thin_lines, f"{stem}.jpg", h, w)
    print(f"  总 shapes: {len(labelme_data['shapes'])}")

    # 复制原图
    dst_img = labelme_dir / f"{stem}.jpg"
    if not dst_img.exists():
        shutil.copy2(str(input_path), str(dst_img))

    # 保存 JSON
    json_path = labelme_dir / f"{stem}.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(labelme_data, f, ensure_ascii=False, indent=2)

    # 可视化
    vis_path = vis_dir / f"{stem}_2class_vis.jpg"
    visualize_two_levels(img, thick_lines, thin_lines, vis_path)
    print(f"  -> {json_path.name}, {vis_path.name}\n")


def main():
    input_dir = Path("/Users/bytedance/Work/qgis_only/map_line_dataset/gcp")
    labelme_dir = Path("/Users/bytedance/Work/qgis_only/boundaries/labelme")
    vis_dir = Path("/Users/bytedance/Work/qgis_only/boundaries/output")
    labelme_dir.mkdir(parents=True, exist_ok=True)
    vis_dir.mkdir(parents=True, exist_ok=True)

    targets = ["05-54淮南道", "07-11岭北行省北部", "08-48云南"]

    for stem in targets:
        input_path = input_dir / f"{stem}.jpg"
        if not input_path.exists():
            print(f"跳过（文件不存在）: {input_path}")
            continue
        process_one(input_path, labelme_dir, vis_dir)

    print("全部完成")


if __name__ == "__main__":
    main()
