"""
行政边界提取脚本
Step 1: 提取带状边界 ROI（将后续所有边界查找限制在此范围内）
"""

import cv2
import numpy as np
from pathlib import Path


def extract_band_roi(img_bgr, a_threshold=10, dilate_size=5, close_size=9, min_area=500):
    """
    提取带状边界 ROI：所有洋红/紫色系像素区域的膨胀闭合结果。
    后续边界线查找全部限制在此 ROI 内。

    参数：
        a_threshold: Lab a* 通道阈值（>threshold 视为洋红/紫色）
        dilate_size: 膨胀核大小（连接虚线间隙）
        close_size: 闭合核大小（填充带内空洞）
        min_area: 连通域最小面积（过滤噪点）
    """
    lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2Lab)
    a_ch = lab[:, :, 1].astype(float) - 128
    l_ch = lab[:, :, 0].astype(float)

    # 提取所有洋红/紫色像素
    pink_mask = (
        (a_ch > a_threshold) &
        (l_ch > 60) & (l_ch < 230)
    ).astype(np.uint8) * 255

    # 开运算去除孤立噪点
    kernel_open = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2, 2))
    pink_mask = cv2.morphologyEx(pink_mask, cv2.MORPH_OPEN, kernel_open)

    # 膨胀：连接虚线间隙，使带状区域连续
    kernel_dilate = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (dilate_size, dilate_size))
    band = cv2.dilate(pink_mask, kernel_dilate)

    # 闭合：填充带内小空洞
    kernel_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (close_size, close_size))
    band = cv2.morphologyEx(band, cv2.MORPH_CLOSE, kernel_close)

    # 连通域过滤：去除面积过小的碎片
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(band, connectivity=8)
    roi_mask = np.zeros_like(band)
    for i in range(1, num_labels):
        if stats[i, cv2.CC_STAT_AREA] > min_area:
            roi_mask[labels == i] = 255

    return roi_mask


def main():
    input_dir = Path("/Users/bytedance/Work/qgis_only/map_line_dataset/gcp")
    output_dir = Path("/Users/bytedance/Work/qgis_only/boundaries/output")
    output_dir.mkdir(parents=True, exist_ok=True)

    jpg_files = sorted(input_dir.glob("*.jpg"))
    jpg_files = [f for f in jpg_files if "_infer" not in f.stem]
    print(f"共 {len(jpg_files)} 张地图\n")

    for input_path in jpg_files:
        stem = input_path.stem
        print(f"处理: {stem}")
        img = cv2.imread(str(input_path))
        h, w = img.shape[:2]
        print(f"  尺寸: {w}x{h}")

        roi_mask = extract_band_roi(img)
        roi_pixels = np.count_nonzero(roi_mask)
        print(f"  ROI 像素: {roi_pixels} ({roi_pixels / roi_mask.size * 100:.2f}%)")

        # 保存 ROI mask
        cv2.imwrite(str(output_dir / f"{stem}_roi.png"), roi_mask)

        # 保存可视化
        vis = img.copy()
        overlay = np.zeros_like(vis)
        overlay[roi_mask > 0] = (200, 100, 255)
        vis = cv2.addWeighted(vis, 1.0, overlay, 0.4, 0)
        cv2.imwrite(str(output_dir / f"{stem}_roi_vis.png"), vis)

        print(f"  -> {stem}_roi.png, {stem}_roi_vis.png\n")

    print("全部完成")


if __name__ == "__main__":
    main()
