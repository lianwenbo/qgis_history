"""
将骨架线转为 LabelMe JSON 标注（linestrip 格式）
对骨架做连通域追踪，每条线段生成一个 shape
"""

import cv2
import numpy as np
import json
from pathlib import Path
from skimage.morphology import skeletonize
from extract_boundaries import extract_band_roi


def bridge_gaps(skeleton, max_gap=20):
    """断裂弥合"""
    skel_bin = (skeleton > 0).astype(np.uint8)
    kernel = np.ones((3, 3), np.uint8)
    neighbor_count = cv2.filter2D(skel_bin, -1, kernel)

    endpoint_mask = (skel_bin == 1) & (neighbor_count == 2)
    endpoint_pts = np.column_stack(np.where(endpoint_mask))

    bridged = skeleton.copy()

    for ey, ex in endpoint_pts:
        direction = _get_direction(skel_bin, ey, ex, trace_len=8)
        if direction is None:
            continue
        dy, dx = direction
        for dist in range(2, max_gap + 1):
            ny = int(ey + dy * dist)
            nx = int(ex + dx * dist)
            if ny < 0 or ny >= skeleton.shape[0] or nx < 0 or nx >= skeleton.shape[1]:
                break
            found = False
            for offset in range(-2, 3):
                sy = int(ny + offset * (-dx))
                sx = int(nx + offset * dy)
                if 0 <= sy < skeleton.shape[0] and 0 <= sx < skeleton.shape[1]:
                    if skel_bin[sy, sx] == 1 and not endpoint_mask[sy, sx]:
                        cv2.line(bridged, (ex, ey), (sx, sy), 255, 1)
                        found = True
                        break
            if found:
                break

    return bridged


def _get_direction(skel_bin, start_y, start_x, trace_len=8):
    """从端点回溯计算延伸方向"""
    h, w = skel_bin.shape
    visited = set()
    path = [(start_y, start_x)]
    visited.add((start_y, start_x))

    cy, cx = start_y, start_x
    for _ in range(trace_len):
        found_next = False
        for dy in [-1, 0, 1]:
            for dx in [-1, 0, 1]:
                if dy == 0 and dx == 0:
                    continue
                ny, nx = cy + dy, cx + dx
                if 0 <= ny < h and 0 <= nx < w:
                    if skel_bin[ny, nx] == 1 and (ny, nx) not in visited:
                        visited.add((ny, nx))
                        path.append((ny, nx))
                        cy, cx = ny, nx
                        found_next = True
                        break
            if found_next:
                break
        if not found_next:
            break

    if len(path) < 3:
        return None
    tail_y, tail_x = path[-1]
    dy = start_y - tail_y
    dx = start_x - tail_x
    length = np.sqrt(dy * dy + dx * dx)
    if length < 1e-6:
        return None
    return (dy / length, dx / length)


def trace_skeleton_lines(skeleton, min_length=10):
    """
    追踪骨架线上的每条独立线段。
    策略：移除交点像素将骨架打断为独立片段，对每个片段追踪为一条线。
    返回线段列表，每条线段是 [(x,y), ...] 的点序列。
    """
    skel_bin = (skeleton > 0).astype(np.uint8)
    h, w = skel_bin.shape

    # 计算邻域数
    kernel = np.ones((3, 3), np.uint8)
    neighbor_count = cv2.filter2D(skel_bin, -1, kernel)

    # 交点像素（邻域>=4）：移除以打断线段
    junction_mask = (skel_bin == 1) & (neighbor_count >= 4)

    # 移除交点后的骨架片段
    fragments = skel_bin.copy()
    fragments[junction_mask] = 0

    # 对片段做连通域分析
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(fragments, connectivity=8)

    lines = []
    for i in range(1, num_labels):
        area = stats[i, cv2.CC_STAT_AREA]
        if area < min_length:
            continue

        # 提取该片段的所有像素
        frag_pts = np.column_stack(np.where(labels == i))  # (y, x)

        # 对片段内像素进行有序追踪
        path = _order_fragment(frag_pts, fragments)
        if len(path) >= min_length:
            lines.append([(int(x), int(y)) for y, x in path])

    return lines


def _order_fragment(points, skel_img):
    """对一个线段片段的像素排序为有序路径"""
    if len(points) <= 2:
        return points

    h, w = skel_img.shape
    # 建立像素集合用于快速查找
    pt_set = set(map(tuple, points))

    # 找端点（只有1个邻居在片段内的像素）
    endpoints = []
    for y, x in points:
        neighbors = 0
        for dy in [-1, 0, 1]:
            for dx in [-1, 0, 1]:
                if dy == 0 and dx == 0:
                    continue
                if (y + dy, x + dx) in pt_set:
                    neighbors += 1
        if neighbors <= 1:
            endpoints.append((y, x))

    # 从端点开始追踪；没有端点则是环，从任意点开始
    start = endpoints[0] if endpoints else tuple(points[0])

    path = [start]
    visited = {start}
    cy, cx = start
    max_steps = len(pt_set) * 2 + 100
    steps = 0
    while steps < max_steps:
        steps += 1
        found = False
        for dy in [-1, 0, 1]:
            for dx in [-1, 0, 1]:
                if dy == 0 and dx == 0:
                    continue
                ny, nx = cy + dy, cx + dx
                if (ny, nx) in pt_set and (ny, nx) not in visited:
                    visited.add((ny, nx))
                    path.append((ny, nx))
                    cy, cx = ny, nx
                    found = True
                    break
        if not found:
            break

    return path


def simplify_line(points, epsilon=2.0):
    """Douglas-Peucker 简化线段点数"""
    if len(points) < 3:
        return points
    pts = np.array(points, dtype=np.float32).reshape(-1, 1, 2)
    simplified = cv2.approxPolyDP(pts, epsilon, False)
    return simplified.reshape(-1, 2).tolist()


def _endpoint_direction(points, end_type, look_back=3):
    """计算线段端点处的方向向量（归一化）"""
    if len(points) < 2:
        return None
    if end_type == "start":
        seg = points[:min(look_back + 1, len(points))]
        dx = seg[0][0] - seg[-1][0]
        dy = seg[0][1] - seg[-1][1]
    else:
        seg = points[max(0, len(points) - look_back - 1):]
        dx = seg[-1][0] - seg[0][0]
        dy = seg[-1][1] - seg[0][1]
    length = np.sqrt(dx * dx + dy * dy)
    if length < 1e-6:
        return None
    return (dx / length, dy / length)


def _build_endpoint_tree(merged, used_mask):
    """
    构建当前所有未使用线段的 4 端点空间索引（start/end × 每个线段）。
    返回 (tree, meta)：meta[i] = (line_idx, end_type) 对应该树第 i 个点。
    """
    pts = []
    meta = []
    for idx in range(len(merged)):
        if used_mask[idx]:
            continue
        if len(merged[idx]) < 2:
            continue
        for end_type in ("start", "end"):
            pt = merged[idx][0] if end_type == "start" else merged[idx][-1]
            pts.append((float(pt[0]), float(pt[1])))
            meta.append((idx, end_type))
    if not pts:
        return None, None
    from scipy.spatial import cKDTree
    return cKDTree(np.array(pts, dtype=np.float64)), meta


def merge_lines(lines, dist_thresh=5.0, angle_thresh_deg=90, _max_iter=None):
    """
    贪心链式合并：将端点距离 < dist_thresh 且方向连续（角度变化 < angle_thresh）的线段合并。
    优化：用 cKDTree 做端点空间索引，O(N³) → O(N log N)；同时加迭代上限防类死循环。
    """
    if not lines:
        return lines

    cos_thresh = np.cos(np.radians(angle_thresh_deg))
    merged = [list(pts) for pts in lines]
    n = len(merged)
    used = [False] * n

    if _max_iter is None:
        _max_iter = max(50, n * 3)

    def get_endpoint(idx, end):
        if end == "start":
            return merged[idx][0]
        return merged[idx][-1]

    iter_count = 0
    changed = True
    while changed:
        changed = False
        iter_count += 1
        if iter_count > _max_iter:
            break

        tree, meta = _build_endpoint_tree(merged, used)
        if tree is None:
            break

        for i in range(n):
            if used[i] or len(merged[i]) < 2:
                continue
            for i_end in ("start", "end"):
                pt_i = get_endpoint(i, i_end)
                dir_i = _endpoint_direction(merged[i], i_end)

                # 查距离内所有候选端点（最多 20 个，足够近邻）
                try:
                    cand_idx = tree.query_ball_point(
                        [float(pt_i[0]), float(pt_i[1])], dist_thresh,
                    )
                except Exception:
                    cand_idx = []

                best_j = -1
                best_j_end = None
                best_dist = dist_thresh

                for c in cand_idx:
                    j_idx, j_end = meta[c]
                    if j_idx == i or used[j_idx]:
                        continue
                    pt_j = get_endpoint(j_idx, j_end)
                    dx = pt_i[0] - pt_j[0]
                    dy = pt_i[1] - pt_j[1]
                    dist = (dx * dx + dy * dy) ** 0.5
                    if dist >= best_dist:
                        continue
                    dir_j = _endpoint_direction(merged[j_idx], j_end)
                    if dir_i is not None and dir_j is not None:
                        dot = dir_i[0] * dir_j[0] + dir_i[1] * dir_j[1]
                        if dot > -cos_thresh:
                            continue
                    best_j = j_idx
                    best_j_end = j_end
                    best_dist = dist

                if best_j >= 0:
                    line_j = merged[best_j]
                    if best_j_end == "end":
                        line_j = list(reversed(line_j))
                    if i_end == "end":
                        merged[i] = merged[i] + line_j
                    else:
                        merged[i] = list(reversed(line_j)) + merged[i]
                    used[best_j] = True
                    changed = True
                    break

    return [merged[i] for i in range(n) if (not used[i]) and len(merged[i]) >= 2]


def to_labelme_json(lines, image_path, img_height, img_width):
    """生成 LabelMe JSON"""
    shapes = []
    for line_pts in lines:
        # 简化点数
        simplified = simplify_line(line_pts, epsilon=2.0)
        if len(simplified) < 2:
            continue
        shapes.append({
            "label": "boundary",
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


def main():
    import shutil

    input_dir = Path("/Users/bytedance/Work/qgis_only/map_line_dataset/gcp")
    labelme_dir = Path("/Users/bytedance/Work/qgis_only/boundaries/labelme")
    labelme_dir.mkdir(parents=True, exist_ok=True)

    targets = ["05-54淮南道", "07-11岭北行省北部", "08-48云南"]

    for stem in targets:
        input_path = input_dir / f"{stem}.jpg"
        print(f"处理: {stem}")

        img = cv2.imread(str(input_path))
        h, w = img.shape[:2]

        # ROI → 骨架化 → 弥合
        roi_mask = extract_band_roi(img)
        skeleton = skeletonize((roi_mask > 0).astype(np.uint8)).astype(np.uint8) * 255
        bridged = bridge_gaps(skeleton, max_gap=20)
        bridged = skeletonize((bridged > 0).astype(np.uint8)).astype(np.uint8) * 255

        # 追踪线段
        lines = trace_skeleton_lines(bridged, min_length=15)
        print(f"  原始线段: {len(lines)}")

        # 合并首尾相接的线段
        lines = merge_lines(lines, dist_thresh=5.0, angle_thresh_deg=90)
        print(f"  合并后: {len(lines)}")

        # 生成 LabelMe JSON
        labelme_data = to_labelme_json(lines, f"{stem}.jpg", h, w)
        print(f"  shapes: {len(labelme_data['shapes'])}")

        # 复制原图到 labelme 目录
        dst_img = labelme_dir / f"{stem}.jpg"
        if not dst_img.exists():
            shutil.copy2(str(input_path), str(dst_img))

        # 保存 JSON
        json_path = labelme_dir / f"{stem}.json"
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(labelme_data, f, ensure_ascii=False, indent=2)

        print(f"  -> {json_path.name}\n")

    print("全部完成")


if __name__ == "__main__":
    main()
