"""
分隔线检测独立模块。

提供两种检测方式：
1. OpenCV 形态学（快速，无需模型）
2. 从 UNet mask 提取（精度高，需要推理结果）
   - 骨架化 → HoughLinesP 线段拟合 → 同方向合并 → 跨方向连接 → 边界延伸

用法:
    from detect_splitters import detect_splitters_opencv, detect_splitters_from_mask, partition_regions

    # 仅用 OpenCV
    splitters = detect_splitters_opencv(image_path)

    # 从 mask 提取
    splitters = detect_splitters_from_mask(pred_mask)

    # 划分区域
    regions = partition_regions(h, w, splitters, pred_mask=pred_mask)
"""
import numpy as np
import cv2
from skimage.morphology import skeletonize


# ============================================================
# 方式1: OpenCV 形态学检测
# ============================================================
def detect_splitters_opencv(image_path: str, dark_threshold: int = 80,
                            min_coverage: float = 0.3) -> list:
    """
    用形态学操作检测深色长直线（分隔线）。
    """
    img = cv2.imread(image_path)
    if img is None:
        return []
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape

    _, binary = cv2.threshold(gray, dark_threshold, 255, cv2.THRESH_BINARY_INV)

    results = []

    # 检测水平线
    h_kernel_len = max(w // 5, 100)
    h_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (h_kernel_len, 1))
    h_mask = cv2.morphologyEx(binary, cv2.MORPH_OPEN, h_kernel)

    num_h, labels_h = cv2.connectedComponents(h_mask)
    for lbl in range(1, num_h):
        ys, xs = np.where(labels_h == lbl)
        if len(ys) == 0:
            continue
        y_mean = float(ys.mean())
        x_min, x_max = int(xs.min()), int(xs.max())
        x_span = x_max - x_min
        coverage = x_span / w
        if coverage >= min_coverage and 10 < y_mean < h - 10:
            pts = _sample_line_points_opencv(h_mask, labels_h, lbl, 'horizontal', 20)
            results.append({
                'points': pts,
                'orientation': 'horizontal',
                'coverage': coverage,
            })

    # 检测垂直线
    v_kernel_len = max(h // 5, 100)
    v_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, v_kernel_len))
    v_mask = cv2.morphologyEx(binary, cv2.MORPH_OPEN, v_kernel)

    num_v, labels_v = cv2.connectedComponents(v_mask)
    for lbl in range(1, num_v):
        ys, xs = np.where(labels_v == lbl)
        if len(ys) == 0:
            continue
        x_mean = float(xs.mean())
        y_min, y_max = int(ys.min()), int(ys.max())
        y_span = y_max - y_min
        coverage = y_span / h
        if coverage >= min_coverage and 10 < x_mean < w - 10:
            pts = _sample_line_points_opencv(v_mask, labels_v, lbl, 'vertical', 20)
            results.append({
                'points': pts,
                'orientation': 'vertical',
                'coverage': coverage,
            })

    return results


def _sample_line_points_opencv(mask, labels, lbl, orientation, n_points):
    """沿检测到的线段均匀采样点"""
    ys, xs = np.where(labels == lbl)
    if orientation == 'horizontal':
        x_min, x_max = xs.min(), xs.max()
        sample_xs = np.linspace(x_min, x_max, n_points)
        pts = []
        for sx in sample_xs:
            nearby_y = ys[np.abs(xs - sx) < max(20, (x_max - x_min) / n_points)]
            if len(nearby_y) > 0:
                pts.append([round(float(sx), 1), round(float(nearby_y.mean()), 1)])
        return pts if pts else [[float(x_min), float(ys.mean())], [float(x_max), float(ys.mean())]]
    else:
        y_min, y_max = ys.min(), ys.max()
        sample_ys = np.linspace(y_min, y_max, n_points)
        pts = []
        for sy in sample_ys:
            nearby_x = xs[np.abs(ys - sy) < max(20, (y_max - y_min) / n_points)]
            if len(nearby_x) > 0:
                pts.append([round(float(nearby_x.mean()), 1), round(float(sy), 1)])
        return pts if pts else [[float(xs.mean()), float(y_min)], [float(xs.mean()), float(y_max)]]


# ============================================================
# 方式2: 从 UNet mask 提取（闭合 + 骨架化 + 连通域路径追踪）
# ============================================================

# 形态学闭合 kernel 大小（弥合空隙）
CLOSING_KERNEL_SIZE = 25
# Douglas-Peucker 简化容差（像素）
DP_EPSILON_RATIO = 0.01
# 边界延伸阈值
EDGE_THRESHOLD = 200
# 路径端点合并距离
PATH_MERGE_DIST = 200


def detect_splitters_from_mask(pred_mask: np.ndarray, class_id: int = 3,
                               min_coverage: float = 0.15) -> list:
    """
    从 UNet 推理 mask 中提取分隔线。
    闭合弥合空隙 → 骨架化 → 连通域路径追踪 → 合并近端路径 → DP简化 → 延伸过滤。
    """
    binary = (pred_mask == class_id).astype(np.uint8)
    if binary.sum() < 100:
        return []

    h, w = pred_mask.shape

    # Step 1: 形态学闭合弥合空隙 + 骨架化
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,
                                       (CLOSING_KERNEL_SIZE, CLOSING_KERNEL_SIZE))
    closed = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
    skeleton = skeletonize(closed > 0).astype(np.uint8) * 255

    # Step 2: 连通域路径追踪
    num_labels, labels = cv2.connectedComponents(skeleton, connectivity=8)

    raw_paths = []
    for label_id in range(1, num_labels):
        ys, xs = np.where(labels == label_id)
        if len(ys) < 10:
            continue
        path = _trace_skeleton_path(skeleton, labels, label_id, ys, xs)
        if len(path) >= 10:
            raw_paths.append(path)

    if not raw_paths:
        return []

    # Step 3: 合并端点接近的路径
    merged_paths = _merge_nearby_paths(raw_paths, PATH_MERGE_DIST)

    # Step 4: Douglas-Peucker 简化 + 延伸 + 过滤
    results = []
    for path in merged_paths:
        path_arr = np.array(path, dtype=np.float32).reshape(-1, 1, 2)
        arc_len = cv2.arcLength(path_arr, closed=False)
        epsilon = arc_len * DP_EPSILON_RATIO
        simplified = cv2.approxPolyDP(path_arr, epsilon, closed=False)
        pts = simplified.squeeze()
        if pts.ndim != 2 or len(pts) < 2:
            continue

        # 端点延伸到边界
        pts_list = [pt.astype(float) for pt in pts]
        pts_list[0] = _extend_to_boundary(pts_list[0], pts_list[1], h, w, EDGE_THRESHOLD)
        pts_list[-1] = _extend_to_boundary(pts_list[-1], pts_list[-2], h, w, EDGE_THRESHOLD)

        # 覆盖率过滤
        x_coords = [p[0] for p in pts_list]
        y_coords = [p[1] for p in pts_list]
        x_span = max(x_coords) - min(x_coords)
        y_span = max(y_coords) - min(y_coords)
        coverage = max(x_span / w, y_span / h)

        if coverage < min_coverage:
            continue

        if x_span > y_span * 2:
            orientation = 'horizontal'
        elif y_span > x_span * 2:
            orientation = 'vertical'
        else:
            orientation = 'mixed'

        results.append({
            'points': [[round(float(p[0]), 1), round(float(p[1]), 1)] for p in pts_list],
            'orientation': orientation,
            'coverage': coverage,
        })

    return results


def _merge_nearby_paths(paths, threshold):
    """合并端点接近的路径为连续折线"""
    if len(paths) <= 1:
        return paths

    merged = [list(p) for p in paths]
    changed = True
    while changed:
        changed = False
        new_merged = []
        used = [False] * len(merged)

        for i in range(len(merged)):
            if used[i]:
                continue
            current = merged[i]
            for j in range(i + 1, len(merged)):
                if used[j]:
                    continue
                candidate = merged[j]
                result = _try_join_paths(current, candidate, threshold)
                if result is not None:
                    current = result
                    used[j] = True
                    changed = True
            used[i] = True
            new_merged.append(current)
        merged = new_merged

    return merged


def _try_join_paths(path_a, path_b, threshold):
    """尝试连接两条路径，返回合并后的路径或 None"""
    a_start = np.array(path_a[0])
    a_end = np.array(path_a[-1])
    b_start = np.array(path_b[0])
    b_end = np.array(path_b[-1])

    pairs = [
        (np.linalg.norm(a_end - b_start), 'a_end_b_start'),
        (np.linalg.norm(a_end - b_end), 'a_end_b_end'),
        (np.linalg.norm(a_start - b_start), 'a_start_b_start'),
        (np.linalg.norm(a_start - b_end), 'a_start_b_end'),
    ]
    best_dist, best_type = min(pairs, key=lambda x: x[0])
    if best_dist > threshold:
        return None

    if best_type == 'a_end_b_start':
        return path_a + path_b
    elif best_type == 'a_end_b_end':
        return path_a + path_b[::-1]
    elif best_type == 'a_start_b_start':
        return path_a[::-1] + path_b
    else:  # a_start_b_end
        return path_b + path_a


def _trace_skeleton_path(skeleton, labels, label_id, ys, xs):
    """从骨架连通域提取有序路径：按主轴排序像素点"""
    x_span = xs.max() - xs.min()
    y_span = ys.max() - ys.min()

    if x_span >= y_span:
        # 主要水平方向：按 x 排序
        order = np.argsort(xs)
    else:
        # 主要垂直方向：按 y 排序
        order = np.argsort(ys)

    path = [[int(xs[i]), int(ys[i])] for i in order]
    return path


def _extend_to_boundary(endpoint, interior_point, h, w, threshold):
    """如果端点距离边界近，沿方向延伸到边界"""
    pt = endpoint.copy()
    direction = endpoint - interior_point
    dir_len = np.linalg.norm(direction)
    if dir_len < 1:
        return pt

    direction = direction / dir_len

    dist_left = pt[0]
    dist_right = w - pt[0]
    dist_top = pt[1]
    dist_bottom = h - pt[1]
    min_dist = min(dist_left, dist_right, dist_top, dist_bottom)

    if min_dist > threshold:
        return pt

    # 沿方向延伸到最近的边界
    if min_dist == dist_left and direction[0] < -0.3:
        t = -pt[0] / direction[0] if abs(direction[0]) > 0.01 else 0
        pt = endpoint + direction * t
        pt[0] = max(0, pt[0])
    elif min_dist == dist_right and direction[0] > 0.3:
        t = (w - pt[0]) / direction[0] if abs(direction[0]) > 0.01 else 0
        pt = endpoint + direction * t
        pt[0] = min(w, pt[0])
    elif min_dist == dist_top and direction[1] < -0.3:
        t = -pt[1] / direction[1] if abs(direction[1]) > 0.01 else 0
        pt = endpoint + direction * t
        pt[1] = max(0, pt[1])
    elif min_dist == dist_bottom and direction[1] > 0.3:
        t = (h - pt[1]) / direction[1] if abs(direction[1]) > 0.01 else 0
        pt = endpoint + direction * t
        pt[1] = min(h, pt[1])

    pt[0] = np.clip(pt[0], 0, w)
    pt[1] = np.clip(pt[1], 0, h)
    return pt


# ============================================================
# 区域划分（基于连通区域分析）
# ============================================================
def partition_regions(h: int, w: int, splitters: list,
                     pred_mask: np.ndarray = None) -> list:
    """
    用分隔线作为墙壁，通过连通区域分析找到各独立区域。

    Args:
        h, w: 图片尺寸
        splitters: 检测到的分隔线列表
        pred_mask: 可选，UNet mask，用 class=3 像素补充墙壁

    Returns:
        [{'bbox': (y_min, y_max, x_min, x_max), 'area': int, 'group_id': int,
          'region_mask': np.ndarray}, ...]
        group_id=0 为最大区域（主图），region_mask 是全图尺寸的布尔掩码
    """
    if not splitters:
        return [{'bbox': (0, h, 0, w), 'area': h * w, 'group_id': 0,
                 'region_mask': np.ones((h, w), dtype=np.uint8)}]

    # 缩小图加速
    scale = 4
    small_h, small_w = h // scale, w // scale
    wall = np.zeros((small_h, small_w), dtype=np.uint8)

    # 画分隔线墙壁
    for sp in splitters:
        pts = np.array(sp['points'], dtype=np.float64)
        pts_scaled = (pts / scale).astype(np.int32)
        for i in range(len(pts_scaled) - 1):
            cv2.line(wall, tuple(pts_scaled[i]), tuple(pts_scaled[i + 1]), 255, thickness=3)

    # 用 pred_mask 中 class=3 的像素补充
    if pred_mask is not None:
        splitter_binary = (pred_mask == 3).astype(np.uint8)
        small_splitter = cv2.resize(splitter_binary, (small_w, small_h),
                                    interpolation=cv2.INTER_NEAREST)
        wall = np.maximum(wall, small_splitter * 255)

    # 膨胀确保封闭
    kernel = np.ones((5, 5), dtype=np.uint8)
    wall = cv2.dilate(wall, kernel, iterations=1)

    # 连通区域分析
    free_space = (wall == 0).astype(np.uint8)
    num_labels, labels = cv2.connectedComponents(free_space, connectivity=4)

    # 提取每个区域的 bounding box + 全图区域掩码
    regions = []
    for label_id in range(1, num_labels):
        ys, xs = np.where(labels == label_id)
        if len(ys) == 0:
            continue
        y_min = int(ys.min()) * scale
        y_max = min(int(ys.max() + 1) * scale, h)
        x_min = int(xs.min()) * scale
        x_max = min(int(xs.max() + 1) * scale, w)
        area = (y_max - y_min) * (x_max - x_min)
        if area < h * w * 0.02:
            continue

        # 将小图 label 放大到全图尺寸作为区域掩码
        small_mask = (labels == label_id).astype(np.uint8)
        region_mask = cv2.resize(small_mask, (w, h), interpolation=cv2.INTER_NEAREST)

        regions.append({
            'bbox': (y_min, y_max, x_min, x_max),
            'area': area,
            'region_mask': region_mask,
        })

    if not regions:
        return [{'bbox': (0, h, 0, w), 'area': h * w, 'group_id': 0,
                 'region_mask': np.ones((h, w), dtype=np.uint8)}]

    regions.sort(key=lambda r: r['area'], reverse=True)
    for idx, region in enumerate(regions):
        region['group_id'] = idx

    return regions

