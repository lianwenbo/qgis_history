"""
从 UNet 推理 mask 直接拟合经纬线，生成高质量 labelme JSON。

核心思路：跳过骨架化，用逐行/逐列扫描 + 线性拟合：
1. 逐行扫描经线 mask，在每个 y 聚类出各条经线的 x 坐标
2. 对每条经线用 RANSAC 拟合直线 x = a*y + b
3. 逐列扫描纬线 mask，在每个 x 聚类出各条纬线的 y 坐标
4. 对每条纬线用多点采样拟合折线
5. 输出 labelme JSON

用法:
    python generate_labelme_from_mask.py --image <图片> [--model <模型路径>] [--output <输出JSON>]
"""
import sys
import json
import argparse
import numpy as np
import cv2
from pathlib import Path
from scipy.cluster.hierarchy import fclusterdata
from scipy.interpolate import interp1d

sys.path.insert(0, str(Path(__file__).parent))
from unet_model import UNet, _build_weight_map
from detect_splitters import detect_splitters_from_mask, partition_regions
import torch


# ============================================================
# 配置
# ============================================================
DEFAULT_MODEL = Path.home() / "Downloads/unet_map_lines_autodl_colab_20260727_234942.pth"
PATCH_SIZE = 512
OVERLAP = 0.5
FILL_COLOR = (245, 235, 210)

# 线条检测参数
V_CLUSTER_GAP = 30       # 经线 x 聚类间距 (px)
H_CLUSTER_GAP = 30       # 纬线 y 聚类间距 (px)
MIN_LINE_COVERAGE = 0.25 # 经线最少需要覆盖图高的比例
MIN_ARC_COVERAGE = 0.25  # 纬线最少需要覆盖图宽的比例
SAMPLE_STEP = 10         # 行/列扫描步长
RANSAC_THRESHOLD = 5     # RANSAC 内点阈值 (px)
ARC_SAMPLE_POINTS = 30   # 纬线输出采样点数


# ============================================================
# Step 1: 推理
# ============================================================
def run_inference(image_path: str, model_path: str = None) -> np.ndarray:
    """滑窗推理，返回 argmax mask (H, W)"""
    if model_path is None:
        model_path = str(DEFAULT_MODEL)

    device = torch.device('mps' if torch.backends.mps.is_available() else
                          'cuda' if torch.cuda.is_available() else 'cpu')
    model = UNet(n_channels=3, n_classes=4).to(device)
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    img = cv2.imread(image_path)
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

    total = n_rows * n_cols
    print(f"  滑窗推理: {n_rows}x{n_cols}={total} patches")

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

    weight_safe = np.maximum(weight_accum, 1e-6)
    prob_avg = prob_accum / weight_safe[:, :, np.newaxis]
    pred_mask = np.argmax(prob_avg, axis=2).astype(np.uint8)
    pred_mask = pred_mask[:original_h, :original_w]

    return pred_mask


# ============================================================
# Step 2: 从 mask 逐行扫描拟合经线
# ============================================================
def fit_vertical_lines(mask: np.ndarray, class_id: int = 1) -> list:
    """
    从 mask 拟合经线。
    逐行扫描 → 聚类 → 每条线积累 (y, x) 点对 → RANSAC 拟合直线。

    Returns:
        [{'points': [[x_top, y_top], [x_bot, y_bot]], 'coverage': float}, ...]
    """
    h, w = mask.shape
    binary = (mask == class_id).astype(np.uint8)

    # 逐行扫描，收集每行的经线 x 坐标
    # 用步长加速
    row_samples = range(0, h, SAMPLE_STEP)
    all_observations = []  # [(y, x), ...]

    for y in row_samples:
        row = binary[y, :]
        x_positions = np.where(row > 0)[0]
        if len(x_positions) == 0:
            continue
        for x in x_positions:
            all_observations.append((y, x))

    if len(all_observations) < 10:
        return []

    obs_arr = np.array(all_observations)  # (N, 2), columns: y, x

    # 按 x 方向聚类，分离不同经线
    # 多行投票：在多个参考行各自独立聚类，跨行按顺序匹配合并
    ref_ys = [int(h * r) for r in [0.1, 0.25, 0.4, 0.5, 0.6, 0.75, 0.9]]
    row_line_sets = []  # 每行检测到的线中心列表

    for ref_y in ref_ys:
        ref_row = binary[ref_y, :]
        x_pos = np.where(ref_row > 0)[0]
        if len(x_pos) < 1:
            row_line_sets.append([])
            continue
        if len(x_pos) == 1:
            row_line_sets.append([float(x_pos[0])])
        else:
            clusters = fclusterdata(x_pos.reshape(-1, 1), t=V_CLUSTER_GAP, criterion='distance')
            centers = []
            for cl in sorted(set(clusters)):
                centers.append(float(x_pos[clusters == cl].mean()))
            centers.sort()
            row_line_sets.append(centers)

    # 选检测到最多线的那行作为主参考
    best_row_idx = max(range(len(row_line_sets)), key=lambda i: len(row_line_sets[i]))
    line_centers = list(row_line_sets[best_row_idx])
    best_ref_y = ref_ys[best_row_idx]

    # 用等间距规律补全边缘缺失的线：
    # 计算主参考行中相邻线的间距中位数
    if len(line_centers) >= 3:
        spacings = [line_centers[i+1] - line_centers[i] for i in range(len(line_centers)-1)]
        typical_spacing = float(np.median(spacings))

        # 向左补全：如果最左线左侧还有空间
        while line_centers[0] - typical_spacing > -typical_spacing * 0.3:
            new_x = line_centers[0] - typical_spacing
            # 验证：在其他行的该位置附近是否有像素支持
            has_support = False
            for row_centers in row_line_sets:
                for cx in row_centers:
                    if abs(cx - new_x) < typical_spacing * 0.4:
                        has_support = True
                        break
                if has_support:
                    break
            if has_support:
                line_centers.insert(0, new_x)
            else:
                break

        # 向右补全
        while line_centers[-1] + typical_spacing < w + typical_spacing * 0.3:
            new_x = line_centers[-1] + typical_spacing
            has_support = False
            for row_centers in row_line_sets:
                for cx in row_centers:
                    if abs(cx - new_x) < typical_spacing * 0.4:
                        has_support = True
                        break
                if has_support:
                    break
            if has_support:
                line_centers.append(new_x)
            else:
                break

    line_centers.sort()

    if len(line_centers) < 2:
        return []

    # 将所有观测点分配到最近的线
    # 经线倾斜容忍：用相邻线间距的一半作为阈值
    if len(line_centers) >= 2:
        spacings = [line_centers[i+1] - line_centers[i] for i in range(len(line_centers)-1)]
        max_tilt = max(150, int(min(spacings) * 0.6))
    else:
        max_tilt = 150

    # 两阶段分配：
    # 阶段1：只用参考行附近的观测（倾斜小，分配无歧义）做初始 RANSAC
    # 阶段2：用拟合线方程对全图观测重新分配
    band_half = h // 4  # 参考行上下各 1/4 高度的窄带
    band_obs = [(y, x) for y, x in all_observations
                if abs(y - best_ref_y) < band_half]

    # 阶段1：窄带分配 + RANSAC
    line_points_band = {i: [] for i in range(len(line_centers))}
    for y, x in band_obs:
        dists = [abs(x - c) for c in line_centers]
        min_dist = min(dists)
        if min_dist < max_tilt:
            nearest = dists.index(min_dist)
            line_points_band[nearest].append((y, x))

    line_equations = {}  # i -> (a, b) where x = a*y + b
    for i, pts in line_points_band.items():
        if len(pts) < 10:
            line_equations[i] = (0.0, line_centers[i])
            continue
        pts_arr = np.array(pts)
        ys_l, xs_l = pts_arr[:, 0].astype(float), pts_arr[:, 1].astype(float)
        best_inliers = 0
        best_a, best_b = 0.0, line_centers[i]
        for _ in range(30):
            idx = np.random.choice(len(ys_l), 2, replace=False)
            y1, x1 = ys_l[idx[0]], xs_l[idx[0]]
            y2, x2 = ys_l[idx[1]], xs_l[idx[1]]
            if abs(y2 - y1) < 5:
                continue
            a = (x2 - x1) / (y2 - y1)
            b = x1 - a * y1
            predicted_x = a * ys_l + b
            inliers = (np.abs(xs_l - predicted_x) < RANSAC_THRESHOLD).sum()
            if inliers > best_inliers:
                best_inliers = inliers
                best_a, best_b = a, b
        line_equations[i] = (best_a, best_b)

    # 阶段2：用拟合线方程对全图观测重新分配
    valid_indices = sorted(line_equations.keys())
    line_points = {i: [] for i in valid_indices}
    for y, x in all_observations:
        best_dist = max_tilt
        best_idx = -1
        for i in valid_indices:
            a, b = line_equations[i]
            predicted_x = a * y + b
            dist = abs(x - predicted_x)
            if dist < best_dist:
                best_dist = dist
                best_idx = i
        if best_idx >= 0:
            line_points[best_idx].append((y, x))

    # 对每条线做 RANSAC 拟合
    results = []
    for i, pts in line_points.items():
        if len(pts) < 20:
            continue

        pts_arr = np.array(pts)  # (N, 2): y, x
        ys = pts_arr[:, 0]
        xs = pts_arr[:, 1]

        # RANSAC: 拟合 x = a*y + b
        best_inliers = 0
        best_a, best_b = 0, line_centers[i]

        for _ in range(50):
            # 随机取2个点
            idx = np.random.choice(len(ys), 2, replace=False)
            y1, x1 = ys[idx[0]], xs[idx[0]]
            y2, x2 = ys[idx[1]], xs[idx[1]]
            if abs(y2 - y1) < 10:
                continue
            a = (x2 - x1) / (y2 - y1)
            b = x1 - a * y1

            # 计算内点
            predicted_x = a * ys + b
            errors = np.abs(xs - predicted_x)
            inliers = (errors < RANSAC_THRESHOLD).sum()

            if inliers > best_inliers:
                best_inliers = inliers
                best_a, best_b = a, b

        # 用所有内点重新拟合
        predicted_x = best_a * ys + best_b
        inlier_mask = np.abs(xs - predicted_x) < RANSAC_THRESHOLD
        if inlier_mask.sum() >= 10:
            ys_in = ys[inlier_mask]
            xs_in = xs[inlier_mask]
            # 最小二乘
            coeffs = np.polyfit(ys_in, xs_in, 1)
            best_a, best_b = coeffs[0], coeffs[1]

        # 计算覆盖范围
        y_min = int(ys.min())
        y_max = int(ys.max())
        coverage = (y_max - y_min) / h

        if coverage >= MIN_LINE_COVERAGE:
            x_top = best_a * y_min + best_b
            x_bot = best_a * y_max + best_b
            results.append({
                'points': [[round(x_top, 1), y_min], [round(x_bot, 1), y_max]],
                'coverage': coverage,
                'n_inliers': int(inlier_mask.sum()),
            })

    # 按 x 排序
    results.sort(key=lambda r: (r['points'][0][0] + r['points'][1][0]) / 2)

    # 去重：合并 x_mid 过近的相邻线条（由倾斜导致的分配重叠）
    if len(results) >= 3:
        mids = [(r['points'][0][0] + r['points'][1][0]) / 2 for r in results]
        all_spacings = [mids[i+1] - mids[i] for i in range(len(mids)-1)]
        typical_sp = float(np.median(all_spacings))
        merge_thr = typical_sp * 0.3
        deduped = []
        for r in results:
            r_mid = (r['points'][0][0] + r['points'][1][0]) / 2
            if deduped:
                prev_mid = (deduped[-1]['points'][0][0] + deduped[-1]['points'][1][0]) / 2
                if r_mid - prev_mid < merge_thr:
                    if r['n_inliers'] > deduped[-1]['n_inliers']:
                        deduped[-1] = r
                    continue
            deduped.append(r)
        results = deduped

    return results


# ============================================================
# Step 3: 从 mask 逐列扫描拟合纬线
# ============================================================
def fit_horizontal_arcs(mask: np.ndarray, class_id: int = 2) -> list:
    """
    从 mask 拟合纬线。
    逐列扫描 → 聚类 → 每条线积累 (x, y) 点对 → 均匀采样输出折线。

    Returns:
        [{'points': [[x1,y1], [x2,y2], ...], 'coverage': float}, ...]
    """
    h, w = mask.shape
    binary = (mask == class_id).astype(np.uint8)

    # 逐列扫描
    col_samples = range(0, w, SAMPLE_STEP)
    all_observations = []

    for x in col_samples:
        col = binary[:, x]
        y_positions = np.where(col > 0)[0]
        if len(y_positions) == 0:
            continue
        for y in y_positions:
            all_observations.append((x, y))

    if len(all_observations) < 10:
        return []

    obs_arr = np.array(all_observations)  # (N, 2): x, y

    # 用中间列的 y 分布做聚类参考
    mid_x = w // 2
    mid_col = binary[:, mid_x]
    mid_y_pos = np.where(mid_col > 0)[0]

    if len(mid_y_pos) < 2:
        for try_x in range(w // 4, 3 * w // 4, w // 10):
            mid_col = binary[:, try_x]
            mid_y_pos = np.where(mid_col > 0)[0]
            if len(mid_y_pos) >= 2:
                mid_x = try_x
                break

    if len(mid_y_pos) < 2:
        return []

    mid_clusters = fclusterdata(mid_y_pos.reshape(-1, 1), t=H_CLUSTER_GAP, criterion='distance')
    line_centers = []
    for cl in sorted(set(mid_clusters)):
        line_centers.append(mid_y_pos[mid_clusters == cl].mean())
    line_centers.sort()

    # 分配观测点到各纬线
    line_points = {i: [] for i in range(len(line_centers))}
    for x, y in all_observations:
        dists = [abs(y - c) for c in line_centers]
        min_dist = min(dists)
        if min_dist < 80:
            nearest = dists.index(min_dist)
            line_points[nearest].append((x, y))

    # 对每条纬线做均匀采样拟合
    results = []
    for i, pts in line_points.items():
        if len(pts) < 20:
            continue

        pts_arr = np.array(pts)  # (N, 2): x, y
        xs = pts_arr[:, 0]
        ys = pts_arr[:, 1]

        x_min = xs.min()
        x_max = xs.max()
        coverage = (x_max - x_min) / w

        if coverage < MIN_ARC_COVERAGE:
            continue

        # 均匀采样 x，对每个 x 取 y 的中值（抗噪）
        sample_xs = np.linspace(x_min, x_max, ARC_SAMPLE_POINTS)
        sample_points = []

        for sx in sample_xs:
            nearby = pts_arr[np.abs(xs - sx) < max(20, (x_max - x_min) / ARC_SAMPLE_POINTS)]
            if len(nearby) > 0:
                y_median = np.median(nearby[:, 1])
                sample_points.append([round(float(sx), 1), round(float(y_median), 1)])

        if len(sample_points) >= 5:
            # 用二次多项式拟合弧线
            sp_xs = np.array([p[0] for p in sample_points])
            sp_ys = np.array([p[1] for p in sample_points])
            coeffs = np.polyfit(sp_xs, sp_ys, 2)
            poly_func = np.poly1d(coeffs)

            # 沿曲线向两端搜索 mask 实际像素支持的范围
            search_margin = 15  # 曲线上下多少像素内算有支持
            step = SAMPLE_STEP

            # 向左搜索
            real_x_min = int(x_min)
            for sx in range(int(x_min) - step, -1, -step):
                ey = int(round(poly_func(sx)))
                if 0 <= ey < h:
                    region = binary[max(0, ey-search_margin):min(h, ey+search_margin+1), sx]
                    if region.sum() > 0:
                        real_x_min = sx
                    else:
                        break
                else:
                    break

            # 向右搜索
            real_x_max = int(x_max)
            for sx in range(int(x_max) + step, w, step):
                ey = int(round(poly_func(sx)))
                if 0 <= ey < h:
                    region = binary[max(0, ey-search_margin):min(h, ey+search_margin+1), sx]
                    if region.sum() > 0:
                        real_x_max = sx
                    else:
                        break
                else:
                    break

            # 在实际支持范围内均匀采样
            extended_xs = np.linspace(real_x_min, real_x_max, ARC_SAMPLE_POINTS)
            extended_points = []
            for ex in extended_xs:
                ey = float(poly_func(ex))
                ey = max(0, min(h - 1, ey))
                extended_points.append([round(ex, 1), round(ey, 1)])

            results.append({
                'points': extended_points,
                'coverage': (real_x_max - real_x_min) / w,
                'n_points': len(pts),
            })

    # 按 y 排序
    results.sort(key=lambda r: np.mean([p[1] for p in r['points']]))
    return results


# ============================================================
# Step 4 & 5: 分隔线检测和区域划分（委托给 detect_splitters 模块）
# ============================================================
def fit_splitter_lines(mask: np.ndarray, class_id: int = 3) -> list:
    """从 mask 检测分隔线"""
    return detect_splitters_from_mask(mask, class_id=class_id)


def partition_by_splitters(mask_shape: tuple, splitters: list,
                           pred_mask: np.ndarray = None) -> list:
    """根据分隔线划分区域"""
    h, w = mask_shape
    return partition_regions(h, w, splitters, pred_mask=pred_mask)


# ============================================================
# Step 6: 生成 labelme JSON
# ============================================================
def generate_labelme_json(image_path: str, region_results: list,
                          splitters: list, output_path: str):
    """
    生成 labelme 格式的 JSON。
    region_results: [{'group_id': int, 'v_lines': [...], 'h_arcs': [...]}, ...]
    """
    img = cv2.imread(image_path)
    h, w = img.shape[:2]

    shapes = []

    for region in region_results:
        gid = region['group_id']

        # 经线 → line
        for v in region['v_lines']:
            shapes.append({
                "label": "vertical_line",
                "points": v['points'],
                "group_id": gid,
                "description": f"coverage={v['coverage']:.2f}",
                "shape_type": "line",
                "flags": {},
                "mask": None
            })

        # 纬线 → linestrip
        for arc in region['h_arcs']:
            shapes.append({
                "label": "horizontal_arc",
                "points": arc['points'],
                "group_id": gid,
                "description": f"coverage={arc['coverage']:.2f}",
                "shape_type": "linestrip",
                "flags": {},
                "mask": None
            })

    # 分隔线 → linestrip
    for sp in splitters:
        shapes.append({
            "label": "splitter",
            "points": sp['points'],
            "group_id": None,
            "description": f"orientation={sp.get('orientation', 'unknown')}",
            "shape_type": "linestrip",
            "flags": {},
            "mask": None
        })

    labelme_data = {
        "version": "5.4.1",
        "flags": {},
        "shapes": shapes,
        "imagePath": Path(image_path).name,
        "imageData": None,
        "imageHeight": h,
        "imageWidth": w
    }

    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(labelme_data, f, ensure_ascii=False, indent=2)


# ============================================================
# 主流程
# ============================================================
def fit_region(pred_mask: np.ndarray, bbox: tuple,
               region_mask: np.ndarray = None) -> dict:
    """对单个区域做经纬线拟合。bbox = (y_min, y_max, x_min, x_max)

    region_mask: 全图尺寸的布尔掩码，True 表示属于本区域的像素。
                 若提供，则在 bbox 内将不属于本区域的像素（如插图）清零，
                 防止主图经纬线穿过子图区域。
    """
    y_min, y_max, x_min, x_max = bbox
    region_mask_crop = pred_mask[y_min:y_max, x_min:x_max].copy()

    if region_mask is not None:
        keep = region_mask[y_min:y_max, x_min:x_max].astype(bool)
        region_mask_crop[~keep] = 0

    v_lines = fit_vertical_lines(region_mask_crop, class_id=1)
    h_arcs = fit_horizontal_arcs(region_mask_crop, class_id=2)

    # 将坐标从区域局部坐标转回全图坐标
    for v in v_lines:
        for pt in v['points']:
            pt[0] += x_min
            pt[1] += y_min
    for arc in h_arcs:
        for pt in arc['points']:
            pt[0] += x_min
            pt[1] += y_min

    return {'v_lines': v_lines, 'h_arcs': h_arcs}


def main():
    parser = argparse.ArgumentParser(description='从 UNet mask 拟合经纬线生成 labelme JSON')
    parser.add_argument('--image', required=True, help='图片路径')
    parser.add_argument('--model', default=None, help='模型路径')
    parser.add_argument('--mask', default=None, help='已有 mask 路径（跳过推理）')
    parser.add_argument('--output', default=None, help='输出 JSON 路径')
    args = parser.parse_args()

    image_path = args.image
    output_path = args.output or str(Path(image_path).with_suffix('.json'))

    # 推理或加载已有 mask
    if args.mask:
        print(f"加载已有 mask: {args.mask}")
        mask_color = cv2.imread(args.mask)
        h, w = mask_color.shape[:2]
        pred_mask = np.zeros((h, w), dtype=np.uint8)
        pred_mask[(mask_color[:,:,2] > 200) & (mask_color[:,:,1] < 50) & (mask_color[:,:,0] < 50)] = 1
        pred_mask[(mask_color[:,:,1] > 200) & (mask_color[:,:,2] < 50) & (mask_color[:,:,0] < 50)] = 2
        pred_mask[(mask_color[:,:,2] > 200) & (mask_color[:,:,1] > 100) & (mask_color[:,:,1] < 200)] = 3
    else:
        model_path = args.model or str(DEFAULT_MODEL)
        print(f"模型: {Path(model_path).name}")
        pred_mask = run_inference(image_path, model_path)

    img = cv2.imread(image_path)
    h, w = img.shape[:2]
    print(f"图片: {Path(image_path).name} ({w}×{h})")

    # 先检测分隔线
    print("\n检测分隔线...")
    splitters = fit_splitter_lines(pred_mask, class_id=3)
    if splitters:
        print(f"  检测到 {len(splitters)} 条分隔线")
        for i, sp in enumerate(splitters):
            print(f"    S{i}: {sp['orientation']}, coverage={sp['coverage']:.2f}")
    else:
        print("  无分隔线")

    # 根据分隔线划分区域
    regions = partition_by_splitters((h, w), splitters, pred_mask)
    print(f"\n划分为 {len(regions)} 个区域:")
    for region in regions:
        bbox = region['bbox']
        label = "主图" if region['group_id'] == 0 else f"子图{region['group_id']}"
        print(f"  {label}: y=[{bbox[0]},{bbox[1]}], x=[{bbox[2]},{bbox[3]}]")

    # 对每个区域独立拟合经纬线
    region_results = []
    for region in regions:
        bbox = region['bbox']
        gid = region['group_id']
        label = "主图" if gid == 0 else f"子图{gid}"
        print(f"\n拟合{label}经纬线...")

        result = fit_region(pred_mask, bbox, region.get('region_mask'))
        result['group_id'] = gid

        v_lines = result['v_lines']
        h_arcs = result['h_arcs']
        print(f"  经线 {len(v_lines)} 条, 纬线 {len(h_arcs)} 条")

        for i, v in enumerate(v_lines):
            x_mid = (v['points'][0][0] + v['points'][1][0]) / 2
            print(f"    V{i}: x_mid={x_mid:.0f}, coverage={v['coverage']:.2f}")
        for i, arc in enumerate(h_arcs):
            y_mean = np.mean([p[1] for p in arc['points']])
            print(f"    H{i}: y_mean={y_mean:.0f}, coverage={arc['coverage']:.2f}")

        region_results.append(result)

    # 生成合并的 JSON
    generate_labelme_json(image_path, region_results, splitters, output_path)

    total_v = sum(len(r['v_lines']) for r in region_results)
    total_h = sum(len(r['h_arcs']) for r in region_results)
    print(f"\n✅ 输出: {output_path}")
    print(f"   经线 {total_v} 条, 纬线 {total_h} 条, 分隔线 {len(splitters)} 条")


if __name__ == '__main__':
    main()
