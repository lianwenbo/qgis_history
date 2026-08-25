"""
从源目录抽取未在 raw_data 中出现的地图图片，
利用 UNet 推理生成 Labelme 格式的标注数据（经线=line, 纬线/分隔线=linestrip）。
"""
import sys
import re
import random
import shutil
import json
from pathlib import Path

import cv2
import numpy as np
import torch
from skimage.morphology import skeletonize

sys.path.insert(0, str(Path(__file__).parent))
from unet_model import UNet, _build_weight_map
from detect_splitters import detect_splitters_from_mask, partition_regions


# ============================================================
# 配置
# ============================================================
SOURCE_DIR = Path.home() / "Work/historical_map/古代中国地图"
RAW_DATA_DIR = Path(__file__).parent / "map_line_dataset/raw_data"
VERIFY_RAW_DIR = Path(__file__).parent / "map_line_dataset/verify_raw"
VERIFY_DATA_DIR = Path(__file__).parent / "map_line_dataset/verify_data"
MODEL_PATH = Path.home() / "Downloads/unet_map_lines_autodl_colab_20260727_234942.pth"

PATCH_SIZE = 512
OVERLAP = 0.5
FILL_COLOR = (245, 235, 210)
NUM_SAMPLES = 10

# 线条合并阈值
MERGE_X_THRESHOLD = 50   # 经线 x 坐标差阈值（果断合并）
MERGE_Y_THRESHOLD = 80   # 纬线 y 质心差阈值（果断合并）
ENDPOINT_GAP_THRESHOLD = 100  # 端点相连距离阈值（同方向线段端点接近即合并）
SAMPLE_POINTS = 30        # 纬线折线采样点数


# ============================================================
# Step 1: 选图（去重、排除 raw_data）
# ============================================================
def select_images():
    """从源目录选取去重后且不在 raw_data 中的图片"""
    prefix_pattern = re.compile(r'^(\d+-\d+)')

    # 获取 raw_data 已有的前缀
    used_prefixes = set()
    if RAW_DATA_DIR.exists():
        for f in RAW_DATA_DIR.iterdir():
            m = prefix_pattern.match(f.name)
            if m:
                used_prefixes.add(m.group(1))

    # 扫描源目录
    all_images = []
    for ext in ('*.jpg', '*.png', '*.jpeg', '*.tif'):
        all_images.extend(SOURCE_DIR.rglob(ext))

    # 按前缀去重，排除已用前缀、01~03 开头、含"全图"
    prefix_map = {}  # prefix -> [paths]
    for img_path in all_images:
        m = prefix_pattern.match(img_path.name)
        if not m:
            continue
        prefix = m.group(1)
        vol = int(prefix.split('-')[0])
        if vol <= 3:
            continue
        if '全图' in img_path.name:
            continue
        if prefix in used_prefixes:
            continue
        if prefix not in prefix_map:
            prefix_map[prefix] = []
        prefix_map[prefix].append(img_path)

    # 每个前缀随机取一张
    candidates = []
    for prefix, paths in prefix_map.items():
        candidates.append((prefix, random.choice(paths)))

    # 随机抽取 N 张
    if len(candidates) < NUM_SAMPLES:
        print(f"警告: 可用图片不足 {NUM_SAMPLES} 张，仅有 {len(candidates)} 张")
        selected = candidates
    else:
        selected = random.sample(candidates, NUM_SAMPLES)

    print(f"已选取 {len(selected)} 张图片:")
    for prefix, path in selected:
        print(f"  [{prefix}] {path.name}")

    return selected


# ============================================================
# Step 2: 滑窗推理
# ============================================================
def run_inference(image_path):
    """对单张图片进行滑窗推理，返回全图 pred_mask"""
    device = torch.device('mps' if torch.backends.mps.is_available() else
                          'cuda' if torch.cuda.is_available() else 'cpu')

    model = UNet(n_channels=3, n_classes=4).to(device)
    checkpoint = torch.load(str(MODEL_PATH), map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

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

    weight_accum_safe = np.maximum(weight_accum, 1e-6)
    prob_avg = prob_accum / weight_accum_safe[:, :, np.newaxis]
    pred_mask = np.argmax(prob_avg, axis=2).astype(np.uint8)
    pred_mask = pred_mask[:original_h, :original_w]

    return pred_mask


# ============================================================
# Step 3: Mask → 骨架化 → 矢量折线 → 合并 → Labelme JSON
# ============================================================
def mask_to_skeleton_lines(mask, class_id):
    """将某类别的 mask 骨架化后提取连通域折线"""
    binary = (mask == class_id).astype(np.uint8)
    if binary.sum() == 0:
        return []

    skeleton = skeletonize(binary > 0).astype(np.uint8) * 255
    contours, _ = cv2.findContours(skeleton, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    lines = []
    for cnt in contours:
        if len(cnt) < 2:
            continue
        pts = cnt.squeeze()
        if pts.ndim == 1:
            continue
        lines.append(pts.tolist())
    return lines


def merge_vertical_lines(lines, img_height=None):
    """经线合并：将骨架碎片投影到参考 y 线后按 x 分组，线性拟合保留真实倾斜角"""
    if not lines:
        return []

    # 估算图片高度用作参考线
    all_points = np.concatenate([np.array(pts) for pts in lines])
    if img_height is None:
        img_height = int(all_points[:, 1].max())
    y_ref = img_height // 2

    # 每段碎片先做局部拟合，然后投影到 y_ref 计算参考 x
    line_infos = []
    for pts in lines:
        arr = np.array(pts)
        if len(arr) < 2:
            continue
        y_span = arr[:, 1].max() - arr[:, 1].min()
        if y_span > 10 and len(arr) >= 2:
            coeffs = np.polyfit(arr[:, 1], arr[:, 0], 1)
            x_at_ref = coeffs[0] * y_ref + coeffs[1]
        else:
            x_at_ref = arr[:, 0].mean()
        line_infos.append({'x_ref': x_at_ref, 'points': arr})

    # 按投影 x 排序后分组
    line_infos.sort(key=lambda l: l['x_ref'])
    groups = []
    current_group = [line_infos[0]]

    for info in line_infos[1:]:
        if abs(info['x_ref'] - current_group[-1]['x_ref']) < MERGE_X_THRESHOLD:
            current_group.append(info)
        else:
            groups.append(current_group)
            current_group = [info]
    groups.append(current_group)

    # 每组合并所有点做全局线性拟合
    result = []
    for group in groups:
        all_pts = np.concatenate([g['points'] for g in group])
        y_min = int(all_pts[:, 1].min())
        y_max = int(all_pts[:, 1].max())
        if y_max - y_min < img_height * 0.25:
            continue

        coeffs = np.polyfit(all_pts[:, 1], all_pts[:, 0], 1)  # x = a*y + b
        x_top = int(coeffs[0] * y_min + coeffs[1])
        x_bot = int(coeffs[0] * y_max + coeffs[1])
        result.append([[x_top, y_min], [x_bot, y_max]])

    return result


def _estimate_y_at_x(pts_arr, target_x, window=50):
    """估算某条碎片在指定 x 处的 y 值"""
    nearby = pts_arr[np.abs(pts_arr[:, 0] - target_x) < window]
    if len(nearby) > 0:
        return nearby[:, 1].mean()
    # 用最近点外推
    dists = np.abs(pts_arr[:, 0] - target_x)
    closest_idx = np.argmin(dists)
    return pts_arr[closest_idx, 1]


def _x_gap(info_a, info_b):
    """计算两段碎片在 x 方向的空隙距离（重叠时返回0）"""
    if info_a['x_max'] < info_b['x_min']:
        return info_b['x_min'] - info_a['x_max']
    elif info_b['x_max'] < info_a['x_min']:
        return info_a['x_min'] - info_b['x_max']
    return 0


def _local_y_distance(info_a, info_b):
    """计算两段碎片在 x 重叠区域的局部 y 偏差"""
    x_overlap_min = max(info_a['x_min'], info_b['x_min'])
    x_overlap_max = min(info_a['x_max'], info_b['x_max'])

    if x_overlap_min < x_overlap_max:
        # 有 x 重叠：在重叠中点比较 y
        mid_x = (x_overlap_min + x_overlap_max) / 2
        y_a = _estimate_y_at_x(info_a['points'], mid_x)
        y_b = _estimate_y_at_x(info_b['points'], mid_x)
        return abs(y_a - y_b)
    else:
        # 无重叠：用最近端外推
        if info_a['x_max'] < info_b['x_min']:
            gap_x = (info_a['x_max'] + info_b['x_min']) / 2
        else:
            gap_x = (info_b['x_max'] + info_a['x_min']) / 2
        y_a = _estimate_y_at_x(info_a['points'], gap_x)
        y_b = _estimate_y_at_x(info_b['points'], gap_x)
        return abs(y_a - y_b)


def merge_horizontal_lines(lines, img_height=None):
    """纬线/分隔线合并：基于局部 y 偏差自适应合并，适应高纬度弧线"""
    if not lines:
        return []

    # 自适应阈值：图越高允许越大的 y 偏差
    base_threshold = MERGE_Y_THRESHOLD
    if img_height:
        adaptive_threshold = max(base_threshold, img_height * 0.025)
    else:
        adaptive_threshold = base_threshold

    # 每条线计算基础信息
    line_infos = []
    for pts in lines:
        arr = np.array(pts)
        y_center = arr[:, 1].mean()
        x_min = arr[:, 0].min()
        x_max = arr[:, 0].max()
        line_infos.append({'y_center': y_center, 'points': arr, 'x_min': x_min, 'x_max': x_max})

    line_infos.sort(key=lambda l: l['y_center'])
    groups = []
    current_group = [line_infos[0]]

    for info in line_infos[1:]:
        # 与当前组中每个成员计算局部 y 距离，取最小值
        min_dist = min(_local_y_distance(info, g) for g in current_group)
        # 同时检查 x 连续性：如果与组中所有成员的 x 范围都存在大空隙，不合并
        x_continuous = any(
            _x_gap(info, g) < adaptive_threshold * 3
            for g in current_group
        )
        if min_dist < adaptive_threshold and x_continuous:
            current_group.append(info)
        else:
            groups.append(current_group)
            current_group = [info]
    groups.append(current_group)

    # 每组合并为采样折线
    result = []
    for group in groups:
        all_pts = np.concatenate([g['points'] for g in group])
        x_min = all_pts[:, 0].min()
        x_max = all_pts[:, 0].max()

        if x_max - x_min < 50:
            continue

        # 在 x 方向均匀采样
        sample_x = np.linspace(x_min, x_max, SAMPLE_POINTS)
        sample_pts = []
        for sx in sample_x:
            nearby = all_pts[np.abs(all_pts[:, 0] - sx) < max(20, (x_max - x_min) / SAMPLE_POINTS)]
            if len(nearby) > 0:
                y_avg = nearby[:, 1].mean()
                sample_pts.append([int(sx), int(y_avg)])
            else:
                # 线性插值
                if sample_pts:
                    sample_pts.append([int(sx), sample_pts[-1][1]])

        if len(sample_pts) >= 2:
            result.append(sample_pts)

    return result


def generate_labelme_json(image_path, pred_mask, output_path):
    """从推理 mask 生成 Labelme JSON，先检测分隔线划分区域，再分别处理"""
    img = cv2.imread(str(image_path))
    h, w = img.shape[:2]

    # Step 1: 检测分隔线（使用独立模块）
    all_splitter_objs = detect_splitters_from_mask(pred_mask, class_id=3)

    # Step 2: 划分区域（使用独立模块）
    regions = partition_regions(h, w, all_splitter_objs, pred_mask=pred_mask)

    # Step 3: 每个区域独立做经纬线骨架化+合并
    shapes = []
    total_v, total_h = 0, 0

    for region in regions:
        y_min, y_max, x_min, x_max = region['bbox']
        gid = region['group_id']
        region_mask = region['region_mask']

        # 用区域掩码隔离当前区域的经纬线像素（防止跨分隔线）
        masked_pred = pred_mask.copy()
        masked_pred[region_mask == 0] = 0

        region_pred = masked_pred[y_min:y_max, x_min:x_max]
        region_h = y_max - y_min

        # 经线
        v_lines = mask_to_skeleton_lines(region_pred, 1)
        v_merged = merge_vertical_lines(v_lines, img_height=region_h)
        for pts in v_merged:
            offset_pts = [[p[0] + x_min, p[1] + y_min] for p in pts]
            shapes.append({
                "label": "vertical_line",
                "points": offset_pts,
                "group_id": gid,
                "description": "",
                "shape_type": "line",
                "flags": {},
                "mask": None
            })
        total_v += len(v_merged)

        # 纬线
        h_lines = mask_to_skeleton_lines(region_pred, 2)
        h_merged = merge_horizontal_lines(h_lines, img_height=region_h)
        for pts in h_merged:
            offset_pts = [[p[0] + x_min, p[1] + y_min] for p in pts]
            shapes.append({
                "label": "horizontal_arc",
                "points": offset_pts,
                "group_id": gid,
                "description": "",
                "shape_type": "linestrip",
                "flags": {},
                "mask": None
            })
        total_h += len(h_merged)

    # 分隔线
    for sp in all_splitter_objs:
        shapes.append({
            "label": "splitter",
            "points": sp['points'],
            "group_id": None,
            "description": f"orientation={sp['orientation']}",
            "shape_type": "linestrip",
            "flags": {},
            "mask": None
        })

    # 构造 Labelme JSON
    labelme_data = {
        "version": "5.4.1",
        "flags": {},
        "shapes": shapes,
        "imagePath": image_path.name,
        "imageData": None,
        "imageHeight": h,
        "imageWidth": w
    }

    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(labelme_data, f, ensure_ascii=False, indent=2)

    print(f"  生成标注: {len(shapes)} 条 (经线 {total_v}, 纬线 {total_h}, 分隔线 {len(all_splitter_objs)}, 区域 {len(regions)})")


def _merge_vertical_splitters(lines, img_width=None):
    """垂直分隔线合并：与经线合并类似，但对 y 方向采样输出折线"""
    if not lines:
        return []

    all_points = np.concatenate([np.array(pts) for pts in lines])
    if img_width is None:
        img_width = int(all_points[:, 0].max())

    # 按 x 坐标分组
    line_infos = []
    for pts in lines:
        arr = np.array(pts)
        if len(arr) < 2:
            continue
        x_center = arr[:, 0].mean()
        y_span = arr[:, 1].max() - arr[:, 1].min()
        line_infos.append({'x_center': x_center, 'points': arr, 'y_span': y_span})

    if not line_infos:
        return []

    line_infos.sort(key=lambda l: l['x_center'])
    groups = []
    current_group = [line_infos[0]]

    for info in line_infos[1:]:
        if abs(info['x_center'] - current_group[-1]['x_center']) < MERGE_X_THRESHOLD:
            current_group.append(info)
        else:
            groups.append(current_group)
            current_group = [info]
    groups.append(current_group)

    result = []
    for group in groups:
        all_pts = np.concatenate([g['points'] for g in group])
        y_min = all_pts[:, 1].min()
        y_max = all_pts[:, 1].max()
        y_span = y_max - y_min

        img_height_est = int(all_pts[:, 1].max()) + 100
        if y_span < img_height_est * 0.2:
            continue

        sample_ys = np.linspace(y_min, y_max, SAMPLE_POINTS)
        sample_pts = []
        for sy in sample_ys:
            nearby = all_pts[np.abs(all_pts[:, 1] - sy) < max(20, y_span / SAMPLE_POINTS)]
            if len(nearby) > 0:
                x_avg = nearby[:, 0].mean()
                sample_pts.append([int(x_avg), int(sy)])

        if len(sample_pts) >= 2:
            result.append(sample_pts)

    return result


def _partition_by_connected_components(pred_mask, h, w, all_splitters):
    """用分隔线像素作为墙壁，通过连通区域分析找到各独立区域"""
    if not all_splitters:
        return [{'bbox': (0, h, 0, w), 'area': h * w, 'group_id': 0}]

    # 缩小图加速
    scale = 4
    small_h, small_w = h // scale, w // scale
    wall = np.zeros((small_h, small_w), dtype=np.uint8)

    # 用分隔线点集画墙，端点延伸到图片边界
    edge_threshold = 50
    for orientation, pts in all_splitters:
        pts_f = np.array(pts, dtype=np.float64)
        if orientation == 'horizontal':
            if pts_f[0][0] < edge_threshold:
                pts_f[0][0] = 0
            if pts_f[-1][0] > w - edge_threshold:
                pts_f[-1][0] = w
        elif orientation == 'vertical':
            if pts_f[0][1] < edge_threshold:
                pts_f[0][1] = 0
            if pts_f[-1][1] > h - edge_threshold:
                pts_f[-1][1] = h
        pts_arr = (pts_f / scale).astype(np.int32)
        for i in range(len(pts_arr) - 1):
            cv2.line(wall, tuple(pts_arr[i]), tuple(pts_arr[i+1]), 255, thickness=3)

    # 用 pred_mask 中 class=3 像素补充
    splitter_binary = (pred_mask == 3).astype(np.uint8)
    small_splitter = cv2.resize(splitter_binary, (small_w, small_h),
                                interpolation=cv2.INTER_NEAREST)
    wall = np.maximum(wall, small_splitter * 255)

    # 膨胀确保封闭
    kernel = np.ones((5, 5), dtype=np.uint8)
    wall = cv2.dilate(wall, kernel, iterations=1)

    # 连通区域
    free_space = (wall == 0).astype(np.uint8)
    num_labels, labels = cv2.connectedComponents(free_space, connectivity=4)

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
        regions.append({'bbox': (y_min, y_max, x_min, x_max), 'area': area})

    if not regions:
        return [{'bbox': (0, h, 0, w), 'area': h * w, 'group_id': 0}]

    regions.sort(key=lambda r: r['area'], reverse=True)
    for idx, region in enumerate(regions):
        region['group_id'] = idx

    return regions


def save_inference_result(image_path, pred_mask, output_dir):
    """保存推理产物：彩色 mask + 彩色叠加可视化"""
    stem = image_path.stem
    img = cv2.imread(str(image_path))

    # 保存彩色 mask（背景黑，经线红，纬线绿，分隔线橙）
    color_mask = np.zeros_like(img)
    color_mask[pred_mask == 1] = [0, 0, 255]
    color_mask[pred_mask == 2] = [0, 255, 0]
    color_mask[pred_mask == 3] = [0, 165, 255]
    cv2.imwrite(str(Path(output_dir) / f"{stem}_mask.png"), color_mask)

    # 生成彩色叠加图
    overlay = img.copy()
    overlay[pred_mask == 1] = [0, 0, 255]
    overlay[pred_mask == 2] = [0, 255, 0]
    overlay[pred_mask == 3] = [0, 165, 255]
    blended = cv2.addWeighted(img, 0.5, overlay, 0.5, 0)
    cv2.imwrite(str(Path(output_dir) / f"{stem}_infer.jpg"), blended)

    print(f"  推理结果已保存: {stem}_mask.png, {stem}_infer.jpg")


# ============================================================
# Main
# ============================================================
def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--test_data', action='store_true',
                        help='从 test_data 目录读取所有图片进行推理')
    args = parser.parse_args()

    random.seed(42)

    TEST_DATA_DIR = Path(__file__).parent / "map_line_dataset/test_data"

    # 准备目录
    VERIFY_DATA_DIR.mkdir(parents=True, exist_ok=True)

    # 清空 verify_data 已有数据
    for f in VERIFY_DATA_DIR.iterdir():
        f.unlink()

    if args.test_data:
        # 从 test_data 目录读取
        image_paths = sorted([
            p for p in TEST_DATA_DIR.iterdir()
            if p.suffix.lower() in ('.jpg', '.png', '.jpeg', '.tif')
        ])
        print("=" * 60)
        print(f"从 test_data 推理: {len(image_paths)} 张图片")
        print("=" * 60)
    else:
        # 原有逻辑：从源目录选图
        VERIFY_RAW_DIR.mkdir(parents=True, exist_ok=True)
        for f in VERIFY_RAW_DIR.iterdir():
            f.unlink()

        print("=" * 60)
        print("Step 1: 选取图片")
        print("=" * 60)
        selected = select_images()
        image_paths = []
        for _, src_path in selected:
            dst_img = VERIFY_RAW_DIR / src_path.name
            shutil.copy2(src_path, dst_img)
            image_paths.append(src_path)

    # 推理 + 生成标注
    print("\n" + "=" * 60)
    print("推理 & 生成 Labelme 标注")
    print("=" * 60)

    for src_path in image_paths:
        print(f"\n处理: {src_path.name}")

        # 复制图片到 verify_data（Labelme 需要图片和 JSON 在同一目录）
        dst_img_data = VERIFY_DATA_DIR / src_path.name
        shutil.copy2(src_path, dst_img_data)

        # 推理
        pred_mask = run_inference(src_path)

        # 保存推理结果（mask + 可视化叠加图）
        save_inference_result(src_path, pred_mask, VERIFY_DATA_DIR)

        # 生成 Labelme JSON
        json_name = src_path.stem + ".json"
        json_path = VERIFY_DATA_DIR / json_name
        generate_labelme_json(dst_img_data, pred_mask, json_path)

    print("\n" + "=" * 60)
    print("完成!")
    print(f"  标注目录: {VERIFY_DATA_DIR}")
    print(f"\n使用 Labelme 校准:")
    print(f"  labelme {VERIFY_DATA_DIR}")
    print("=" * 60)


if __name__ == "__main__":
    main()
