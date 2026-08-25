"""
从 labelme JSON + 图片边缘刻度生成 QGIS GCP 文件。

流程：
1. 解析 labelme 标注，提取主图经纬线（排除插图区域）
2. 按位置排序（经线按 x，纬线按 y）
3. 从图片边缘提取刻度区域，交互确认锚点度数
4. 等差推断未标注线的度数
5. 计算交点，生成 .points 文件

用法:
    python generate_gcp.py --image <图片> --json <标注> [--output <输出>]
    python generate_gcp.py --image <图片> --json <标注> --anchors "v:4=92,5=96;h:2=52,3=48"
"""
import json
import argparse
import numpy as np
from pathlib import Path
from dataclasses import dataclass, field
from typing import List, Tuple, Optional
from scipy.interpolate import interp1d
import cv2


@dataclass
class GridLine:
    index: int
    degree: Optional[float] = None
    confirmed: bool = False


@dataclass
class VerticalLine(GridLine):
    top_point: Tuple[float, float] = (0, 0)
    bottom_point: Tuple[float, float] = (0, 0)
    x_mid: float = 0
    y_span: float = 0


@dataclass
class HorizontalArc(GridLine):
    points: np.ndarray = field(default_factory=lambda: np.array([]))
    y_mean: float = 0
    x_span: float = 0


# ============================================================
# Step 1: 解析标注，分离主图线与插图线
# ============================================================
def parse_shapes(shapes: list, img_h: int, img_w: int):
    """解析 labelme shapes，返回经线、纬线、splitter 列表"""
    v_lines_raw = []
    h_arcs_raw = []
    splitters = []

    for i, s in enumerate(shapes):
        label = s['label']
        shape_type = s['shape_type']
        pts = s['points']

        if label == 'vertical_line' and shape_type == 'line':
            p1, p2 = pts[0], pts[1]
            if p1[1] > p2[1]:
                p1, p2 = p2, p1
            y_span = abs(p2[1] - p1[1])
            x_mid = (p1[0] + p2[0]) / 2
            v_lines_raw.append(VerticalLine(
                index=i, top_point=tuple(p1), bottom_point=tuple(p2),
                x_mid=x_mid, y_span=y_span
            ))
        elif label == 'horizontal_arc' and shape_type == 'linestrip':
            arr = np.array(pts)
            y_mean = arr[:, 1].mean()
            x_span = arr[:, 0].max() - arr[:, 0].min()
            h_arcs_raw.append(HorizontalArc(
                index=i, points=arr, y_mean=y_mean, x_span=x_span
            ))
        elif label == 'splitter':
            splitters.append(np.array(pts))

    return v_lines_raw, h_arcs_raw, splitters


def filter_inset_lines(v_lines: list, h_arcs: list, splitters: list,
                       img_h: int, img_w: int):
    """根据 splitter 边界过滤插图区域的线"""
    if not splitters:
        return v_lines, h_arcs

    # 计算 splitter 覆盖区域的包围盒
    inset_regions = []
    for sp in splitters:
        x_min, y_min = sp.min(axis=0)
        x_max, y_max = sp.max(axis=0)
        inset_regions.append((x_min, y_min, x_max, y_max))

    def is_in_inset(line):
        """判断一条线是否完全在某个插图区域内"""
        if isinstance(line, VerticalLine):
            lx_min = min(line.top_point[0], line.bottom_point[0])
            lx_max = max(line.top_point[0], line.bottom_point[0])
            ly_min = line.top_point[1]
            ly_max = line.bottom_point[1]
        else:
            lx_min = line.points[:, 0].min()
            lx_max = line.points[:, 0].max()
            ly_min = line.points[:, 1].min()
            ly_max = line.points[:, 1].max()

        for (rx_min, ry_min, rx_max, ry_max) in inset_regions:
            x_overlap = max(0, min(lx_max, rx_max) - max(lx_min, rx_min))
            y_overlap = max(0, min(ly_max, ry_max) - max(ly_min, ry_min))
            line_area = max(1, (lx_max - lx_min)) * max(1, (ly_max - ly_min))
            overlap_ratio = (x_overlap * y_overlap) / line_area
            if overlap_ratio > 0.5:
                return True
        return False

    v_filtered = [v for v in v_lines if not is_in_inset(v)]
    h_filtered = [h for h in h_arcs if not is_in_inset(h)]
    return v_filtered, h_filtered


# ============================================================
# Step 2: 排序 + 去重（合并距离过近的线）
# ============================================================
def sort_and_dedup(v_lines: list, h_arcs: list,
                   v_min_gap: float = 100, h_min_gap: float = 100):
    """排序并合并间距过近的线（保留较长的）"""
    # 经线按 x_mid 排序
    v_sorted = sorted(v_lines, key=lambda l: l.x_mid)
    # 纬线按 y_mean 排序
    h_sorted = sorted(h_arcs, key=lambda l: l.y_mean)

    # 去重经线
    v_deduped = []
    for v in v_sorted:
        if not v_deduped or v.x_mid - v_deduped[-1].x_mid > v_min_gap:
            v_deduped.append(v)
        else:
            if v.y_span > v_deduped[-1].y_span:
                v_deduped[-1] = v

    # 去重纬线
    h_deduped = []
    for h in h_sorted:
        if not h_deduped or h.y_mean - h_deduped[-1].y_mean > h_min_gap:
            h_deduped.append(h)
        else:
            if h.x_span > h_deduped[-1].x_span:
                h_deduped[-1] = h

    return v_deduped, h_deduped


# ============================================================
# Step 3: 锚点确认 + 等差推断
# ============================================================
def assign_degrees(lines: list, anchors: dict, line_type: str = 'v'):
    """
    根据锚点和等差规律赋值度数。

    Args:
        lines: 排序后的线列表
        anchors: {排序序号: 度数} 至少需要2个
        line_type: 'v'(经度) 或 'h'(纬度)

    Returns:
        更新 degree 和 confirmed 的线列表
    """
    if len(anchors) < 2:
        raise ValueError(f"至少需要2个锚点，当前只有 {len(anchors)} 个")

    # 从锚点计算等差间距
    anchor_indices = sorted(anchors.keys())
    intervals = []
    for i in range(len(anchor_indices) - 1):
        idx1, idx2 = anchor_indices[i], anchor_indices[i + 1]
        deg_diff = anchors[idx2] - anchors[idx1]
        idx_diff = idx2 - idx1
        intervals.append(deg_diff / idx_diff)

    # 检查间距一致性
    avg_interval = np.mean(intervals)
    if max(intervals) - min(intervals) > 0.1:
        print(f"  ⚠️ 间距不完全一致: {intervals}，使用平均值 {avg_interval:.2f}")

    # 选取主锚点（第一个确认的）
    ref_idx = anchor_indices[0]
    ref_deg = anchors[ref_idx]

    # 等差赋值
    for i, line in enumerate(lines):
        deg = ref_deg + (i - ref_idx) * avg_interval
        # 四舍五入到合理精度
        if abs(avg_interval) >= 1:
            deg = round(deg)
        else:
            deg = round(deg, 1)
        line.degree = deg
        line.confirmed = (i in anchors)

    return lines


# ============================================================
# Step 4: 计算交点 + 生成 GCP
# ============================================================
def line_y_to_x(v_line: VerticalLine, y: float) -> float:
    """经线在指定 y 处的 x 坐标"""
    x1, y1 = v_line.top_point
    x2, y2 = v_line.bottom_point
    if y2 == y1:
        return (x1 + x2) / 2
    t = (y - y1) / (y2 - y1)
    return x1 + t * (x2 - x1)


def arc_x_to_y(h_arc: HorizontalArc, x: float) -> Optional[float]:
    """纬线在指定 x 处的 y 坐标（插值，不外推）"""
    pts = h_arc.points
    sorted_idx = np.argsort(pts[:, 0])
    xs = pts[sorted_idx, 0]
    ys = pts[sorted_idx, 1]
    if x < xs[0] or x > xs[-1]:
        return None
    f = interp1d(xs, ys, kind='linear')
    return float(f(x))


def compute_intersections(v_lines: List[VerticalLine],
                          h_arcs: List[HorizontalArc],
                          img_w: int = None, img_h: int = None) -> list:
    """计算所有经纬线交点（交点必须同时落在经线和纬线的标注范围内）"""
    gcps = []
    for v in v_lines:
        v_y_min = min(v.top_point[1], v.bottom_point[1])
        v_y_max = max(v.top_point[1], v.bottom_point[1])

        for h in h_arcs:
            # 纬线是弧线，y 随 x 变化大，用纬线实际 y 范围判断是否可能与经线相交
            h_pts = h.points
            h_y_min = float(h_pts[:, 1].min())
            h_y_max = float(h_pts[:, 1].max())

            # 两条线的 y 范围必须有重叠才可能相交
            if v_y_max < h_y_min or v_y_min > h_y_max:
                continue

            # 初始 y 猜测：取两者 y 范围重叠区的中点
            overlap_y_min = max(v_y_min, h_y_min)
            overlap_y_max = min(v_y_max, h_y_max)
            y_guess = (overlap_y_min + overlap_y_max) / 2

            for _ in range(30):
                x_at_y = line_y_to_x(v, y_guess)
                y_at_x = arc_x_to_y(h, x_at_y)
                if y_at_x is None:
                    break
                if abs(y_at_x - y_guess) < 0.5:
                    # 确认交点在经线 y 范围内
                    if v_y_min <= y_at_x <= v_y_max:
                        gcps.append({
                            'lon': v.degree,
                            'lat': h.degree,
                            'pixel_x': round(x_at_y, 2),
                            'pixel_y': round(y_at_x, 2),
                        })
                    break
                y_guess = y_at_x

    return gcps


def write_points_file(gcps: list, output_path: str):
    """写入 QGIS .points 文件"""
    lines = ['mapX,mapY,sourceX,sourceY,enable,dX,dY,residual']
    for gcp in gcps:
        lines.append(f'{gcp["lon"]},{gcp["lat"]},{gcp["pixel_x"]},{-gcp["pixel_y"]},1,0,0,0')
    with open(output_path, 'w') as f:
        f.write('\n'.join(lines))


# ============================================================
# Step 5: 边缘刻度提取（辅助人工确认）
# ============================================================
def extract_scale_regions(image_path: str, v_lines: list, h_arcs: list,
                          output_dir: str = '/tmp'):
    """提取边缘刻度区域图片，供人工查看"""
    img = cv2.imread(image_path)
    h, w = img.shape[:2]

    print("\n=== 经线刻度区域（底部端点） ===")
    for i, v in enumerate(v_lines):
        bot_x = int(v.bottom_point[0])
        strip = img[h - 40:h, max(0, bot_x - 50):min(w, bot_x + 50), :]
        path = f'{output_dir}/scale_v{i}_bot.jpg'
        cv2.imwrite(path, strip)
        print(f"  V{i}: bot_x={bot_x}, 保存 → {path}")

    print("\n=== 纬线刻度区域（右侧端点） ===")
    for i, h_arc in enumerate(h_arcs):
        y_at_right = int(h_arc.y_mean)
        strip = img[max(0, y_at_right - 40):y_at_right + 40, w - 80:w, :]
        path = f'{output_dir}/scale_h{i}_right.jpg'
        cv2.imwrite(path, strip)
        print(f"  H{i}: y_mean={y_at_right}, 保存 → {path}")


# ============================================================
# 报告生成
# ============================================================
def generate_report(v_lines: list, h_arcs: list, gcps: list) -> str:
    """生成摘要报告"""
    v_confirmed = sum(1 for v in v_lines if v.confirmed)
    h_confirmed = sum(1 for h in h_arcs if h.confirmed)

    report = []
    report.append("=" * 50)
    report.append("GCP 生成报告")
    report.append("=" * 50)
    report.append(f"经线: {len(v_lines)} 条 (确认 {v_confirmed}, 推断 {len(v_lines)-v_confirmed})")
    report.append(f"  范围: {v_lines[0].degree}° ~ {v_lines[-1].degree}°E")
    report.append(f"  间距: {v_lines[1].degree - v_lines[0].degree}°")
    report.append(f"纬线: {len(h_arcs)} 条 (确认 {h_confirmed}, 推断 {len(h_arcs)-h_confirmed})")
    report.append(f"  范围: {h_arcs[0].degree}° ~ {h_arcs[-1].degree}°N")
    report.append(f"  间距: {h_arcs[1].degree - h_arcs[0].degree}°")
    report.append(f"GCP 总数: {len(gcps)}")
    report.append("")

    # 标记推断的线
    inferred_v = [v for v in v_lines if not v.confirmed]
    inferred_h = [h for h in h_arcs if not h.confirmed]
    if inferred_v:
        report.append("推断经线:")
        for v in inferred_v:
            report.append(f"  {v.degree}°E (x_mid={v.x_mid:.0f}, span={v.y_span:.0f})")
    if inferred_h:
        report.append("推断纬线:")
        for h in inferred_h:
            report.append(f"  {h.degree}°N (y_mean={h.y_mean:.0f}, span={h.x_span:.0f})")

    report.append("=" * 50)
    return '\n'.join(report)


# ============================================================
# 锚点解析（命令行格式）
# ============================================================
def parse_anchors_string(anchor_str: str) -> Tuple[dict, dict]:
    """
    解析锚点字符串格式: "v:4=92,5=96,6=100;h:2=52,3=48"
    Returns: (v_anchors, h_anchors)
    """
    v_anchors = {}
    h_anchors = {}

    for part in anchor_str.split(';'):
        part = part.strip()
        if part.startswith('v:'):
            pairs = part[2:].split(',')
            for p in pairs:
                idx, val = p.split('=')
                v_anchors[int(idx)] = float(val)
        elif part.startswith('h:'):
            pairs = part[2:].split(',')
            for p in pairs:
                idx, val = p.split('=')
                h_anchors[int(idx)] = float(val)

    return v_anchors, h_anchors


# ============================================================
# 主流程
# ============================================================
def main():
    parser = argparse.ArgumentParser(description='从 labelme 标注生成 QGIS GCP 文件')
    parser.add_argument('--image', required=True, help='图片路径')
    parser.add_argument('--json', required=True, help='labelme JSON 路径')
    parser.add_argument('--output', default=None, help='输出 .points 文件路径')
    parser.add_argument('--anchors', default=None,
                        help='锚点字符串，格式: "v:4=92,5=96;h:2=52,3=48"')
    parser.add_argument('--extract-scales', action='store_true',
                        help='只提取边缘刻度区域图片（用于人工确认）')
    parser.add_argument('--no-dedup', action='store_true',
                        help='不去重（保留所有间距过近的线）')
    args = parser.parse_args()

    # 加载数据
    image_path = args.image
    with open(args.json, 'r') as f:
        data = json.load(f)

    img_h = data['imageHeight']
    img_w = data['imageWidth']
    shapes = data['shapes']

    print(f"图片: {Path(image_path).name} ({img_w}×{img_h})")
    print(f"标注: {len(shapes)} 个")

    # Step 1: 解析标注
    v_lines, h_arcs, splitters = parse_shapes(shapes, img_h, img_w)
    print(f"原始: 经线 {len(v_lines)} 条, 纬线 {len(h_arcs)} 条, splitter {len(splitters)} 个")

    # 过滤插图线
    if splitters:
        v_lines, h_arcs = filter_inset_lines(v_lines, h_arcs, splitters, img_h, img_w)
        print(f"过滤后: 经线 {len(v_lines)} 条, 纬线 {len(h_arcs)} 条")

    # Step 2: 排序 + 去重
    if args.no_dedup:
        v_lines = sorted(v_lines, key=lambda l: l.x_mid)
        h_arcs = sorted(h_arcs, key=lambda l: l.y_mean)
    else:
        v_lines, h_arcs = sort_and_dedup(v_lines, h_arcs)
    print(f"去重后: 经线 {len(v_lines)} 条, 纬线 {len(h_arcs)} 条")

    # 显示排序结果
    print("\n经线排序 (按 x_mid):")
    for i, v in enumerate(v_lines):
        short = "(短)" if v.y_span < img_h * 0.5 else ""
        print(f"  V{i}: x_mid={v.x_mid:.0f}, bot_x={v.bottom_point[0]:.0f}, span={v.y_span:.0f} {short}")

    print("\n纬线排序 (按 y_mean):")
    for i, h in enumerate(h_arcs):
        short = "(短)" if h.x_span < img_w * 0.5 else ""
        print(f"  H{i}: y_mean={h.y_mean:.0f}, x_span={h.x_span:.0f} {short}")

    # 提取刻度区域（如果只是预览）
    if args.extract_scales:
        extract_scale_regions(image_path, v_lines, h_arcs)
        print("\n刻度区域已提取，请查看后用 --anchors 参数指定度数")
        return

    # Step 3: 赋值度数
    if not args.anchors:
        print("\n❌ 需要指定 --anchors 参数")
        print("格式: --anchors \"v:索引=度数,...;h:索引=度数,...\"")
        print("示例: --anchors \"v:4=92,5=96,6=100;h:2=52,3=48\"")
        print("\n提示: 先运行 --extract-scales 查看边缘刻度，确认后再指定锚点")
        return

    v_anchors, h_anchors = parse_anchors_string(args.anchors)
    print(f"\n经线锚点: {v_anchors}")
    print(f"纬线锚点: {h_anchors}")

    v_lines = assign_degrees(v_lines, v_anchors, 'v')
    h_arcs = assign_degrees(h_arcs, h_anchors, 'h')

    # Step 4: 计算交点（限制在图片边界内）
    gcps = compute_intersections(v_lines, h_arcs, img_w, img_h)

    # 输出
    output_path = args.output or str(Path(image_path).with_suffix('.points'))
    write_points_file(gcps, output_path)

    # 生成预览图
    preview_path = str(Path(output_path).with_suffix('')) + '_preview.jpg'
    img = cv2.imread(image_path)
    if img is None:
        from PIL import Image as PILImage
        pil = PILImage.open(image_path).convert("RGB")
        img = cv2.cvtColor(np.array(pil), cv2.COLOR_RGB2BGR)
    for gcp in gcps:
        x, y = int(round(gcp['pixel_x'])), int(round(gcp['pixel_y']))
        cv2.circle(img, (x, y), 8, (0, 0, 255), -1)
        cv2.circle(img, (x, y), 9, (255, 255, 255), 2)
    cv2.imwrite(preview_path, img, [cv2.IMWRITE_JPEG_QUALITY, 85])

    # 报告
    report = generate_report(v_lines, h_arcs, gcps)
    print(f"\n{report}")
    print(f"✅ GCP 文件已保存: {output_path}")
    print(f"📍 预览图已保存: {preview_path}")


if __name__ == '__main__':
    main()
