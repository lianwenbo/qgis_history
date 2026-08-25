"""
OCR 刻度识别 + 与 labelme 经纬线端点对齐，自动推断每条线度数。

策略：
  1. 解析 labelme JSON，得到经线 V 的 top/bottom 端点，纬线 H 的左右端点
  2. 在图像上/下边缘沿经线 bot_x/top_x 附近裁刻度条 → OCR → 数字 → 最近邻匹配 V.x
  3. 在图像左/右边缘沿纬线左/右点 y 附近裁刻度条 → OCR → 数字 → 最近邻匹配 H.y
  4. 对匹配结果做等差一致性校验 → 输出 anchors（可直接塞 generate_gcp）

用法:
  python _ocr_scales.py --image map_line_dataset/gcp/08-7直隶.jpg --json map_line_dataset/gcp/08-7直隶.json
  python _ocr_scales.py --dir map_line_dataset/gcp --pattern '08-*.jpg'
"""
import os
# PaddleOCR 启动时会联网检查模型源，跳过以加速
os.environ.setdefault("DISABLE_MODEL_SOURCE_CHECK", "1")

import argparse
import json
import re
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

from generate_gcp import parse_shapes, sort_and_dedup, filter_inset_lines

# 经纬度合理范围（中国历史地图集）
LON_RANGE = (70, 140)
LAT_RANGE = (15, 56)


# ============================================================
# OCR 单例（懒加载，避免 --help 也加载模型）
# ============================================================
_ocr = None


def get_ocr():
    global _ocr
    if _ocr is None:
        from paddleocr import PaddleOCR
        # 关键：
        # 1) 关闭 UVDoc / 方向分类（否则窄条被旋转到几万像素）
        # 2) text_det_limit_side_len=960：阻止 PaddleOCR v5 内部 ResizeNorm
        #    把 240x2800 的窄图先放大 4 倍再送入检测；我们自己先做预降采样到 1400 长边，
        #    数字 30-80px 已足够，精度基本无损
        # 3) 显式使用 server 版模型（本地已缓存，mobile 模型沙箱禁止写入 ~/.paddlex）
        kwargs = dict(
            lang="ch",
            doc_orientation_classify_model_name=None,
            doc_orientation_classify_model_dir=None,
            doc_unwarping_model_name=None,
            doc_unwarping_model_dir=None,
            use_doc_orientation_classify=False,
            use_doc_unwarping=False,
            use_textline_orientation=False,
            textline_orientation_model_name=None,
            text_detection_model_name="PP-OCRv5_server_det",
            text_recognition_model_name="PP-OCRv5_server_rec",
            text_det_unclip_ratio=1.8,
            text_det_limit_side_len=1600,
            text_det_limit_type="max",
            text_rec_score_thresh=0.3,
        )
        try:
            _ocr = PaddleOCR(**kwargs)
        except ValueError:
            # 旧版 PaddleOCR 参数名兼容
            kwargs_old = dict(
                use_angle_cls=False, lang="ch",
                det_db_unclip_ratio=1.8, det_limit_side_len=1600,
            )
            _ocr = PaddleOCR(**kwargs_old)
    return _ocr


def ocr_digit_region(img_bgr, max_long_side: int = 2400) -> list:
    """对裁切区域做 OCR，返回 [{text,cx,cy,w,h,conf}, ...]

    额外做一次"长边缩到 max_long_side 以下"的预降采样：
      - 上下刻度条（2000x200）压到 2400×240 → 数字高度约 25-35px，够识别
      - 左右刻度条（240x2800）压到 240×2400 → 数字宽度约 30-40px，够识别
    返回的 cx/cy/w/h 全部还原到原始图像坐标。
    """
    if img_bgr is None or img_bgr.size == 0:
        return []
    h0, w0 = img_bgr.shape[:2]
    long_side = max(w0, h0)
    scale = 1.0
    if long_side > max_long_side:
        scale = max_long_side / long_side
        img = cv2.resize(
            img_bgr, (int(round(w0 * scale)), int(round(h0 * scale))),
            interpolation=cv2.INTER_AREA,
        )
    else:
        img = img_bgr
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    ocr = get_ocr()
    try:
        result = ocr.predict(img_rgb)
    except Exception:
        try:
            result = ocr.ocr(img_rgb, cls=False)
        except Exception:
            result = ocr.ocr(img_rgb)

    def _bbox_to_xywh(bbox):
        arr = np.asarray(bbox, dtype=np.float32)
        if arr.size == 0:
            return None
        if arr.ndim == 1 and arr.size == 4:
            x1, y1, x2, y2 = arr.tolist()
        else:
            arr = arr.reshape(-1, 2)
            x1, y1 = arr.min(axis=0).tolist()
            x2, y2 = arr.max(axis=0).tolist()
        return (x1, y1, x2, y2)

    items = []

    # ---- PaddleOCR v5 predict() 新格式 ----
    if isinstance(result, list) and result and isinstance(result[0], dict):
        page = result[0]
        texts = list(page.get("rec_texts") or [])
        scores = list(page.get("rec_scores") or [])
        # rec_boxes / dt_polys / rec_polys 都是候选（numpy 不能直接 or 链，手动判空）
        boxes = None
        for key in ("rec_boxes", "rec_polys", "dt_polys"):
            val = page.get(key)
            if val is None:
                continue
            if isinstance(val, np.ndarray) and val.size == 0:
                continue
            if isinstance(val, (list, tuple)) and len(val) == 0:
                continue
            boxes = val
            break
        if boxes is None:
            boxes = []
        n = max(len(texts), len(boxes), len(scores))
        for i in range(n):
            text = str(texts[i]) if i < len(texts) else ""
            conf = float(scores[i]) if i < len(scores) and scores[i] is not None else 0.5
            if conf < 0.35:
                continue
            bbox = boxes[i] if i < len(boxes) else None
            if bbox is None:
                continue
            try:
                # dt_polys 里常常是 list[list[ndarray]] — 降一层
                if isinstance(bbox, list) and len(bbox) == 1 and not np.isscalar(bbox[0]):
                    bbox = bbox[0]
                xywh = _bbox_to_xywh(bbox)
            except Exception:
                continue
            if xywh is None:
                continue
            x1, y1, x2, y2 = xywh
            cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
            # 还原到调用方传入的原始 img_bgr 坐标
            if scale != 1.0:
                cx /= scale; cy /= scale
                w = (x2 - x1) / scale; h = (y2 - y1) / scale
            else:
                w, h = x2 - x1, y2 - y1
            items.append({"text": text, "cx": cx, "cy": cy, "w": w, "h": h, "conf": conf})
        return items

    # ---- 旧版 PaddleOCR ocr() 格式: list[ [ [box, [text,conf]], ... ] ] ----
    if isinstance(result, list):
        page = None
        for candidate in result:
            if isinstance(candidate, list):
                page = candidate
                break
        if page:
            for line in page:
                try:
                    if isinstance(line, list) and len(line) >= 2:
                        box = line[0]
                        text_rec = line[1]
                        text = str(text_rec[0]) if isinstance(text_rec, (list, tuple)) else str(text_rec)
                        conf = float(text_rec[1]) if isinstance(text_rec, (list, tuple)) and len(text_rec) > 1 else 0.5
                    elif isinstance(line, dict):
                        box = line.get("bbox") or line.get("polygon")
                        text = str(line.get("text", ""))
                        conf = float(line.get("confidence", 0.5))
                    else:
                        continue
                    if conf < 0.35:
                        continue
                    xywh = _bbox_to_xywh(box)
                    if xywh is None:
                        continue
                except Exception:
                    continue
                x1, y1, x2, y2 = xywh
                cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
                if scale != 1.0:
                    cx /= scale; cy /= scale
                    w = (x2 - x1) / scale; h = (y2 - y1) / scale
                else:
                    w, h = x2 - x1, y2 - y1
                items.append({"text": text, "cx": cx, "cy": cy, "w": w, "h": h, "conf": conf})
    return items


# ============================================================
# 数字提取与纠错
# ============================================================
NUM_RE = re.compile(r"\d+(?:\.\d+)?")

CHAR_FIX = {
    'O': '0', 'Q': '0', 'D': '0',
    'I': '1', 'l': '1', '|': '1', '！': '1',
    'Z': '2', 'S': '5', 'B': '8', 'G': '6', 'g': '9',
    'A': '4', 'T': '7', 'Y': '4', 'b': '6',
    '。': '.', '，': '.', ',': '.', ' ': '',
}


def normalize_text(s: str) -> str:
    s = s.strip()
    for k, v in CHAR_FIX.items():
        s = s.replace(k, v)
    return s


def extract_lon_candidates(items: list, offset_x: int, offset_y: int) -> list:
    """返回 [(度数float, 全局cx, 全局cy)]"""
    out = []
    for it in items:
        norm = normalize_text(it["text"])
        for m in NUM_RE.findall(norm):
            try:
                v = float(m)
            except ValueError:
                continue
            # 经度常见是 1xx 或 两位数(带 °E 时OCR容易漏前导1)。若 <70 可能是纬度或非刻度，跳过
            if not (LON_RANGE[0] - 10 <= v <= LON_RANGE[1] + 10):
                continue
            out.append((v, it["cx"] + offset_x, it["cy"] + offset_y))
    return out


def extract_lat_candidates(items: list, offset_x: int, offset_y: int) -> list:
    out = []
    for it in items:
        norm = normalize_text(it["text"])
        for m in NUM_RE.findall(norm):
            try:
                v = float(m)
            except ValueError:
                continue
            if not (LAT_RANGE[0] - 5 <= v <= LAT_RANGE[1] + 5):
                continue
            out.append((v, it["cx"] + offset_x, it["cy"] + offset_y))
    return out


# ============================================================
# 分割线 → 主图/插图 区域判定
# ============================================================
def compute_regions(splitters: list, img_h: int, img_w: int):
    """
    从 labelme splitter（分割线围成的插图框）计算：
      - inset_boxes: 每个插图区域的 (x1,y1,x2,y2)
      - main_box:    主图实际边界（由最长经线/纬线的端点包围盒确定，作为刻度裁剪基准）
    若没有 splitter，则主图就是整张图。
    """
    inset_boxes = []
    if splitters:
        for sp in splitters:
            arr = np.asarray(sp, dtype=np.float32).reshape(-1, 2)
            inset_boxes.append((
                int(arr[:, 0].min()), int(arr[:, 1].min()),
                int(arr[:, 0].max()), int(arr[:, 1].max()),
            ))

    def in_any_inset(x, y, margin=0):
        for (x1, y1, x2, y2) in inset_boxes:
            if (x1 - margin) <= x <= (x2 + margin) and (y1 - margin) <= y <= (y2 + margin):
                return True
        return False

    # main_box 初始 = 全图
    main_box = [0, 0, img_w - 1, img_h - 1]
    return inset_boxes, in_any_inset, main_box


def refine_main_box_by_lines(main_box: list, v_lines: list, h_arcs: list):
    """
    根据"保留下来的主图经纬线端点"收紧主图边界：
      左右边界 = 最长 5 条经线 bot_x 的 min/max
      上下边界 = 最长 5 条纬线左右端点 y 的 min/max
    """
    if not v_lines and not h_arcs:
        return main_box
    # 经线端点
    v_xs, v_ys = [], []
    for v in sorted(v_lines, key=lambda l: l.y_span, reverse=True)[:10]:
        v_xs.extend([v.bottom_point[0], v.top_point[0]])
        v_ys.extend([v.bottom_point[1], v.top_point[1]])
    # 纬线端点
    h_xs, h_ys = [], []
    for h in sorted(h_arcs, key=lambda l: l.x_span, reverse=True)[:10]:
        pts = h.points.reshape(-1, 2)
        h_xs.extend([pts[0, 0], pts[-1, 0]])
        h_ys.extend([pts[0, 1], pts[-1, 1]])

    xs = v_xs + h_xs
    ys = v_ys + h_ys
    if not xs or not ys:
        return main_box
    x1 = int(max(0, min(main_box[0], min(xs))))
    y1 = int(max(0, min(main_box[1], min(ys))))
    x2 = int(min(main_box[2], max(xs)))
    y2 = int(min(main_box[3], max(ys)))
    return [x1, y1, x2, y2]


# ============================================================
# 刻度区域裁剪（约束在主图边界内 + 自动避开插图）
# ============================================================
def _clip(x, lo, hi):
    return int(max(lo, min(hi, x)))


def crop_bottom_strip(img, main_box=None, h_strip=200):
    """只裁主图底部 h_strip 高的一条（不进入插图区的部分）"""
    H, W = img.shape[:2]
    x1, y1, x2, y2 = main_box if main_box else (0, 0, W - 1, H - 1)
    cy1 = _clip(y2 - h_strip + 1, 0, H - 1)
    cy2 = _clip(y2 + 1, 0, H)
    cx1 = _clip(x1, 0, W - 1)
    cx2 = _clip(x2 + 1, 0, W)
    return img[cy1:cy2, cx1:cx2, :], cx1, cy1


def crop_top_strip(img, main_box=None, h_strip=200):
    H, W = img.shape[:2]
    x1, y1, x2, y2 = main_box if main_box else (0, 0, W - 1, H - 1)
    cy1 = _clip(y1, 0, H - 1)
    cy2 = _clip(y1 + h_strip, 0, H)
    cx1 = _clip(x1, 0, W - 1)
    cx2 = _clip(x2 + 1, 0, W)
    return img[cy1:cy2, cx1:cx2, :], cx1, cy1


def crop_left_strip(img, main_box=None, w_strip=240):
    H, W = img.shape[:2]
    x1, y1, x2, y2 = main_box if main_box else (0, 0, W - 1, H - 1)
    cx1 = _clip(x1, 0, W - 1)
    cx2 = _clip(x1 + w_strip, 0, W)
    cy1 = _clip(y1, 0, H - 1)
    cy2 = _clip(y2 + 1, 0, H)
    return img[cy1:cy2, cx1:cx2, :], cx1, cy1


def crop_right_strip(img, main_box=None, w_strip=240):
    H, W = img.shape[:2]
    x1, y1, x2, y2 = main_box if main_box else (0, 0, W - 1, H - 1)
    cx1 = _clip(x2 - w_strip + 1, 0, W - 1)
    cx2 = _clip(x2 + 1, 0, W)
    cy1 = _clip(y1, 0, H - 1)
    cy2 = _clip(y2 + 1, 0, H)
    return img[cy1:cy2, cx1:cx2, :], cx1, cy1


def filter_inset_candidates(candidates: list, in_any_inset, margin: int = 18) -> list:
    """过滤"中心点落在插图框内或边缘"的OCR数字候选（避免把插图里的序号/日期当作刻度）"""
    if not in_any_inset:
        return candidates
    out = []
    for deg, cx, cy in candidates:
        if in_any_inset(cx, cy, margin=margin):
            continue
        out.append((deg, cx, cy))
    return out


# ============================================================
# 最近邻匹配 + 等差一致性校验
# ============================================================
def match_lines(lines_deg_xcyc: list, lines_sorted_by_coord: list,
                axis: str, threshold: float = 150.0) -> dict:
    """
    匹配 OCR 识别的度数到 labelme 线列表。
    axis='x' 表示用 cx 去匹配经线的 x_mid / bot_x
    axis='y' 表示用 cy 去匹配纬线的 y_mean / 端点 y
    返回 {线索引: 度数}
    """
    matched = {}
    used = set()
    # 按距离从小到大做稳定匹配
    pairs = []
    for i_deg, (deg, lx, ly) in enumerate(lines_deg_xcyc):
        for j_line, line in enumerate(lines_sorted_by_coord):
            if axis == 'x':
                line_coord = getattr(line, 'bot_x_for_match', line.x_mid)
                dist = abs(lx - line_coord)
            else:
                line_coord = getattr(line, 'y_for_match', line.y_mean)
                dist = abs(ly - line_coord)
            pairs.append((dist, i_deg, j_line, deg))
    pairs.sort()
    for dist, i_deg, j_line, deg in pairs:
        if dist > threshold:
            break
        if i_deg in used or j_line in matched:
            continue
        matched[j_line] = deg
        used.add(i_deg)
    return matched


def verify_arithmetic(matches: dict, n_lines: int, min_anchors: int = 3):
    """从匹配集合中选取等差一致性最高的子集，返回最终 anchors dict（至少 min_anchors 个点）"""
    if len(matches) < min_anchors:
        return matches
    indices = sorted(matches.keys())
    # 穷举任意两点作为基准，检查其余点的偏差
    best = (1e9, None)
    for i in range(len(indices)):
        for j in range(i + 1, len(indices)):
            ia, ib = indices[i], indices[j]
            if ia == ib:
                continue
            da, db = matches[ia], matches[ib]
            interval = (db - da) / (ib - ia)
            if interval == 0:
                continue
            # 收集所有支持的锚点(残差 < 0.5°)
            support = {}
            for idx in indices:
                deg_est = da + (idx - ia) * interval
                deg_act = matches[idx]
                # 保留整度，先四舍五入
                if abs(interval) >= 1:
                    deg_est_r = round(deg_est)
                else:
                    deg_est_r = round(deg_est, 1)
                if abs(deg_est_r - deg_act) <= 0.51:
                    support[idx] = deg_est_r
            if len(support) >= min_anchors:
                # 评分：残差均方 + 奖励点数多
                resid = np.std([support[idx] - (da + (idx - ia) * interval) for idx in support])
                score = resid * 10 - len(support) * 0.5
                if score < best[0]:
                    best = (score, support)
    return best[1] if best[1] is not None else matches


# ============================================================
# 主流程（单图）
# ============================================================
def process_one(image_path: Path, json_path: Path):
    img = cv2.imread(str(image_path))
    if img is None:
        # 兼容中文路径
        pil = Image.open(str(image_path)).convert("RGB")
        img = cv2.cvtColor(np.array(pil), cv2.COLOR_RGB2BGR)
    img_h, img_w = img.shape[:2]

    data = json.loads(json_path.read_text(encoding="utf-8"))
    v_lines_raw, h_arcs_raw, splitters = parse_shapes(
        data["shapes"], data["imageHeight"], data["imageWidth"]
    )

    # ---------- 分割线约束：先计算插图/主图边界 ----------
    inset_boxes, in_any_inset, main_box = compute_regions(splitters, img_h, img_w)
    debug_info = {
        "n_splitters": len(splitters),
        "inset_boxes": inset_boxes,
    }

    # 过滤掉完全落在插图里的经纬线（保留主图线）
    if splitters:
        v_lines, h_arcs = filter_inset_lines(
            v_lines_raw, h_arcs_raw, splitters, img_h, img_w
        )
    else:
        v_lines, h_arcs = list(v_lines_raw), list(h_arcs_raw)
    v_lines, h_arcs = sort_and_dedup(v_lines, h_arcs)

    # 用主图最长的那些经纬线端点收紧 main_box（避免按全图裁到插图区留白或插图内部的"伪刻度"）
    main_box = refine_main_box_by_lines(main_box, v_lines, h_arcs)
    debug_info["main_box"] = tuple(main_box)

    # 给每条线预计算"匹配用坐标"（仅主图范围内）
    for v in v_lines:
        v.bot_x_for_match = v.bottom_point[0]
        v.top_x_for_match = v.top_point[0]
    for h in h_arcs:
        arr = h.points
        h.left_y_for_match = arr[0, 1]
        h.right_y_for_match = arr[-1, 1]
        h.y_for_match = (arr[0, 1] + arr[-1, 1]) / 2

    # ---------- 经度（经线）：仅在主图上下边缘做 OCR，且过滤落在插图内的候选 ----------
    lon_all = []
    for crop_fn in (crop_bottom_strip, crop_top_strip):
        crop, off_x, off_y = crop_fn(img, main_box)
        items = ocr_digit_region(crop)
        lon_all.extend(extract_lon_candidates(items, off_x, off_y))
    n_lon_before_filter = len(lon_all)
    lon_all = filter_inset_candidates(lon_all, in_any_inset, margin=18)
    debug_info["lon_removed_by_inset"] = n_lon_before_filter - len(lon_all)

    v_matches = {}
    for axis_name in ("bot_x_for_match", "top_x_for_match"):
        lines_with_attr = []
        for v in v_lines:
            class _Proxy:
                pass
            p = _Proxy()
            p.x_mid = getattr(v, axis_name)
            lines_with_attr.append(p)
        res = match_lines(lon_all, lines_with_attr, axis="x", threshold=180)
        for idx, deg in res.items():
            if idx in v_matches:
                prev = v_matches[idx]
                if abs(interval_hint(v_matches, idx, deg)) < abs(interval_hint(v_matches, idx, prev)):
                    v_matches[idx] = deg
            else:
                v_matches[idx] = deg

    # ---------- 纬度（纬线）：仅在主图左右边缘做 OCR，过滤落在插图内的候选 ----------
    lat_all = []
    for crop_fn in (crop_left_strip, crop_right_strip):
        crop, off_x, off_y = crop_fn(img, main_box)
        items = ocr_digit_region(crop)
        lat_all.extend(extract_lat_candidates(items, off_x, off_y))
    n_lat_before_filter = len(lat_all)
    lat_all = filter_inset_candidates(lat_all, in_any_inset, margin=18)
    debug_info["lat_removed_by_inset"] = n_lat_before_filter - len(lat_all)

    h_matches = {}
    for attr_name in ("y_for_match", "left_y_for_match", "right_y_for_match"):
        lines_with_attr = []
        for h in h_arcs:
            class _Proxy:
                pass
            p = _Proxy()
            p.y_mean = getattr(h, attr_name)
            lines_with_attr.append(p)
        res = match_lines(lat_all, lines_with_attr, axis="y", threshold=160)
        for idx, deg in res.items():
            if idx in h_matches:
                # 两端来源冲突时暂以主y为准（第一个匹配）
                continue
            h_matches[idx] = round(deg) if deg == round(deg) else deg

    # 整度数化 + 两位数经度补前导 1
    def fix_lon(idx, deg):
        if deg < 80 and deg > 10:
            deg2 = deg + 100
            if LON_RANGE[0] <= deg2 <= LON_RANGE[1]:
                return deg2
        return deg

    v_matches = {i: fix_lon(i, round(d)) for i, d in v_matches.items()}
    h_matches = {i: round(d) for i, d in h_matches.items()}

    # 等差一致性精修（主图边界上不存在的刻度 → 等差推断补足首末）
    v_anchors = verify_arithmetic(v_matches, len(v_lines), min_anchors=2)
    h_anchors = verify_arithmetic(h_matches, len(h_arcs), min_anchors=2)

    return {
        "image": image_path.name,
        "n_v": len(v_lines),
        "n_h": len(h_arcs),
        "ocr_lon_hits": len(lon_all),
        "ocr_lat_hits": len(lat_all),
        "v_raw_matches": v_matches,
        "h_raw_matches": h_matches,
        "v_anchors": v_anchors,
        "h_anchors": h_anchors,
        "debug": debug_info,
    }


def interval_hint(matches: dict, idx: int, new_deg: float) -> float:
    """返回加入新值后与最近邻居的间距残差（供选择更一致的那一个）"""
    items = sorted(matches.items())
    if not items:
        return 0
    tmp = dict(items)
    tmp[idx] = new_deg
    ks = sorted(tmp.keys())
    diffs = [tmp[ks[i+1]] - tmp[ks[i]] for i in range(len(ks)-1)]
    if not diffs:
        return 0
    return float(np.std(diffs))


def print_report(r):
    print("\n" + "=" * 70)
    print(f"图片: {r['image']}   经线{r['n_v']}条  纬线{r['n_h']}条")
    d = r.get("debug", {})
    extra_bits = []
    if d.get("n_splitters"):
        extra_bits.append(f"插图框:{d['n_splitters']}个")
    if d.get("main_box"):
        x1, y1, x2, y2 = d["main_box"]
        extra_bits.append(f"主图box:({x1},{y1})-({x2},{y2})")
    if d.get("lon_removed_by_inset") or d.get("lat_removed_by_inset"):
        extra_bits.append(
            f"插图过滤掉: 经度{d.get('lon_removed_by_inset',0)} 纬度{d.get('lat_removed_by_inset',0)}"
        )
    if extra_bits:
        print("  [" + " | ".join(extra_bits) + "]")
    print(f"OCR 数字候选: 经度{r['ocr_lon_hits']}个 / 纬度{r['ocr_lat_hits']}个")
    print("-" * 70)
    if r["v_anchors"]:
        vs = sorted(r["v_anchors"].items())
        span = f"{vs[0][1]}~{vs[-1][1]}°E"
        interval = (vs[-1][1] - vs[0][1]) / (vs[-1][0] - vs[0][0]) if len(vs) >= 2 else None
        print(f"✅ 经线锚点({len(vs)}个): {vs}   → 范围 {span}  间距 {interval}°")
    else:
        print("❌ 经线锚点: 未能确定（首末刻度不落在主图边界→请人工补锚点后等差推断）")
    if r["h_anchors"]:
        hs = sorted(r["h_anchors"].items())
        span = f"{hs[0][1]}~{hs[-1][1]}°N"
        interval = (hs[-1][1] - hs[0][1]) / (hs[-1][0] - hs[0][0]) if len(hs) >= 2 else None
        print(f"✅ 纬线锚点({len(hs)}个): {hs}   → 范围 {span}  间距 {interval}°")
    else:
        print("❌ 纬线锚点: 未能确定（首末刻度不落在主图边界→请人工补锚点后等差推断）")
    if r["v_raw_matches"]:
        print(f"  原始经度匹配: {sorted(r['v_raw_matches'].items())}")
    if r["h_raw_matches"]:
        print(f"  原始纬度匹配: {sorted(r['h_raw_matches'].items())}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", type=Path)
    parser.add_argument("--json", type=Path)
    parser.add_argument("--dir", type=Path, help="批量模式：扫描该目录下的jpg+json")
    parser.add_argument("--pattern", default="08-*.jpg")
    args = parser.parse_args()

    if args.dir:
        jpgs = sorted(Path(args.dir).glob(args.pattern))
        print(f"批量: {args.dir}/{args.pattern}  →  {len(jpgs)} 张")
        for jpg in jpgs:
            js = jpg.with_suffix(".json")
            if not js.exists():
                print(f"[SKIP] {jpg.name}: 缺少json")
                continue
            try:
                r = process_one(jpg, js)
                print_report(r)
            except Exception as e:
                print(f"[ERR] {jpg.name}: {e}")
    else:
        if not args.image or not args.json:
            parser.error("请提供 --image + --json 或 --dir")
        r = process_one(args.image, args.json)
        print_report(r)


if __name__ == "__main__":
    main()
