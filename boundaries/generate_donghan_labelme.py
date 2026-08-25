"""
东汉刺史部/郡国边界反投影生成 LabelMe

将手绘的东汉郡国多边形（按 cishibu 字段归属刺史部）dissolve 后，
利用 QGIS GCP 二阶多项式逆映射到扫描图像素坐标，输出 boundary_1（刺史部界）
和 boundary_2（郡国间内部界）两类 linestrip。

用法:
    python boundaries/generate_donghan_labelme.py \
        --image  map_line_dataset/gcp/04-青徐兖豫四州刺史部.jpg \
        --gcp    map_line_dataset/gcp/04-青徐兖豫四州刺史部.points \
        --shp    ~/Work/historical_map/donghan/东汉郡国图.shp \
        --cishibu 青州刺史部 徐州刺史部 兖州刺史部 豫州刺史部
"""

import argparse
import json
import shutil
import sys
from pathlib import Path

import cv2
import geopandas as gpd
import numpy as np
from PIL import Image
from shapely.geometry import GeometryCollection, LineString, MultiLineString
from shapely.validation import make_valid
from shapely.ops import linemerge, unary_union

sys.path.insert(0, str(Path(__file__).parent))
from generate_chgis_labelme import (
    Polynomial2GeoToPixel,
    densify_line,
    iter_lines,
    merge_clean_lines,
    transform_lines,
)


def dissolve_cishibu(gdf: gpd.GeoDataFrame, cishibu_names: list[str]):
    """按刺史部 dissolve，返回 (cishibu_geoms, county_geoms) 两个 GeoSeries。"""
    gdf = gdf.to_crs("EPSG:4326")
    gdf = gdf.loc[gdf.geometry.notna() & ~gdf.geometry.is_empty].copy()
    gdf["geometry"] = gdf.geometry.apply(
        lambda g: g if g.is_valid else make_valid(g)
    )

    target = gdf.loc[
        gdf["cishibu"].astype(str).str.strip().isin([n.strip() for n in cishibu_names])
    ].copy()
    if target.empty:
        raise ValueError(f"未找到任何指定刺史部: {cishibu_names}")
    found = sorted(target["cishibu"].unique().tolist())
    missing = set(cishibu_names) - set(found)
    if missing:
        print(f"  ⚠️ 未找到: {sorted(missing)}")
    print(f"  选中 {len(target)} 个郡/国, 归属 {len(found)} 个刺史部: {found}")

    dissolved = target.dissolve(by="cishibu")
    return dissolved.geometry, target.geometry


def build_linework(cishibu_geoms, county_geoms):
    """生成两级边界线。

    boundary_1 = 各刺史部多边形边界的并集（外轮廓 + 刺史部之间的分界）
    boundary_2 = 所有郡国边界线 - 刺史部级边界线（郡国之间的内部分界）
    """
    cishibu_lines = unary_union([g.boundary for g in cishibu_geoms])
    if not cishibu_lines.is_empty:
        cishibu_lines = linemerge(cishibu_lines)

    county_lines_all = unary_union([g.boundary for g in county_geoms])
    internal = county_lines_all.difference(cishibu_lines)
    if not internal.is_empty:
        internal = linemerge(internal)

    return cishibu_lines, internal


def line_to_shape(line: LineString, label: str, desc: str) -> dict:
    pts = [[round(float(x), 2), round(float(y), 2)] for x, y in line.coords]
    return {
        "label": label,
        "points": pts,
        "group_id": None,
        "description": desc,
        "shape_type": "linestrip",
        "flags": {},
    }


def visualize_preview(img_bgr, b1_lines, b2_lines, out_path: Path):
    vis = img_bgr.copy()
    for line in b1_lines:
        pts = np.array(line.coords, dtype=np.int32).reshape(-1, 1, 2)
        cv2.polylines(vis, [pts], False, (0, 0, 255), 2, lineType=cv2.LINE_AA)
    for line in b2_lines:
        pts = np.array(line.coords, dtype=np.int32).reshape(-1, 1, 2)
        cv2.polylines(vis, [pts], False, (0, 200, 0), 1, lineType=cv2.LINE_AA)
    cv2.imwrite(str(out_path), vis, [cv2.IMWRITE_JPEG_QUALITY, 90])


def main():
    ap = argparse.ArgumentParser(description="东汉刺史部边界反投影生成 LabelMe")
    ap.add_argument("--image", required=True, type=Path)
    ap.add_argument("--gcp", required=True, type=Path)
    ap.add_argument("--shp", required=True, type=Path)
    ap.add_argument("--out-dir", type=Path, default=Path("boundaries/chgis_labelme"))
    ap.add_argument("--preview-dir", type=Path, default=Path("boundaries/chgis_output"))
    ap.add_argument(
        "--cishibu", nargs="+",
        default=["青州刺史部", "徐州刺史部", "兖州刺史部", "豫州刺史部"],
    )
    ap.add_argument("--densify-degrees", type=float, default=0.01)
    ap.add_argument("--simplify-px", type=float, default=2.0)
    ap.add_argument("--min-line-length-px", type=float, default=8.0)
    args = ap.parse_args()

    for p in (args.image, args.gcp, args.shp):
        if not p.exists():
            raise FileNotFoundError(p)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    args.preview_dir.mkdir(parents=True, exist_ok=True)

    pil = Image.open(args.image).convert("RGB")
    w, h = pil.size
    img_bgr = cv2.cvtColor(np.array(pil), cv2.COLOR_RGB2BGR)
    print(f"图像: {args.image.name} ({w}x{h})")

    print("\n[1/5] 加载郡国矢量并 dissolve...")
    gdf = gpd.read_file(args.shp)
    cishibu_geoms, county_geoms = dissolve_cishibu(gdf, args.cishibu)

    print("\n[2/5] 构建两级边界...")
    b1_geo, b2_geo = build_linework(cishibu_geoms, county_geoms)

    print("\n[3/5] 拟合 GCP 二阶多项式逆映射...")
    model = Polynomial2GeoToPixel.from_qgis_points(args.gcp)
    print(f"  GCP 残差: mean={model.residuals.mean():.2f}px, "
          f"max={model.residuals.max():.2f}px")

    print("\n[4/5] 逆映射 + 合并简化...")
    b1_raw = transform_lines(b1_geo, model, args.densify_degrees)
    b2_raw = transform_lines(b2_geo, model, args.densify_degrees)
    b1_lines = merge_clean_lines(b1_raw, w, h, args.simplify_px, args.min_line_length_px)
    b2_lines = merge_clean_lines(b2_raw, w, h, args.simplify_px, args.min_line_length_px)
    print(f"  boundary_1(刺史部界): {len(b1_lines)} 条")
    print(f"  boundary_2(郡国界):  {len(b2_lines)} 条")

    print("\n[5/5] 写 LabelMe + 预览...")
    shapes = []
    for line in b1_lines:
        shapes.append(line_to_shape(line, "boundary_1", "东汉刺史部界"))
    for line in b2_lines:
        shapes.append(line_to_shape(line, "boundary_2", "东汉郡国间内部分界"))

    labelme = {
        "version": "5.3.1",
        "flags": {},
        "shapes": shapes,
        "imagePath": args.image.name,
        "imageData": None,
        "imageHeight": h,
        "imageWidth": w,
        "description": (
            f"source=donghan_junguo.shp; cishibu={'+'.join(args.cishibu)}; "
            f"gcp={args.gcp.name}; transform=polynomial_2_inverse_gcp"
        ),
    }
    out_json = args.out_dir / f"{args.image.stem}.json"
    out_jpg = args.out_dir / f"{args.image.stem}.jpg"
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(labelme, f, ensure_ascii=False, indent=2)
    if args.image.resolve() != out_jpg.resolve():
        shutil.copy2(args.image, out_jpg)

    preview_path = args.preview_dir / f"{args.image.stem}_donghan_overlay.jpg"
    visualize_preview(img_bgr, b1_lines, b2_lines, preview_path)

    print(f"\n✅ 输出: {out_json}")
    print(f"   预览: {preview_path}")
    print(f"   总计 {len(shapes)} 条线段 (b1={len(b1_lines)}, b2={len(b2_lines)})")


if __name__ == "__main__":
    main()
