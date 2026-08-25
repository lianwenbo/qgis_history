#!/usr/bin/env python3
"""Generate LabelMe boundary lines from CHGIS polygons and QGIS GCPs."""

from __future__ import annotations

import argparse
import json
import math
import shutil
from pathlib import Path
from typing import Iterable, Iterator

import cv2
import geopandas as gpd
import numpy as np
import pandas as pd
from PIL import Image
from shapely.geometry import (
    GeometryCollection,
    LineString,
    MultiLineString,
    MultiPolygon,
    Polygon,
    box,
)
from shapely import make_valid
from shapely.ops import linemerge, unary_union, snap


DEFAULT_PREFECTURE = Path(
    "/Users/bytedance/Work/historical_map/复旦历史地图/"
    "州府界/PII_Boun_1820_Pref.TAB"
)
DEFAULT_PROVINCE = Path(
    "/Users/bytedance/Work/historical_map/复旦历史地图/"
    "省界/PII_Boun_1820_Prov.TAB"
)
DEFAULT_IMAGE = Path(
    "/Users/bytedance/Work/qgis_only/map_line_dataset/gcp/08-48云南.jpg"
)
DEFAULT_GCP = Path(
    "/Users/bytedance/Work/qgis_only/map_line_dataset/gcp/08-48云南.jpg.points"
)
DEFAULT_OUTPUT_DIR = Path(
    "/Users/bytedance/Work/qgis_only/boundaries/chgis_labelme"
)
DEFAULT_PREVIEW_DIR = Path(
    "/Users/bytedance/Work/qgis_only/boundaries/chgis_output"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="将CHGIS省界/州府界逆向映射到原始扫描图并生成LabelMe。",
    )
    parser.add_argument("--province-name", default="云南")
    parser.add_argument("--prefecture-data", type=Path, default=DEFAULT_PREFECTURE)
    parser.add_argument("--province-data", type=Path, default=DEFAULT_PROVINCE)
    parser.add_argument("--image", type=Path, default=DEFAULT_IMAGE)
    parser.add_argument("--gcp", type=Path, default=DEFAULT_GCP)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--preview-dir", type=Path, default=DEFAULT_PREVIEW_DIR)
    parser.add_argument(
        "--densify-degrees",
        type=float,
        default=0.01,
        help="地理空间加密间隔（度），默认约对应原图2-3像素。",
    )
    parser.add_argument(
        "--min-line-length-px",
        type=float,
        default=8.0,
        help="裁剪后保留线段的最小像素长度（过滤图幅边缘碎段）。",
    )
    parser.add_argument(
        "--simplify-px",
        type=float,
        default=2.0,
        help="全局合并后折线的D-P简化容差（像素），默认2px。",
    )
    return parser.parse_args()


def validate_inputs(args: argparse.Namespace) -> None:
    for path in (
        args.prefecture_data,
        args.province_data,
        args.image,
        args.gcp,
    ):
        if not path.is_file():
            raise FileNotFoundError(path)
    if args.densify_degrees <= 0:
        raise ValueError("--densify-degrees 必须大于0")
    if args.min_line_length_px < 0:
        raise ValueError("--min-line-length-px 不能小于0")
    if args.simplify_px < 0:
        raise ValueError("--simplify-px 不能小于0")


def polynomial_terms(points: np.ndarray) -> np.ndarray:
    """Return normalized second-order polynomial terms."""
    x = points[:, 0]
    y = points[:, 1]
    return np.column_stack(
        [
            np.ones(len(points)),
            x,
            y,
            x * x,
            x * y,
            y * y,
        ]
    )


class Polynomial2GeoToPixel:
    """Second-order inverse transform fitted from QGIS GCP rows."""

    def __init__(
        self,
        mean: np.ndarray,
        scale: np.ndarray,
        coefficients: np.ndarray,
        residuals: np.ndarray,
    ) -> None:
        self.mean = mean
        self.scale = scale
        self.coefficients = coefficients
        self.residuals = residuals

    @classmethod
    def from_qgis_points(cls, points_path: Path) -> "Polynomial2GeoToPixel":
        frame = pd.read_csv(points_path, comment="#")
        required = {"mapX", "mapY", "sourceX", "sourceY", "enable"}
        missing = required - set(frame.columns)
        if missing:
            raise ValueError(f"控制点缺少字段：{sorted(missing)}")

        enabled = frame.loc[frame["enable"].eq(1)].copy()
        if len(enabled) < 6:
            raise ValueError("二阶多项式至少需要6个启用控制点")

        geographic = enabled[["mapX", "mapY"]].to_numpy(dtype=float)
        pixels = np.column_stack(
            [
                enabled["sourceX"].to_numpy(dtype=float),
                -enabled["sourceY"].to_numpy(dtype=float),
            ]
        )
        mean = geographic.mean(axis=0)
        scale = geographic.std(axis=0)
        if np.any(scale == 0):
            raise ValueError("控制点地理坐标退化，无法拟合二阶多项式")

        normalized = (geographic - mean) / scale
        design = polynomial_terms(normalized)
        coefficients, _, rank, _ = np.linalg.lstsq(design, pixels, rcond=None)
        if rank < 6:
            raise ValueError(f"二阶多项式设计矩阵秩不足：{rank}/6")

        predictions = design @ coefficients
        residuals = np.linalg.norm(predictions - pixels, axis=1)
        return cls(mean, scale, coefficients, residuals)

    def transform(self, points: np.ndarray) -> np.ndarray:
        normalized = (np.asarray(points, dtype=float) - self.mean) / self.scale
        return polynomial_terms(normalized) @ self.coefficients


def read_chgis(
    province_path: Path,
    prefecture_path: Path,
    province_name: str,
) -> tuple[LineString | MultiLineString, Polygon | MultiPolygon, gpd.GeoDataFrame]:
    province_frame = gpd.read_file(
        province_path,
        engine="pyogrio",
        use_arrow=False,
        on_invalid="ignore",
    )
    prefecture_frame = gpd.read_file(
        prefecture_path,
        engine="pyogrio",
        use_arrow=False,
        on_invalid="ignore",
    )
    if province_frame.crs is None or prefecture_frame.crs is None:
        raise ValueError("CHGIS图层缺少CRS")

    prefecture_rows = prefecture_frame.loc[
        prefecture_frame["LEV1_CH"].astype(str).str.strip().eq(province_name)
    ].copy()
    if prefecture_rows.empty:
        raise ValueError(f"州府界中未找到LEV1_CH={province_name!r}的记录")

    province_frame = province_frame.to_crs("EPSG:4326")
    prefecture_rows = prefecture_rows.to_crs("EPSG:4326")
    province_frame = province_frame.loc[
        province_frame.geometry.notna() & ~province_frame.geometry.is_empty
    ].copy()
    if province_frame.empty:
        raise ValueError("省界图层没有可用几何")
    province_frame["geometry"] = province_frame.geometry.apply(
        lambda geometry: geometry if geometry.is_valid else make_valid(geometry)
    )

    prefecture_rows = prefecture_rows.loc[
        prefecture_rows.geometry.notna() & ~prefecture_rows.geometry.is_empty
    ].copy()
    if not prefecture_rows.geometry.is_valid.all():
        prefecture_rows["geometry"] = prefecture_rows.geometry.apply(
            lambda geometry: geometry if geometry.is_valid else make_valid(geometry)
        )

    target_province = province_frame.loc[
        province_frame["NAME_CH"].astype(str).str.strip().eq(province_name)
    ]
    if len(target_province) != 1:
        raise ValueError(
            f"省界中应唯一匹配{province_name!r}，实际匹配{len(target_province)}条"
        )
    target_geometry = target_province.geometry.iloc[0]
    all_province_boundaries = unary_union(
        [geometry.boundary for geometry in province_frame.geometry]
    )
    if not all_province_boundaries.is_empty:
        all_province_boundaries = linemerge(all_province_boundaries)
    return all_province_boundaries, target_geometry, prefecture_rows


def build_boundary_linework(
    all_province_boundaries: LineString | MultiLineString,
    target_province_geometry: Polygon | MultiPolygon,
    prefecture_rows: gpd.GeoDataFrame,
) -> tuple[LineString | MultiLineString, LineString | MultiLineString]:
    prefecture_union = prefecture_rows.geometry.union_all()
    symmetric_area = target_province_geometry.symmetric_difference(
        prefecture_union
    ).area
    if symmetric_area > 2.0:
        raise ValueError(
            "州府界并集与省界不一致，不能可靠区分外边界和内部边界："
            f"symmetric_difference={symmetric_area:.12g}"
        )

    all_prefecture_boundaries = unary_union(
        [geometry.boundary for geometry in prefecture_rows.geometry]
    )
    internal = all_prefecture_boundaries.difference(
        target_province_geometry.boundary
    )
    if not internal.is_empty:
        internal = linemerge(internal)
    return all_province_boundaries, internal


def iter_lines(geometry: object) -> Iterator[LineString]:
    if geometry is None or getattr(geometry, "is_empty", True):
        return
    if isinstance(geometry, LineString):
        yield geometry
        return
    if isinstance(geometry, (MultiLineString, GeometryCollection)):
        for part in geometry.geoms:
            yield from iter_lines(part)


def densify_line(line: LineString, max_spacing: float) -> np.ndarray:
    coordinates = np.asarray(line.coords, dtype=float)
    output: list[np.ndarray] = [coordinates[0]]
    for start, end in zip(coordinates[:-1], coordinates[1:]):
        distance = float(np.linalg.norm(end - start))
        steps = max(1, int(math.ceil(distance / max_spacing)))
        for index in range(1, steps + 1):
            output.append(start + (end - start) * (index / steps))
    return np.asarray(output)


def transform_lines(
    geometry: object,
    transform_model: Polynomial2GeoToPixel,
    densify_degrees: float,
) -> list[LineString]:
    raw_lines: list[LineString] = []
    for line in iter_lines(geometry):
        geographic_points = densify_line(line, densify_degrees)
        pixel_points = transform_model.transform(geographic_points)
        if len(pixel_points) < 2:
            continue
        pixel_line = LineString(pixel_points)
        if not pixel_line.is_empty and pixel_line.length > 0:
            raw_lines.append(pixel_line)
    return raw_lines


def merge_clean_lines(
    raw_lines: list[LineString],
    image_width: int,
    image_height: int,
    simplify_px: float = 2.0,
    min_length_px: float = 8.0,
) -> list[LineString]:
    if not raw_lines:
        return []
    image_box = box(0.0, 0.0, image_width - 1.0, image_height - 1.0)

    pre_simplified = [
        line.simplify(1.0, preserve_topology=True)
        for line in raw_lines
        if not line.is_empty and line.length > 0
    ]
    pre_simplified = [line for line in pre_simplified if not line.is_empty and line.length > 0]

    merged = unary_union(pre_simplified)
    merged = linemerge(merged)

    output: list[LineString] = []
    for line in iter_lines(merged):
        simplified = line.simplify(simplify_px, preserve_topology=True)
        for simple_part in iter_lines(simplified):
            clipped = simple_part.intersection(image_box)
            for clip_part in iter_lines(clipped):
                if clip_part.length >= min_length_px and len(clip_part.coords) >= 2:
                    output.append(clip_part)
    return output


def shape_from_line(
    line: LineString,
    label: str,
    source_layer: str,
    description: str,
) -> dict:
    points = [
        [round(float(x), 3), round(float(y), 3)]
        for x, y in line.coords
    ]
    return {
        "label": label,
        "points": points,
        "group_id": None,
        "description": description,
        "shape_type": "linestrip",
        "flags": {
            "source": "Fudan_CHGIS_1820",
            "source_layer": source_layer,
            "transform": "polynomial_2_inverse_gcp",
        },
        "mask": None,
    }


def create_labelme(
    image_name: str,
    image_width: int,
    image_height: int,
    province_name: str,
    outer_lines: Iterable[LineString],
    internal_lines: Iterable[LineString],
    model: Polynomial2GeoToPixel,
    prefecture_count: int,
) -> dict:
    shapes = [
        shape_from_line(
            line,
            "boundary_1",
            "PII_Boun_1820_Prov",
            "当前图幅覆盖范围内的省界",
        )
        for line in outer_lines
    ]
    shapes.extend(
        shape_from_line(
            line,
            "boundary_2",
            "PII_Boun_1820_Pref",
            f"{province_name}州府内部边界（公共边已去重）",
        )
        for line in internal_lines
    )
    return {
        "version": "5.4.1",
        "flags": {},
        "shapes": shapes,
        "imagePath": image_name,
        "imageData": None,
        "imageHeight": image_height,
        "imageWidth": image_width,
        "description": f"source=Fudan_CHGIS_1820; province={province_name}; "
                       f"gcp_count={len(model.residuals)}; "
                       f"gcp_mean_residual_px={float(model.residuals.mean()):.3f}; "
                       f"gcp_max_residual_px={float(model.residuals.max()):.3f}",
    }


def write_preview(
    image_path: Path,
    output_path: Path,
    outer_lines: Iterable[LineString],
    internal_lines: Iterable[LineString],
) -> None:
    image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"OpenCV无法读取图像：{image_path}")

    overlay = image.copy()
    for line in internal_lines:
        points = np.rint(np.asarray(line.coords)).astype(np.int32).reshape((-1, 1, 2))
        cv2.polylines(overlay, [points], False, (0, 255, 0), 2, cv2.LINE_AA)
    for line in outer_lines:
        points = np.rint(np.asarray(line.coords)).astype(np.int32).reshape((-1, 1, 2))
        cv2.polylines(overlay, [points], False, (0, 0, 255), 3, cv2.LINE_AA)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(output_path), overlay):
        raise OSError(f"无法写入预览图：{output_path}")


def main() -> None:
    args = parse_args()
    validate_inputs(args)

    with Image.open(args.image) as image:
        width, height = image.size

    model = Polynomial2GeoToPixel.from_qgis_points(args.gcp)
    all_province_boundaries, target_province_geometry, prefecture_rows = read_chgis(
        args.province_data,
        args.prefecture_data,
        args.province_name,
    )
    outer_geographic, internal_geographic = build_boundary_linework(
        all_province_boundaries,
        target_province_geometry,
        prefecture_rows,
    )
    outer_raw = transform_lines(
        outer_geographic,
        model,
        args.densify_degrees,
    )
    internal_raw = transform_lines(
        internal_geographic,
        model,
        args.densify_degrees,
    )
    outer_lines = merge_clean_lines(
        outer_raw,
        width,
        height,
        simplify_px=args.simplify_px,
        min_length_px=args.min_line_length_px,
    )
    internal_lines = merge_clean_lines(
        internal_raw,
        width,
        height,
        simplify_px=args.simplify_px,
        min_length_px=args.min_line_length_px,
    )
    if not outer_lines:
        raise ValueError("省界映射后未落入原图范围")
    if not internal_lines:
        raise ValueError("州府内部边界映射后未落入原图范围")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    output_image = args.output_dir / args.image.name
    output_json = args.output_dir / f"{args.image.stem}.json"
    preview_path = args.preview_dir / f"{args.image.stem}_chgis_overlay.jpg"
    shutil.copy2(args.image, output_image)

    labelme = create_labelme(
        image_name=output_image.name,
        image_width=width,
        image_height=height,
        province_name=args.province_name,
        outer_lines=outer_lines,
        internal_lines=internal_lines,
        model=model,
        prefecture_count=len(prefecture_rows),
    )
    output_json.write_text(
        json.dumps(labelme, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    write_preview(args.image, preview_path, outer_lines, internal_lines)

    summary = {
        "image": str(output_image),
        "labelme": str(output_json),
        "preview": str(preview_path),
        "image_size": [width, height],
        "prefecture_features": len(prefecture_rows),
        "boundary_1_shapes": len(outer_lines),
        "boundary_2_shapes": len(internal_lines),
        "total_shapes": len(labelme["shapes"]),
        "gcp_count": len(model.residuals),
        "gcp_mean_residual_px": round(float(model.residuals.mean()), 6),
        "gcp_max_residual_px": round(float(model.residuals.max()), 6),
        "simplify_px": args.simplify_px,
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
