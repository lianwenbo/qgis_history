"""
LabelMe 标注排序整理工具

手动编辑 labelme JSON 后运行，将 shapes 按类型分组、按空间位置排序，
使 LabelMe 右侧列表展示有条理。

排序规则：
  1. 按 label 分组：boundary_1 在前，boundary_2 在后，其他标签最后
  2. 组内按空间位置排序：先按线条最上端 y（从上到下），再按最左端 x（从左到右）
  3. 同组内可选择按线条长度排序（--by-length）

用法：
    # 整理单个文件
    python boundaries/sort_labelme_shapes.py boundaries/chgis_labelme/04-青徐兖豫四州刺史部.json

    # 整理目录下所有 json
    python boundaries/sort_labelme_shapes.py boundaries/chgis_labelme/

    # 按长度排序（长的在前）
    python boundaries/sort_labelme_shapes.py boundaries/chgis_labelme/xxx.json --by-length
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np


LABEL_ORDER = {"boundary_1": 0, "boundary_2": 1}


def shape_sort_key(shape, by_length: bool = False):
    """生成 shape 的排序键。

    返回 (label_order, y_min, x_min) 或 (label_order, -length)
    """
    label = shape.get("label", "")
    label_order = LABEL_ORDER.get(label, 99)

    pts = shape.get("points", [])
    if not pts:
        return (label_order, 0, 0)

    pts = np.array(pts, dtype=np.float64)
    y_min = float(pts[:, 1].min())
    x_min = float(pts[:, 0].min())

    if by_length:
        if len(pts) >= 2:
            diffs = np.diff(pts, axis=0)
            length = float(np.sqrt((diffs ** 2).sum(axis=1)).sum())
        else:
            length = 0.0
        return (label_order, -length, y_min, x_min)

    return (label_order, y_min, x_min)


def sort_shapes(shapes: list, by_length: bool = False) -> list:
    return sorted(shapes, key=lambda s: shape_sort_key(s, by_length))


def process_file(json_path: Path, by_length: bool = False, dry_run: bool = False) -> dict:
    with open(str(json_path), "r", encoding="utf-8") as f:
        data = json.load(f)

    old_shapes = data.get("shapes", [])
    new_shapes = sort_shapes(old_shapes, by_length)

    # 统计
    from collections import Counter
    old_counts = Counter(s.get("label", "?") for s in old_shapes)
    new_counts = Counter(s.get("label", "?") for s in new_shapes)

    changed = old_shapes != new_shapes
    if changed:
        data["shapes"] = new_shapes
        if not dry_run:
            with open(str(json_path), "w", encoding="utf-8") as f:
                json.dump(data, f, ensure_ascii=False, indent=2)

    return {
        "path": str(json_path),
        "changed": changed,
        "old_order": [s.get("label", "?") for s in old_shapes],
        "new_order": [s.get("label", "?") for s in new_shapes],
        "counts": dict(new_counts),
    }


def main():
    parser = argparse.ArgumentParser(description="LabelMe shapes 排序整理")
    parser.add_argument("path", type=str, help="JSON 文件或目录路径")
    parser.add_argument("--by-length", action="store_true",
                        help="同组内按线条长度降序排列（默认按空间位置从上到下、从左到右）")
    parser.add_argument("--dry-run", action="store_true",
                        help="只显示将要做的修改，不写文件")
    args = parser.parse_args()

    target = Path(args.path)
    if target.is_file():
        files = [target]
    elif target.is_dir():
        files = sorted(target.glob("*.json"))
        files = [f for f in files if not f.name.startswith("._")]
    else:
        print(f"❌ 路径不存在: {target}")
        sys.exit(1)

    if not files:
        print("❌ 未找到 JSON 文件")
        sys.exit(1)

    print(f"共 {len(files)} 个文件")
    print("=" * 60)

    total_changed = 0
    for fp in files:
        result = process_file(fp, by_length=args.by_length, dry_run=args.dry_run)
        status = "✏️  已排序" if result["changed"] else "  无需修改"
        dry_tag = " (dry-run)" if args.dry_run else ""
        counts_str = ", ".join(f"{k}:{v}" for k, v in sorted(result["counts"].items()))
        print(f"{status}{dry_tag}  {fp.name}")
        print(f"          {counts_str}")
        if result["changed"]:
            total_changed += 1

    print("=" * 60)
    print(f"完成: {total_changed}/{len(files)} 个文件需要排序")


if __name__ == "__main__":
    main()
