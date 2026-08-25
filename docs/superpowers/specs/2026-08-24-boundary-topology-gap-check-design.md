# 行政边界拓扑检查与缺口标记设计

日期：2026-08-24
状态：已批准进入实现

## 1. 背景

当前 `boundaries/chgis_labelme/06-16河北东路.json` 包含：

- 38 条 `boundary_1`：红色，路级一级行政区划分界线；
- 53 条 `boundary_2`：绿色，州、府、军级二级行政区划分界线。

这些标注在视觉上基本贴合原图，但仍是彼此独立的开放 `linestrip`。直接执行矢量
`polygonize` 时，内部边界被识别为悬挂线，只能得到整张图框对应的一个面。要可靠地
生成二级行政区划面并恢复所属一级区划，必须先检查边界网络的连接关系，并标记需要
人工处理的缺口。

本阶段只实现只读检查和缺口标记，不自动修改原始 LabelMe 边界。

## 2. 目标

1. 检查一级红线网络的连续性。
2. 检查红线与绿线联合网络能否形成二级闭合面。
3. 找出未连接端点、近距离缺口、方向冲突和候选连接歧义。
4. 生成可在 LabelMe 中打开的审阅副本，并在缺口位置显示分类标记。
5. 生成机器可读报告和可视化预览，为下一阶段的保守吸附与 polygonize 提供输入。
6. 保证检查过程不覆盖、不重排、不修改原始人工标注。

## 3. 非目标

本阶段不包括：

- 自动移动或延长边界线；
- 自动删除、合并或改变边界等级；
- 生成最终一级、二级行政区划面；
- OCR、地名识别或行政区名称绑定；
- 计算实际平方公里面积；
- 将审阅副本自动回写训练集。

## 4. 输入

命令输入为一个 LabelMe JSON：

```bash
python boundaries/check_boundary_topology.py \
  boundaries/chgis_labelme/06-16河北东路.json
```

要求：

- JSON 包含 `imageWidth`、`imageHeight` 和 `imagePath`；
- 支持 `boundary_1` 和 `boundary_2` 的 `linestrip`；
- 每条参与检查的线至少包含两个有效像素坐标；
- 与 JSON 同目录的原图必须存在。

遇到未知标签时忽略并在报告中统计，不修改未知标注。

## 5. 两套拓扑网络

### 5.1 一级连续性网络

一级网络由以下几何构成：

```text
boundary_1 + 主图图框
```

用途：

- 检查路级红线是否连续；
- 标记红线自身的断点；
- 确认红线是否能够作为不同一级行政区之间的阻断边。

一级红线端点不能仅因靠近绿线就被判定为已连接。它必须连接到另一条红线、红线交点
或主图图框，否则即使二级网络闭合，一级区划分组仍可能错误。

### 5.2 二级闭合网络

二级网络由以下几何构成：

```text
boundary_1 + boundary_2 + 主图图框
```

用途：

- 检查州、府、军级最小行政面是否能够闭合；
- 允许绿色州界连接到红色路界；
- 在红绿交点和同级边界交点处分割线段；
- 为下一阶段生成二级最小面。

红线同时是一级边界和二级行政面的外边界，因此生成二级面时不能只使用绿线。

## 6. 几何预处理

检查器在内存中进行以下标准化，不写回输入文件：

1. 删除连续重复点。
2. 拒绝 NaN、无限值和少于两个不同点的退化线。
3. 保持原始 shape 索引、label 和点序不变。
4. 使用线段真实交点对网络进行 noding。
5. 将图片矩形边界作为主图图框。
6. 保留像素浮点坐标，不通过整数栅格化改变权威几何。

如果未来存在插图区域，应由独立的排除框配置处理；本阶段不自动识别插图。

## 7. 端点连接判定

对每条线的首尾端点，搜索其他合法边界线及主图图框上的最近点。自身线条不参与候选
搜索。

### 7.1 已连接

满足以下任一条件：

- 端点到目标网络距离不超过 `1.5px`；
- 端点位于另一条线的真实交点；
- 端点到主图图框距离不超过 `1.5px`。

### 7.2 可保守吸附

候选距离满足：

```text
1.5px < distance <= 5px
```

并且：

- 端点延伸方向与“端点到候选点”的夹角不超过 `30°`；
- 最近候选唯一；
- 最近候选与第二候选的距离差至少为 `2px`。

该类只标记为 `topology_gap_snappable`，本阶段不执行吸附。

### 7.3 需要人工确认

满足以下任一条件：

- `5px < distance <= 10px`；
- 距离虽不超过 `5px`，但方向不兼容；
- 存在多个距离接近的候选连接；
- 一级红线端点只连接到绿线，未连接到红线或图框。

标记为 `topology_gap_review` 或 `topology_gap_ambiguous`。

### 7.4 明显开放

在 `10px` 内没有合法候选时，标记为：

```text
topology_gap_open
```

报告中仍记录最近候选及距离，便于判断是缺线、漏标还是合法的图外终止。

## 8. 方向计算

每个端点使用邻近端点的一段折线估算切向：

- 默认累计回看不超过 `20px` 的线段；
- 起点方向取“内部点指向起点”；
- 终点方向取“内部点指向终点”；
- 与候选连接向量比较最小夹角；
- 线段过短、无法稳定估计时，不作为自动可吸附证据，转入人工确认。

方向只用于排除明显错误候选，不能覆盖距离和唯一性约束。

## 9. 交点与缺口记录

每个问题端点记录：

```json
{
  "gap_id": "L2-S0042-end",
  "network": "level_2",
  "shape_index": 42,
  "label": "boundary_2",
  "endpoint": "end",
  "point_px": [1512.3, 874.6],
  "status": "review",
  "nearest_distance_px": 7.4,
  "candidate_shape_index": 17,
  "candidate_label": "boundary_1",
  "candidate_point_px": [1517.9, 879.4],
  "direction_angle_deg": 18.2,
  "candidate_count": 1
}
```

一级、二级网络可能对同一个物理端点产生不同结论，必须分别保存。例如一个红线端点
连接到绿线后，二级网络可以闭合，但一级网络仍应报告红线连续性缺口。

## 10. 输出

默认输出目录：

```text
boundaries/chgis_output/
boundaries/topology_review/
```

### 10.1 机器可读报告

```text
boundaries/chgis_output/06-16河北东路_topology_report.json
```

包含：

- 输入文件和图片信息；
- 两类边界数量；
- 一级、二级网络的端点统计；
- 已连接、可吸附、人工确认、歧义和开放端点数量；
- 每个问题端点的完整候选信息；
- 原始网络 `polygonize_full` 的面、悬挂线和无效环数量；
- 模拟应用“仅唯一且不超过 5px 的连接”后的潜在面数量；
- 参数和脚本版本。

### 10.2 GeoJSON 缺口图层

```text
boundaries/chgis_output/06-16河北东路_topology_gaps.geojson
```

每个缺口为点要素，属性与报告一致，可直接在 QGIS 中查看。

### 10.3 可视化预览

```text
boundaries/chgis_output/06-16河北东路_topology_preview.jpg
```

颜色：

- 绿色：已连接，仅在调试模式显示；
- 橙色：`snappable`；
- 黄色：`review`；
- 紫色：`ambiguous`；
- 红色：`open`；
- 青色连线：端点到建议候选点。

预览必须同时绘制原始红、绿边界，但不得改变原图。

### 10.4 LabelMe 审阅副本

```text
boundaries/topology_review/06-16河北东路.jpg
boundaries/topology_review/06-16河北东路.json
```

审阅 JSON 是原始 JSON 的副本，并在 shapes 末尾追加 `point` 类型的缺口标记：

- `topology_gap_l1_snappable`
- `topology_gap_l1_review`
- `topology_gap_l1_ambiguous`
- `topology_gap_l1_open`
- `topology_gap_l2_snappable`
- `topology_gap_l2_review`
- `topology_gap_l2_ambiguous`
- `topology_gap_l2_open`

每个标记通过 `flags` 保存 `gap_id`、距离和候选 shape 索引。原始边界 shape 不重排、
不改点、不改 label。原图只在不存在时复制。

## 11. CLI

```bash
python boundaries/check_boundary_topology.py \
  boundaries/chgis_labelme/06-16河北东路.json \
  --output-dir boundaries/chgis_output \
  --review-dir boundaries/topology_review \
  --connected-tolerance 1.5 \
  --snap-tolerance 5 \
  --review-tolerance 10 \
  --direction-tolerance 30
```

额外参数：

- `--no-review-copy`：不生成 LabelMe 审阅副本；
- `--show-connected`：在预览中显示已连接点；
- `--min-face-area-px`：报告中忽略极小碎面的阈值，默认 `1000`；
- `--overwrite-review`：显式允许覆盖已有审阅副本。

默认不覆盖已有审阅 JSON，避免丢失人工修改。报告和预览可重复生成并原子替换。

## 12. 错误处理

以下情况返回非零状态且不生成不完整审阅副本：

- JSON 或图片不存在；
- 图像尺寸与 LabelMe 尺寸不一致；
- 没有可用边界；
- 坐标非法；
- 输出审阅文件已存在但未传 `--overwrite-review`。

报告与预览先写临时文件，成功后再替换目标文件。

单条退化 shape 不终止全部检查，但在报告中列入 `invalid_shapes`。

## 13. 测试

单元测试覆盖：

- 端点已精确连接到另一条线；
- 端点连接到线段中部；
- 红线与绿线形成合法 T 形交点；
- 距离 `1.5px`、`5px`、`10px` 的阈值边界；
- 方向兼容与方向冲突；
- 两个候选距离接近时的歧义；
- 图框连接；
- 一级红线只接绿线时，一级失败而二级通过；
- 退化线和未知标签；
- polygonize 统计。

集成测试覆盖：

- 对合成闭合网络生成零缺口报告；
- 对合成断裂网络生成对应 LabelMe 点标记；
- 原始 LabelMe 文件字节保持不变；
- 已存在审阅副本时默认拒绝覆盖；
- 河北东路数据能够生成报告、GeoJSON、预览和审阅副本。

## 14. 验收标准

1. 原始 `06-16河北东路.json` 和原图不发生任何修改。
2. 每个开放端点都有稳定、可复现的分类和候选信息。
3. 一级红线连续性与二级红绿联合闭合性分别报告。
4. 审阅副本可由 LabelMe 正常打开，原边界保持原顺序。
5. GeoJSON 可由 QGIS 正常加载。
6. 重复运行得到相同的 gap ID、分类和统计。
7. 仅在显式指定时覆盖已有审阅副本。
8. 检查结果足以指导下一阶段的保守吸附和方案 B 行政面恢复。
