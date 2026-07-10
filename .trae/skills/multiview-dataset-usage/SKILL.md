---
name: "multiview-dataset-usage"
description: "解释多视角相机控制数据集的结构、字段含义和在LoRA训练中的使用方法。当用户需要理解poses.json/train_manifest.json的数据格式、字段含义，或需要将数据集接入训练流程时调用。"
---

# 多视角相机控制数据集 - 结构与用法

本 Skill 解释 `dataset_output/` 下多视角数据集的格式约定、每个字段的几何/渲染含义，以及如何在 LoRA 训练中使用 `rotation_vector_relative_to_ref` 作为条件输入。

## 目录结构

```
dataset_output/
  dataset_manifest.json          # 汇总清单（生成时间、场景数、总样本数、分辨率、性能参数等）
  scene_0000/
    poses.json                   # 该场景所有视角的位姿标注列表
    view_ref.png                 # 参考视角 RGB 图像（azimuth=0, elevation=0，正对物体）
    depth_ref.png                # 参考视角 16bit 深度图（与 view_ref.png 像素对齐）
    edge_ref.png                 # 参考视角 8bit 轮廓图（与 view_ref.png 像素对齐）
    view_0.png                   # 其他视角 RGB 图像（view_id 从 0 递增）
    depth_0.png                  # 对应视角的深度图
    edge_0.png                   # 对应视角的轮廓图
    view_1.png / depth_1.png / edge_1.png
    ...
  scene_0001/
    ...
  train_manifest.json            # （通过 split_dataset.py 生成）训练集样本清单
  val_manifest.json              # 验证集样本清单
```

所有图像统一分辨率（默认 `512×512`）：RGB 图为 PNG/RGB 8bit，由 Blender 渲染产出；深度图为 PNG 单通道 16bit（保留深度精度），由 DepthAnything V2 对 RGB 图推理产出；轮廓图为 PNG 单通道 8bit（边缘掩码，0或255），由 `cv2.Canny` 直接对 RGB 原图检测产出。三者基于同一张 RGB 图生成，像素坐标严格一一对应，无需额外配准。深度图/轮廓图并非与 RGB 图同一次渲染产生，而是渲染完成后由独立脚本 `export_conditions.py` 后处理生成（Blender 5.0 的 Compositor API 不再支持在 `--background` 模式下程序化导出深度 Pass）。

## poses.json 字段详解

每个场景的 `poses.json` 是一个 JSON 数组，每个元素对应一个视角（同时关联 RGB/深度/轮廓三张图）：

| 字段 | 类型 | 示例 | 含义 |
|---|---|---|---|
| `view_id` | string 或 int | `"ref"` 或 `0` | 视角标识符，`"ref"` 为参考视角，其余为递增整数 |
| `image_path` | string | `"scene_0000/view_0.png"` | RGB 图像相对数据集根目录的路径 |
| `depth_path` | string | `"scene_0000/depth_0.png"` | 16bit 深度图相对路径（DepthAnything V2 对 RGB 图推理导出，与 RGB 图像素对齐） |
| `edge_path` | string | `"scene_0000/edge_0.png"` | 8bit 轮廓图相对路径（对 RGB 原图做 Canny 边缘检测导出） |
| `azimuth_deg` | float | `0.83656` | 相机方位角（度），水平绕物体旋转角度，0°为正对物体正面 |
| `elevation_deg` | float | `-30.0` | 相机俯仰角（度），正值为仰视，负值为俯视 |
| `distance` | float | `5.0` | 相机与物体几何中心的距离（Blender 单位） |
| `euler_xyz_deg` | [float,float,float] | `[120.0, -0.0, 107.15]` | 相机在世界坐标系下的欧拉角 (rx, ry, rz)，Blender 惯例（度） |
| `camera_position_xyz` | [float,float,float] | `[4.33, 0.06, -2.5]` | 相机在三维世界空间中的位置 (x, y, z) |
| `rotation_vector_relative_to_ref` | [float,float,float] | `[0.84, -30.0, 0.84]` | **相对参考视角的三维旋转向量** (delta_azimuth, delta_elevation, delta_roll)，单位度。**这是 LoRA 训练的核心条件向量** |
| `scene_id` | int | `0` | 场景编号，映射到特定的 3D 物体 |
| `object_source` | string | `"e:\\Multimodal\\assets\\models\\Barrel_01\\Barrel_01.gltf"` | 该场景使用的 3D 模型来源路径或名称 |

> 若某场景的 `poses.json` 不含 `depth_path`/`edge_path` 字段，说明该数据是用改造前的旧版导出脚本生成的（仅有 RGB 图）；若字段存在但文件缺失，说明只跑过 `render_pipeline.py` 尚未运行 `export_conditions.py`。两种情况 `training/dataset.py` 加载时都会直接报错阻止训练，需要先跑最新版 `render_pipeline.py` 再跑 `export_conditions.py` 补全。

## 深度图与轮廓图的生成方式（为什么不再实时计算，也不再由 Blender 渲染阶段产生）

Blender 5.0 大幅重构了 Compositor 节点 API（`scene.node_tree` 被移除、Viewer 节点在 `--background` 模式下不产生像素数据等），在 Blender 内部程序化导出深度 Pass 的技术路径不可行。因此深度图/轮廓图改为渲染完成后，由独立的 `dataset_pipeline/export_conditions.py` 脚本（纯 Python，不依赖 bpy）基于已落盘的 RGB 图后处理生成：

- **深度图**：用 DepthAnything V2 模型（`depth-anything/Depth-Anything-V2-Large-hf`）对 RGB 图做单目深度估计推理，输出的相对深度图归一化到 `[0,1]` 后编码为 16bit PNG（`0~65535` 对应 `0.0~1.0` 归一化深度）。16bit 位深相比 8bit 能保留更精细的深度层次，避免量化误差影响后续 ControlNet/LoRA 对深度结构的学习。
- **轮廓图**：**直接对同一张 RGB 原图**做灰度化后 `cv2.Canny` 边缘检测（低/高阈值可通过 `--canny_threshold1`/`--canny_threshold2` 调整，默认 100/200），而不是对深度图求梯度——这样轮廓图反映的是物体在真实纹理图像上的边缘，而非深度不连续处的边界。
- 两者均基于同一张已落盘的 RGB 图独立计算，天然保证像素级配准（分辨率、坐标系完全一致），无需额外对齐步骤。

## 核心字段：rotation_vector_relative_to_ref

```
rotation_vector_relative_to_ref = [delta_azimuth, delta_elevation, delta_roll]
```

这是**相机条件注入向量**（即 ComfyUI 工作流 `camera_control.json` 中的"相机以原图像为中心进行旋转的三维向量提示词"）的原始标注。其含义为：

- **delta_azimuth**：目标视角相对于参考视角的水平旋转量（度）。正值表示相机向右水平旋转（物体左转），负值表示向左旋转。
- **delta_elevation**：目标视角相对于参考视角的俯仰变化量（度）。正值表示相机上仰（俯视物体减小），负值表示相机下俯。
- **delta_roll**：目标视角相对于参考视角的滚动变化量（度）。在球面采样中该值约等于 delta_azimuth（因相机在球面上沿切线方向旋转产生欧拉角滚动耦合）。

**参考视角**（`view_id="ref"`）的 `azimuth_deg=0, elevation_deg=0`，因此其 `rotation_vector_relative_to_ref = [0, 0, 0]`。

**LoRA 训练输入**：将参考图的深度图/轮廓图（二选一，取决于训练哪个 LoRA）与目标视角图像配对，以 `rotation_vector_relative_to_ref` 作为条件注入向量，训练 LoRA 模块学习从参考视角结构图到目标视角图像的相机旋转映射。**两个 LoRA（canny/depth）分别独立训练，不共享权重，训练数据分别绑定 `edge_path`/`depth_path`**，详见下一节。

## 两个 LoRA 分别绑定的条件图字段

`training/dataset.py` 的 `RotationPairDataset(condition_type=...)` 参数决定读取哪个字段作为 `condition_image`：

| `--condition_type` | 读取字段 | 对应图像 | 训练输出的 LoRA |
|---|---|---|---|
| `canny` | `edge_path` | 8bit 轮廓图 | `lora_canny`（原方案中用 Canny 边缘图训练，现改为读取导出的轮廓图，语义等价，命名沿用 `canny` 以兼容 `camera_control.json` 工作流的 Canny 分支挂载点） |
| `depth` | `depth_path` | 16bit 深度图 | `lora_depth` |

条件图在训练前统一转换为 3 通道 `[-1,1]` 归一化张量（复制单通道到 RGB 三通道，与 VAE 输入约定对齐），16bit 深度图会先按 `value/65535*255` 映射到 8bit 值域后再复制，不丢失相对深度层次关系。

## 训练数据格式（train_manifest.json / val_manifest.json）

`split_dataset.py` 按 9:1 比例拆分后输出 `train_manifest.json` 和 `val_manifest.json`，每个文件是一个 JSON 数组，**每项是单张图像的完整位姿记录**（内联了 `poses.json` 中对应条目的全部字段，含 `depth_path`/`edge_path`），训练脚本无需再关联查询 `poses.json`：

```json
{
  "image_path": "scene_0000/view_0.png",
  "depth_path": "scene_0000/depth_0.png",
  "edge_path": "scene_0000/edge_0.png",
  "scene_id": 0,
  "view_id": 0,
  "azimuth_deg": 0.8366,
  "elevation_deg": -30.0,
  "distance": 5.0,
  "euler_xyz_deg": [120.0, -0.0, 107.15],
  "camera_position_xyz": [4.33, 0.06, -2.5],
  "rotation_vector_relative_to_ref": [0.84, -30.0, 0.84],
  "object_source": "e:\\Multimodal\\assets\\models\\Barrel_01\\Barrel_01.gltf"
}
```

## 拆分层级说明

按 `--split_by scene`（默认，推荐）拆分时，同场景的所有视角（包括其参考视角）全部划入同一个集合（train 或 val），避免**数据泄漏**（同一物体的不同视角不会出现在两个集合中）。按 `--split_by sample` 拆分时，不区分场景边界，完全随机分配。

## 与 camera_control.json 的对应关系

`camera_control.json` 工作流接受一个参考图像（的结构图：轮廓图或深度图）和一个三维旋转向量，通过对应分支的 LoRA + ControlNet 生成目标视角的结构图。数据集中每个 `(ref结构图, 目标视角结构图, rotation_vector_relative_to_ref)` 三元组构成一个训练样本：

- **input_1**: `edge_ref.png`（Canny分支）或 `depth_ref.png`（Depth分支）
- **input_2**: `rotation_vector_relative_to_ref`（条件向量，[d_az, d_el, d_roll]）
- **target**: `edge_N.png`（Canny分支）或 `depth_N.png`（Depth分支）

训练产出的两个 LoRA 权重分别对应工作流中 `LoraLoader(183)`（Canny 分支）与 `LoraLoader(191)`（Depth 分支）。

## LoRA 训练收敛验证

`training/train_lora.py` 训练过程中会将 loss 落盘为 `<output_dir>/loss_history.json`（每 10 step 记录一次）。可用 `training/verify_lora_convergence.py` 分别校验两个 LoRA 的收敛状态（loss 是否呈下降趋势、末尾是否趋于稳定、是否出现 NaN/Inf 发散）：

```powershell
python training\verify_lora_convergence.py --loss_history training\output\lora_canny\loss_history.json --loss_history training\output\lora_depth\loss_history.json
```
