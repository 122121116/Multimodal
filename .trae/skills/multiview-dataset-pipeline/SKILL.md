---
name: "multiview-dataset-pipeline"
description: "生成/拆分基于Blender的多视角相机控制数据集（渲染pipeline、相机轨迹、位姿标注、train/val拆分）。当用户需要为扩散模型相机控制LoRA任务构建、扩产或调试多角度图像数据集时调用。"
---

# Multiview Dataset Pipeline

本 Skill 用于操作 `dataset_pipeline/` 目录下的 Blender 多视角数据集自动化生成管线，服务于"扩散模型 + 相机位姿控制 LoRA"训练任务（配套 ComfyUI 工作流 `camera_control.json`）。

## 架构说明（重要）
Blender 5.0 大幅重构了 Compositor 节点 API（`scene.node_tree` 被移除、`CompositorNodeComposite`/`MapRange`/`OutputFile` 等节点类型全部变更，且 Viewer 节点在 `--background` 无界面渲染模式下不产生像素数据），在 Blender 内部程序化导出深度图/轮廓图的技术路径不可行。因此采用两阶段方案：
1. **Blender 渲染阶段**（`render_pipeline.py`）：只负责渲染并落盘 RGB 图，同时在 `poses.json` 中预写 `depth_path`/`edge_path` 字段（此时对应文件尚不存在）。
2. **Python 后处理阶段**（`export_conditions.py`，纯 Python，不依赖 bpy）：基于已落盘的 RGB 图，用 DepthAnything V2 模型推理生成深度图，用 `cv2.Canny` **直接对 RGB 原图**做边缘检测生成轮廓图（不是对深度图求梯度），补全上一步预留的文件路径。

两阶段必须依次执行，`export_conditions.py` 会基于 `poses.json` 中预写的 depth_path/edge_path 路径生成对应文件。

## 适用场景
- 用户需要新生成、扩产（如从验证规模扩到 5000+ 样本）多视角图像数据集
- 用户需要调整相机轨迹采样方式、场景光照、物体来源（内置几何体 or 外部 .obj/.fbx/.glb 模型）
- 用户需要按 9:1（或自定义比例）拆分 train/val
- 用户询问如何验证 pipeline 是否可跑通、如何排查渲染失败

## 目录与文件职责
位于 `e:\Multimodal\dataset_pipeline\`：

| 文件 | 运行环境 | 职责 |
|---|---|---|
| `scene_builder.py` | Blender bpy（被 import，不单独调用） | 清场景、建地面、建三点光照、加载物体（外部模型或内置几何体降级）、物体归一化对中 |
| `camera_trajectory.py` | Blender bpy（被 import） | 分层网格采样相机欧拉角（azimuth/elevation均匀分布）、生成含参考视角(ref)的位姿列表、创建/更新相机并look-at物体中心 |
| `render_pipeline.py` | Blender bpy，命令行入口 | 批量渲染主流程：搭场景→生成位姿→逐视角渲染RGB PNG→写 `poses.json`（预写depth_path/edge_path字段）→汇总 `dataset_manifest.json` |
| `export_conditions.py` | 纯 Python（torch+transformers+opencv） | 渲染完成后独立运行：用 DepthAnything V2 对 RGB 图批量推理生成16bit深度图，用 cv2.Canny 对 RGB 原图批量生成8bit轮廓图，按 poses.json 中预写的路径落盘 |
| `benchmark_export.py` | Blender bpy，命令行入口 | RGB 渲染性能基准对比：以"优化前基线参数"与"优化后参数"各渲染一个场景，输出 `benchmark_report.json`，验证提速是否达标（>=40%） |
| `split_dataset.py` | 纯 Python（标准库） | 按 scene 或 sample 粒度拆分 train/val，输出 `train_manifest.json` / `val_manifest.json` |

## 核心数据格式
- 数据集根目录下每个场景一个子目录 `scene_{id:04d}/`，每个视角对应三类图像：`view_*.png`（RGB原图，Blender渲染产出）、`depth_*.png`（16bit单通道深度图，DepthAnything V2推理产出）、`edge_*.png`（8bit单通道轮廓图，对RGB原图Canny边缘检测产出），三者分辨率一致、像素坐标严格对齐（后处理直接基于已落盘的RGB图计算，不涉及二次渲染或坐标变换）。`view_ref.png`/`depth_ref.png`/`edge_ref.png` 为参考视角。该场景的标注汇总在 `poses.json`。
- `poses.json` 是列表，每项字段：`view_id`, `image_path`, `depth_path`, `edge_path`, `azimuth_deg`, `elevation_deg`, `distance`, `euler_xyz_deg`, `camera_position_xyz`, `rotation_vector_relative_to_ref`（相对参考视角的三维旋转向量，度，即相机控制LoRA训练所需的核心条件向量）, `scene_id`, `object_source`。`depth_path`/`edge_path` 由 `render_pipeline.py` 预写，实际文件需运行 `export_conditions.py` 后才存在。
- 根目录 `dataset_manifest.json` 汇总场景数、总样本数、分辨率、种子、`export_depth`/`export_edge`/`samples`/`use_gpu` 等生成参数，供拆分脚本读取。

## 常用命令
> 以下命令均以用户实际 Blender 安装路径 `F:\blender\blender.exe`（Blender 5.0.1）为例，请根据实际安装位置替换；输出路径统一用绝对路径 `e:\Multimodal\...` 避免因 Blender 后台模式 cwd 不确定导致产物跑到其他盘符。

```powershell
# 1. 第一阶段：渲染 RGB 图（GPU加速，采样率32）
& "F:\blender\blender.exe" --background --python e:\Multimodal\dataset_pipeline\render_pipeline.py -- --output_dir e:\Multimodal\dataset_output --num_scenes 5 --views_per_scene 30 --resolution 512 --engine EEVEE_NEXT --seed 42

# 2. 使用外部模型资产目录（.obj/.fbx/.glb），并显式指定性能参数
& "F:\blender\blender.exe" --background --python e:\Multimodal\dataset_pipeline\render_pipeline.py -- --output_dir e:\Multimodal\dataset_output --num_scenes 9 --views_per_scene 300 --object_dir e:\Multimodal\assets\models --seed 42 --samples 32 --use_gpu 1

# 3. 第二阶段：补全深度图与轮廓图（在普通 Python 环境运行，无需 Blender）
python e:\Multimodal\dataset_pipeline\export_conditions.py --dataset_dir e:\Multimodal\dataset_output --canny_threshold1 100 --canny_threshold2 200 --depth_batch_size 8

# 4. 导出性能基准对比（验证 RGB 渲染提速>=40%目标）
& "F:\blender\blender.exe" --background --python e:\Multimodal\dataset_pipeline\benchmark_export.py -- --output_dir e:\Multimodal\dataset_output\_benchmark --views_per_scene 30 --object_dir e:\Multimodal\assets\models

# 5. train/val 拆分（默认按场景9:1，避免同场景视角跨集泄漏）
python e:\Multimodal\dataset_pipeline\split_dataset.py --dataset_dir e:\Multimodal\dataset_output --train_ratio 0.9 --split_by scene
```

## 验证测试执行命令（本次性能优化任务专用）

### 1) Blender RGB 渲染速度提升验证：确认 ≥40% 提速目标达标
```powershell
& "F:\blender\blender.exe" --background --python e:\Multimodal\dataset_pipeline\benchmark_export.py -- --output_dir e:\Multimodal\dataset_output\_benchmark --views_per_scene 30 --resolution 512 --object_dir e:\Multimodal\assets\models
```
运行结束后终端会打印类似：
```
基准测试完成：优化前 XX.XXs -> 优化后 XX.XXs，提速 XX.X%（目标 >= 40%，达标/未达标）
```
同时生成 `e:\Multimodal\dataset_output\_benchmark\benchmark_report.json`，其中 `target_met` 字段为 `true` 即视为达标。若未达标，优先检查终端是否打印过"未检测到可用的 OPTIX/CUDA GPU 设备"的警告（说明 GPU 加速未生效，是最主要的提速来源）。

### 2) 两个 LoRA 训练收敛状态验证
先分别训练两个 LoRA（各自使用 `export_conditions.py` 生成的 depth/edge 图像作为条件输入）：
```powershell
cd e:\Multimodal\training
python train_lora.py --dataset_root e:\Multimodal\dataset_output --condition_type canny --num_epochs 10
python train_lora.py --dataset_root e:\Multimodal\dataset_output --condition_type depth --num_epochs 10
```
训练过程中会在 `e:\Multimodal\training\output\lora_canny\loss_history.json` 与 `...\lora_depth\loss_history.json` 持续记录 loss。训练完成（或训练中途，只要已产生足够多 checkpoint）后运行收敛校验：
```powershell
python e:\Multimodal\training\verify_lora_convergence.py --loss_history e:\Multimodal\training\output\lora_canny\loss_history.json --loss_history e:\Multimodal\training\output\lora_depth\loss_history.json --report_path e:\Multimodal\training\output\convergence_report.json
```
终端打印每个 LoRA 的"前段均值loss -> 末段均值loss"下降幅度、末尾波动比与"收敛判定：通过/未通过"；最后一行"整体收敛验证结果"为"全部通过"即视为两个 LoRA 均收敛达标。

## 训练阶段的样本配对与文本条件方案（`training/dataset.py` / `training/train_lora.py`）
- **任意双视角配对**：训练样本不再固定"以参考视角(ref)为基准 -> 其他视角"的搭配，而是 `RotationPairDataset` 在每个场景内部的全部视角中随机采样任意两张图片，一张作为条件图（相当于参考图），另一张作为目标图；两者的相对旋转向量按各自实际的 `azimuth_deg`/`elevation_deg`/`euler_xyz_deg` 现算得到，不局限于必须以 `view_id="ref"` 为基准。这样可以让模型学习任意视角之间的旋转映射，而不是只学习"从固定正面参考图出发"的狭窄先验，从而提升对任意参考视角输入的泛化能力。
- **文本条件动态化**：CLIP 文本编码器的输入不再是固定的空字符串，而是从该场景 `poses.json` 中的 `object_source` 字段解析出的物品名称文本（如 `"CoffeeTable_01"` 解析为 `"coffee table 01"`），每个样本逐样本 tokenize + 动态编码后，作为 UNet cross-attention 的文本条件基底，再与旋转向量编码在 token 维度拼接，使文本条件携带物品语义信息。

## 性能优化参数说明（`render_pipeline.py`）
- `--samples`：渲染采样数，默认 32（原工程模板常见默认值 128），越低越快，配合 `--denoise 1` 补偿画质损失。
- `--use_gpu`：1=启用 GPU 加速（Cycles 自动探测 OPTIX/CUDA；EEVEE 系列本身即 GPU 光栅化渲染），0=强制 CPU。
- `--tile_size`：渲染分块尺寸，仅在旧版本 Blender 暴露 `tile_x/tile_y` 属性时生效，Blender 4.x/5.x 已自动管理分块。
- `--export_depth` / `--export_edge`：是否在 `poses.json` 中预写 `depth_path`/`edge_path` 字段，默认均为 1（开启）。实际图像文件由 `export_conditions.py` 生成，与本参数控制的字段预写是两回事。

## `export_conditions.py` 参数说明
- `--export_depth` / `--export_edge`：是否分别执行深度图/轮廓图生成，默认均为 1。
- `--overwrite`：已存在的输出文件是否重新生成覆盖，默认 0（跳过已存在文件，支持中断续跑）。
- `--depth_model_id`：DepthAnything V2 模型标识符，默认 `depth-anything/Depth-Anything-V2-Large-hf`。
- `--depth_batch_size`：深度图推理批大小，默认 8（RTX 4060 16G 下可安全运行，显存不足可调低）。
- `--depth_device`：深度模型推理设备，默认 `cuda`。
- `--canny_threshold1` / `--canny_threshold2`：`cv2.Canny` 的低/高阈值，默认 100/200，轮廓过淡或过密时调整。
- `--edge_num_workers`：轮廓图生成的并行线程数（CPU计算，可安全多线程），默认 4。

## 扩产到 5000+ 样本的调整方式
- 提高 `--num_scenes`，保持 `--views_per_scene >= 30`；例如 200 场景 × 30 视角 ≈ 6000 样本。
- 优先准备外部 CC0/自有 `.obj/.fbx/.glb` 模型资产放入 `--object_dir` 指向的目录，让场景物体形态多样化（内置几何体仅用于快速验证管线，不建议作为正式交付数据集的唯一物体来源）。
- 若显存/渲染时长吃紧，`--engine` 使用 `EEVEE_NEXT`（默认，速度快）而非 `CYCLES`；`CYCLES` 引擎下务必保持 `--use_gpu 1` 以启用 OPTIX/CUDA 加速。
- 扩产后务必两阶段都要跑：先 `render_pipeline.py` 出全部 RGB 图，再用一次 `export_conditions.py`（不加 `--overwrite`）补全所有新增场景的深度图/轮廓图，脚本会自动跳过已生成的旧场景。

## 排查建议
- 若 Blender 环境无法运行脚本（如未安装/不在 PATH），需在真实装有 Blender 的机器上执行 `render_pipeline.py`；`export_conditions.py`/`split_dataset.py` 均为纯 Python，可在任意机器上对已生成的数据集离线运行（`export_conditions.py` 需要 GPU 加速深度推理，建议在训练机上跑）。
- 修改 `scene_builder.py` / `camera_trajectory.py` 后，务必确认 `render_pipeline.py` 中调用的函数签名（`build_scene(config)` 的 config key、`generate_camera_poses(...)` 参数顺序、`create_camera(pose, scene_center)`）与被修改后的定义保持一致。
- 若 GPU 加速未生效（终端打印"未检测到可用的 OPTIX/CUDA GPU 设备"），检查 Blender 内 Edit > Preferences > System 中是否已识别到显卡，以及显卡驱动是否支持 OPTIX（RTX 系列建议优先用 OPTIX）；`export_conditions.py` 的深度推理若报 CUDA 不可用，检查本机 PyTorch 是否为 CUDA 版本（`torch.cuda.is_available()`）。
