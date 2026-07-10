# Multimodal Application

基于扩散模型的相机镜头控制多视角图像生成项目：给定参考图与相机旋转三维向量提示词，生成物体在目标视角下的图像。采用 LoRA 微调方案，通过 ComfyUI 工作流 [camera_control.json](file:///e:/Multimodal/camera_control.json) 部署，运行硬件为 NVIDIA RTX 4060 16G。

## 功能与可应用场景

- 输入参考图 + 相机相对旋转向量（azimuth/elevation/roll），生成同一物体在目标视角下的一致性图像
- 智能电影摄影、3D 内容生成中间件（多视角图像序列可用于 NeRF/Gaussian Splatting 重建）、交互式叙事等场景，详见 [Project_Proposal_Camera_Control.md](file:///e:/Multimodal/doc/Project_Proposal_Camera_Control.md)

## 快速开始

#### 环境要求

- Windows / Linux，Blender（用于数据集生成，需可在命令行调用）
- Python 3，依赖 `numpy`、`Pillow`（数据集拆分脚本）
- ComfyUI + LoRA 训练环境，显存 16G（RTX 4060 16G 验证通过）

#### 安装

- 安装 Blender 并确保 `blender` 命令可在终端调用
- `pip install numpy Pillow`
- 将 [camera_control.json](file:///e:/Multimodal/camera_control.json) 导入 ComfyUI 工作流

#### 使用：多视角数据集生成

数据集生成管线位于 [dataset_pipeline](file:///e:/Multimodal/dataset_pipeline)，覆盖场景搭建、相机轨迹采样、批量渲染、位姿标注、train/val 拆分全流程。详细用法见 Skill：[multiview-dataset-pipeline](file:///e:/Multimodal/.trae/skills/multiview-dataset-pipeline/SKILL.md)。

核心文件：

| 文件 | 职责 |
|---|---|
| [scene_builder.py](file:///e:/Multimodal/dataset_pipeline/scene_builder.py) | 搭建统一光照/地面的 Blender 场景，加载物体（外部 .obj/.fbx/.glb 或内置几何体降级）并归一化对中 |
| [camera_trajectory.py](file:///e:/Multimodal/dataset_pipeline/camera_trajectory.py) | 分层网格采样相机欧拉角（方位角/俯仰角均匀分布），生成含参考视角的位姿列表 |
| [render_pipeline.py](file:///e:/Multimodal/dataset_pipeline/render_pipeline.py) | 批量渲染命令行入口，输出统一分辨率图像 + 相机位姿标注（含相对参考视角的三维旋转向量） |
| [split_dataset.py](file:///e:/Multimodal/dataset_pipeline/split_dataset.py) | 按场景（默认，避免数据泄漏）或样本粒度拆分 train/val，默认 9:1 |

示例命令：

```powershell
blender --background --python dataset_pipeline\render_pipeline.py -- --output_dir dataset_output --num_scenes 200 --views_per_scene 30 --resolution 512 --engine EEVEE_NEXT --object_dir assets\models --seed 42
python dataset_pipeline\split_dataset.py --dataset_dir dataset_output --train_ratio 0.9 --split_by scene
```

无外部模型资产时，`render_pipeline.py` 会自动降级为内置几何体（cube/sphere/cylinder/cone/torus/monkey）轮换生成场景，用于先行验证 pipeline 是否可跑通；正式交付数据集建议替换为 CC0 或自有 `.obj/.fbx/.glb` 模型资产，并将场景数提高至 200+（× 30 视角/场景 ≈ 6000+ 样本），满足总规模不低于 5000 组有效样本的要求。

## 模型架构和训练方法

#### 模型架构

- 参考图经 VAE 编码 + Canny/Depth（DepthAnything V2）双 ControlNet 条件注入，配合 LoRA 微调的扩散模型，两级 KSampler 级联生成目标视角图像，详见 [camera_control.json](file:///e:/Multimodal/camera_control.json)
- 相机位姿以相对参考视角的三维旋转向量（azimuth/elevation/roll，度）形式作为条件注入，理论方案见 [研究方向说明.md](file:///e:/Multimodal/doc/text/研究方向说明.md)

#### 训练方法

- LoRA 微调方案，基础模型推荐（适配 RTX 4060 16G 显存约束）：**Stable Diffusion 1.5** 系列（如 `runwayml/stable-diffusion-v1-5` 或社区优化底模，如 `epiCRealism`/`RealisticVision` 等 SD1.5 微调版）作为主底模：
  - 参数量约 0.98B，全流程（模型权重 + 优化器状态 + LoRA 梯度 + 激活）在 16G 显存下可开启 `fp16`/`bf16` 混合精度训练，无需 CPU offload 即可稳定跑通，相比 SDXL（2.6B UNet）显存余量更充足，便于叠加 ControlNet 条件与更大 batch size
  - 生态成熟，ControlNet（Canny/Depth）、LoRA 训练工具链（kohya_ss、diffusers）对 SD1.5 支持最完善，与当前工作流中 ControlNetApplyAdvanced 节点直接兼容
  - 若后续显存/质量要求提升，可平行评估 **SDXL Turbo** 或 **SD 2.1**（分辨率更高但显存开销显著上升，16G 下需降低 batch size 或开启梯度检查点）
  - 数据集训练输入采用本项目 [dataset_pipeline](file:///e:/Multimodal/dataset_pipeline) 产出的 (参考图, 目标图, 相对旋转向量) 三元组，train/val 按 9:1 划分

