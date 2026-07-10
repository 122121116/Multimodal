# Multimodal Application

<<<<<<< HEAD
基于扩散模型的相机镜头控制多视角图像生成项目：给定参考图与相机旋转三维向量提示词，生成物体在目标视角下的图像。采用 LoRA 微调方案，通过 ComfyUI 工作流 [camera_control.json](file:///e:/Multimodal/camera_control.json) 部署，运行硬件为 NVIDIA RTX 4060 16G。

## 功能与可应用场景

- 输入参考图 + 相机相对旋转向量（azimuth/elevation/roll），生成同一物体在目标视角下的一致性图像
- 智能电影摄影、3D 内容生成中间件（多视角图像序列可用于 NeRF/Gaussian Splatting 重建）、交互式叙事等场景，详见 [Project_Proposal_Camera_Control.md](file:///e:/Multimodal/doc/Project_Proposal_Camera_Control.md)
=======
## 功能与可应用场景

本项目面向 **AI 图像生成中的镜头控制任务**，核心目标是让扩散模型不仅能够根据文本生成图像，还能够根据输入图像、镜头条件和结构控制信息，生成符合目标视角和空间结构约束的图像。

传统文生图模型主要依赖 prompt 控制图像内容与风格，但在镜头角度、空间结构、物体轮廓、多视角一致性等方面控制能力有限。本项目基于 Stable Diffusion、ControlNet、LoRA、Depth Estimation 和 Canny Edge Detection 构建多模态图像生成流程，将文本、图像、结构条件和相机/镜头关键词结合起来，实现更稳定的可控图像生成。

主要功能包括：

* **图像结构控制生成**
  输入原始图像后，系统可自动提取 Depth 深度图与 Canny 边缘图，并将其作为扩散模型的结构控制条件。

* **镜头条件引导生成**
  通过相机关键词、镜头语言或 LoRA 条件，使模型在生成过程中尽可能遵循特定视角、构图和空间变化。

* **多模态条件融合**
  同时使用文本 prompt、输入图像、Depth 深度图、Canny 边缘图和 LoRA 权重，实现文本语义与图像结构的联合控制。

* **可控图像生成与风格保持**
  在保留原始图像主要结构的基础上，生成具有新语义、新风格或新镜头效果的图像。

* **多视角一致性探索**
  为后续实现相机轨迹控制、多视角一致生成、身份保持和连续镜头生成提供原型基础。

可应用场景包括：

* AI 绘画中的精确构图控制；
* 图像到图像生成；
* 摄影镜头模拟；
* 动画分镜与概念设计；
* 游戏、影视、虚拟场景中的视角生成；
* 多模态生成模型课程实验与原型验证。

---
>>>>>>> 0cbc223442b4b27c2ebbbda715db13b23280f218

## 快速开始

### 环境要求

<<<<<<< HEAD
- Windows / Linux，Blender（用于数据集生成，需可在命令行调用）
- Python 3，依赖 `numpy`、`Pillow`（数据集拆分脚本）
- ComfyUI + LoRA 训练环境，显存 16G（RTX 4060 16G 验证通过）
=======
建议使用支持 CUDA 的 NVIDIA GPU 运行本项目，以获得较好的生成速度和稳定性。
>>>>>>> 0cbc223442b4b27c2ebbbda715db13b23280f218

推荐环境如下：

<<<<<<< HEAD
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
=======
* Python >= 3.10
* PyTorch >= 2.0
* CUDA >= 11.8
* ComfyUI
* Stable Diffusion Checkpoint
* ControlNet 相关模型
* LoRA 权重文件
* Depth Anything V2 模型
* Canny Edge Detection 节点或相关图像处理模块

如果使用 ComfyUI 工作流，需提前安装以下组件或节点：

* ComfyUI 本体；
* ComfyUI-Manager；
* Depth Anything V2 节点；
* ControlNet 相关节点；
* LoRA 加载节点；
* VAE 编码与解码节点；
* KSampler 采样节点。

---

### 安装

1. 克隆项目或进入本地项目目录：

```bash
git clone <your-repository-url>
cd <your-project-folder>
```

2. 创建 Python 虚拟环境：

```bash
conda create -n multimodal-camera-control python=3.10
conda activate multimodal-camera-control
```

3. 安装依赖：

```bash
pip install -r requirements.txt
```

如果使用 ComfyUI，可按照 ComfyUI 官方方式安装：

```bash
git clone https://github.com/comfyanonymous/ComfyUI.git
cd ComfyUI
pip install -r requirements.txt
```

4. 下载并放置模型文件：

建议按照以下结构存放模型：

```text
ComfyUI/
├── models/
│   ├── checkpoints/
│   │   └── stable_diffusion_model.safetensors
│   ├── controlnet/
│   │   └── controlnet_model.safetensors
│   ├── loras/
│   │   └── camera_control_lora.safetensors
│   ├── vae/
│   │   └── vae_model.safetensors
│   └── depth_anything/
│       └── depth_anything_v2_model.safetensors
```

模型文件名可根据实际下载内容进行调整。

---

### 使用

本项目主要支持两种使用方式：一种是基于 ComfyUI 的可视化工作流，另一种是基于脚本的推理流程。

#### 方式一：使用 ComfyUI 工作流

1. 启动 ComfyUI：

```bash
python main.py
```

2. 在浏览器中打开 ComfyUI 页面：

```text
http://127.0.0.1:8188
```

3. 导入项目工作流文件：

```text
workflow/camera_control_workflow.json
```

4. 上传输入图像。

5. 设置文本 prompt，例如：

```text
a high-quality image, cinematic lighting, realistic style, camera view control
```

6. 加载 LoRA 权重，并设置 LoRA 强度。

7. 运行工作流，生成结果图像。

工作流的基本流程为：

```text
输入图像
↓
Depth Anything V2 提取深度图
↓
Canny 边缘检测提取轮廓图
↓
LoRA 注入镜头控制能力
↓
CLIP 文本编码
↓
VAE 编码进入 latent 空间
↓
KSampler 扩散采样
↓
VAE 解码
↓
输出图像
```

#### 方式二：使用脚本推理

如果项目包含推理脚本，可使用如下命令：

```bash
python inference.py \
  --input examples/input.png \
  --prompt "a high-quality cinematic image with controlled camera view" \
  --lora models/loras/camera_control_lora.safetensors \
  --output outputs/result.png
```

其中：

* `--input` 表示输入图像；
* `--prompt` 表示文本提示词；
* `--lora` 表示镜头控制 LoRA 权重；
* `--output` 表示生成图像保存路径。

---
>>>>>>> 0cbc223442b4b27c2ebbbda715db13b23280f218

## 模型架构和训练方法

### 模型架构

<<<<<<< HEAD
- 参考图经 VAE 编码 + Canny/Depth（DepthAnything V2）双 ControlNet 条件注入，配合 LoRA 微调的扩散模型，两级 KSampler 级联生成目标视角图像，详见 [camera_control.json](file:///e:/Multimodal/camera_control.json)
- 相机位姿以相对参考视角的三维旋转向量（azimuth/elevation/roll，度）形式作为条件注入，理论方案见 [研究方向说明.md](file:///e:/Multimodal/doc/text/研究方向说明.md)
=======
本项目采用基于扩散模型的多模态条件控制架构。整体思路是：先将输入图像和相机/镜头条件转化为模型可理解的结构控制图，再通过 ControlNet 将这些结构条件注入 Stable Diffusion 的生成过程。
>>>>>>> 0cbc223442b4b27c2ebbbda715db13b23280f218

整体架构如下：

<<<<<<< HEAD
- LoRA 微调方案，基础模型推荐（适配 RTX 4060 16G 显存约束）：**Stable Diffusion 1.5** 系列（如 `runwayml/stable-diffusion-v1-5` 或社区优化底模，如 `epiCRealism`/`RealisticVision` 等 SD1.5 微调版）作为主底模：
  - 参数量约 0.98B，全流程（模型权重 + 优化器状态 + LoRA 梯度 + 激活）在 16G 显存下可开启 `fp16`/`bf16` 混合精度训练，无需 CPU offload 即可稳定跑通，相比 SDXL（2.6B UNet）显存余量更充足，便于叠加 ControlNet 条件与更大 batch size
  - 生态成熟，ControlNet（Canny/Depth）、LoRA 训练工具链（kohya_ss、diffusers）对 SD1.5 支持最完善，与当前工作流中 ControlNetApplyAdvanced 节点直接兼容
  - 若后续显存/质量要求提升，可平行评估 **SDXL Turbo** 或 **SD 2.1**（分辨率更高但显存开销显著上升，16G 下需降低 batch size 或开启梯度检查点）
  - 数据集训练输入采用本项目 [dataset_pipeline](file:///e:/Multimodal/dataset_pipeline) 产出的 (参考图, 目标图, 相对旋转向量) 三元组，train/val 按 9:1 划分
=======
```text
输入图像 / 文本 prompt / 相机条件
↓
条件控制模块
↓
Depth 深度图 + Canny 边缘图
↓
ControlNet 结构控制
↓
Stable Diffusion 扩散生成
↓
输出图像
```

各模块作用如下：

#### 1. Stable Diffusion 主生成模型

Stable Diffusion 是项目的主生成模型，负责在 latent 空间中进行逐步去噪，并生成最终图像。它具有较强的语义理解和图像生成能力，但单独依赖 prompt 时，对镜头角度、透视关系和空间结构的控制不够稳定。

#### 2. ControlNet 结构控制模块

ControlNet 用于向扩散模型注入额外的空间控制条件。它可以接收 Depth、Canny、Pose、Segmentation 等结构图，使生成图像在内容自由生成的同时，遵循输入结构约束。

在本项目中，ControlNet 主要负责接收 Depth 深度图和 Canny 边缘图，并将这些信息传入扩散生成过程，使输出图像尽可能保持合理的空间结构和轮廓关系。

#### 3. LoRA 镜头控制模块

LoRA 是一种轻量化微调方法，可以在不重新训练整个大模型的情况下，让模型学习特定条件或风格。

在本项目中，LoRA 主要用于学习相机关键词、镜头语言或特定视角控制信息，使模型能够在生成过程中响应镜头变化需求。例如：

```text
low angle view
top view
close-up shot
wide angle
camera moving left
dolly in
```

通过 LoRA，模型可以在较低训练成本下增强对镜头条件的响应能力。

#### 4. Depth 深度图

Depth 深度图用于表达画面中不同区域的远近关系。它可以帮助模型理解空间层次、物体前后关系和透视变化。

在镜头控制任务中，Depth 主要承担“空间结构约束”的作用，使模型在生成时不只是根据文本自由发挥，而是尽量遵循合理的三维空间关系。

#### 5. Canny 边缘图

Canny 边缘图用于表达物体轮廓、边界和主要结构线条。它可以帮助模型保持输入图像的构图、物体形状和局部结构。

在本项目中，Canny 与 Depth 形成互补关系：

* Depth 负责控制空间远近和透视结构；
* Canny 负责控制物体轮廓和边缘结构。

两者结合可以提高生成图像的结构稳定性。

---

### 训练方法

本项目的训练和实验流程可以分为数据准备、条件生成、LoRA 训练、扩散推理和结果评估五个部分。

#### 1. 数据准备

训练数据可以来自真实图像数据或合成图像数据。

如果需要构建更严格的镜头控制数据集，可以使用 Blender、Cycles 等渲染引擎生成同一场景在不同相机位姿下的图像，并记录对应的相机参数，包括：

* 相机位置；
* 相机旋转角度；
* 焦距；
* 视场角；
* 景深；
* 相机轨迹；
* 对应深度图；
* 对应边缘图。

这类数据有助于模型学习镜头条件与图像结构变化之间的关系。

#### 2. 条件控制图生成

对于每张输入图像，首先生成对应的结构控制图：

```text
输入图像
├── Depth Anything V2 → Depth 深度图
└── Canny Edge Detection → Canny 边缘图
```

Depth 图提供空间层次信息，Canny 图提供轮廓边界信息。二者共同构成扩散模型的结构约束。

#### 3. LoRA 训练

LoRA 训练的目标是让模型学习镜头关键词或相机条件与图像变化之间的对应关系。

训练样本可以组织为：

```text
图像 + 镜头关键词 / 相机条件 + 目标图像
```

示例镜头关键词包括：

```text
front view
side view
top view
low angle
close-up
wide angle
camera moves left
camera moves forward
```

训练时只更新 LoRA 的少量参数，而冻结主模型参数。这样可以降低训练成本，并减少对原始 Stable Diffusion 生成能力的破坏。

#### 4. 扩散推理

在推理阶段，系统同时接收文本 prompt、输入图像、Depth 图、Canny 图和 LoRA 条件，并通过 KSampler 完成扩散采样。

推理流程如下：

```text
文本 prompt → CLIP 文本编码
输入图像 → VAE 编码
输入图像 → Depth / Canny 条件图
LoRA → 镜头控制权重注入
Depth / Canny → ControlNet 结构约束
ControlNet + Stable Diffusion → KSampler 采样
VAE 解码 → 输出图像
```

#### 5. 结果评估

生成结果可以从以下几个角度进行评估：

* **图像质量**：生成图像是否清晰、自然、具有较高视觉质量；
* **结构一致性**：输出图像是否保留输入图像的主要结构；
* **边缘一致性**：输出图像是否遵循 Canny 轮廓；
* **空间合理性**：输出图像是否符合 Depth 深度关系；
* **镜头响应能力**：生成结果是否体现指定视角或镜头变化；
* **身份保持能力**：同一对象在不同视角下是否保持一致；
* **多视角稳定性**：连续镜头或多视角生成中是否出现结构漂移、材质跳变或物体重绘。

---

## 项目总结

本项目构建了一个面向 AI 图像生成镜头控制的多模态应用原型。其核心思路是将抽象的相机条件和镜头语言转化为Depth与Canny等结构控制图，再通过ControlNet注入扩散模型生成过程，最终实现更加稳定的结构约束图像生成。

相比普通文生图，本项目强调的不只是图像质量，而是图像生成过程中的可控性、结构稳定性和镜头逻辑。当前系统可以作为镜头控制生成的基础原型，后续可进一步扩展到显式 6DoF 相机位姿编码、多视角一致性约束、连续镜头轨迹生成和视频生成任务。




>>>>>>> 0cbc223442b4b27c2ebbbda715db13b23280f218

