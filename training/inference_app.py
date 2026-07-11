# -*- coding: utf-8 -*-
"""
带 UI 的推理调用示例脚本：加载参考图 + 目标相对旋转向量，生成目标视角的
结构图（轮廓图/深度图），用于快速验证 train_lora.py 训练出的 LoRA 效果。

设计说明：
- 模型构建逻辑（UNet conv_in 4->8 通道扩展、LoRA 挂载点、RotationEncoder
  条件编码方式）与 train_lora.py 完全一致，直接复用同样的初始化步骤，
  确保推理时的模型结构与训练时严格对应，参见 train_lora.py 顶部注释与
  expand_unet_conv_in 函数、rotation_encoder.py。
- 条件注入方式参考 camera_control.json（ComfyUI 工作流）的整体思路：
  参考图 -> VAE 编码 -> 与 noisy latent 拼接 -> UNet 去噪 -> VAE 解码，
  但本脚本不依赖 ComfyUI，而是用 diffusers 原生 API 直接在 Python 内实现
  同样的推理流程，方便本地快速调试。
- 双分支工作流：train_lora.py 训练出的 canny / depth 两个 LoRA 是完全独立
  训练的两套权重，分别对应轮廓图与深度图两种条件图输入。本 UI 不再要求
  用户手动上传已经是条件图的图像，而是只上传一张原始 RGB 参考图，一旦
  上传（或调整 Canny 阈值），立即自动提取出深度图与轮廓图并展示（无需
  等待点击生成按钮）；点击"生成"后，再分别喂给各自独立的 InferencePipeline
  （depth_pipeline / canny_pipeline）做多步迭代推理，两分支的中间结果与
  最终输出互不影响，各自独立产出一张预测图，只共享同一组旋转角度/物品
  prompt/seed/采样参数。
- 多步迭代推理：train_lora.py 现在只训练"相邻视角小角度旋转"这一更简单、
  更容易收敛的子任务（参见 dataset.py 的 AdjacentPairDataset），因此模型
  单次前向只擅长小角度旋转变换。要生成任意大角度的目标视角，需要将目标
  旋转向量 [Δaz, Δel, Δroll] 拆分为 N 个小步长，每步用上一步的生成结果
  作为下一步的条件图，串联调用模型 N 次（类似自回归式相机轨迹游走）。
  单步角度步长建议不超过训练数据中相邻视角的典型间隔（可在 UI 中调节）。
- LoRA 权重当前尚未训练完成，因此 LoRA 路径在 UI 中留空即可运行（此时仅
  使用底模 + 扩展后的 conv_in，不具备"结构图视角变换"能力，仅用于验证
  UI/推理链路本身是否跑通）；训练完成后，在 UI 中填入类似
  e:/Multimodal/training/output/lora_depth/final 的 checkpoint 目录，
  即可加载真正训练好的 LoRA/conv_in/RotationEncoder 权重进行测试。

依赖（需预先安装，另见 requirements.txt）：
    torch, diffusers, transformers, accelerate, peft, Pillow, numpy, opencv-python, gradio

启动方式：
    python inference_app.py
默认会在本地启动一个 Gradio 网页界面（http://127.0.0.1:7860）。
"""
import os

# 必须在 import transformers/huggingface_hub 之前设置，才能让 hf_hub 的下载请求
# 走镜像端点。若用户已在外部环境变量中手动设置 HF_ENDPOINT，则不覆盖。
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")

import numpy as np
import torch
from diffusers import AutoencoderKL, DDIMScheduler, UNet2DConditionModel
from PIL import Image
from transformers import CLIPTextModel, CLIPTokenizer

from checkpoint_utils import ensure_checkpoint
from rotation_encoder import RotationEncoder

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE = torch.float16 if DEVICE == "cuda" else torch.float32
RESOLUTION = 512

# 训练完成后，可将默认路径改为对应 checkpoint 目录，
# 或直接在 UI 的 "LoRA checkpoint 目录" 输入框中填写，无需改代码。
DEFAULT_LORA_DIR_DEPTH = "e:/Multimodal/training/output/lora_depth/final"
DEFAULT_LORA_DIR_CANNY = "e:/Multimodal/training/output/lora_canny/final"

# DepthAnything V2 模型标识符，与 export_conditions.py 保持一致。
DEPTH_MODEL_ID = "depth-anything/Depth-Anything-V2-Large-hf"


def expand_unet_conv_in(unet: UNet2DConditionModel) -> UNet2DConditionModel:
    """与 train_lora.py 中同名函数逻辑完全一致：conv_in 从 4 通道扩展为 8 通道，
    前 4 通道保留预训练权重，后 4 通道置零初始化（推理时若加载了 LoRA
    checkpoint 中的 conv_in.pt，后 4 通道会被训练好的权重覆盖）。
    """
    old_conv_in = unet.conv_in
    in_channels = old_conv_in.in_channels
    if in_channels == 8:
        return unet
    assert in_channels == 4, f"预期 conv_in 输入通道为4，实际为{in_channels}"

    new_conv_in = torch.nn.Conv2d(
        in_channels=8,
        out_channels=old_conv_in.out_channels,
        kernel_size=old_conv_in.kernel_size,
        stride=old_conv_in.stride,
        padding=old_conv_in.padding,
    )
    new_conv_in.weight.data.zero_()
    new_conv_in.weight.data[:, :4, :, :] = old_conv_in.weight.data
    new_conv_in.bias.data = old_conv_in.bias.data.clone()
    unet.conv_in = new_conv_in
    unet.config.in_channels = 8
    return unet


# ----------------------------------------------------------------------------
# 深度图 / 轮廓图提取（从原始 RGB 参考图自动生成，逻辑与
# dataset_pipeline/export_conditions.py 的 export_depth_maps / _canny_single
# 保持一致，确保与训练时的条件图分布对齐）
# ----------------------------------------------------------------------------
class _DepthModelCache:
    """DepthAnything V2 模型懒加载缓存，避免每次生成都重新加载模型。"""

    processor = None
    model = None

    @classmethod
    def get(cls):
        if cls.model is None:
            from transformers import AutoImageProcessor, AutoModelForDepthEstimation

            print(f"[inference_app] 首次使用，加载深度估计模型 {DEPTH_MODEL_ID} ...")
            cls.processor = AutoImageProcessor.from_pretrained(DEPTH_MODEL_ID)
            cls.model = AutoModelForDepthEstimation.from_pretrained(DEPTH_MODEL_ID)
            cls.model.to(DEVICE)
            cls.model.eval()
        return cls.processor, cls.model


def _extract_depth_condition_image(rgb_image: Image.Image) -> Image.Image:
    """对原始 RGB 参考图推理生成深度图，返回 3 通道 PIL 图像。

    处理方式与 export_conditions.py 的 export_depth_maps 一致：深度值
    归一化到 [0, 1] 后（这里直接归一化到 [0, 255] uint8，等价于
    dataset.py 中 _load_condition_image 对 16bit 深度图的后处理结果），
    复制为 3 通道，与 _pil_to_normalized_tensor 的输入约定对齐。
    """
    processor, model = _DepthModelCache.get()
    rgb_image = rgb_image.convert("RGB")

    with torch.no_grad():
        inputs = processor(images=[rgb_image], return_tensors="pt").to(DEVICE)
        outputs = model(**inputs)
        predicted_depth = outputs.predicted_depth[0]  # (h, w)

        width, height = rgb_image.size
        depth_resized = torch.nn.functional.interpolate(
            predicted_depth.unsqueeze(0).unsqueeze(0),
            size=(height, width),
            mode="bicubic",
            align_corners=False,
        ).squeeze().cpu().numpy()

    depth_min = float(depth_resized.min())
    depth_max = float(depth_resized.max())
    if depth_max - depth_min > 1e-8:
        depth_norm = (depth_resized - depth_min) / (depth_max - depth_min)
    else:
        depth_norm = np.zeros_like(depth_resized)
    depth_uint8 = (depth_norm * 255.0).astype(np.uint8)

    depth_3ch = np.stack([depth_uint8, depth_uint8, depth_uint8], axis=-1)
    return Image.fromarray(depth_3ch, mode="RGB")


def _extract_edge_condition_image(
    rgb_image: Image.Image, threshold1: float, threshold2: float
) -> Image.Image:
    """对原始 RGB 参考图做 Canny 边缘检测，返回 3 通道 PIL 图像。

    处理方式与 export_conditions.py 的 _canny_single 一致：转灰度图后用
    cv2.Canny 提取边缘，再复制为 3 通道，与 _pil_to_normalized_tensor 的
    输入约定对齐。
    """
    import cv2

    rgb_array = np.array(rgb_image.convert("RGB"))
    bgr_array = cv2.cvtColor(rgb_array, cv2.COLOR_RGB2BGR)
    gray = cv2.cvtColor(bgr_array, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, threshold1, threshold2)

    edges_3ch = np.stack([edges, edges, edges], axis=-1)
    return Image.fromarray(edges_3ch, mode="RGB")


class InferencePipeline:
    """封装底模加载 + 可选 LoRA 加载 + 单步推理，供 Gradio 回调复用。

    通过 lora_dir 是否变化来判断是否需要重新加载/切换 LoRA 权重，
    避免每次点击"生成"都重新加载一遍底模（底模加载耗时最长）。

    condition_type 仅用于区分该实例属于 depth 分支还是 canny 分支
    （影响提示文本/日志），两个分支的模型结构本身完全一致，各自独立
    持有一整套底模 + LoRA + RotationEncoder，互不共享，避免引入
    共享状态带来的额外复杂度。
    """

    def __init__(self, condition_type: str):
        assert condition_type in ("depth", "canny")
        self.condition_type = condition_type

        self.pretrained_model_path = None
        self.tokenizer = None
        self.text_encoder = None
        self.vae = None
        self.unet_base = None  # 未挂载 LoRA 的基础 UNet（含扩展后的 conv_in）
        self.scheduler = None
        self.rotation_encoder = None

        self.unet = None  # 当前实际用于推理的 UNet（可能是 unet_base 或挂载了 LoRA 的版本）
        self.loaded_lora_dir = None  # 记录当前已加载的 LoRA 目录，避免重复加载

    def ensure_base_model_loaded(self):
        if self.unet_base is not None:
            return
        self.pretrained_model_path = ensure_checkpoint()
        print(f"[inference_app][{self.condition_type}] 从 {self.pretrained_model_path} 加载 SD1.5 各子模块 ...")
        self.tokenizer = CLIPTokenizer.from_pretrained(self.pretrained_model_path, subfolder="tokenizer")
        self.text_encoder = CLIPTextModel.from_pretrained(
            self.pretrained_model_path, subfolder="text_encoder"
        ).to(DEVICE, dtype=DTYPE)
        self.vae = AutoencoderKL.from_pretrained(
            self.pretrained_model_path, subfolder="vae"
        ).to(DEVICE, dtype=DTYPE)
        unet = UNet2DConditionModel.from_pretrained(self.pretrained_model_path, subfolder="unet")
        unet = expand_unet_conv_in(unet)
        self.unet_base = unet

        self.scheduler = DDIMScheduler.from_pretrained(self.pretrained_model_path, subfolder="scheduler")

        self.text_encoder.requires_grad_(False)
        self.vae.requires_grad_(False)
        self.unet_base.requires_grad_(False)
        self.unet_base.eval()

        self.rotation_encoder = RotationEncoder(cross_attention_dim=self.text_encoder.config.hidden_size)
        self.rotation_encoder.to(DEVICE, dtype=DTYPE)
        self.rotation_encoder.eval()

        self.unet = self.unet_base.to(DEVICE, dtype=DTYPE)

    def ensure_lora_loaded(self, lora_dir: str):
        """按需加载/切换 LoRA + conv_in + RotationEncoder 权重。
        lora_dir 为空字符串时表示不使用 LoRA，仅用扩展后的底模做推理
        （LoRA 尚未训练完成时，用于验证 UI/推理链路是否跑通）。
        """
        lora_dir = (lora_dir or "").strip()

        if lora_dir == self.loaded_lora_dir:
            return  # 已是当前状态，无需重复加载

        if lora_dir == "":
            self.unet = self.unet_base.to(DEVICE, dtype=DTYPE)
            self.loaded_lora_dir = ""
            print(f"[inference_app][{self.condition_type}] 未指定 LoRA 目录，使用未挂载 LoRA 的底模（含扩展 conv_in）进行推理。")
            return

        conv_in_path = os.path.join(lora_dir, "conv_in.pt")
        unet_lora_path = os.path.join(lora_dir, "unet_lora")
        rotation_encoder_path = os.path.join(lora_dir, "rotation_encoder.pt")
        for p in (conv_in_path, unet_lora_path, rotation_encoder_path):
            if not os.path.exists(p):
                raise FileNotFoundError(
                    f"LoRA checkpoint 目录不完整，缺少 {p}。"
                    f"请确认路径下同时包含 conv_in.pt / unet_lora/ / rotation_encoder.pt"
                    f"（训练完成后由 train_lora.py 的 save_checkpoint 一并生成）。"
                )

        from peft import PeftModel

        print(f"[inference_app][{self.condition_type}] 从 {lora_dir} 加载 conv_in 权重 ...")
        self.unet_base.conv_in.load_state_dict(torch.load(conv_in_path, map_location="cpu"))

        print(f"[inference_app][{self.condition_type}] 从 {unet_lora_path} 加载 LoRA adapter 权重 ...")
        unet_with_lora = PeftModel.from_pretrained(self.unet_base, unet_lora_path, is_trainable=False)
        unet_with_lora.eval()
        self.unet = unet_with_lora.to(DEVICE, dtype=DTYPE)

        print(f"[inference_app][{self.condition_type}] 从 {rotation_encoder_path} 加载 RotationEncoder 权重 ...")
        self.rotation_encoder.load_pretrained(rotation_encoder_path, map_location="cpu")
        self.rotation_encoder.to(DEVICE, dtype=DTYPE)
        self.rotation_encoder.eval()

        self.loaded_lora_dir = lora_dir
        print(f"[inference_app][{self.condition_type}] LoRA checkpoint 加载完成：{lora_dir}")

    @torch.no_grad()
    def generate(
        self,
        condition_image: Image.Image,
        object_prompt: str,
        delta_azimuth: float,
        delta_elevation: float,
        delta_roll: float,
        num_inference_steps: int,
        guidance_scale: float,
        seed: int,
        lora_dir: str,
    ) -> Image.Image:
        """单步推理：仅适合小角度旋转（与训练时的相邻视角间隔量级相当）。
        大角度旋转请使用 generate_multi_step。
        """
        self.ensure_base_model_loaded()
        self.ensure_lora_loaded(lora_dir)
        return self._generate_single_step(
            condition_image=condition_image,
            object_prompt=object_prompt,
            delta_azimuth=delta_azimuth,
            delta_elevation=delta_elevation,
            delta_roll=delta_roll,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            seed=seed,
        )

    @torch.no_grad()
    def generate_multi_step(
        self,
        condition_image: Image.Image,
        object_prompt: str,
        delta_azimuth: float,
        delta_elevation: float,
        delta_roll: float,
        step_degree: float,
        num_inference_steps: int,
        guidance_scale: float,
        seed: int,
        lora_dir: str,
    ):
        """多步迭代推理：将总旋转量 [delta_azimuth, delta_elevation, delta_roll]
        按 step_degree（单步最大角度步长，度）拆分为 N 个小步长，每步用上一步
        的生成结果作为下一步的条件图，串联调用模型 N 次，逐步逼近目标视角。

        拆分方式：以三个分量中绝对值最大的一个确定步数 N =
        ceil(max(|Δaz|, |Δel|, |Δroll|) / step_degree)，其余分量按相同步数
        均分，保证每一步的三个分量都同步、线性地趋近目标值。

        返回：(final_image, intermediate_images)
            final_image: 最后一步的生成结果（PIL.Image）
            intermediate_images: 每一步生成结果组成的列表（含最后一步），
                供 UI 展示多步生成过程。
        """
        self.ensure_base_model_loaded()
        self.ensure_lora_loaded(lora_dir)

        step_degree = max(float(step_degree), 1e-3)
        max_abs_delta = max(abs(delta_azimuth), abs(delta_elevation), abs(delta_roll))
        num_steps = max(1, int(np.ceil(max_abs_delta / step_degree)))

        step_azimuth = delta_azimuth / num_steps
        step_elevation = delta_elevation / num_steps
        step_roll = delta_roll / num_steps

        current_image = condition_image
        intermediate_images = []
        for step_idx in range(num_steps):
            current_image = self._generate_single_step(
                condition_image=current_image,
                object_prompt=object_prompt,
                delta_azimuth=step_azimuth,
                delta_elevation=step_elevation,
                delta_roll=step_roll,
                num_inference_steps=num_inference_steps,
                guidance_scale=guidance_scale,
                # 每一步用不同的种子偏移，避免多步都采样到完全相同的噪声模式
                seed=int(seed) + step_idx,
            )
            intermediate_images.append(current_image)

        return current_image, intermediate_images

    @torch.no_grad()
    def _generate_single_step(
        self,
        condition_image: Image.Image,
        object_prompt: str,
        delta_azimuth: float,
        delta_elevation: float,
        delta_roll: float,
        num_inference_steps: int,
        guidance_scale: float,
        seed: int,
    ) -> Image.Image:
        """单次前向 UNet 去噪循环，是 generate / generate_multi_step 的公共实现。
        调用前需确保 ensure_base_model_loaded / ensure_lora_loaded 已执行。
        """
        condition_tensor = _pil_to_normalized_tensor(condition_image, RESOLUTION).unsqueeze(0)
        condition_tensor = condition_tensor.to(DEVICE, dtype=DTYPE)

        generator = torch.Generator(device=DEVICE).manual_seed(int(seed))

        condition_latent = self.vae.encode(condition_tensor).latent_dist.sample() * self.vae.config.scaling_factor

        prompt_input_ids = self.tokenizer(
            [object_prompt], padding="max_length",
            max_length=self.tokenizer.model_max_length, truncation=True,
            return_tensors="pt",
        ).input_ids.to(DEVICE)
        text_embedding = self.text_encoder(prompt_input_ids)[0]  # (1, 77, 768)

        rotation_vector = torch.tensor(
            [[delta_azimuth, delta_elevation, delta_roll]], dtype=DTYPE, device=DEVICE
        )
        rotation_embedding = self.rotation_encoder(rotation_vector)  # (1, 1, 768)
        encoder_hidden_states = torch.cat([text_embedding, rotation_embedding], dim=1)  # (1, 78, 768)

        # classifier-free guidance：无条件分支复用空文本 + 同一个 condition_latent/rotation
        if guidance_scale > 1.0:
            uncond_input_ids = self.tokenizer(
                [""], padding="max_length", max_length=self.tokenizer.model_max_length,
                truncation=True, return_tensors="pt",
            ).input_ids.to(DEVICE)
            uncond_embedding = self.text_encoder(uncond_input_ids)[0]
            uncond_hidden_states = torch.cat([uncond_embedding, rotation_embedding], dim=1)
            encoder_hidden_states = torch.cat([uncond_hidden_states, encoder_hidden_states], dim=0)
            condition_latent_input = torch.cat([condition_latent, condition_latent], dim=0)
        else:
            condition_latent_input = condition_latent

        self.scheduler.set_timesteps(num_inference_steps, device=DEVICE)
        latents = torch.randn(
            (1, 4, RESOLUTION // 8, RESOLUTION // 8), generator=generator, device=DEVICE, dtype=DTYPE
        )
        latents = latents * self.scheduler.init_noise_sigma

        for t in self.scheduler.timesteps:
            latent_model_input = torch.cat([latents] * 2) if guidance_scale > 1.0 else latents
            latent_model_input = self.scheduler.scale_model_input(latent_model_input, t)
            unet_input = torch.cat([latent_model_input, condition_latent_input], dim=1)

            noise_pred = self.unet(unet_input, t, encoder_hidden_states=encoder_hidden_states).sample

            if guidance_scale > 1.0:
                noise_pred_uncond, noise_pred_cond = noise_pred.chunk(2)
                noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_cond - noise_pred_uncond)

            latents = self.scheduler.step(noise_pred, t, latents).prev_sample

        image = self.vae.decode(latents / self.vae.config.scaling_factor).sample
        return _tensor_to_pil(image[0])


def _pil_to_normalized_tensor(img: Image.Image, resolution: int) -> torch.Tensor:
    img = img.convert("RGB")
    if img.size != (resolution, resolution):
        img = img.resize((resolution, resolution), Image.BICUBIC)
    arr = np.array(img).astype(np.float32)
    tensor = torch.from_numpy(arr) / 127.5 - 1.0
    return tensor.permute(2, 0, 1).contiguous()


def _tensor_to_pil(tensor: torch.Tensor) -> Image.Image:
    tensor = (tensor.float().clamp(-1, 1) + 1.0) / 2.0
    arr = (tensor.permute(1, 2, 0).cpu().numpy() * 255.0).round().astype(np.uint8)
    return Image.fromarray(arr)


def build_ui():
    import gradio as gr

    depth_pipeline = InferencePipeline(condition_type="depth")
    canny_pipeline = InferencePipeline(condition_type="canny")

    def on_image_uploaded(rgb_image, canny_threshold1, canny_threshold2):
        """图片一上传/更换就立即提取深度图与轮廓图并展示，不必等点击生成按钮。"""
        if rgb_image is None:
            return None, None
        depth_condition_image = _extract_depth_condition_image(rgb_image)
        edge_condition_image = _extract_edge_condition_image(
            rgb_image, float(canny_threshold1), float(canny_threshold2)
        )
        return depth_condition_image, edge_condition_image

    def on_generate(rgb_image, depth_condition_image, edge_condition_image,
                     object_prompt, depth_lora_dir, canny_lora_dir,
                     delta_azimuth, delta_elevation, delta_roll,
                     step_degree, num_inference_steps, guidance_scale, seed):
        if rgb_image is None:
            raise gr.Error("请先上传一张原始 RGB 参考图。")
        if depth_condition_image is None or edge_condition_image is None:
            raise gr.Error("深度图/轮廓图尚未提取完成，请稍候或重新上传图片。")

        try:
            # 2) depth 分支与 canny 分支完全独立地做多步迭代推理，
            #    彼此只用各自上一步的输出作为下一步条件图，互不影响。
            depth_final_image, depth_intermediate_images = depth_pipeline.generate_multi_step(
                condition_image=depth_condition_image,
                object_prompt=object_prompt,
                delta_azimuth=delta_azimuth,
                delta_elevation=delta_elevation,
                delta_roll=delta_roll,
                step_degree=float(step_degree),
                num_inference_steps=int(num_inference_steps),
                guidance_scale=float(guidance_scale),
                seed=int(seed),
                lora_dir=depth_lora_dir,
            )
            canny_final_image, canny_intermediate_images = canny_pipeline.generate_multi_step(
                condition_image=edge_condition_image,
                object_prompt=object_prompt,
                delta_azimuth=delta_azimuth,
                delta_elevation=delta_elevation,
                delta_roll=delta_roll,
                step_degree=float(step_degree),
                num_inference_steps=int(num_inference_steps),
                guidance_scale=float(guidance_scale),
                seed=int(seed),
                lora_dir=canny_lora_dir,
            )
        except FileNotFoundError as e:
            raise gr.Error(str(e))

        return (
            depth_final_image, depth_intermediate_images,
            canny_final_image, canny_intermediate_images,
        )

    with gr.Blocks(title="多视角相机控制 LoRA 推理调试") as demo:
        gr.Markdown(
            "## 多视角相机控制 LoRA 推理调试\n"
            "上传一张**原始 RGB 参考图**，自动提取出深度图与轮廓图两种条件图，"
            "设置目标视角相对参考视角的旋转角度后，depth 分支与 canny 分支会各自"
            "独立加载对应的 LoRA 并生成目标视角的结构图（互不影响）。\n\n"
            "**模型只训练了相邻视角的小角度旋转**，因此大角度旋转会自动拆分为多步"
            "小角度迭代生成（每步用上一步的输出作为下一步的输入），可通过「单步角度"
            "步长」控制拆分粒度，下方会分别展示两个分支每一步的中间结果。\n\n"
            "**LoRA 尚未训练完成时**，将下方对应「LoRA checkpoint 目录」留空即可运行"
            "（仅用底模验证链路是否跑通，不具备视角变换能力）；训练完成后填入类似 "
            "`e:/Multimodal/training/output/lora_depth/final` / "
            "`e:/Multimodal/training/output/lora_canny/final` 的目录即可测试真实效果。"
        )

        with gr.Row():
            with gr.Column():
                rgb_image = gr.Image(label="原始 RGB 参考图", type="pil")
                object_prompt = gr.Textbox(label="物品名称 prompt", value="coffee table")
                with gr.Row():
                    depth_lora_dir = gr.Textbox(
                        label="depth 分支 LoRA checkpoint 目录（留空 = 不使用 LoRA）",
                        value=DEFAULT_LORA_DIR_DEPTH,
                        placeholder="e:/Multimodal/training/output/lora_depth/final",
                    )
                    canny_lora_dir = gr.Textbox(
                        label="canny 分支 LoRA checkpoint 目录（留空 = 不使用 LoRA）",
                        value=DEFAULT_LORA_DIR_CANNY,
                        placeholder="e:/Multimodal/training/output/lora_canny/final",
                    )
                with gr.Row():
                    delta_azimuth = gr.Slider(-180, 180, value=0, step=1, label="Δ方位角 azimuth（度，总量）")
                    delta_elevation = gr.Slider(-90, 90, value=0, step=1, label="Δ俯仰角 elevation（度，总量）")
                    delta_roll = gr.Slider(
                        -180, 180, value=0, step=1,
                        label="Δ滚转角 roll（训练数据未提供可靠标签，已禁用，恒为0）",
                        interactive=False,
                    )
                with gr.Row():
                    canny_threshold1 = gr.Slider(0, 255, value=100, step=1, label="Canny 低阈值 threshold1")
                    canny_threshold2 = gr.Slider(0, 255, value=200, step=1, label="Canny 高阈值 threshold2")
                step_degree = gr.Slider(
                    1, 45, value=10, step=1,
                    label="单步角度步长（度，与训练数据中相邻视角间隔量级相当，越小越稳但步数越多）",
                )
                with gr.Row():
                    num_inference_steps = gr.Slider(10, 50, value=20, step=1, label="每步采样步数")
                    guidance_scale = gr.Slider(1.0, 15.0, value=7.5, step=0.5, label="CFG guidance scale")
                    seed = gr.Number(value=42, precision=0, label="随机种子")
                generate_btn = gr.Button("生成目标视角结构图", variant="primary")

            with gr.Column():
                with gr.Row():
                    depth_condition_preview = gr.Image(label="自动提取的深度图（depth 分支初始条件图）", type="pil")
                    edge_condition_preview = gr.Image(label="自动提取的轮廓图（canny 分支初始条件图）", type="pil")
                gr.Markdown("### depth 分支生成结果")
                depth_output_image = gr.Image(label="depth 分支最终生成结果")
                depth_intermediate_gallery = gr.Gallery(label="depth 分支多步迭代中间结果", columns=4)
                gr.Markdown("### canny 分支生成结果")
                canny_output_image = gr.Image(label="canny 分支最终生成结果")
                canny_intermediate_gallery = gr.Gallery(label="canny 分支多步迭代中间结果", columns=4)

        # 图片一上传/更换、或 Canny 阈值调整，立即重新提取深度图/轮廓图并展示，
        # 不必等点击「生成」按钮。
        rgb_image.upload(
            fn=on_image_uploaded,
            inputs=[rgb_image, canny_threshold1, canny_threshold2],
            outputs=[depth_condition_preview, edge_condition_preview],
        )
        for threshold_slider in (canny_threshold1, canny_threshold2):
            threshold_slider.release(
                fn=on_image_uploaded,
                inputs=[rgb_image, canny_threshold1, canny_threshold2],
                outputs=[depth_condition_preview, edge_condition_preview],
            )

        generate_btn.click(
            fn=on_generate,
            inputs=[rgb_image, depth_condition_preview, edge_condition_preview,
                    object_prompt, depth_lora_dir, canny_lora_dir,
                    delta_azimuth, delta_elevation, delta_roll,
                    step_degree, num_inference_steps, guidance_scale, seed],
            outputs=[depth_output_image, depth_intermediate_gallery,
                     canny_output_image, canny_intermediate_gallery],
        )

    return demo


if __name__ == "__main__":
    app = build_ui()
    app.launch()
