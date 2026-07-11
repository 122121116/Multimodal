# -*- coding: utf-8 -*-
import os
# 设置 HuggingFace 缓存到 E 盘（避免 C 盘空间不足），路径需与
# checkpoint_utils.py 中的设置保持一致，避免同一份缓存被下载到两个不同目录。
os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS", "1")
os.environ.setdefault("HF_HOME", "e:/Multimodal/.hf_cache")
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
# Gradio 默认会在 launch() 时联网上报 analytics 并检测新版本，国内网络下
# 该请求经常长时间无响应，导致 launch() 卡住且没有任何报错输出。这里彻底
# 关闭该行为（必须在 import gradio 之前设置才生效）。
os.environ.setdefault("GRADIO_ANALYTICS_ENABLED", "False")

"""
带 UI 的推理调用示例脚本：加载参考图 + 目标相对旋转向量，生成目标视角的
结构图（轮廓图/深度图），用于快速验证 train_lora.py 训练出的 LoRA 效果；
LoRA 未训练完成时，自动退化为 InstructPix2Pix 原生图生图（图像编辑）能力，
保证 App 在任何时候都能"读图 -> 出图"，不会因为 LoRA 缺失而输出无意义结果。

工作流程：
1. 用户上传一张原始 RGB 参考图后，立即自动提取出该图的深度图与轮廓图并
   展示（不必等点击生成按钮），分别对应 depth 分支与 canny 分支的初始条件图。
2. 点击"生成"后，depth 分支与 canny 分支各自独立运行、互不影响：
   - 若该分支填写了有效的 LoRA checkpoint 目录：
       走"结构图视角变换"任务：以对应条件图（深度图/轮廓图）作为参考结构图，
       [Δazimuth, Δelevation, Δroll] 旋转向量作为条件，用训练时同款的
       VAE-latent 拼接 + RotationEncoder 方案预测目标视角的结构图。因为
       LoRA 只训练了"相邻视角小角度旋转"，大角度旋转会自动拆分成多个小步长、
       串联迭代生成（每步用上一步输出作为下一步条件图）。
   - 若该分支未填写 LoRA 目录（默认状态，LoRA 尚未训练完成）：
       不走上述结构图预测逻辑（此时 RotationEncoder 是随机初始化、未训练的，
       结构图预测管线本身没有意义），而是改用 InstructPix2Pix 原生的
       StableDiffusionInstructPix2PixPipeline，对原始 RGB 参考图 + 文本编辑
       指令做标准的图生图（图像编辑），这是底模本身已经在大规模数据上预训练
       收敛好的能力，保证"没有 LoRA 时也能读取图片输出有意义结果"。

依赖（需预先安装，另见 requirements.txt）：
    torch, diffusers, transformers, accelerate, peft, Pillow, numpy, opencv-python, gradio

启动方式：
    python inference_app.py
默认会在本地启动一个 Gradio 网页界面（http://127.0.0.1:7860）。
"""
print("[inference_app] [1/4] 开始导入依赖库（torch/diffusers/transformers），可能需要几秒到几十秒 ...", flush=True)
import numpy as np
import torch
from diffusers import (
    AutoencoderKL,
    DDIMScheduler,
    StableDiffusionInstructPix2PixPipeline,
    UNet2DConditionModel,
)
from PIL import Image
from transformers import CLIPTextModel, CLIPTokenizer

from checkpoint_utils import ensure_checkpoint, ensure_depth_model
print(f"[inference_app] [1/4] 依赖库导入完成。torch.cuda.is_available()={torch.cuda.is_available()}", flush=True)
from rotation_encoder import RotationEncoder

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE = torch.float16 if DEVICE == "cuda" else torch.float32
RESOLUTION = 512

# LoRA 尚未训练完成时默认留空，确保 App 默认即可在"无 LoRA、仅用底模原生
# 图生图能力"的状态下跑通链路；训练完成后可在 UI 的输入框中手动填入对应
# 目录，例如 e:/Multimodal/training/output/lora_depth/final，无需改代码。
DEFAULT_LORA_DIR_DEPTH = ""
DEFAULT_LORA_DIR_CANNY = ""


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

            print("[inference_app] 首次使用，正在检查/下载 DepthAnything V2 深度模型 ...", flush=True)
            depth_model_path = ensure_depth_model()
            print(f"[inference_app] 从 {depth_model_path} 加载深度估计模型 ...", flush=True)
            cls.processor = AutoImageProcessor.from_pretrained(depth_model_path)
            cls.model = AutoModelForDepthEstimation.from_pretrained(depth_model_path)
            cls.model.to(DEVICE)
            cls.model.eval()
            print("[inference_app] 深度估计模型加载完成，已就绪。", flush=True)
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


class InferencePipeline:
    """封装底模加载 + 可选 LoRA 加载 + 单步推理，供 Gradio 回调复用。

    两种互斥的推理模式，由 lora_dir 是否为空自动切换：
    - lora_dir 非空：结构图视角变换模式（走 RotationEncoder + condition
      latent 拼接，与 train_lora.py 训练任务完全一致），需要加载对应的
      LoRA adapter + RotationEncoder 权重。
    - lora_dir 为空（默认，LoRA 尚未训练完成）：InstructPix2Pix 原生
      图生图模式，直接调用 diffusers 官方 StableDiffusionInstructPix2PixPipeline，
      用文本编辑指令对原始 RGB 图做图像编辑，不涉及 RotationEncoder /
      condition latent 拼接这套自定义逻辑。

    condition_type 仅用于区分该实例属于 depth 分支还是 canny 分支
    （影响提示文本/日志/结构图预测模式下使用哪种条件图），两个分支的模型
    结构本身完全一致，各自独立持有一整套底模 + LoRA + RotationEncoder，
    互不共享，避免引入共享状态带来的额外复杂度。
    """

    def __init__(self, condition_type: str):
        assert condition_type in ("depth", "canny")
        self.condition_type = condition_type

        self.pretrained_model_path = None
        self.tokenizer = None
        self.text_encoder = None
        self.vae = None
        self.unet_base = None  # 未挂载 LoRA 的基础 UNet（InstructPix2Pix 原生 8 通道 conv_in）
        self.scheduler = None
        self.rotation_encoder = None

        self.unet = None  # 结构图预测模式下实际用于推理的 UNet（可能挂载了 LoRA）
        self.loaded_lora_dir = None  # 记录当前已加载的 LoRA 目录，避免重复加载

        # InstructPix2Pix 原生图生图 pipeline（无 LoRA 时使用），懒加载。
        self.raw_pipe = None

    # ------------------------------------------------------------------
    # 底模 / LoRA 加载
    # ------------------------------------------------------------------
    def ensure_base_model_loaded(self):
        if self.unet_base is not None:
            return
        print(f"[inference_app][{self.condition_type}] 正在检查/下载 InstructPix2Pix 底模 ...", flush=True)
        self.pretrained_model_path = ensure_checkpoint()
        print(f"[inference_app][{self.condition_type}] 从 {self.pretrained_model_path} 加载 InstructPix2Pix 各子模块 ...", flush=True)
        self.tokenizer = CLIPTokenizer.from_pretrained(self.pretrained_model_path, subfolder="tokenizer")
        print(f"[inference_app][{self.condition_type}]   - tokenizer 加载完成", flush=True)
        self.text_encoder = CLIPTextModel.from_pretrained(
            self.pretrained_model_path, subfolder="text_encoder"
        ).to(DEVICE, dtype=DTYPE)
        print(f"[inference_app][{self.condition_type}]   - text_encoder 加载完成", flush=True)
        self.vae = AutoencoderKL.from_pretrained(
            self.pretrained_model_path, subfolder="vae"
        ).to(DEVICE, dtype=DTYPE)
        print(f"[inference_app][{self.condition_type}]   - vae 加载完成", flush=True)
        unet = UNet2DConditionModel.from_pretrained(self.pretrained_model_path, subfolder="unet")
        assert unet.conv_in.in_channels == 8, (
            f"预期 InstructPix2Pix UNet conv_in 输入通道为8，实际为{unet.conv_in.in_channels}。"
        )
        self.unet_base = unet
        print(f"[inference_app][{self.condition_type}]   - unet 加载完成（conv_in.in_channels={unet.conv_in.in_channels}）", flush=True)

        self.scheduler = DDIMScheduler.from_pretrained(self.pretrained_model_path, subfolder="scheduler")

        self.text_encoder.requires_grad_(False)
        self.vae.requires_grad_(False)
        self.unet_base.requires_grad_(False)
        self.unet_base.eval()

        self.rotation_encoder = RotationEncoder(cross_attention_dim=self.text_encoder.config.hidden_size)
        self.rotation_encoder.to(DEVICE, dtype=DTYPE)
        self.rotation_encoder.eval()

        self.unet = self.unet_base.to(DEVICE, dtype=DTYPE)
        print(f"[inference_app][{self.condition_type}] 结构图预测模式底模加载完成。", flush=True)

    def ensure_lora_loaded(self, lora_dir: str):
        """按需加载/切换 LoRA + RotationEncoder 权重（base model 全程冻结，
        不含单独的 conv_in 权重文件）。仅在结构图预测模式（lora_dir 非空）
        下调用。
        """
        lora_dir = (lora_dir or "").strip()
        assert lora_dir != "", "ensure_lora_loaded 只应在 lora_dir 非空时调用。"

        if lora_dir == self.loaded_lora_dir:
            return  # 已是当前状态，无需重复加载

        unet_lora_path = os.path.join(lora_dir, "unet_lora")
        rotation_encoder_path = os.path.join(lora_dir, "rotation_encoder.pt")
        for p in (unet_lora_path, rotation_encoder_path):
            if not os.path.exists(p):
                raise FileNotFoundError(
                    f"LoRA checkpoint 目录不完整，缺少 {p}。"
                    f"请确认路径下同时包含 unet_lora/ / rotation_encoder.pt"
                    f"（训练完成后由 train_lora.py 的 save_checkpoint 一并生成），"
                    f"或将该输入框留空以使用底模原生图生图能力。"
                )

        from peft import PeftModel

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

    def ensure_raw_pipe_loaded(self):
        """懒加载 InstructPix2Pix 官方图生图 pipeline，无 LoRA 时使用。
        与结构图预测模式共享同一份底模文件（本地磁盘路径相同），但走
        diffusers 官方封装好的标准推理流程，不涉及本项目自定义的
        RotationEncoder / condition latent 拼接逻辑。
        """
        if self.raw_pipe is not None:
            return
        print(f"[inference_app][{self.condition_type}] 正在检查/下载 InstructPix2Pix 底模 ...", flush=True)
        pretrained_model_path = ensure_checkpoint()
        print(f"[inference_app][{self.condition_type}] 从 {pretrained_model_path} 加载 InstructPix2Pix 原生图生图 pipeline ...", flush=True)
        self.raw_pipe = StableDiffusionInstructPix2PixPipeline.from_pretrained(
            pretrained_model_path, torch_dtype=DTYPE, safety_checker=None,
        ).to(DEVICE)
        self.raw_pipe.set_progress_bar_config(disable=True)
        print(f"[inference_app][{self.condition_type}] 原生图生图 pipeline 加载完成，已就绪。", flush=True)

    # ------------------------------------------------------------------
    # 生成入口：根据 lora_dir 是否为空，自动选择"结构图视角变换"或
    # "InstructPix2Pix 原生图生图"两种互斥模式。
    # ------------------------------------------------------------------
    @torch.no_grad()
    def generate_multi_step(
        self,
        rgb_image: Image.Image,
        condition_image: Image.Image,
        object_prompt: str,
        edit_instruction: str,
        delta_azimuth: float,
        delta_elevation: float,
        delta_roll: float,
        step_degree: float,
        num_inference_steps: int,
        guidance_scale: float,
        image_guidance_scale: float,
        seed: int,
        lora_dir: str,
    ):
        """统一的多步生成入口。

        返回：(final_image, intermediate_images)
            final_image: 最后一步的生成结果（PIL.Image）
            intermediate_images: 每一步生成结果组成的列表（含最后一步），
                供 UI 展示多步生成过程；无 LoRA 模式下固定只有 1 步。
        """
        lora_dir = (lora_dir or "").strip()

        if lora_dir == "":
            # 无 LoRA：退化为 InstructPix2Pix 原生图生图（图像编辑），
            # 直接对原始 RGB 图操作，不使用条件图/旋转向量。
            final_image = self._generate_raw_edit(
                rgb_image=rgb_image,
                edit_instruction=edit_instruction,
                num_inference_steps=num_inference_steps,
                guidance_scale=guidance_scale,
                image_guidance_scale=image_guidance_scale,
                seed=seed,
            )
            return final_image, [final_image]

        # 有 LoRA：结构图视角变换模式，大角度旋转拆分为多步小步长迭代。
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
    def _generate_raw_edit(
        self,
        rgb_image: Image.Image,
        edit_instruction: str,
        num_inference_steps: int,
        guidance_scale: float,
        image_guidance_scale: float,
        seed: int,
    ) -> Image.Image:
        """InstructPix2Pix 原生图生图（图像编辑）：无 LoRA 时的默认路径。"""
        self.ensure_raw_pipe_loaded()
        rgb_image = rgb_image.convert("RGB")
        if rgb_image.size != (RESOLUTION, RESOLUTION):
            rgb_image = rgb_image.resize((RESOLUTION, RESOLUTION), Image.BICUBIC)

        generator = torch.Generator(device=DEVICE).manual_seed(int(seed))
        result = self.raw_pipe(
            prompt=edit_instruction,
            image=rgb_image,
            num_inference_steps=int(num_inference_steps),
            guidance_scale=float(guidance_scale),
            image_guidance_scale=float(image_guidance_scale),
            generator=generator,
        )
        return result.images[0]

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
        """单次前向 UNet 去噪循环（结构图视角变换模式），是
        generate_multi_step 在"有 LoRA"分支下的公共实现。调用前需确保
        ensure_base_model_loaded / ensure_lora_loaded 已执行。
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
                     object_prompt, edit_instruction, depth_lora_dir, canny_lora_dir,
                     delta_azimuth, delta_elevation, delta_roll,
                     step_degree, num_inference_steps, guidance_scale,
                     image_guidance_scale, seed):
        if rgb_image is None:
            raise gr.Error("请先上传一张原始 RGB 参考图。")
        if depth_condition_image is None or edge_condition_image is None:
            raise gr.Error("深度图/轮廓图尚未提取完成，请稍候或重新上传图片。")

        try:
            # depth 分支与 canny 分支完全独立地生成，各自根据自己的 LoRA
            # 目录是否为空，自动选择"结构图视角变换"或"原生图生图"模式，
            # 彼此互不影响。
            depth_final_image, depth_intermediate_images = depth_pipeline.generate_multi_step(
                rgb_image=rgb_image,
                condition_image=depth_condition_image,
                object_prompt=object_prompt,
                edit_instruction=edit_instruction,
                delta_azimuth=delta_azimuth,
                delta_elevation=delta_elevation,
                delta_roll=delta_roll,
                step_degree=float(step_degree),
                num_inference_steps=int(num_inference_steps),
                guidance_scale=float(guidance_scale),
                image_guidance_scale=float(image_guidance_scale),
                seed=int(seed),
                lora_dir=depth_lora_dir,
            )
            canny_final_image, canny_intermediate_images = canny_pipeline.generate_multi_step(
                rgb_image=rgb_image,
                condition_image=edge_condition_image,
                object_prompt=object_prompt,
                edit_instruction=edit_instruction,
                delta_azimuth=delta_azimuth,
                delta_elevation=delta_elevation,
                delta_roll=delta_roll,
                step_degree=float(step_degree),
                num_inference_steps=int(num_inference_steps),
                guidance_scale=float(guidance_scale),
                image_guidance_scale=float(image_guidance_scale),
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
            "上传一张**原始 RGB 参考图**，自动提取出深度图与轮廓图两种条件图。\n\n"
            "**depth 分支 / canny 分支各自独立**：若填写了对应的 LoRA checkpoint "
            "目录，则加载该 LoRA 做「结构图视角变换」（按设置的旋转角度生成目标"
            "视角的结构图，大角度会自动拆分为多步小角度迭代）；**若该输入框留空"
            "（默认状态，LoRA 尚未训练完成），则改用 InstructPix2Pix 底模原生的"
            "图生图（图像编辑）能力**，对原始 RGB 图按下方「图像编辑指令」直接编辑，"
            "保证在没有 LoRA 时也能正常读图出图。"
        )

        with gr.Row():
            with gr.Column():
                rgb_image = gr.Image(label="原始 RGB 参考图", type="pil")
                object_prompt = gr.Textbox(
                    label="物品名称 prompt（结构图视角变换模式使用）", value="coffee table"
                )
                edit_instruction = gr.Textbox(
                    label="图像编辑指令（无 LoRA 时，InstructPix2Pix 原生图生图使用）",
                    value="turn it into a pencil sketch",
                )
                with gr.Row():
                    depth_lora_dir = gr.Textbox(
                        label="depth 分支 LoRA checkpoint 目录（留空 = 不使用 LoRA，走原生图生图）",
                        value=DEFAULT_LORA_DIR_DEPTH,
                        placeholder="e:/Multimodal/training/output/lora_depth/final",
                    )
                    canny_lora_dir = gr.Textbox(
                        label="canny 分支 LoRA checkpoint 目录（留空 = 不使用 LoRA，走原生图生图）",
                        value=DEFAULT_LORA_DIR_CANNY,
                        placeholder="e:/Multimodal/training/output/lora_canny/final",
                    )
                with gr.Row():
                    delta_azimuth = gr.Slider(-180, 180, value=0, step=1, label="Δ方位角 azimuth（度，总量，结构图视角变换模式使用）")
                    delta_elevation = gr.Slider(-90, 90, value=0, step=1, label="Δ俯仰角 elevation（度，总量，结构图视角变换模式使用）")
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
                    label="单步角度步长（度，结构图视角变换模式下角度较大时自动分步，越小越稳但步数越多）",
                )
                with gr.Row():
                    num_inference_steps = gr.Slider(10, 50, value=20, step=1, label="每步采样步数")
                    guidance_scale = gr.Slider(1.0, 15.0, value=7.5, step=0.5, label="文本 guidance scale")
                    image_guidance_scale = gr.Slider(1.0, 5.0, value=1.5, step=0.1, label="图像 guidance scale（仅原生图生图模式使用）")
                    seed = gr.Number(value=42, precision=0, label="随机种子")
                generate_btn = gr.Button("生成", variant="primary")

            with gr.Column():
                with gr.Row():
                    depth_condition_preview = gr.Image(label="自动提取的深度图（depth 分支初始条件图）", type="pil")
                    edge_condition_preview = gr.Image(label="自动提取的轮廓图（canny 分支初始条件图）", type="pil")
                gr.Markdown("### depth 分支生成结果")
                depth_output_image = gr.Image(label="depth 分支最终生成结果")
                depth_intermediate_gallery = gr.Gallery(label="depth 分支多步迭代中间结果（无 LoRA 时仅 1 张）", columns=4)
                gr.Markdown("### canny 分支生成结果")
                canny_output_image = gr.Image(label="canny 分支最终生成结果")
                canny_intermediate_gallery = gr.Gallery(label="canny 分支多步迭代中间结果（无 LoRA 时仅 1 张）", columns=4)

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
                    object_prompt, edit_instruction, depth_lora_dir, canny_lora_dir,
                    delta_azimuth, delta_elevation, delta_roll,
                    step_degree, num_inference_steps, guidance_scale,
                    image_guidance_scale, seed],
            outputs=[depth_output_image, depth_intermediate_gallery,
                     canny_output_image, canny_intermediate_gallery],
        )

    return demo


if __name__ == "__main__":
    print("[inference_app] [2/4] 正在构建 Gradio UI ...", flush=True)
    app = build_ui()
    print("[inference_app] [3/4] UI 构建完成。", flush=True)
    print("[inference_app] [4/4] 正在启动本地 Web 服务（首次生成时才会按需加载模型，"
          "不会在这里卡住）...", flush=True)
    # Gradio 默认 launch() 会尝试联网做版本/更新检测（analytics），国内网络
    # 环境下容易卡在这一步且没有任何报错输出，看起来像"卡死"。这里显式关闭
    # analytics，并指定 server_name/server_port，避免额外的网络探测。
    app.launch(
        server_name="127.0.0.1",
        server_port=7860,
        share=False,
        inbrowser=False,
        show_error=True,
        quiet=False,
    )
    print("[inference_app] Web 服务已退出。", flush=True)
