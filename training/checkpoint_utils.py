# -*- coding: utf-8 -*-
"""
底模（InstructPix2Pix）下载与本地落地工具。

设计说明：
- 底模改用 timbrooks/instruct-pix2pix 而非纯文生图的 stable-diffusion-v1-5。
  原因：本项目的 UNet 输入本来就需要 8 通道（4通道 noisy target latent +
  4通道 condition latent 在 channel 维度拼接），而 InstructPix2Pix 的 UNet
  原生 conv_in 就是 8 通道输入，且已经在大规模图像编辑数据上预训练收敛
  （前4通道对应噪声、后4通道对应参考图的 VAE latent，与本项目的拼接方式
  完全一致）。这样可以直接复用其预训练好的 8 通道 conv_in 权重，训练时
  只需要挂载 LoRA 微调"从参考结构图 + 旋转向量 -> 目标结构图"这一新任务，
  不需要再像扩展 SD1.5 的 4 通道 conv_in 那样，从随机/零初始化开始学习
  一套全新的条件图注入权重——后者在小数据集上收敛慢、效果差。
- diffusers 判断"本地目录是否已是一个完整的 diffusers 格式模型"的标准方式，
  就是看该目录下是否存在 model_index.json（这是 pipeline 的顶层索引文件，
  记录了 unet/vae/text_encoder/scheduler 等子模块各自的类名与相对路径）。
  我们复用这个判断标准，避免每次训练都重新触发下载。
- 下载走 huggingface_hub 的标准缓存机制（网络失败时不吞异常，直接抛出并给出
  可操作的排查建议），下载成功后再 save_pretrained 落地到指定目录，方便：
  1) 离线复用（后续训练直接从本地目录加载，不再联网）
  2) 用户后续手动用 diffusers 官方转换脚本导出成 ComfyUI 需要的单文件 .safetensors
"""
import os
import sys

# Windows 上创建符号链接需要额外权限，huggingface_hub 默认的缓存机制会为每个
# 下载文件创建符号链接，权限不足时会报 WinError 14007/1314 等错误。禁用符号
# 链接后 huggingface_hub 会改为直接复制文件到目标路径，规避该问题。
os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS", "1")

# huggingface_hub 默认把下载缓存放在 C 盘用户目录（C:\Users\<user>\.cache\
# huggingface），底模数个 GB，容易在 C 盘空间紧张时下载到一半报
# "Not enough free disk space" 而中断。这里统一把缓存目录转移到 E 盘。
os.environ.setdefault("HF_HOME", "e:/Multimodal/.hf_cache")

# 国内网络直连 huggingface.co 经常超时/连接失败，改用镜像端点。若用户已在
# 外部环境变量中手动设置 HF_ENDPOINT，则不覆盖。
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")

DEFAULT_MODEL_ID = "timbrooks/instruct-pix2pix"
DEFAULT_LOCAL_DIR = "e:/Multimodal/checkpoints/instruct_pix2pix"

DEFAULT_DEPTH_MODEL_ID = "depth-anything/Depth-Anything-V2-Large-hf"
DEFAULT_DEPTH_LOCAL_DIR = "e:/Multimodal/checkpoints/depth_anything_v2"


def _is_complete_diffusers_dir(local_dir: str) -> bool:
    """判断 local_dir 是否已经是一个完整的 diffusers 格式模型目录。"""
    return os.path.isfile(os.path.join(local_dir, "model_index.json"))


def _is_complete_transformers_dir(local_dir: str) -> bool:
    """判断 local_dir 是否已经是一个完整的 transformers 格式模型目录。"""
    return os.path.isfile(os.path.join(local_dir, "config.json"))


def ensure_checkpoint(model_id: str = DEFAULT_MODEL_ID, local_dir: str = DEFAULT_LOCAL_DIR) -> str:
    """
    确保 local_dir 下存在一份完整的 diffusers 格式 SD1.5 模型。
    若已存在则直接返回其绝对路径；若不存在则从 huggingface 下载并落地。

    返回：本地模型目录的绝对路径。
    """
    local_dir_abs = os.path.abspath(local_dir)

    if _is_complete_diffusers_dir(local_dir_abs):
        print(f"[checkpoint_utils] 检测到本地已存在完整模型：{local_dir_abs}，跳过下载。")
        return local_dir_abs

    print(f"[checkpoint_utils] 本地目录 {local_dir_abs} 不是完整的 diffusers 模型，"
          f"开始从 huggingface（{model_id}）下载 ...")
    print("[checkpoint_utils] 下载可能需要几分钟到几十分钟，取决于网络状况，请耐心等待。")

    try:
        from diffusers import StableDiffusionInstructPix2PixPipeline
        import torch

        pipe = StableDiffusionInstructPix2PixPipeline.from_pretrained(
            model_id,
            torch_dtype=torch.float32,
            safety_checker=None,
        )
        print(f"[checkpoint_utils] 下载完成，正在落地到本地目录：{local_dir_abs} ...")
        os.makedirs(local_dir_abs, exist_ok=True)
        pipe.save_pretrained(local_dir_abs)
        print(f"[checkpoint_utils] 模型已保存到：{local_dir_abs}")
        del pipe
    except Exception as e:
        print("[checkpoint_utils] 模型下载/保存失败！", file=sys.stderr)
        print(
            "[checkpoint_utils] 常见原因与排查建议：\n"
            "  1) 网络无法访问 huggingface.co：请检查代理设置，或设置国内镜像环境变量，\n"
            "     例如 PowerShell 中执行： $env:HF_ENDPOINT = 'https://hf-mirror.com'\n"
            "  2) 若使用代理，请确认 HTTP_PROXY / HTTPS_PROXY 环境变量已正确配置。\n"
            "  3) 若为权限/网关超时问题，可重试或更换网络环境后重新运行本函数。\n"
            f"原始异常信息：{e}",
            file=sys.stderr,
        )
        raise

    return local_dir_abs


def ensure_depth_model(model_id: str = DEFAULT_DEPTH_MODEL_ID, local_dir: str = DEFAULT_DEPTH_LOCAL_DIR) -> str:
    """
    确保 local_dir 下存在一份完整的 DepthAnything V2 深度估计模型（transformers 格式）。
    若已存在则直接返回其绝对路径；若不存在则从 huggingface 下载并落地。

    用途：inference_app.py 上传图片后自动提取深度图条件图，需要用到该模型。

    返回：本地模型目录的绝对路径。
    """
    local_dir_abs = os.path.abspath(local_dir)

    if _is_complete_transformers_dir(local_dir_abs):
        print(f"[checkpoint_utils] 检测到本地已存在完整深度模型：{local_dir_abs}，跳过下载。")
        return local_dir_abs

    print(f"[checkpoint_utils] 本地目录 {local_dir_abs} 不是完整的深度模型，"
          f"开始从 huggingface（{model_id}）下载 ...")

    try:
        from transformers import AutoImageProcessor, AutoModelForDepthEstimation

        processor = AutoImageProcessor.from_pretrained(model_id)
        model = AutoModelForDepthEstimation.from_pretrained(model_id)
        print(f"[checkpoint_utils] 下载完成，正在落地到本地目录：{local_dir_abs} ...")
        os.makedirs(local_dir_abs, exist_ok=True)
        processor.save_pretrained(local_dir_abs)
        model.save_pretrained(local_dir_abs)
        print(f"[checkpoint_utils] 深度模型已保存到：{local_dir_abs}")
        del processor, model
    except Exception as e:
        print("[checkpoint_utils] 深度模型下载/保存失败！", file=sys.stderr)
        print(
            "[checkpoint_utils] 常见原因与排查建议：\n"
            "  1) 网络无法访问 huggingface.co：请检查代理设置，或设置国内镜像环境变量，\n"
            "     例如 PowerShell 中执行： $env:HF_ENDPOINT = 'https://hf-mirror.com'\n"
            "  2) 若使用代理，请确认 HTTP_PROXY / HTTPS_PROXY 环境变量已正确配置。\n"
            "  3) 若为权限/网关超时问题，可重试或更换网络环境后重新运行本函数。\n"
            f"原始异常信息：{e}",
            file=sys.stderr,
        )
        raise

    return local_dir_abs


# --------------------------------------------------------------------------
# 可选：导出为 ComfyUI 可用的单文件 .safetensors
# --------------------------------------------------------------------------
# diffusers 历史上在 scripts/convert_diffusers_to_original_stable_diffusion.py
# 中提供过 diffusers -> 单文件 ckpt/safetensors 的转换逻辑，但该脚本并非稳定
# 公开 API（不同 diffusers 版本中路径/函数签名多次变动，甚至在较新版本中被移除
# 或迁移），直接依赖它在这里做"自动转换"并不可靠，因此本文件不提供该函数。
#
# 手动转换方法（训练完成后按需执行，任选其一）：
#   方式一（推荐）：使用 diffusers 官方仓库脚本（版本需与本地 diffusers 匹配）：
#     python scripts/convert_diffusers_to_original_stable_diffusion.py \
#         --model_path e:/Multimodal/checkpoints/instruct_pix2pix \
#         --checkpoint_path e:/Multimodal/checkpoints/instruct_pix2pix_merged.safetensors \
#         --use_safetensors
#     该脚本可在 https://github.com/huggingface/diffusers 的 scripts 目录下找到，
#     需要与本地 diffusers 版本对应的版本（不同版本 API 略有差异）。
#   方式二：LoRA 权重本身已经是 ComfyUI LoraLoader 可直接加载的标准 PEFT/diffusers
#     LoRA 格式（safetensors），无需转换底模本身；只需要把 train_lora.py 训练产出的
#     LoRA 目录中的 safetensors 文件放到 ComfyUI 的 models/loras 目录下，
#     配合原始 InstructPix2Pix 单文件权重使用 LoraLoader 节点加载即可，
#     这是更简单可靠的路径。


if __name__ == "__main__":
    path = ensure_checkpoint()
    print(f"[checkpoint_utils] 最终本地模型路径：{path}")
    depth_path = ensure_depth_model()
    print(f"[checkpoint_utils] 最终本地深度模型路径：{depth_path}")
