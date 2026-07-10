# -*- coding: utf-8 -*-
"""
底模（Stable Diffusion 1.5）下载与本地落地工具。

设计说明：
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

DEFAULT_MODEL_ID = "runwayml/stable-diffusion-v1-5"
DEFAULT_LOCAL_DIR = "e:/Multimodal/checkpoints/sd15"


def _is_complete_diffusers_dir(local_dir: str) -> bool:
    """判断 local_dir 是否已经是一个完整的 diffusers 格式模型目录。"""
    return os.path.isfile(os.path.join(local_dir, "model_index.json"))


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
        from diffusers import StableDiffusionPipeline
        import torch

        pipe = StableDiffusionPipeline.from_pretrained(
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
#         --model_path e:/Multimodal/checkpoints/sd15 \
#         --checkpoint_path e:/Multimodal/checkpoints/sd15_merged.safetensors \
#         --use_safetensors
#     该脚本可在 https://github.com/huggingface/diffusers 的 scripts 目录下找到，
#     需要与本地 diffusers 版本对应的版本（不同版本 API 略有差异）。
#   方式二：LoRA 权重本身已经是 ComfyUI LoraLoader 可直接加载的标准 PEFT/diffusers
#     LoRA 格式（safetensors），无需转换底模本身；只需要把 train_lora.py 训练产出的
#     LoRA 目录中的 safetensors 文件放到 ComfyUI 的 models/loras 目录下，
#     配合原始 SD1.5 单文件权重（可从 huggingface 页面直接下载 v1-5-pruned.safetensors）
#     使用 LoraLoader 节点加载即可，这是更简单可靠的路径。


if __name__ == "__main__":
    path = ensure_checkpoint()
    print(f"[checkpoint_utils] 最终本地模型路径：{path}")
