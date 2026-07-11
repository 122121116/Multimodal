# -*- coding: utf-8 -*-
import os
os.environ.setdefault("HF_HOME", "E:/Multimodal/hf_cache")
os.environ.setdefault("TRANSFORMERS_CACHE", "E:/Multimodal/hf_cache/transformers")
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")

"""
LoRA 训练脚本：训练 canny 或 depth 分支的 LoRA 模块。

任务：输入 (参考图的结构图 + 相机相对旋转向量)，输出目标视角的结构图。

依赖（需预先安装）：
    torch, diffusers, transformers, accelerate, peft, Pillow, numpy

命令行示例：
    python train_lora.py --condition_type canny
    python train_lora.py --condition_type depth --num_epochs 20 --lora_rank 8
    # 断点续训（从中断前保存的 checkpoint 继续，恢复 LoRA/conv_in/RotationEncoder/
    # 优化器状态与训练进度）：
    python train_lora.py --condition_type canny --resume_from e:/Multimodal/training/output/lora_canny/step_500

核心技术方案：
1) condition_image 注入方式（参考 diffusers 官方 train_instruct_pix2pix.py 的做法）：
   UNet 原生 conv_in 只接受 4 通道（纯噪声 latent）。这里把 condition_image 也
   编码到 VAE latent 空间，与 noisy target latent 在 channel 维度拼接成 8 通道，
   因此需要把 conv_in 的输入通道从 4 扩展到 8：新 conv_in 前 4 个通道复制原始
   预训练权重（保留SD对"从噪声预测干净图像"的先验知识），新增的 4 个通道权重
   初始化为 0（保证扩展初期，新增通道对输出没有扰动，训练从等价于原始SD行为
   的状态开始，这是 InstructPix2Pix 论文里验证过的稳定初始化方式）。
   condition_image 直接读取渲染完成后由 export_conditions.py 独立后处理生成的
   深度图/轮廓图文件（不再于训练阶段实时计算 Canny/DepthAnything V2 推理，
   详见 dataset.py）。
2) 参考图/目标图为同一场景内、同一 elevation 行中 azimuth 相邻的两个视角
   （小角度旋转），而非任意两个视角，以降低单步训练任务难度、提升 loss
   收敛稳定性；任意大角度的视角变换改由推理阶段多步迭代小步长旋转实现
   （详见 dataset.py 的 AdjacentPairDataset.__getitem__ 与 inference_app.py）。
3) 旋转向量条件注入：3维角度先做 sin/cos 编码消除周期性，再经 RotationEncoder
   MLP 升维到 768，产出长度为1的伪文本token，与 CLIP 的 (batch,77,768) 输出在
   token 维度拼接得到 (batch,78,768)，作为 UNet cross-attention 的条件。
4) 文本条件：不再使用固定空字符串编码，而是取每个样本所在场景的物品名称
   （dataset.py 中 object_prompt 字段，从 poses.json 的 object_source 字段解析
   得到，如 "CoffeeTable_01" -> "coffee table"）逐样本动态 tokenize + CLIP
   编码，使文本条件携带物品语义信息。
5) LoRA 只挂载在 UNet 的 attention 层（peft.LoraConfig），VAE 与 CLIP 文本编码器
   全程冻结，不参与反向传播，大幅降低显存占用。
6) 显存优化（面向 16G 显存的 RTX 4060）：
   - gradient_checkpointing 默认开启
   - mixed_precision 默认 fp16
   - batch_size=1 + gradient_accumulation_steps=4，用梯度累积模拟等效 batch=4

排障说明：
   每次关键阶段（模型加载/conv_in扩展/LoRA挂载/数据集构建/accelerator.prepare/
   首个训练 step）都会打印带 flush=True 的日志，且训练循环用 try/except 包裹
   并在异常时打印完整 traceback，避免进程崩溃时因 stdout 缓冲区未刷新而看起来
   像是"卡死在上一条日志"，方便定位真实出错位置。
"""
import argparse
import json
import os
import sys
import traceback

import torch
import torch.nn.functional as F
from accelerate import Accelerator
from diffusers import AutoencoderKL, DDPMScheduler, UNet2DConditionModel
from peft import LoraConfig, get_peft_model
from torch.utils.data import DataLoader
from transformers import CLIPTextModel, CLIPTokenizer

from checkpoint_utils import ensure_checkpoint
from dataset import AdjacentPairDataset
from rotation_encoder import RotationEncoder


def _log(msg: str) -> None:
    print(msg, flush=True)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--condition_type", type=str, required=True, choices=["canny", "depth"])
    parser.add_argument("--dataset_root", type=str, default="e:/Multimodal/dataset_output")
    parser.add_argument("--manifest_path", type=str, default=None)
    parser.add_argument("--output_dir", type=str, default=None)
    parser.add_argument("--pretrained_model", type=str, default=None)
    parser.add_argument("--resolution", type=int, default=512)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=4)
    parser.add_argument("--learning_rate", type=float, default=1e-4)
    parser.add_argument("--num_epochs", type=int, default=10)
    parser.add_argument("--lora_rank", type=int, default=4)
    parser.add_argument("--mixed_precision", type=str, default="fp16", choices=["no", "fp16", "bf16"])
    parser.add_argument("--save_steps", type=int, default=500)
    parser.add_argument(
        "--resume_from", type=str, default=None,
        help="断点续训：指定此前保存的 checkpoint 目录（如 "
             "e:/Multimodal/training/output/lora_canny/step_500），将恢复 LoRA/"
             "conv_in/RotationEncoder/优化器权重与训练进度（epoch、global_step），"
             "从中断处继续训练，不再从零开始。默认 None 表示从头训练。",
    )
    parser.add_argument(
        "--num_workers", type=int, default=0,
        help="DataLoader 子进程数。Windows 下多进程 spawn 需要重新 import 整个训练"
             "脚本及其重型依赖，首次启动较慢，默认 0（主进程内加载，最稳妥）；"
             "如需加速可按需调大。",
    )
    args = parser.parse_args()

    if args.output_dir is None:
        args.output_dir = f"e:/Multimodal/training/output/lora_{args.condition_type}"
    return args


def expand_unet_conv_in(unet: UNet2DConditionModel) -> UNet2DConditionModel:
    """
    将 UNet 的 conv_in 从 4 通道输入扩展为 8 通道输入（参考 InstructPix2Pix 做法）。
    前 4 个通道保留原始预训练权重（对应 noisy target latent），
    新增的 4 个通道初始化为 0（对应 condition latent，初始时不影响输出）。
    """
    old_conv_in = unet.conv_in
    in_channels = old_conv_in.in_channels
    if in_channels == 8:
        _log("[train_lora] UNet conv_in 已经是 8 通道输入，跳过扩展。")
        return unet

    assert in_channels == 4, f"预期 conv_in 输入通道为4，实际为{in_channels}，无法按标准流程扩展。"

    out_channels = old_conv_in.out_channels
    kernel_size = old_conv_in.kernel_size
    stride = old_conv_in.stride
    padding = old_conv_in.padding

    new_conv_in = torch.nn.Conv2d(
        in_channels=8,
        out_channels=out_channels,
        kernel_size=kernel_size,
        stride=stride,
        padding=padding,
    )
    new_conv_in.weight.data.zero_()
    new_conv_in.weight.data[:, :4, :, :] = old_conv_in.weight.data
    new_conv_in.bias.data = old_conv_in.bias.data.clone()

    unet.conv_in = new_conv_in
    # 让 UNet 配置也同步记录新的输入通道数，避免后续 save_pretrained/from_pretrained 时通道数不一致
    unet.config.in_channels = 8
    _log("[train_lora] 已将 UNet conv_in 从 4 通道扩展为 8 通道（前4通道保留预训练权重，后4通道置零初始化）。")
    return unet


def build_models(args, accelerator):
    """加载底模、扩展 conv_in、挂载 LoRA、构建 RotationEncoder。返回训练所需的各个模块。"""
    pretrained_model_path = args.pretrained_model or ensure_checkpoint()

    _log(f"[train_lora] 从 {pretrained_model_path} 加载 SD1.5 各子模块 ...")
    tokenizer = CLIPTokenizer.from_pretrained(pretrained_model_path, subfolder="tokenizer")
    text_encoder = CLIPTextModel.from_pretrained(pretrained_model_path, subfolder="text_encoder")
    vae = AutoencoderKL.from_pretrained(pretrained_model_path, subfolder="vae")
    unet = UNet2DConditionModel.from_pretrained(pretrained_model_path, subfolder="unet")
    noise_scheduler = DDPMScheduler.from_pretrained(pretrained_model_path, subfolder="scheduler")
    _log("[train_lora] SD1.5 各子模块加载完成。")

    vae.requires_grad_(False)
    text_encoder.requires_grad_(False)

    unet = expand_unet_conv_in(unet)

    if args.resume_from is not None:
        conv_in_path = os.path.join(args.resume_from, "conv_in.pt")
        unet.conv_in.load_state_dict(torch.load(conv_in_path, map_location="cpu"))
        _log(f"[train_lora] 断点续训：已从 {conv_in_path} 恢复 conv_in 权重。")

    if args.resume_from is not None:
        from peft import PeftModel
        unet = PeftModel.from_pretrained(unet, os.path.join(args.resume_from, "unet_lora"), is_trainable=True)
        _log(f"[train_lora] 断点续训：已从 {args.resume_from}/unet_lora 恢复 LoRA adapter 权重。")
    else:
        lora_config = LoraConfig(
            r=args.lora_rank,
            lora_alpha=args.lora_rank,
            target_modules=["to_q", "to_k", "to_v", "to_out.0"],
            init_lora_weights="gaussian",
        )
        unet = get_peft_model(unet, lora_config)
    unet.print_trainable_parameters()

    # get_peft_model 默认冻结所有非-LoRA参数，但 conv_in 是本任务新扩展的、
    # 承载 condition_image 信息的关键层，必须显式设为可训练，否则新增的4个
    # 通道会永远停留在零初始化状态，模型学不到如何利用 condition 图像。
    unet.get_base_model().conv_in.requires_grad_(True)

    if args.mixed_precision != "no":
        unet.enable_gradient_checkpointing()
    _log("[train_lora] conv_in 扩展 + LoRA 挂载完成。")

    rotation_encoder = RotationEncoder(cross_attention_dim=text_encoder.config.hidden_size)
    if args.resume_from is not None:
        rotation_encoder_path = os.path.join(args.resume_from, "rotation_encoder.pt")
        rotation_encoder.load_pretrained(rotation_encoder_path, map_location="cpu")
        _log(f"[train_lora] 断点续训：已从 {rotation_encoder_path} 恢复 RotationEncoder 权重。")

    return tokenizer, text_encoder, vae, unet, noise_scheduler, rotation_encoder


def main():
    args = parse_args()

    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision=args.mixed_precision,
    )
    _log(f"[train_lora] accelerator 就绪，device={accelerator.device}，mixed_precision={args.mixed_precision}")

    tokenizer, text_encoder, vae, unet, noise_scheduler, rotation_encoder = build_models(args, accelerator)

    # ------------------------------------------------------------------
    # 数据集 / DataLoader
    # ------------------------------------------------------------------
    train_dataset = AdjacentPairDataset(
        dataset_root=args.dataset_root,
        manifest_path=args.manifest_path,
        condition_type=args.condition_type,
        resolution=args.resolution,
    )
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
    )
    _log(f"[train_lora] 数据集就绪，共 {len(train_dataset)} 个训练样本。")

    # ------------------------------------------------------------------
    # 优化器：只优化 LoRA 参数 + RotationEncoder 参数
    # ------------------------------------------------------------------
    trainable_params = [p for p in unet.parameters() if p.requires_grad] + list(rotation_encoder.parameters())
    optimizer = torch.optim.AdamW(trainable_params, lr=args.learning_rate)

    unet, rotation_encoder, optimizer, train_dataloader = accelerator.prepare(
        unet, rotation_encoder, optimizer, train_dataloader
    )
    vae.to(accelerator.device)
    text_encoder.to(accelerator.device)
    _log("[train_lora] accelerator.prepare 完成，模型已就绪，准备开始训练循环。")

    start_epoch = 0
    global_step = 0
    loss_history = []
    if args.resume_from is not None:
        optimizer_path = os.path.join(args.resume_from, "optimizer.pt")
        training_state_path = os.path.join(args.resume_from, "training_state.json")
        if os.path.isfile(optimizer_path):
            optimizer.load_state_dict(torch.load(optimizer_path, map_location=accelerator.device))
            _log(f"[train_lora] 断点续训：已从 {optimizer_path} 恢复优化器状态。")
        if os.path.isfile(training_state_path):
            with open(training_state_path, "r", encoding="utf-8") as f:
                training_state = json.load(f)
            global_step = training_state["global_step"]
            start_epoch = training_state["epoch"]
            _log(f"[train_lora] 断点续训：从 epoch={start_epoch} global_step={global_step} 继续训练。")
        loss_history_path = os.path.join(os.path.dirname(args.resume_from), "loss_history.json")
        if os.path.isfile(loss_history_path):
            with open(loss_history_path, "r", encoding="utf-8") as f:
                loss_history = json.load(f)

    # ------------------------------------------------------------------
    # 训练循环
    # ------------------------------------------------------------------
    try:
        for epoch in range(start_epoch, args.num_epochs):
            unet.train()
            rotation_encoder.train()
            for step, batch in enumerate(train_dataloader):
                if epoch == start_epoch and step == 0:
                    _log("[train_lora] 已取到第一个 batch，开始第一次前向/反向 ...")

                with accelerator.accumulate(unet):
                    condition_image = batch["condition_image"].to(accelerator.device)
                    target_image = batch["target_image"].to(accelerator.device)
                    rotation_vector = batch["rotation_vector"].to(accelerator.device)
                    object_prompt = batch["object_prompt"]  # list[str]，长度为 bsz
                    bsz = target_image.shape[0]

                    with torch.no_grad():
                        # target latent：加噪声的对象（VAE latent 需要按官方缩放系数缩放）
                        target_latent = vae.encode(target_image).latent_dist.sample() * vae.config.scaling_factor
                        # condition latent：与 noisy latent 拼接的额外条件通道
                        condition_latent = vae.encode(condition_image).latent_dist.sample() * vae.config.scaling_factor

                        # 逐样本物品名称文本 -> CLIP 文本编码，替代固定空字符串编码
                        prompt_input_ids = tokenizer(
                            list(object_prompt), padding="max_length",
                            max_length=tokenizer.model_max_length, truncation=True,
                            return_tensors="pt",
                        ).input_ids.to(accelerator.device)
                        text_embedding = text_encoder(prompt_input_ids)[0]  # (bsz, 77, 768)

                    noise = torch.randn_like(target_latent)
                    timesteps = torch.randint(
                        0, noise_scheduler.config.num_train_timesteps, (bsz,), device=accelerator.device
                    ).long()
                    noisy_target_latent = noise_scheduler.add_noise(target_latent, noise, timesteps)

                    # channel维度拼接：(batch, 4+4, H/8, W/8) -> conv_in 8通道输入
                    unet_input = torch.cat([noisy_target_latent, condition_latent], dim=1)

                    # 文本条件 + 旋转向量条件在 token 维度拼接： (batch,77,768)+(batch,1,768) -> (batch,78,768)
                    rotation_embedding = rotation_encoder(rotation_vector)
                    encoder_hidden_states = torch.cat([text_embedding, rotation_embedding], dim=1)

                    model_pred = unet(unet_input, timesteps, encoder_hidden_states=encoder_hidden_states).sample

                    loss = F.mse_loss(model_pred.float(), noise.float(), reduction="mean")

                    accelerator.backward(loss)
                    optimizer.step()
                    optimizer.zero_grad()

                if epoch == start_epoch and step == 0:
                    _log(f"[train_lora] 第一个 step 完成，loss={loss.item():.5f}，训练流程正常运行。")

                if accelerator.sync_gradients:
                    global_step += 1
                    if global_step % 10 == 0:
                        _log(f"[train_lora] epoch={epoch} step={global_step} loss={loss.item():.5f}")
                        if accelerator.is_main_process:
                            loss_history.append({"step": global_step, "epoch": epoch, "loss": loss.item()})

                    if global_step % args.save_steps == 0:
                        save_checkpoint(accelerator, unet, rotation_encoder, optimizer,
                                         args.output_dir, global_step, epoch)
                        _save_loss_history(args.output_dir, loss_history)
    except Exception:
        _log("[train_lora] 训练循环发生异常，完整 traceback：")
        traceback.print_exc(file=sys.stdout)
        sys.stdout.flush()
        raise

    save_checkpoint(accelerator, unet, rotation_encoder, optimizer,
                     args.output_dir, global_step, args.num_epochs - 1, final=True)
    _save_loss_history(args.output_dir, loss_history)
    _log("[train_lora] 训练完成。")


def _save_loss_history(output_dir, loss_history):
    """将 loss 曲线落盘为 JSON，供 verify_lora_convergence.py 做收敛趋势校验。"""
    if not loss_history:
        return
    os.makedirs(output_dir, exist_ok=True)
    with open(os.path.join(output_dir, "loss_history.json"), "w", encoding="utf-8") as f:
        json.dump(loss_history, f, ensure_ascii=False, indent=2)


def save_checkpoint(accelerator, unet, rotation_encoder, optimizer, output_dir, step, epoch, final=False):
    if not accelerator.is_main_process:
        return
    tag = "final" if final else f"step_{step}"
    save_dir = os.path.join(output_dir, tag)
    os.makedirs(save_dir, exist_ok=True)

    unwrapped_unet = accelerator.unwrap_model(unet)
    # peft 的 save_pretrained 只保存 LoRA adapter 权重（几MB），
    # 但 conv_in 被 expand_unet_conv_in 手动扩展为8通道后，其权重属于 base model
    # 的一部分（不在 LoRA adapter 范围内），且新增的4个通道会随训练更新，
    # 因此必须单独保存 conv_in 权重，否则推理时重新扩展 conv_in 只能拿到
    # 零初始化的后4通道，丢失训练成果。
    unwrapped_unet.save_pretrained(os.path.join(save_dir, "unet_lora"))
    conv_in_state = unwrapped_unet.get_base_model().conv_in.state_dict()
    torch.save(conv_in_state, os.path.join(save_dir, "conv_in.pt"))

    unwrapped_rotation_encoder = accelerator.unwrap_model(rotation_encoder)
    unwrapped_rotation_encoder.save_pretrained(os.path.join(save_dir, "rotation_encoder.pt"))

    # 断点续训所需：优化器状态（动量等）与当前训练进度（epoch / global_step）。
    # accelerator.prepare() 返回的 optimizer 是 AcceleratedOptimizer 包装，
    # state_dict()/load_state_dict() 已经透传到底层 torch optimizer，无需 unwrap。
    torch.save(optimizer.state_dict(), os.path.join(save_dir, "optimizer.pt"))
    with open(os.path.join(save_dir, "training_state.json"), "w", encoding="utf-8") as f:
        json.dump({"epoch": epoch, "global_step": step}, f, ensure_ascii=False, indent=2)

    _log(f"[train_lora] 已保存 checkpoint 到：{save_dir}（含 LoRA adapter + 扩展后的 conv_in 权重 + RotationEncoder）")


if __name__ == "__main__":
    main()
