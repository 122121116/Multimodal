# -*- coding: utf-8 -*-
"""
verify_lora_convergence.py
LoRA 训练收敛状态验证脚本（纯 Python，不依赖 GPU/训练框架）。

用法：
    python verify_lora_convergence.py \
        --loss_history e:/Multimodal/training/output/lora_canny/loss_history.json \
        --loss_history e:/Multimodal/training/output/lora_depth/loss_history.json

职责：
    读取 train_lora.py 训练过程中落盘的 loss_history.json（每 10 step 记录一次
    loss），对每个 LoRA 分别判断收敛状态：
    1. 趋势下降：将 loss 序列均分为前/中/后三段，后段均值应显著低于前段均值
       （允许一定噪声容忍度，默认要求下降幅度 >= min_drop_ratio）。
    2. 稳定性：末尾一段（默认最后 20%）的 loss 标准差不应过大，判断是否收敛到
       稳定区间而非仍在大幅震荡。
    3. 数值有效性：不应出现 NaN/Inf（梯度爆炸的典型信号）。
    输出每个 LoRA 的判定结果与整体摘要。
"""
import argparse
import json
import math
import os


def _split_three_segments(values):
    n = len(values)
    if n < 3:
        return values, values, values
    third = max(1, n // 3)
    head = values[:third]
    mid = values[third:2 * third]
    tail = values[2 * third:]
    return head, mid, tail


def _mean(values):
    return sum(values) / len(values) if values else 0.0


def _std(values):
    if len(values) < 2:
        return 0.0
    m = _mean(values)
    variance = sum((v - m) ** 2 for v in values) / len(values)
    return math.sqrt(variance)


def check_convergence(loss_history, min_drop_ratio=0.15, tail_ratio=0.2, max_tail_std_ratio=0.5):
    """判断单个 LoRA 的收敛状态，返回 (is_converged: bool, details: dict)。"""
    losses = [record["loss"] for record in loss_history]

    if not losses:
        return False, {"error": "loss_history 为空"}

    if any(math.isnan(v) or math.isinf(v) for v in losses):
        return False, {"error": "loss 序列中出现 NaN/Inf，训练可能已发散"}

    head, _mid, tail_all = _split_three_segments(losses)
    head_mean = _mean(head)
    tail_mean = _mean(tail_all)
    drop_ratio = (head_mean - tail_mean) / head_mean if head_mean > 1e-12 else 0.0

    tail_n = max(1, int(len(losses) * tail_ratio))
    tail_window = losses[-tail_n:]
    tail_std = _std(tail_window)
    tail_window_mean = _mean(tail_window)
    tail_std_ratio = tail_std / tail_window_mean if tail_window_mean > 1e-12 else float("inf")

    trend_ok = drop_ratio >= min_drop_ratio
    stability_ok = tail_std_ratio <= max_tail_std_ratio
    is_converged = trend_ok and stability_ok

    details = {
        "num_records": len(losses),
        "head_mean_loss": head_mean,
        "tail_mean_loss": tail_mean,
        "drop_ratio": drop_ratio,
        "min_drop_ratio_required": min_drop_ratio,
        "trend_ok": trend_ok,
        "tail_window_std": tail_std,
        "tail_window_std_ratio": tail_std_ratio,
        "max_tail_std_ratio_allowed": max_tail_std_ratio,
        "stability_ok": stability_ok,
        "final_loss": losses[-1],
    }
    return is_converged, details


def parse_args():
    parser = argparse.ArgumentParser(description="LoRA 训练收敛状态验证")
    parser.add_argument("--loss_history", type=str, action="append", required=True,
                         help="loss_history.json 路径，可重复传入以同时校验多个 LoRA（如 canny 和 depth）")
    parser.add_argument("--min_drop_ratio", type=float, default=0.15,
                         help="要求 loss 从前段到后段至少下降的比例，默认 0.15（即下降>=15%%）")
    parser.add_argument("--tail_ratio", type=float, default=0.2,
                         help="用于稳定性判断的末尾窗口占比，默认 0.2（最后20%%的step）")
    parser.add_argument("--max_tail_std_ratio", type=float, default=0.5,
                         help="末尾窗口 loss 标准差/均值 的上限，超过则判定为仍在震荡未收敛，默认 0.5")
    parser.add_argument("--report_path", type=str, default=None,
                         help="校验结果 JSON 输出路径，默认不写文件，仅打印到终端")
    return parser.parse_args()


def main():
    args = parse_args()
    all_results = {}
    all_converged = True

    for path in args.loss_history:
        name = os.path.basename(os.path.dirname(path)) or path
        if not os.path.isfile(path):
            print("[verify_lora_convergence] 警告：文件不存在，跳过：{}".format(path))
            all_results[name] = {"error": "file_not_found", "path": path}
            all_converged = False
            continue

        with open(path, "r", encoding="utf-8") as f:
            loss_history = json.load(f)

        is_converged, details = check_convergence(
            loss_history,
            min_drop_ratio=args.min_drop_ratio,
            tail_ratio=args.tail_ratio,
            max_tail_std_ratio=args.max_tail_std_ratio,
        )
        details["path"] = path
        details["is_converged"] = is_converged
        all_results[name] = details
        all_converged = all_converged and is_converged

        print("=" * 60)
        print("LoRA: {}".format(name))
        print("  记录数: {}".format(details.get("num_records")))
        if "error" in details:
            print("  错误: {}".format(details["error"]))
        else:
            print("  前段均值loss: {:.5f}  ->  末段均值loss: {:.5f}  (下降 {:.1%})".format(
                details["head_mean_loss"], details["tail_mean_loss"], details["drop_ratio"]))
            print("  末尾窗口波动比: {:.3f}（阈值 <= {}）".format(
                details["tail_window_std_ratio"], details["max_tail_std_ratio_allowed"]))
            print("  最终loss: {:.5f}".format(details["final_loss"]))
        print("  收敛判定: {}".format("通过" if is_converged else "未通过"))

    print("=" * 60)
    print("整体收敛验证结果: {}".format("全部通过" if all_converged else "存在未通过项"))

    if args.report_path:
        with open(args.report_path, "w", encoding="utf-8") as f:
            json.dump({"results": all_results, "all_converged": all_converged}, f, ensure_ascii=False, indent=2)
        print("报告已写出：{}".format(args.report_path))


if __name__ == "__main__":
    main()
