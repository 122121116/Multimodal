# -*- coding: utf-8 -*-
"""
split_dataset.py
train/val 数据集拆分脚本（纯 Python，不依赖 bpy，仅依赖标准库）。

用法：
    python split_dataset.py --dataset_dir E:/Multimodal/dataset_output \
        --train_ratio 0.9 --seed 42 --split_by scene

职责：
    1. 读取 dataset_manifest.json 获取所有场景列表。
    2. 支持两种拆分粒度：
       - scene（默认，推荐）：按场景整体划分，同一场景的所有视角（含参考视角）
         不会同时出现在 train/val 中，避免数据泄漏。
       - sample：按单张图片样本随机划分，不考虑场景边界。
    3. 按 train_ratio 划分，固定 seed 保证可复现。
    4. 输出 train_manifest.json / val_manifest.json，每项内联展开完整位姿标注字段，
       训练脚本无需再关联查询 poses.json。
    5. 打印拆分摘要。
"""

import argparse
import json
import os
import random


def load_all_samples(dataset_dir, scenes):
    """遍历所有场景的 poses.json，加载全部样本标注，并按 scene_id 分组返回。

    返回：dict[scene_id] -> list[pose_record]（pose_record 即 poses.json 中的原始字段）。
    """
    samples_by_scene = {}
    for scene_record in scenes:
        scene_id = scene_record["scene_id"]
        scene_dir_name = scene_record["scene_dir"]
        poses_json_path = os.path.join(dataset_dir, scene_dir_name, "poses.json")

        if not os.path.isfile(poses_json_path):
            print("警告：场景 {} 缺少 poses.json，已跳过：{}".format(scene_id, poses_json_path))
            samples_by_scene[scene_id] = []
            continue

        with open(poses_json_path, "r", encoding="utf-8") as f:
            pose_list = json.load(f)
        samples_by_scene[scene_id] = pose_list

    return samples_by_scene


def split_by_scene(scenes, samples_by_scene, train_ratio, seed):
    """按场景整体划分：同一场景的所有视角只会出现在 train 或 val 其中一侧。"""
    scene_ids = [s["scene_id"] for s in scenes]
    rng = random.Random(seed)
    shuffled_scene_ids = list(scene_ids)
    rng.shuffle(shuffled_scene_ids)

    num_train_scenes = round(len(shuffled_scene_ids) * train_ratio)
    train_scene_ids = set(shuffled_scene_ids[:num_train_scenes])
    val_scene_ids = set(shuffled_scene_ids[num_train_scenes:])

    train_samples = []
    val_samples = []
    for scene_id in scene_ids:
        target_list = train_samples if scene_id in train_scene_ids else val_samples
        target_list.extend(samples_by_scene.get(scene_id, []))

    return train_samples, val_samples, len(train_scene_ids), len(val_scene_ids)


def split_by_sample(scenes, samples_by_scene, train_ratio, seed):
    """按单张样本随机划分：不考虑场景边界，直接对全部样本随机打乱后切分。"""
    all_samples = []
    for scene_record in scenes:
        all_samples.extend(samples_by_scene.get(scene_record["scene_id"], []))

    rng = random.Random(seed)
    shuffled = list(all_samples)
    rng.shuffle(shuffled)

    num_train = round(len(shuffled) * train_ratio)
    train_samples = shuffled[:num_train]
    val_samples = shuffled[num_train:]
    return train_samples, val_samples


def build_manifest_entries(pose_records):
    """将 poses.json 中的原始字段直接内联展开为拆分清单条目，
    确保训练脚本无需再关联 poses.json 即可拿到完整位姿标注。
    """
    entries = []
    for record in pose_records:
        entry = dict(record)  # 直接内联展开所有原始字段（image_path/scene_id/view_id/位姿等）
        entries.append(entry)
    return entries


def run_split(args):
    manifest_path = os.path.join(args.dataset_dir, "dataset_manifest.json")
    with open(manifest_path, "r", encoding="utf-8") as f:
        manifest = json.load(f)

    scenes = manifest.get("scenes", [])
    samples_by_scene = load_all_samples(args.dataset_dir, scenes)

    if args.split_by == "scene":
        train_samples, val_samples, num_train_scenes, num_val_scenes = split_by_scene(
            scenes, samples_by_scene, args.train_ratio, args.seed
        )
    else:
        train_samples, val_samples = split_by_sample(
            scenes, samples_by_scene, args.train_ratio, args.seed
        )
        num_train_scenes = None
        num_val_scenes = None

    train_entries = build_manifest_entries(train_samples)
    val_entries = build_manifest_entries(val_samples)

    train_manifest_path = os.path.join(args.dataset_dir, "train_manifest.json")
    val_manifest_path = os.path.join(args.dataset_dir, "val_manifest.json")

    with open(train_manifest_path, "w", encoding="utf-8") as f:
        json.dump(train_entries, f, ensure_ascii=False, indent=2)
    with open(val_manifest_path, "w", encoding="utf-8") as f:
        json.dump(val_entries, f, ensure_ascii=False, indent=2)

    return {
        "train_manifest_path": train_manifest_path,
        "val_manifest_path": val_manifest_path,
        "num_train_samples": len(train_entries),
        "num_val_samples": len(val_entries),
        "num_train_scenes": num_train_scenes,
        "num_val_scenes": num_val_scenes,
    }


def print_summary(args, result):
    total_samples = result["num_train_samples"] + result["num_val_samples"]
    actual_ratio = (result["num_train_samples"] / total_samples) if total_samples else 0.0

    print("=" * 60)
    print("数据集拆分摘要（split_by={}）".format(args.split_by))
    if args.split_by == "scene":
        print("train 场景数：{}".format(result["num_train_scenes"]))
        print("val   场景数：{}".format(result["num_val_scenes"]))
    print("train 样本数：{}".format(result["num_train_samples"]))
    print("val   样本数：{}".format(result["num_val_samples"]))
    print("目标 train_ratio：{:.4f}，实际比例：{:.4f}（约 {:.1f} : {:.1f}）".format(
        args.train_ratio, actual_ratio, actual_ratio * 10, (1 - actual_ratio) * 10
    ))
    print("train_manifest：{}".format(result["train_manifest_path"]))
    print("val_manifest：{}".format(result["val_manifest_path"]))
    print("=" * 60)


def parse_args():
    parser = argparse.ArgumentParser(description="将 render_pipeline.py 生成的数据集拆分为 train/val 两个清单")
    parser.add_argument("--dataset_dir", type=str, required=True, help="数据集根目录（包含 dataset_manifest.json）")
    parser.add_argument("--train_ratio", type=float, default=0.9, help="训练集比例，默认 0.9（即 9:1）")
    parser.add_argument("--seed", type=int, default=42, help="随机种子，保证拆分结果可复现，默认 42")
    parser.add_argument("--split_by", type=str, default="scene", choices=["scene", "sample"],
                         help="拆分粒度：scene（默认，按场景整体划分，避免同场景视角跨集泄漏）或 sample（按单张样本随机划分）")
    return parser.parse_args()


def main():
    args = parse_args()
    result = run_split(args)
    print_summary(args, result)


if __name__ == "__main__":
    main()
