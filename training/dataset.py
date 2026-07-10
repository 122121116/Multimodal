# -*- coding: utf-8 -*-
"""
同场景双视角配对数据集。

设计说明（泛化能力增强版本）：
- 每个训练样本不再固定"参考视角(ref)-> 目标视角"的搭配。为兼顾"覆盖所有视角"
  与"配对距离分布均衡"两个目标，不采用完全随机采样，而是在数据集构建时为
  每个场景预生成固定数量（默认 1000）的 (view_a, view_b) 组合列表：
    1) 覆盖阶段：保证场景内每个视角至少出现在一个组合中，不遗漏任何样本；
    2) 分层补齐阶段：按两视角间的球面角距离（基于 azimuth/elevation 现算，
       与旋转向量的度量口径一致）分桶（近/中近/中远/远，默认4档），
       将剩余组合数尽量均匀分配到各距离档位，使得训练组合里"距离近的"
       （如相邻视角）与"距离远的"（如接近正对面的视角）比例相近，让模型
       同时学到细粒度微调和大幅度旋转的映射关系。
  训练时按预生成组合列表顺序索引取样（仍配合 DataLoader 的 shuffle=True
  打乱组合顺序），不再是每次 __getitem__ 都临时随机挑两张图。
- 深度图 / 轮廓图不再于训练阶段实时计算（原 cv2.Canny / DepthAnything V2
  在线推理开销很大），而是直接读取渲染完成后由 export_conditions.py 独立
  后处理生成的：
    - depth_path：16bit 单通道深度图 PNG（DepthAnything V2 对 RGB 图推理生成）
    - edge_path： 8bit  单通道轮廓图 PNG（对同一张 RGB 原图做 cv2.Canny 边缘检测）
  两者均基于同一张已落盘的 RGB 图生成，分辨率一致、像素坐标严格对齐
  （Blender 渲染阶段只产出 RGB 图，不再涉及 Compositor 深度导出）。
- condition_type="canny" 对应读取 edge_path（轮廓图，替代原实时 Canny 边缘图），
  condition_type="depth" 对应读取 depth_path（16bit 深度图），两者分别绑定到
  各自独立训练的 LoRA 输入分支，不会互相混用。
- 若某场景的 poses.json 是旧版本（不含 depth_path/edge_path 字段，即数据是在
  本次改造前生成的），会在首次访问时报错提示，避免静默使用错误/缺失的条件图。
- 每个样本额外返回 object_prompt：从该场景的 object_source 字段解析出的物品
  名称文本（如 "CoffeeTable_01" -> "coffee table 01"），用于替代训练时空的
  CLIP 文本编码，让文本条件携带物品语义信息。
"""
import json
import math
import os
import random
import re

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset


def _load_and_resize_rgb(image_path: str, resolution: int) -> Image.Image:
    img = Image.open(image_path).convert("RGB")
    if img.size != (resolution, resolution):
        img = img.resize((resolution, resolution), Image.BICUBIC)
    return img


def _to_normalized_tensor(rgb_uint8: np.ndarray) -> torch.Tensor:
    """HWC uint8 [0,255] -> CHW float32 [-1, 1]，与 SD VAE 输入范围对齐。"""
    tensor = torch.from_numpy(rgb_uint8).float() / 127.5 - 1.0
    return tensor.permute(2, 0, 1).contiguous()


def _load_condition_image(image_path: str, resolution: int) -> torch.Tensor:
    """加载导出的单通道条件图（深度图或轮廓图），统一转换为 3 通道张量。

    - depth_path 为 16bit PNG（mode "I;16"），edge_path 为 8bit PNG（mode "L"）；
      这里统一按各自原始位深读入并归一化到 [0,255] uint8 值域后复制为 3 通道，
      与 VAE 3 通道输入约定保持一致。
    """
    img = Image.open(image_path)
    if img.mode == "I;16" or img.mode == "I":
        arr16 = np.array(img).astype(np.float32)
        if arr16.max() > 0:
            arr8 = (arr16 / 65535.0 * 255.0).astype(np.uint8)
        else:
            arr8 = arr16.astype(np.uint8)
    else:
        arr8 = np.array(img.convert("L"))

    if arr8.shape != (resolution, resolution):
        arr8 = np.array(
            Image.fromarray(arr8).resize((resolution, resolution), Image.BICUBIC)
        )

    arr_3ch = np.stack([arr8, arr8, arr8], axis=-1)
    return _to_normalized_tensor(arr_3ch)


def _object_source_to_prompt(object_source: str) -> str:
    """将 object_source 字段（外部模型文件路径或 "builtin:<primitive>"）转换为
    人类可读的物品名称文本，作为 CLIP 文本编码器的输入 prompt。

    示例：
        "e:\\Multimodal\\assets\\models\\CoffeeTable_01\\CoffeeTable_01.gltf" -> "coffee table"
        "builtin:sphere" -> "sphere"
    """
    if object_source.startswith("builtin:"):
        return object_source.split(":", 1)[1].replace("_", " ").strip().lower()

    stem = os.path.splitext(os.path.basename(object_source))[0]
    # 驼峰命名拆分：CoffeeTable -> Coffee Table
    stem = re.sub(r"(?<=[a-z0-9])(?=[A-Z])", " ", stem)
    # 下划线/连字符统一转空格
    stem = re.sub(r"[_\-]+", " ", stem)
    # 去除末尾无意义的数字编号（如 CoffeeTable_01 -> CoffeeTable, model_2 -> model）
    stem = re.sub(r"\s\d+$", "", stem)
    # 合并多余空格
    stem = re.sub(r"\s+", " ", stem).strip().lower()
    return stem if stem else "object"


def _angular_distance_deg(view_a, view_b):
    """计算两个视角之间的球面角距离（度），仅基于 azimuth/elevation
    （不含 roll，roll 只由相机 look-at 约束决定，不反映视角在球面上的
    远近关系），用球面余弦公式计算大圆角距离，范围 [0, 180]。
    """
    az_a, el_a = math.radians(view_a["azimuth_deg"]), math.radians(view_a["elevation_deg"])
    az_b, el_b = math.radians(view_b["azimuth_deg"]), math.radians(view_b["elevation_deg"])
    cos_d = (
        math.sin(el_a) * math.sin(el_b)
        + math.cos(el_a) * math.cos(el_b) * math.cos(az_a - az_b)
    )
    cos_d = max(-1.0, min(1.0, cos_d))
    return math.degrees(math.acos(cos_d))


def _build_stratified_pairs(views, num_pairs, rng, num_distance_bins=4):
    """为单个场景预生成固定数量的 (view_a, view_b) 组合，兼顾覆盖性与距离分布均衡。

    步骤：
        1) 覆盖阶段：将全部视角随机打乱后两两相邻配对（首尾循环相接），
           保证每个视角至少出现在一个组合中。
        2) 分层补齐阶段：计算所有可能组合的球面角距离，按距离切分为
           num_distance_bins 个桶（近->远），在剩余额度内尽量均匀地从
           各个桶中补充采样，使最终组合集里"距离近"与"距离远"的配对
           比例相近，而不是被完全随机采样主导（完全随机采样会因为球面上
           中等距离的点对数量远多于远距离点对，导致远距离配对被稀释）。

    返回：长度为 num_pairs 的 (view_a_record, view_b_record) 列表。
    """
    n = len(views)
    pairs = []
    seen_pairs = set()

    def _add_pair(i, j):
        key = (i, j) if i < j else (j, i)
        if key in seen_pairs:
            return False
        seen_pairs.add(key)
        pairs.append((views[i], views[j]))
        return True

    # ---- 阶段1：覆盖阶段，保证每个视角至少出现一次 ----
    order = list(range(n))
    rng.shuffle(order)
    for k in range(n):
        i, j = order[k], order[(k + 1) % n]
        if i != j:
            _add_pair(i, j)

    # ---- 阶段2：按球面角距离分桶，均匀补齐到 num_pairs ----
    if len(pairs) < num_pairs:
        # 候选组合池：全部未使用过的 (i, j) 组合及其角距离
        candidates_by_bin = [[] for _ in range(num_distance_bins)]
        max_dist = 180.0
        bin_width = max_dist / num_distance_bins
        for i in range(n):
            for j in range(i + 1, n):
                key = (i, j)
                if key in seen_pairs:
                    continue
                dist = _angular_distance_deg(views[i], views[j])
                bin_idx = min(int(dist / bin_width), num_distance_bins - 1)
                candidates_by_bin[bin_idx].append((i, j))
        for bucket in candidates_by_bin:
            rng.shuffle(bucket)

        remaining = num_pairs - len(pairs)
        # 轮询各桶，每轮从每个非空桶取一个，直到补满或所有桶耗尽
        bucket_cursors = [0] * num_distance_bins
        while remaining > 0:
            progressed = False
            for b in range(num_distance_bins):
                if remaining <= 0:
                    break
                cursor = bucket_cursors[b]
                bucket = candidates_by_bin[b]
                if cursor >= len(bucket):
                    continue
                i, j = bucket[cursor]
                bucket_cursors[b] += 1
                if _add_pair(i, j):
                    remaining -= 1
                    progressed = True
            if not progressed:
                # 所有桶的候选组合都已用尽（场景视角数过少，不放回组合已耗尽），
                # 允许重复采样已有组合以补满数量，保证 __len__ 恒等于 num_pairs。
                i, j = rng.sample(range(n), 2)
                pairs.append((views[i], views[j]))
                remaining -= 1

    # 若视角数很多、覆盖阶段已经超过 num_pairs，做截断（罕见，仅当 n > num_pairs 时可能发生）
    if len(pairs) > num_pairs:
        rng.shuffle(pairs)
        pairs = pairs[:num_pairs]

    return pairs


class RotationPairDataset(Dataset):
    def __init__(self, dataset_root, manifest_path=None, condition_type="canny",
                 resolution=512, seed=42, pairs_per_scene=1000, num_distance_bins=4):
        assert condition_type in ("canny", "depth"), \
            f"condition_type 必须是 'canny' 或 'depth'，实际收到：{condition_type}"
        self.dataset_root = os.path.abspath(dataset_root)
        self.condition_type = condition_type
        self.resolution = resolution
        # canny -> edge_path（轮廓图），depth -> depth_path（16bit深度图）
        self._condition_field = "edge_path" if condition_type == "canny" else "depth_path"
        self._rng = random.Random(seed)
        self.pairs_per_scene = pairs_per_scene
        self.num_distance_bins = num_distance_bins

        # self.scenes: list[dict]，每个元素为一个场景的信息：
        #   {"views": [record, ...], "object_prompt": str}
        self.scenes = self._build_scenes(manifest_path)
        if len(self.scenes) == 0:
            raise RuntimeError(f"未能在 {self.dataset_root} 下找到任何场景，请检查数据集路径。")
        self._validate_condition_field()

        # 预生成每个场景的固定组合列表，并展开为全局可索引的 (scene_idx, view_a, view_b) 列表
        self._all_pairs = self._build_all_pairs()

    # ------------------------------------------------------------------
    # 样本清单构建：按场景聚合所有视角（含 ref），而不是按目标图展开
    # ------------------------------------------------------------------
    def _build_scenes(self, manifest_path):
        scene_records = {}

        def _add_records(records):
            for r in records:
                scene_id = r["scene_id"]
                scene_records.setdefault(scene_id, []).append(r)

        if manifest_path is not None and os.path.isfile(manifest_path):
            print(f"[RotationPairDataset] 使用 manifest 文件：{manifest_path}")
            with open(manifest_path, "r", encoding="utf-8") as f:
                records = json.load(f)
            _add_records(records)
        else:
            print(f"[RotationPairDataset] 未提供有效 manifest_path，"
                  f"尝试通过 dataset_manifest.json / 目录扫描汇总所有场景样本 ...")

            dataset_manifest_path = os.path.join(self.dataset_root, "dataset_manifest.json")
            scene_dirs = []
            if os.path.isfile(dataset_manifest_path):
                with open(dataset_manifest_path, "r", encoding="utf-8") as f:
                    manifest = json.load(f)
                if isinstance(manifest, dict) and "scenes" in manifest:
                    scene_dirs = [s if isinstance(s, str) else s.get("scene_id") for s in manifest["scenes"]]
                    scene_dirs = [f"scene_{int(s):04d}" if not str(s).startswith("scene_") else s for s in scene_dirs]
                elif isinstance(manifest, list):
                    scene_dirs = [f"scene_{int(s):04d}" if not str(s).startswith("scene_") else s for s in manifest]

            if not scene_dirs:
                scene_dirs = sorted(
                    d for d in os.listdir(self.dataset_root)
                    if d.startswith("scene_") and os.path.isdir(os.path.join(self.dataset_root, d))
                )

            for scene_dir in scene_dirs:
                poses_path = os.path.join(self.dataset_root, scene_dir, "poses.json")
                if not os.path.isfile(poses_path):
                    continue
                with open(poses_path, "r", encoding="utf-8") as f:
                    records = json.load(f)
                _add_records(records)

        scenes = []
        total_views = 0
        for scene_id in sorted(scene_records.keys()):
            views = scene_records[scene_id]
            if len(views) < 2:
                # 单视角场景无法组成"任意两张图"的配对，跳过
                continue
            object_source = views[0].get("object_source", "builtin:cube")
            scenes.append({
                "scene_id": scene_id,
                "views": views,
                "object_prompt": _object_source_to_prompt(object_source),
            })
            total_views += len(views)

        print(f"[RotationPairDataset] 共汇总 {len(scenes)} 个可用场景，{total_views} 个视角。")
        return scenes

    def _build_all_pairs(self):
        """为每个场景预生成 pairs_per_scene 个 (view_a, view_b) 组合，
        并展开为全局列表，供 __getitem__ 按索引直接取用。
        """
        all_pairs = []
        for scene in self.scenes:
            scene_pairs = _build_stratified_pairs(
                scene["views"], self.pairs_per_scene, self._rng,
                num_distance_bins=self.num_distance_bins,
            )
            for view_a, view_b in scene_pairs:
                all_pairs.append((scene, view_a, view_b))

        print(f"[RotationPairDataset] 已为每个场景预生成 {self.pairs_per_scene} 个双视角组合"
              f"（覆盖全部视角 + 按球面角距离分 {self.num_distance_bins} 档均衡采样），"
              f"共 {len(all_pairs)} 个训练样本。")
        return all_pairs

    def _validate_condition_field(self):
        """检查样本标注中是否包含 depth_path/edge_path 字段。

        若数据是在导出流程改造前生成的旧数据（只有 RGB 图，没有同步导出的
        深度图/轮廓图），这里主动报错并提示用户重新导出，而不是静默降级，
        避免训练用到不存在的文件路径或产生错位的条件图。
        """
        sample = self.scenes[0]["views"][0]
        if self._condition_field not in sample:
            raise RuntimeError(
                f"当前数据集样本标注中缺少 '{self._condition_field}' 字段，"
                f"这通常说明数据是在导出流程改造前生成的旧数据（仅含 RGB 图），"
                f"或渲染完成后尚未运行 export_conditions.py 补全深度图/轮廓图。"
                f"请先用 render_pipeline.py 渲染 RGB 图，再运行 "
                f"export_conditions.py --dataset_dir <数据集目录> 生成深度图与轮廓图。"
            )

    # ------------------------------------------------------------------
    def __len__(self):
        return len(self._all_pairs)

    def _preprocess_target(self, image_path: str) -> torch.Tensor:
        img = _load_and_resize_rgb(image_path, self.resolution)
        return _to_normalized_tensor(np.array(img))

    def _preprocess_condition(self, record) -> torch.Tensor:
        condition_rel_path = record[self._condition_field]
        condition_path = os.path.join(self.dataset_root, condition_rel_path)
        return _load_condition_image(condition_path, self.resolution)

    @staticmethod
    def _compute_relative_rotation_vector(pose, ref_pose):
        """计算 pose 相对 ref_pose 的三维旋转向量（度），公式与
        render_pipeline._compute_relative_rotation_vector 保持一致，
        但这里可以对任意两个视角调用，不局限于必须以 view_id="ref" 为基准。
        """
        delta_azimuth = pose["azimuth_deg"] - ref_pose["azimuth_deg"]
        delta_azimuth = (delta_azimuth + 180.0) % 360.0 - 180.0
        delta_elevation = pose["elevation_deg"] - ref_pose["elevation_deg"]
        delta_roll = pose["euler_xyz_deg"][2] - ref_pose["euler_xyz_deg"][2]
        delta_roll = (delta_roll + 180.0) % 360.0 - 180.0
        return [delta_azimuth, delta_elevation, delta_roll]

    def __getitem__(self, idx):
        scene, view_a, view_b = self._all_pairs[idx]

        condition_image = self._preprocess_condition(view_a)

        target_path = os.path.join(self.dataset_root, view_b["image_path"])
        target_image = self._preprocess_target(target_path)

        rotation_vector = torch.tensor(
            self._compute_relative_rotation_vector(view_b, view_a), dtype=torch.float32
        )

        return {
            "condition_image": condition_image,
            "target_image": target_image,
            "rotation_vector": rotation_vector,
            "object_prompt": scene["object_prompt"],
        }

