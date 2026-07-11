# -*- coding: utf-8 -*-
"""
同场景相邻视角配对数据集。

设计说明（相邻帧训练版本）：
- 训练目标从"任意两视角间的旋转变换"收窄为"相邻视角间的小角度旋转变换"，
  以降低单步训练任务难度、提升 loss 收敛稳定性。任意大角度的视角变换，
  改由推理阶段将目标旋转分解为多个小步长、迭代调用模型多步生成来实现
  （参见 inference_app.py 的多步迭代推理逻辑），不再要求单次前向直接
  学会覆盖 [0°, 180°] 的全部旋转幅度。
- "相邻"的定义：仅同一 elevation 行内、azimuth 相邻的视角对（不跨行）。
  这与 camera_trajectory.generate_camera_poses 的采样方式对应：视角以
  "行(elevation) x 列(azimuth)"网格方式生成，同一行内的 view_id 在
  azimuth 上是连续递增的，因此同一行内 azimuth 相邻的两个视角在空间角度
  上也是相邻的（小角度旋转）。参考视角 "ref"（固定正对物体，不属于任何
  网格行）不参与相邻配对。
  每一行按 azimuth 升序排列后，取所有相邻的 (i, i+1) 组合；若该行视角
  覆盖了完整 360°（首尾 azimuth 间隔与行内平均间隔接近），额外补上首尾
  循环相邻的组合（最后一个视角 -> 第一个视角）。
- 每个 (view_a, view_b) 相邻对会生成两个方向的训练样本（a->b 和 b->a），
  使模型同时学到正向与反向的小角度旋转，不引入方向偏置。
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
import os
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


def _build_adjacent_pairs(views):
    """为单个场景构建"同行(elevation)相邻"的 (view_a, view_b) 组合列表。

    步骤：
        1) 剔除参考视角 "ref"（固定正对物体，不属于网格行，不参与相邻配对）。
        2) 按 elevation_deg 分组（同一采样行 elevation 理论上完全相同，
           因为 camera_trajectory 生成同一行时使用同一个 elevation 值）。
        3) 组内按 azimuth_deg 升序排序，取所有相邻 (i, i+1) 对；
           若该行整体覆盖接近完整 360°（首尾 azimuth 间隔与行内平均间隔
           量级相近），额外补上首尾循环相邻的组合。
        4) 每个相邻对生成 (view_a, view_b) 与 (view_b, view_a) 两个方向。

    返回：list[(view_a_record, view_b_record)]。
    """
    grid_views = [v for v in views if v["view_id"] != "ref"]
    if len(grid_views) < 2:
        return []

    # 按 elevation 分组（同一行内 elevation 理论上相同，用四舍五入做容差分组）
    rows = {}
    for v in grid_views:
        key = round(v["elevation_deg"], 3)
        rows.setdefault(key, []).append(v)

    pairs = []
    for row_views in rows.values():
        if len(row_views) < 2:
            continue
        row_views = sorted(row_views, key=lambda v: v["azimuth_deg"])
        n = len(row_views)

        # 行内相邻 azimuth 间隔的平均值，用于判断首尾是否也应视为相邻
        gaps = [
            (row_views[(i + 1) % n]["azimuth_deg"] - row_views[i]["azimuth_deg"]) % 360.0
            for i in range(n)
        ]
        avg_gap = sum(gaps[:-1]) / (n - 1) if n > 1 else 0.0

        for i in range(n - 1):
            pairs.append((row_views[i], row_views[i + 1]))

        # 首尾循环相邻：仅当首尾间隔与行内平均间隔量级相近（即该行采样覆盖了
        # 完整一圈），才补上循环相邻对，避免把跨越大半圈的首尾误判为相邻。
        wrap_gap = gaps[-1]
        if n > 2 and wrap_gap <= avg_gap * 1.5:
            pairs.append((row_views[-1], row_views[0]))

    # 每个相邻对生成正反两个方向的样本
    bidirectional_pairs = []
    for view_a, view_b in pairs:
        bidirectional_pairs.append((view_a, view_b))
        bidirectional_pairs.append((view_b, view_a))
    return bidirectional_pairs


class AdjacentPairDataset(Dataset):
    """相邻视角配对数据集：每个样本是同一场景内、同一 elevation 行中
    azimuth 相邻的两个视角，用于训练小角度旋转变换（详见模块顶部说明）。
    """

    def __init__(self, dataset_root, manifest_path=None, condition_type="canny",
                 resolution=512, seed=42):
        assert condition_type in ("canny", "depth"), \
            f"condition_type 必须是 'canny' 或 'depth'，实际收到：{condition_type}"
        self.dataset_root = os.path.abspath(dataset_root)
        self.condition_type = condition_type
        self.resolution = resolution
        # canny -> edge_path（轮廓图），depth -> depth_path（16bit深度图）
        self._condition_field = "edge_path" if condition_type == "canny" else "depth_path"

        # self.scenes: list[dict]，每个元素为一个场景的信息：
        #   {"views": [record, ...], "object_prompt": str}
        self.scenes = self._build_scenes(manifest_path)
        if len(self.scenes) == 0:
            raise RuntimeError(f"未能在 {self.dataset_root} 下找到任何场景，请检查数据集路径。")
        self._validate_condition_field()

        # 为每个场景构建相邻视角组合，并展开为全局可索引的 (scene, view_a, view_b) 列表
        self._all_pairs = self._build_all_pairs()
        if len(self._all_pairs) == 0:
            raise RuntimeError(
                "未能构建出任何相邻视角组合，请检查各场景的视角数量是否 >= 2 "
                "且 elevation_deg 分组是否正常。"
            )

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
            print(f"[AdjacentPairDataset] 使用 manifest 文件：{manifest_path}")
            with open(manifest_path, "r", encoding="utf-8") as f:
                records = json.load(f)
            _add_records(records)
        else:
            print(f"[AdjacentPairDataset] 未提供有效 manifest_path，"
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
                # 单视角场景无法组成配对，跳过
                continue
            object_source = views[0].get("object_source", "builtin:cube")
            scenes.append({
                "scene_id": scene_id,
                "views": views,
                "object_prompt": _object_source_to_prompt(object_source),
            })
            total_views += len(views)

        print(f"[AdjacentPairDataset] 共汇总 {len(scenes)} 个可用场景，{total_views} 个视角。")
        return scenes

    def _build_all_pairs(self):
        """为每个场景构建"同行相邻"视角组合（含正反两个方向），
        并展开为全局列表，供 __getitem__ 按索引直接取用。
        """
        all_pairs = []
        for scene in self.scenes:
            scene_pairs = _build_adjacent_pairs(scene["views"])
            for view_a, view_b in scene_pairs:
                all_pairs.append((scene, view_a, view_b))

        print(f"[AdjacentPairDataset] 已为每个场景构建同行(elevation)相邻视角组合"
              f"（含正反两个方向），共 {len(all_pairs)} 个训练样本。")
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
        """计算 pose 相对 ref_pose 的旋转向量 [delta_azimuth, delta_elevation, delta_roll]（度）。

        delta_roll 固定为 0：当前渲染管线的相机采用纯 look-at 约束
        （camera_trajectory._look_at_euler 用 direction.to_track_quat("-Z", "Y")
        计算朝向），相机没有独立可控的 roll 自由度，poses.json 中记录的
        euler_xyz_deg[2] 只是该朝向下四元数转欧拉角的一个数值解，在某些
        朝向（尤其接近上方向奇异点）附近会出现跳变（万向锁），并非真实的
        相机滚转量。若直接用其差值作为训练标签，会引入与实际旋转无关的
        噪声，因此这里不使用 euler_xyz_deg 计算 delta_roll，训练与推理均
        只使用 delta_azimuth / delta_elevation 两个自由度。
        """
        delta_azimuth = pose["azimuth_deg"] - ref_pose["azimuth_deg"]
        delta_azimuth = (delta_azimuth + 180.0) % 360.0 - 180.0
        delta_elevation = pose["elevation_deg"] - ref_pose["elevation_deg"]
        return [delta_azimuth, delta_elevation, 0.0]

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
