# -*- coding: utf-8 -*-
"""
render_pipeline.py
批量渲染 pipeline 主入口脚本。

用法（在装有 Blender 的机器上）：
    blender --background --python render_pipeline.py -- \
        --output_dir E:/Multimodal/dataset_output \
        --num_scenes 5 \
        --views_per_scene 30 \
        --resolution 512 \
        --engine EEVEE_NEXT \
        --object_dir E:/Multimodal/assets/models \
        --seed 42

职责：
    - 解析命令行参数（`--` 之后的部分为脚本自身参数，Blender 会忽略之前的参数）
    - 对每个场景：调用 scene_builder.build_scene 搭建场景
      -> camera_trajectory.generate_camera_poses 生成位姿
      -> 逐视角 create_camera + 渲染 + 保存 PNG
      -> 写出该场景的 poses.json 标注文件
    - 单帧渲染异常不应导致整个 pipeline 崩溃：记录错误并跳过继续下一视角
    - 所有场景渲染完成后，生成数据集根目录下的汇总清单 dataset_manifest.json
"""

import argparse
import concurrent.futures
import datetime
import json
import os
import sys
import time

import bpy
import numpy as np

# 保证可以 import 同目录下的 scene_builder / camera_trajectory
# （Blender --python 执行时，脚本所在目录不一定在 sys.path 中）
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
if _THIS_DIR not in sys.path:
    sys.path.insert(0, _THIS_DIR)

import scene_builder
import camera_trajectory


# ----------------------------------------------------------------------------
# 关键参数集中定义，方便用户后续调整以扩产到 5000+ 样本
# ----------------------------------------------------------------------------
DEFAULT_CONFIG = {
    "output_dir": "./dataset_output",
    "num_scenes": 5,
    "views_per_scene": 30,          # 每个场景视角数（含参考视角），要求 >= 30
    "resolution": 512,              # 统一渲染分辨率 512x512
    "engine": "EEVEE_NEXT",         # EEVEE_NEXT / EEVEE / CYCLES
    "object_dir": None,             # 外部模型资产目录（.obj/.fbx/.glb），可为空
    "seed": 42,
    "radius": 5.0,                  # 相机到物体中心的距离（Blender 单位）
    "elevation_range": (-30.0, 60.0),
    "normalize_size": 2.0,          # 物体归一化最大边长，需与 scene_builder 一致
    # ---------------- 性能优化相关参数 ----------------
    "samples": 32,                  # 渲染采样数（Cycles 路径追踪 / EEVEE_NEXT 光线数），越低越快
    "use_gpu": True,                # 是否启用 GPU 加速（Cycles: CUDA/OPTIX；EEVEE 本身即为 GPU 渲染）
    "denoise": True,                # 启用降噪，允许在低采样下仍保持可用画质
    "tile_size": 256,               # 渲染分块尺寸（仅影响内存占用与调度粒度，GPU 渲染建议与分辨率相近）
    # ---------------- 多模态导出相关参数 ----------------
    # 注：深度图/轮廓图不在 Blender 渲染阶段生成（Blender 5.0 Compositor API
    # 不兼容 --background 模式），这里的 export_depth/export_edge 仅控制是否在
    # poses.json 中预写 depth_path/edge_path 字段；实际图像由渲染完成后独立
    # 运行的 export_conditions.py 生成（DepthAnything V2 + Canny）。
    "export_depth": True,           # 是否在标注中预留 depth_path 字段
    "export_edge": True,            # 是否在标注中预留 edge_path 字段
}

SUPPORTED_MODEL_EXTS = (".obj", ".fbx", ".glb", ".gltf")


# ----------------------------------------------------------------------------
# 命令行参数解析
# ----------------------------------------------------------------------------
def parse_args():
    """解析 `blender --background --python render_pipeline.py -- <args>` 中的自定义参数。"""
    if "--" in sys.argv:
        argv = sys.argv[sys.argv.index("--") + 1:]
    else:
        argv = []

    parser = argparse.ArgumentParser(description="多视角数据集批量渲染 pipeline")
    parser.add_argument("--output_dir", type=str, default=DEFAULT_CONFIG["output_dir"])
    parser.add_argument("--num_scenes", type=int, default=DEFAULT_CONFIG["num_scenes"])
    parser.add_argument("--views_per_scene", type=int, default=DEFAULT_CONFIG["views_per_scene"])
    parser.add_argument("--resolution", type=int, default=DEFAULT_CONFIG["resolution"])
    parser.add_argument("--engine", type=str, default=DEFAULT_CONFIG["engine"],
                         choices=["EEVEE_NEXT", "EEVEE", "CYCLES", "BLENDER_EEVEE", "BLENDER_EEVEE_NEXT"])
    parser.add_argument("--object_dir", type=str, default=DEFAULT_CONFIG["object_dir"])
    parser.add_argument("--seed", type=int, default=DEFAULT_CONFIG["seed"])

    # ---------------- 性能优化相关参数 ----------------
    parser.add_argument("--samples", type=int, default=DEFAULT_CONFIG["samples"],
                         help="渲染采样数，越低越快，默认 32（原默认工程模板通常为 128）")
    parser.add_argument("--use_gpu", type=int, default=int(DEFAULT_CONFIG["use_gpu"]), choices=[0, 1],
                         help="是否启用 GPU 渲染加速，1=启用（默认），0=强制 CPU")
    parser.add_argument("--denoise", type=int, default=int(DEFAULT_CONFIG["denoise"]), choices=[0, 1],
                         help="是否启用降噪以补偿低采样画质损失，1=启用（默认）")
    parser.add_argument("--tile_size", type=int, default=DEFAULT_CONFIG["tile_size"])

    # ---------------- 多模态导出相关参数 ----------------
    parser.add_argument("--export_depth", type=int, default=int(DEFAULT_CONFIG["export_depth"]), choices=[0, 1])
    parser.add_argument("--export_edge", type=int, default=int(DEFAULT_CONFIG["export_edge"]), choices=[0, 1])

    args = parser.parse_args(argv)
    return args


# ----------------------------------------------------------------------------
# 渲染引擎与分辨率设置
# ----------------------------------------------------------------------------
def _normalize_engine_name(engine):
    """兼容不同 Blender 版本的引擎枚举名。

    Blender 4.2 起旧版 "BLENDER_EEVEE" 已被 "BLENDER_EEVEE_NEXT" 取代并最终移除，
    因此 "EEVEE"/"EEVEE_NEXT"/旧版 "BLENDER_EEVEE" 一律归一化为 "BLENDER_EEVEE_NEXT"，
    避免回退到已不存在的旧引擎标识符导致渲染静默失败（不报错但不产出像素）。
    """
    mapping = {
        "EEVEE": "BLENDER_EEVEE_NEXT",
        "EEVEE_NEXT": "BLENDER_EEVEE_NEXT",
        "CYCLES": "CYCLES",
        "BLENDER_EEVEE": "BLENDER_EEVEE_NEXT",
        "BLENDER_EEVEE_NEXT": "BLENDER_EEVEE_NEXT",
    }
    return mapping.get(engine, engine)


def _enable_gpu_acceleration(scene):
    """检测并启用 GPU 渲染加速。

    - Cycles：优先尝试 OPTIX（NVIDIA RTX 系列专用，速度显著优于 CUDA），
      不可用时回退 CUDA，再不可用则保持 CPU 并打印警告。
    - EEVEE/EEVEE_NEXT：本身即通过显卡光栅化管线渲染，无需额外设备切换，
      这里仅做一次性提示。
    返回 bool，表示 GPU 是否实际启用成功。
    """
    if scene.render.engine != "CYCLES":
        # EEVEE 系列始终使用 GPU 光栅化管线渲染，无需切换 compute_device_type
        return True

    prefs = bpy.context.preferences.addons.get("cycles")
    if prefs is None:
        print("警告：未找到 cycles 插件偏好设置，无法配置 GPU 设备，将使用 CPU 渲染。")
        return False

    cprefs = prefs.preferences
    for device_type in ("OPTIX", "CUDA"):
        try:
            cprefs.compute_device_type = device_type
        except TypeError:
            continue
        cprefs.get_devices()
        gpu_devices = [d for d in cprefs.devices if d.type == device_type]
        if gpu_devices:
            for d in cprefs.devices:
                d.use = d.type in (device_type, "CPU") and (d.type == device_type)
            scene.cycles.device = "GPU"
            print("已启用 Cycles GPU 加速：{}，设备数 {}".format(device_type, len(gpu_devices)))
            return True

    print("警告：未检测到可用的 OPTIX/CUDA GPU 设备，Cycles 将回退为 CPU 渲染（速度会明显变慢）。")
    scene.cycles.device = "CPU"
    return False


def _configure_render_settings(scene, resolution, engine, config=None):
    """设置统一渲染分辨率、输出格式（PNG）与渲染引擎，并应用性能优化参数。"""
    config = config or {}
    scene.render.resolution_x = resolution
    scene.render.resolution_y = resolution
    scene.render.resolution_percentage = 100
    scene.render.image_settings.file_format = "PNG"
    scene.render.image_settings.color_mode = "RGBA"
    scene.render.film_transparent = False  # 保持世界背景色一致，便于后续对齐/抠图判断

    # Blender 4.0+/5.0 默认色调映射为 AgX（大幅压低对比度、整体发灰发暗），
    # 数据集渲染需要色彩还原准确、对比度正常的画面，因此显式切换为 Standard
    # （线性值直接映射到显示，不做额外的电影感压缩）。
    scene.view_settings.view_transform = "Standard"
    scene.view_settings.look = "None"
    scene.view_settings.exposure = 0.0
    scene.view_settings.gamma = 1.0

    target_engine = _normalize_engine_name(engine)
    try:
        scene.render.engine = target_engine
    except TypeError:
        # 当前 Blender 版本不支持该引擎标识符，回退到 Cycles（历史版本兼容性最好）
        print("警告：渲染引擎 {} 在当前 Blender 版本不可用，回退为 CYCLES".format(target_engine))
        scene.render.engine = "CYCLES"

    # 直接读取赋值后的实际值做二次确认，避免 Blender 静默忽略非法枚举值
    if scene.render.engine != target_engine and target_engine != "CYCLES":
        print("警告：渲染引擎设置未生效（当前为 {}），请检查 Blender 版本兼容性".format(scene.render.engine))

    # ---------------- 性能优化参数 ----------------
    samples = config.get("samples", 32)
    use_gpu = config.get("use_gpu", True)
    denoise = config.get("denoise", True)
    tile_size = config.get("tile_size", 256)

    if use_gpu:
        _enable_gpu_acceleration(scene)

    if scene.render.engine == "CYCLES":
        scene.cycles.samples = samples
        scene.cycles.use_denoising = denoise
        # Blender 4.x/5.x 已移除手动 tile_x/tile_y（改为自动分块调度），
        # 仅在旧版本仍暴露该属性时才显式设置，新版本静默跳过即可。
        if hasattr(scene.render, "tile_x"):
            scene.render.tile_x = tile_size
            scene.render.tile_y = tile_size
        # Adaptive sampling 进一步减少无效采样，加速收敛
        if hasattr(scene.cycles, "use_adaptive_sampling"):
            scene.cycles.use_adaptive_sampling = True
    else:
        # EEVEE_NEXT：采样数对应光线数/阴影质量，同样可下调以提速
        if hasattr(scene.eevee, "taa_render_samples"):
            scene.eevee.taa_render_samples = samples
        if hasattr(scene.eevee, "use_gtao"):
            scene.eevee.use_gtao = True  # 保留环境光遮蔽，兼顾速度与轮廓/深度可用性

    # 关闭运动模糊、景深等与本任务无关且拖慢渲染的效果（若已默认关闭则无副作用）
    scene.render.use_motion_blur = False


# ----------------------------------------------------------------------------
# 外部模型资产扫描
# ----------------------------------------------------------------------------
def _scan_object_files(object_dir):
    """递归遍历 object_dir 下的 .obj/.fbx/.glb/.gltf 文件，用作各场景的物体来源。

    使用递归扫描（而非仅顶层）是因为下载的 Poly Haven 等资产通常按
    "每个资产一个独立子目录"存放（子目录内还有 textures/ 和 .bin 等引用文件，
    必须保持相对目录结构不能打散），因此模型文件本身可能位于 object_dir 的
    任意子目录下。

    若 object_dir 为空或目录不存在或没有匹配文件，返回空列表，
    由调用方降级为内置几何体轮换。
    """
    if not object_dir or not os.path.isdir(object_dir):
        return []

    model_files = []
    for root, _dirs, files in os.walk(object_dir):
        for fname in sorted(files):
            if fname.lower().endswith(SUPPORTED_MODEL_EXTS):
                model_files.append(os.path.join(root, fname))
    return sorted(model_files)


# ----------------------------------------------------------------------------
# 相对旋转向量计算
# ----------------------------------------------------------------------------
def _compute_relative_rotation_vector(pose, ref_pose):
    """计算某视角相对参考视角的三维旋转向量（度）：
    (delta_azimuth, delta_elevation, delta_roll)。

    这是"相机以参考图像为中心进行旋转的三维向量提示词"所需的核心标注字段，
    用于扩散模型训练时的相对相机位姿条件输入。
    roll 分量对于始终 look-at 物体中心、Y 轴朝上的相机而言恒为 0，
    这里仍显式计算并保留字段，便于未来扩展非 look-at 轨迹（如自由 roll）。
    """
    delta_azimuth = pose["azimuth_deg"] - ref_pose["azimuth_deg"]
    # 归一化到 [-180, 180) 区间，避免出现如 359 度这种不直观的差值
    delta_azimuth = (delta_azimuth + 180.0) % 360.0 - 180.0
    delta_elevation = pose["elevation_deg"] - ref_pose["elevation_deg"]
    delta_roll = pose["euler_xyz_deg"][2] - ref_pose["euler_xyz_deg"][2]
    delta_roll = (delta_roll + 180.0) % 360.0 - 180.0
    return [delta_azimuth, delta_elevation, delta_roll]


def render_scene(scene_id, object_source_path, config, output_dir):
    """搭建并渲染一个场景的所有视角，返回该场景的位姿标注列表与统计信息。

    本函数只负责渲染并落盘 RGB 图（保持简单可靠，不依赖易变的 Blender
    Compositor API）。深度图（16bit PNG）与轮廓图（8bit PNG）由渲染完成后
    独立运行的 export_conditions.py 脚本批量生成（深度图用 DepthAnything V2
    模型推理，轮廓图对 RGB 图做 Canny 边缘检测），二者分辨率与像素坐标
    自然与 RGB 图完全一致（后处理阶段直接基于已保存的 RGB 图计算，无需
    额外的渲染期配准）。
    """
    scene_config = {
        "normalize_size": config["normalize_size"],
    }
    if object_source_path:
        scene_config["object_path"] = object_source_path
    else:
        # 无外部资产：在内置几何体列表中轮换，保证多场景物体形态有变化
        primitive_type = scene_builder.BUILTIN_PRIMITIVES[scene_id % len(scene_builder.BUILTIN_PRIMITIVES)]
        scene_config["primitive_type"] = primitive_type

    scene_info = scene_builder.build_scene(scene_config)
    scene_center = tuple(scene_info["object_center"])

    poses = camera_trajectory.generate_camera_poses(
        num_views=config["views_per_scene"],
        radius=config["radius"],
        elevation_range=config["elevation_range"],
        seed=config["seed"] + scene_id,
        include_reference=True,
    )
    ref_pose = next(p for p in poses if p["is_reference"])

    scene_dir_name = "scene_{:04d}".format(scene_id)
    scene_dir = os.path.join(output_dir, scene_dir_name)
    os.makedirs(scene_dir, exist_ok=True)

    bl_scene = bpy.context.scene
    _configure_render_settings(bl_scene, config["resolution"], config["engine"], config)

    annotations = []
    success_count = 0
    fail_count = 0

    for pose in poses:
        view_id = pose["view_id"]
        image_filename = "view_ref.png" if view_id == "ref" else "view_{}.png".format(view_id)
        image_path_abs = os.path.join(scene_dir, image_filename)

        camera_trajectory.create_camera(pose, scene_center=scene_center)
        bl_scene.render.filepath = image_path_abs

        # 渲染单帧是整个 pipeline 中唯一真正可能因场景/驱动异常而失败的步骤，
        # 因此仅在此处做容错，避免单个视角失败导致整个数据集生成中断
        try:
            bpy.ops.render.render(write_still=True)
        except Exception as exc:
            print("场景 {} 视角 {} 渲染失败，已跳过：{}".format(scene_id, view_id, exc))
            fail_count += 1
            continue

        # bpy.ops.render.render 在引擎/驱动异常时可能不抛异常但也不真正写盘，
        # 因此显式校验文件是否落盘，避免产出"标注存在但图片缺失"的无效样本
        if not os.path.isfile(image_path_abs):
            print("场景 {} 视角 {} 渲染未产生输出文件，已跳过：{}".format(scene_id, view_id, image_path_abs))
            fail_count += 1
            continue

        success_count += 1
        relative_rotation = _compute_relative_rotation_vector(pose, ref_pose)
        depth_filename = "depth_ref.png" if view_id == "ref" else "depth_{}.png".format(view_id)
        edge_filename = "edge_ref.png" if view_id == "ref" else "edge_{}.png".format(view_id)
        record = {
            "view_id": view_id,
            "image_path": "{}/{}".format(scene_dir_name, image_filename),
            "azimuth_deg": pose["azimuth_deg"],
            "elevation_deg": pose["elevation_deg"],
            "distance": pose["distance"],
            "euler_xyz_deg": pose["euler_xyz_deg"],
            "camera_position_xyz": pose["camera_position_xyz"],
            "rotation_vector_relative_to_ref": relative_rotation,
            "scene_id": scene_id,
            "object_source": scene_info["object_source"],
        }
        # depth_path/edge_path 字段预先写入（指向 export_conditions.py 将要生成的路径），
        # 该脚本运行后这两个文件才会真正存在。
        if config.get("export_depth", True):
            record["depth_path"] = "{}/{}".format(scene_dir_name, depth_filename)
        if config.get("export_edge", True):
            record["edge_path"] = "{}/{}".format(scene_dir_name, edge_filename)
        annotations.append(record)

    poses_json_path = os.path.join(scene_dir, "poses.json")
    with open(poses_json_path, "w", encoding="utf-8") as f:
        json.dump(annotations, f, ensure_ascii=False, indent=2)

    return {
        "scene_dir": scene_dir_name,
        "object_source": scene_info["object_source"],
        "success_count": success_count,
        "fail_count": fail_count,
    }


# ----------------------------------------------------------------------------
# 主流程
# ----------------------------------------------------------------------------
def main():
    args = parse_args()

    config = dict(DEFAULT_CONFIG)
    config.update({
        "output_dir": args.output_dir,
        "num_scenes": args.num_scenes,
        "views_per_scene": args.views_per_scene,
        "resolution": args.resolution,
        "engine": args.engine,
        "object_dir": args.object_dir,
        "seed": args.seed,
        "samples": args.samples,
        "use_gpu": bool(args.use_gpu),
        "denoise": bool(args.denoise),
        "tile_size": args.tile_size,
        "export_depth": bool(args.export_depth),
        "export_edge": bool(args.export_edge),
    })

    # Blender 以 --background --python 启动时，进程当前工作目录并不等同于
    # 用户在终端里执行命令时所在的目录（常见坑：相对路径最终写到了 C 盘用户目录
    # 或 blender.exe 安装目录下）。因此这里显式将 output_dir/object_dir 转换为
    # 绝对路径，避免因 cwd 不确定导致输出位置跑偏。
    config["output_dir"] = os.path.abspath(config["output_dir"])
    if config["object_dir"]:
        config["object_dir"] = os.path.abspath(config["object_dir"])

    print("数据集将输出到绝对路径：{}".format(config["output_dir"]))
    os.makedirs(config["output_dir"], exist_ok=True)

    model_files = _scan_object_files(config["object_dir"])
    if model_files:
        print("发现 {} 个外部模型文件，将按场景轮流使用。".format(len(model_files)))
    else:
        print("未提供外部模型资产（或目录为空），将使用内置几何体轮换生成场景。")

    scene_records = []
    total_success = 0
    total_fail = 0

    for scene_id in range(config["num_scenes"]):
        object_source_path = model_files[scene_id % len(model_files)] if model_files else None

        start_time = time.time()
        result = render_scene(scene_id, object_source_path, config, config["output_dir"])
        elapsed = time.time() - start_time

        total_success += result["success_count"]
        total_fail += result["fail_count"]
        print("场景 {:04d} 完成：耗时 {:.2f}s，成功 {} 张，失败 {} 张，物体来源：{}".format(
            scene_id, elapsed, result["success_count"], result["fail_count"], result["object_source"]
        ))

        scene_records.append({
            "scene_id": scene_id,
            "scene_dir": result["scene_dir"],
            "object_source": result["object_source"],
            "success_count": result["success_count"],
            "fail_count": result["fail_count"],
            "render_time_sec": elapsed,
        })

    manifest = {
        "generated_at": datetime.datetime.now().isoformat(),
        "blender_version": bpy.app.version_string,
        "num_scenes": config["num_scenes"],
        # 总有效样本数：所有场景成功渲染的图片总数，含参考视角图片
        "total_valid_samples": total_success,
        "total_failed_renders": total_fail,
        "resolution": config["resolution"],
        "views_per_scene": config["views_per_scene"],
        "seed": config["seed"],
        "engine": config["engine"],
        "object_dir": config["object_dir"],
        "export_depth": config["export_depth"],
        "export_edge": config["export_edge"],
        "samples": config["samples"],
        "use_gpu": config["use_gpu"],
        "scenes": scene_records,
    }
    manifest_path = os.path.join(config["output_dir"], "dataset_manifest.json")
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)

    print("数据集生成完成。总有效样本数：{}，失败次数：{}，清单文件：{}".format(
        total_success, total_fail, manifest_path
    ))


if __name__ == "__main__":
    main()
