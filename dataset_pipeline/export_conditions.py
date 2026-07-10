# -*- coding: utf-8 -*-
"""
export_conditions.py
数据集深度图 / 轮廓图后处理导出脚本（纯 Python，不依赖 bpy）。

背景：
    Blender 5.0 大幅重构了 Compositor 节点 API（scene.node_tree 被移除、
    CompositorNodeComposite/MapRange/OutputFile 等节点类型全部变更，
    且 Viewer 节点在 --background 无界面渲染模式下不产生像素数据），
    在 Blender 内部程序化导出深度图/轮廓图的技术路径已被判定不可行。
    因此改为：render_pipeline.py 只负责渲染并落盘 RGB 图（同时在
    poses.json 中预写 depth_path / edge_path 字段），本脚本在渲染完成后
    独立运行，基于已落盘的 RGB 图做后处理，生成对应的深度图与轮廓图。

职责：
    1. 遍历 dataset_dir 下所有 scene_*/poses.json，收集每条记录的 image_path。
    2. 深度图（depth_path，16bit 单通道 PNG）：使用 DepthAnything V2 模型
       （depth-anything/Depth-Anything-V2-Large-hf）对 RGB 图推理生成，
       深度值归一化到 [0, 65535] 后保存。
    3. 轮廓图（edge_path，8bit 单通道 PNG）：直接对同一张 RGB 原图使用
       cv2.Canny 做边缘检测（而不是对深度图求梯度），保证轮廓图反映的是
       物体在 RGB 图像上的真实边缘。
    4. 深度图与轮廓图均与源 RGB 图分辨率一致、像素坐标严格对齐（后处理
       直接基于已保存的 RGB 图计算，不引入额外的重新渲染或坐标变换）。
    5. 支持跳过已存在的输出文件（--overwrite 0，默认），便于中断后续跑。

用法：
    python export_conditions.py --dataset_dir E:/Multimodal/dataset_output \
        --export_depth 1 --export_edge 1 \
        --canny_threshold1 100 --canny_threshold2 200 \
        --depth_batch_size 8 --edge_num_workers 4
"""

import argparse
import json
import os

import numpy as np
from PIL import Image


# ----------------------------------------------------------------------------
# 命令行参数解析
# ----------------------------------------------------------------------------
def parse_args():
    parser = argparse.ArgumentParser(description="深度图/轮廓图后处理导出脚本")
    parser.add_argument("--dataset_dir", type=str, required=True,
                         help="数据集根目录（包含 scene_*/poses.json 与 dataset_manifest.json）")
    parser.add_argument("--export_depth", type=int, default=1, choices=[0, 1],
                         help="是否生成深度图（DepthAnything V2 推理），默认开启")
    parser.add_argument("--export_edge", type=int, default=1, choices=[0, 1],
                         help="是否生成轮廓图（对 RGB 原图做 Canny 边缘检测），默认开启")
    parser.add_argument("--overwrite", type=int, default=0, choices=[0, 1],
                         help="已存在的输出文件是否重新生成覆盖，默认 0（跳过已存在文件，便于中断续跑）")

    # 深度图相关
    parser.add_argument("--depth_model_id", type=str,
                         default="depth-anything/Depth-Anything-V2-Large-hf",
                         help="DepthAnything V2 模型标识符（HuggingFace Hub）")
    parser.add_argument("--depth_batch_size", type=int, default=8,
                         help="深度图推理批大小，默认 8（在 16G 显存的 RTX 4060 上可安全运行）")
    parser.add_argument("--depth_device", type=str, default="cuda",
                         help="深度模型推理设备，默认 cuda，无 GPU 时可设为 cpu")

    # 轮廓图相关（Canny，直接作用于 RGB 原图）
    parser.add_argument("--canny_threshold1", type=float, default=100.0,
                         help="cv2.Canny 低阈值，默认 100")
    parser.add_argument("--canny_threshold2", type=float, default=200.0,
                         help="cv2.Canny 高阈值，默认 200")
    parser.add_argument("--edge_num_workers", type=int, default=4,
                         help="轮廓图生成的并行线程数（Canny 为 CPU 计算，可安全多线程），默认 4")
    parser.add_argument("--hf_mirror", type=str, default=None,
                         help="HuggingFace 镜像站点，如 https://hf-mirror.com，解决国内连接不稳定问题")

    return parser.parse_args()


# ----------------------------------------------------------------------------
# 样本收集
# ----------------------------------------------------------------------------
def collect_records(dataset_dir):
    """遍历 dataset_dir 下所有 scene_*/poses.json，返回 (scene_dir_abs, record) 列表。"""
    scene_dirs = sorted(
        d for d in os.listdir(dataset_dir)
        if d.startswith("scene_") and os.path.isdir(os.path.join(dataset_dir, d))
    )

    all_records = []
    for scene_dir_name in scene_dirs:
        scene_dir_abs = os.path.join(dataset_dir, scene_dir_name)
        poses_path = os.path.join(scene_dir_abs, "poses.json")
        if not os.path.isfile(poses_path):
            continue
        with open(poses_path, "r", encoding="utf-8") as f:
            records = json.load(f)
        for record in records:
            all_records.append((scene_dir_abs, record))

    print("共发现 {} 个场景，{} 条视角标注记录。".format(len(scene_dirs), len(all_records)))
    return all_records


def _needs_processing(dataset_dir, scene_dir_abs, record, field, overwrite):
    """判断某条记录的某个条件图字段是否需要生成（字段存在、文件缺失或要求覆盖）。"""
    rel_path = record.get(field)
    if not rel_path:
        return False
    abs_path = os.path.join(dataset_dir, rel_path)
    if overwrite:
        return True
    return not os.path.isfile(abs_path)


# ----------------------------------------------------------------------------
# 深度图导出：DepthAnything V2，批处理推理
# ----------------------------------------------------------------------------
def export_depth_maps(dataset_dir, all_records, args):
    """对所有待处理记录批量推理生成 16bit 深度图 PNG。"""
    import torch
    from transformers import AutoImageProcessor, AutoModelForDepthEstimation

    pending = [
        (scene_dir_abs, record) for scene_dir_abs, record in all_records
        if _needs_processing(dataset_dir, scene_dir_abs, record, "depth_path", args.overwrite)
    ]
    if not pending:
        print("深度图：所有目标文件已存在，无需生成（如需重新生成请加 --overwrite 1）。")
        return

    print("深度图：待生成 {} 张，加载模型 {} ...".format(len(pending), args.depth_model_id))
    device = args.depth_device if torch.cuda.is_available() or args.depth_device == "cpu" else "cpu"
    if device != args.depth_device:
        print("警告：CUDA 不可用，深度模型回退为 CPU 推理（速度会明显变慢）。")

    processor = AutoImageProcessor.from_pretrained(args.depth_model_id)
    model = AutoModelForDepthEstimation.from_pretrained(args.depth_model_id)
    model.to(device)
    model.eval()

    batch_size = max(1, args.depth_batch_size)
    total_done = 0

    with torch.no_grad():
        for start in range(0, len(pending), batch_size):
            batch = pending[start:start + batch_size]
            images = []
            out_paths = []
            orig_sizes = []
            for scene_dir_abs, record in batch:
                image_abs_path = os.path.join(dataset_dir, record["image_path"])
                img = Image.open(image_abs_path).convert("RGB")
                images.append(img)
                orig_sizes.append(img.size)  # (W, H)
                out_paths.append(os.path.join(dataset_dir, record["depth_path"]))

            inputs = processor(images=images, return_tensors="pt").to(device)
            outputs = model(**inputs)
            predicted_depth = outputs.predicted_depth  # (B, h, w)

            for i in range(len(batch)):
                depth = predicted_depth[i]
                width, height = orig_sizes[i]
                depth_resized = torch.nn.functional.interpolate(
                    depth.unsqueeze(0).unsqueeze(0),
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
                depth_uint16 = (depth_norm * 65535.0).astype(np.uint16)

                out_path = out_paths[i]
                os.makedirs(os.path.dirname(out_path), exist_ok=True)
                Image.fromarray(depth_uint16, mode="I;16").save(out_path)

            total_done += len(batch)
            print("深度图进度：{}/{}".format(total_done, len(pending)))

    print("深度图生成完成，共 {} 张。".format(total_done))


# ----------------------------------------------------------------------------
# 轮廓图导出：对 RGB 原图直接做 Canny 边缘检测，多线程并行
# ----------------------------------------------------------------------------
def _canny_single(dataset_dir, record, threshold1, threshold2):
    import cv2

    image_abs_path = os.path.join(dataset_dir, record["image_path"])
    out_path = os.path.join(dataset_dir, record["edge_path"])

    img_bgr = cv2.imread(image_abs_path, cv2.IMREAD_COLOR)
    if img_bgr is None:
        return False, record["image_path"]

    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, threshold1, threshold2)

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    cv2.imwrite(out_path, edges)
    return True, record["image_path"]


def export_edge_maps(dataset_dir, all_records, args):
    """对所有待处理记录并行做 Canny 边缘检测，生成 8bit 轮廓图 PNG。"""
    import concurrent.futures

    pending = [
        record for _scene_dir_abs, record in all_records
        if _needs_processing(dataset_dir, _scene_dir_abs, record, "edge_path", args.overwrite)
    ]
    if not pending:
        print("轮廓图：所有目标文件已存在，无需生成（如需重新生成请加 --overwrite 1）。")
        return

    print("轮廓图：待生成 {} 张，Canny 阈值 ({}, {})，并行线程数 {} ...".format(
        len(pending), args.canny_threshold1, args.canny_threshold2, args.edge_num_workers
    ))

    total_done = 0
    fail_count = 0
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.edge_num_workers) as executor:
        futures = [
            executor.submit(_canny_single, dataset_dir, record, args.canny_threshold1, args.canny_threshold2)
            for record in pending
        ]
        for future in concurrent.futures.as_completed(futures):
            ok, image_path = future.result()
            if not ok:
                fail_count += 1
                print("轮廓图生成失败，源图片无法读取：{}".format(image_path))
            total_done += 1
            if total_done % 50 == 0 or total_done == len(pending):
                print("轮廓图进度：{}/{}".format(total_done, len(pending)))

    print("轮廓图生成完成，共 {} 张，失败 {} 张。".format(total_done - fail_count, fail_count))


# ----------------------------------------------------------------------------
# 主流程
# ----------------------------------------------------------------------------
def main():
    args = parse_args()

    # 设置 HuggingFace 镜像（解决国内网络环境连接超时/重置问题）
    if args.hf_mirror:
        os.environ["HF_ENDPOINT"] = args.hf_mirror
        print("已配置 HuggingFace 镜像：{}".format(args.hf_mirror))

    dataset_dir = os.path.abspath(args.dataset_dir)

    all_records = collect_records(dataset_dir)
    if not all_records:
        print("未在 {} 下找到任何 poses.json 标注记录，退出。".format(dataset_dir))
        return

    if args.export_depth:
        export_depth_maps(dataset_dir, all_records, args)
    else:
        print("已跳过深度图生成（--export_depth 0）。")

    if args.export_edge:
        export_edge_maps(dataset_dir, all_records, args)
    else:
        print("已跳过轮廓图生成（--export_edge 0）。")

    print("全部条件图后处理完成。")


if __name__ == "__main__":
    main()
