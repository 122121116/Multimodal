# -*- coding: utf-8 -*-
"""
download_assets.py
从 Poly Haven（https://polyhaven.com）自动下载一批 CC0 授权的免费 3D 模型资产，
供 render_pipeline.py 的 --object_dir 参数使用。

Poly Haven API 说明（无需 API key，可匿名直接调用）：
    - 资产列表：GET https://api.polyhaven.com/assets?type=models
      返回 { slug: {name, categories, tags, polycount, ...}, ... }
    - 资产文件清单：GET https://api.polyhaven.com/files/{slug}
      返回该资产各格式（gltf/blend/fbx...）各分辨率（1k/2k/4k/8k）的下载直链，
      本脚本只关心 gltf 格式，其结构大致为：
          files["gltf"]["1k"]["gltf"] = {
              "url": "<.gltf 主文件下载直链>",
              "include": {
                  "textures/xxx_diff_1k.jpg": {"url": "<贴图直链>", ...},
                  "Camera_01.bin": {"url": "<几何数据直链>", ...},
                  ...
              }
          }
      .gltf 文件内部使用相对路径引用 include 里的贴图/bin 文件，因此下载时必须
      原样保留 include 字典 key 所描述的相对目录结构（如 textures/ 子目录），
      否则 Blender 导入该 .gltf 时会因为找不到引用文件而报错或丢失材质。

用法：
    python download_assets.py --output_dir e:/Multimodal/assets/models
"""

import argparse
import json
import os
import time
import urllib.request

# 默认下载的资产 slug 列表：均为 Poly Haven 上有一定几何/材质复杂度的
# 免费日常物品模型（非简单几何体），CC0 授权，无需人像（Poly Haven 暂无 CC0 人像资源）。
DEFAULT_ASSET_SLUGS = [
    "Camera_01",        # 相机
    "Drill_01",         # 电钻
    "Lantern_01",       # 灯笼
    "Barrel_01",        # 油桶
    "CoffeeTable_01",   # 咖啡桌
    "GreenChair_01",    # 椅子
    "CashRegister_01",  # 收银机
    "Megaphone_01",     # 扩音器
]

API_FILES_URL = "https://api.polyhaven.com/files/{slug}"
USER_AGENT = "Mozilla/5.0 (compatible; MultimodalDatasetPipeline/1.0; +https://polyhaven.com)"
REQUEST_TIMEOUT = 30  # 秒


def _http_get_json(url):
    """请求 url 并把返回内容解析为 JSON，设置 User-Agent 与超时时间。"""
    req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(req, timeout=REQUEST_TIMEOUT) as resp:
        return json.loads(resp.read().decode("utf-8"))


def _download_file(url, dest_path):
    """下载单个文件到 dest_path，若已存在且非空则跳过（支持断点续跑）。

    返回 (是否实际下载, 文件大小字节数, 耗时秒)。单个文件下载失败会抛出异常，
    由调用方 try/except 捕获，避免影响其他文件的下载。
    """
    if os.path.isfile(dest_path) and os.path.getsize(dest_path) > 0:
        return False, os.path.getsize(dest_path), 0.0

    os.makedirs(os.path.dirname(dest_path), exist_ok=True)

    req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    t0 = time.time()
    with urllib.request.urlopen(req, timeout=REQUEST_TIMEOUT) as resp:
        data = resp.read()
    with open(dest_path, "wb") as f:
        f.write(data)
    elapsed = time.time() - t0
    return True, len(data), elapsed


def _resolve_gltf_entry(files_json, resolution):
    """从 /files/{slug} 返回的 JSON 中取出指定分辨率下的 gltf 文件条目。

    若该分辨率没有 gltf 数据，兜底尝试 1k；仍不存在则返回 None。
    """
    gltf_formats = files_json.get("gltf", {})

    entry = gltf_formats.get(resolution, {}).get("gltf")
    if entry:
        return entry

    if resolution != "1k":
        entry = gltf_formats.get("1k", {}).get("gltf")
        if entry:
            return entry

    return None


def download_asset(slug, output_dir, resolution, index, total):
    """下载单个资产（.gltf 主文件 + include 中的贴图/bin 文件）到 output_dir/slug/ 下。

    返回 True 表示成功（至少主文件下载成功），False 表示该资产被跳过。
    """
    print("[{}/{}] 处理资产: {}".format(index, total, slug))

    try:
        files_json = _http_get_json(API_FILES_URL.format(slug=slug))
    except Exception as e:
        print("  警告：获取 {} 的文件清单失败，跳过该资产。错误：{}".format(slug, e))
        return False

    gltf_entry = _resolve_gltf_entry(files_json, resolution)
    if not gltf_entry:
        print("  警告：{} 在分辨率 {}（含 1k 兜底）下没有 gltf 数据，跳过。".format(slug, resolution))
        return False

    asset_dir = os.path.join(output_dir, slug)
    os.makedirs(asset_dir, exist_ok=True)

    total_bytes = 0
    ok_count = 0
    fail_count = 0

    # 下载主 .gltf 文件，统一命名为 {slug}.gltf
    main_url = gltf_entry.get("url")
    main_dest = os.path.join(asset_dir, "{}.gltf".format(slug))
    try:
        downloaded, size, elapsed = _download_file(main_url, main_dest)
        total_bytes += size
        ok_count += 1
        if downloaded:
            print("  主文件 {}.gltf 下载完成，{} 字节，耗时 {:.1f}s".format(slug, size, elapsed))
        else:
            print("  主文件 {}.gltf 已存在，跳过下载".format(slug))
    except Exception as e:
        print("  错误：主文件 {}.gltf 下载失败：{}".format(slug, e))
        return False

    # 下载 include 中的所有关联文件（贴图、.bin 几何数据等），
    # 必须原样保留其相对路径（如 textures/xxx_diff_1k.jpg），
    # 因为 .gltf 内部通过该相对路径引用这些文件，目录结构一变 Blender 导入即失败。
    include = gltf_entry.get("include", {})
    for rel_path, file_info in include.items():
        file_url = file_info.get("url")
        dest_path = os.path.join(asset_dir, rel_path.replace("/", os.sep))
        try:
            downloaded, size, elapsed = _download_file(file_url, dest_path)
            total_bytes += size
            ok_count += 1
            if downloaded:
                print("    附属文件 {} 下载完成，{} 字节，耗时 {:.1f}s".format(rel_path, size, elapsed))
            else:
                print("    附属文件 {} 已存在，跳过下载".format(rel_path))
        except Exception as e:
            fail_count += 1
            print("    错误：附属文件 {} 下载失败：{}".format(rel_path, e))
            continue

    print("  资产 {} 完成：成功 {} 个文件，失败 {} 个，累计 {} 字节".format(
        slug, ok_count, fail_count, total_bytes))
    return True


def main():
    parser = argparse.ArgumentParser(description="从 Poly Haven 下载免费 CC0 3D 模型资产")
    parser.add_argument("--output_dir", type=str, default="e:/Multimodal/assets/models",
                         help="模型资产保存的根目录")
    parser.add_argument("--assets", type=str, default=None,
                         help="逗号分隔的 slug 列表，覆盖默认资产列表")
    parser.add_argument("--resolution", type=str, default="1k", choices=["1k", "2k"],
                         help="下载的贴图分辨率，默认 1k")
    args = parser.parse_args()

    slugs = [s.strip() for s in args.assets.split(",") if s.strip()] if args.assets else DEFAULT_ASSET_SLUGS

    output_dir = os.path.abspath(args.output_dir)
    os.makedirs(output_dir, exist_ok=True)

    print("准备下载 {} 个资产到: {}".format(len(slugs), output_dir))
    print("分辨率: {}".format(args.resolution))
    print("-" * 60)

    success_count = 0
    fail_count = 0
    t_start = time.time()

    for i, slug in enumerate(slugs, start=1):
        ok = download_asset(slug, output_dir, args.resolution, i, len(slugs))
        if ok:
            success_count += 1
        else:
            fail_count += 1
        print("-" * 60)

    total_elapsed = time.time() - t_start
    print("下载汇总：成功 {} 个资产，失败/跳过 {} 个，总耗时 {:.1f}s".format(
        success_count, fail_count, total_elapsed))
    print("资产保存目录: {}".format(output_dir))
    print("可将该目录作为 render_pipeline.py 的 --object_dir 参数使用。")


if __name__ == "__main__":
    main()
