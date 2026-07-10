# -*- coding: utf-8 -*-
"""
camera_trajectory.py
相机轨道自动化轨迹生成模块。

职责：
    - 使用球面均匀采样（分层网格采样 azimuth/elevation）生成一组相机位姿，
      保证方位角/俯仰角方向分布均匀，而非随机乱撒点
    - 强制包含一个固定的参考视角（正对物体，view_id="ref"），
      用于构成 (reference_image, target_image, relative_camera_pose) 训练三元组
    - 提供 create_camera，根据位姿在 Blender 场景中创建/更新相机对象，
      并始终 look-at 物体几何中心

坐标系与角度约定（与 scene_builder._spherical_to_cartesian 保持一致）：
    - 世界坐标系 Z 轴向上
    - azimuth（方位角）：从 +X 轴绕 Z 轴逆时针旋转的角度，范围 [0, 360)
    - elevation（俯仰角）：相机位置与 XY 平面的夹角，范围由 elevation_range 决定
    - 相机默认朝向 -Z 轴，因此需要计算 track-to 旋转使其朝向物体中心
"""

import math

import bpy
import mathutils


# ----------------------------------------------------------------------------
# 球坐标 <-> 笛卡尔坐标
# ----------------------------------------------------------------------------
def _spherical_to_cartesian(azimuth_deg, elevation_deg, distance):
    """球坐标（方位角/俯仰角/距离）转世界坐标系笛卡尔坐标。"""
    az = math.radians(azimuth_deg)
    el = math.radians(elevation_deg)
    x = distance * math.cos(el) * math.cos(az)
    y = distance * math.cos(el) * math.sin(az)
    z = distance * math.sin(el)
    return (x, y, z)


def _look_at_euler(camera_location, target):
    """计算使相机从 camera_location 朝向 target 所需的欧拉角（度）。"""
    direction = mathutils.Vector((
        target[0] - camera_location[0],
        target[1] - camera_location[1],
        target[2] - camera_location[2],
    ))
    # 相机默认朝向 -Z 轴，上方向为 Y 轴
    quat = direction.to_track_quat("-Z", "Y")
    euler = quat.to_euler()
    return (math.degrees(euler.x), math.degrees(euler.y), math.degrees(euler.z))


# ----------------------------------------------------------------------------
# 相机位姿采样：分层网格采样，保证 azimuth/elevation 均匀分布
# ----------------------------------------------------------------------------
def _generate_uniform_grid_angles(num_samples, elevation_range, seed=None):
    """在 azimuth [0, 360) 与 elevation range 内做分层网格采样。

    做法：将 num_samples 分解为 (n_elevation 行 x n_azimuth 列) 的近似网格，
    每行固定一个 elevation，在该行内把 azimuth 均匀分布在 [0, 360) 上，
    并额外加一个小的随机抖动（同一 seed 下可复现），避免网格状伪影，
    同时保持整体均匀性（不是纯随机撒点）。

    返回：长度为 num_samples 的 (azimuth_deg, elevation_deg) 列表。
    """
    rng = __import__("random").Random(seed)

    # 估算行数（elevation 方向）与每行列数（azimuth 方向），使网格尽量接近正方形
    n_elevation = max(1, round(math.sqrt(num_samples)))
    base_cols = num_samples // n_elevation
    remainder = num_samples % n_elevation

    elev_min, elev_max = elevation_range
    angles = []

    for row in range(n_elevation):
        # 行数为 1 时直接取中间值，避免除零
        if n_elevation == 1:
            elevation = (elev_min + elev_max) / 2.0
        else:
            elevation = elev_min + (elev_max - elev_min) * row / (n_elevation - 1)

        # 将余数分摊到前面的行，保证总数精确等于 num_samples
        n_cols = base_cols + (1 if row < remainder else 0)
        if n_cols <= 0:
            continue

        for col in range(n_cols):
            azimuth = 360.0 * col / n_cols
            # 加入小幅随机抖动（<= 半个网格间隔的一半），保持均匀性的同时避免机械感
            jitter = rng.uniform(-0.5, 0.5) * (360.0 / n_cols) * 0.3
            azimuth = (azimuth + jitter) % 360.0
            angles.append((azimuth, elevation))

    return angles


def generate_camera_poses(num_views, radius, elevation_range=(-30, 60), seed=None, include_reference=True):
    """生成一个场景的相机位姿列表。

    参数：
        num_views: 需要生成的位姿总数（含参考视角，若 include_reference=True）。
                   建议 >= 30，以满足数据集单场景视角数量要求。
        radius: 相机到物体中心的距离（Blender 单位）。
        elevation_range: 俯仰角采样范围（度），(min, max)。
        seed: 随机种子，用于 azimuth 抖动的可复现性。
        include_reference: 是否强制生成固定参考视角（azimuth=0, elevation=0），
                            view_id 固定为 "ref"。

    返回：
        list[dict]，每个 dict 是一组相机位姿，字段包括：
            - view_id: 唯一标识（参考视角为 "ref"，其余为从 0 开始的整数）
            - azimuth_deg / elevation_deg: 球坐标角度（度）
            - distance: 相机到物体中心的距离
            - euler_xyz_deg: 相机欧拉角 (rx, ry, rz)，单位度
            - camera_position_xyz: 相机在世界坐标系下的位置 (x, y, z)
            - is_reference: 是否为参考视角
    """
    poses = []
    scene_center = (0.0, 0.0, 0.0)

    if include_reference:
        ref_azimuth, ref_elevation = 0.0, 0.0
        ref_location = _spherical_to_cartesian(ref_azimuth, ref_elevation, radius)
        ref_euler = _look_at_euler(ref_location, scene_center)
        poses.append({
            "view_id": "ref",
            "azimuth_deg": ref_azimuth,
            "elevation_deg": ref_elevation,
            "distance": radius,
            "euler_xyz_deg": list(ref_euler),
            "camera_position_xyz": list(ref_location),
            "is_reference": True,
        })
        num_remaining = num_views - 1
    else:
        num_remaining = num_views

    angles = _generate_uniform_grid_angles(num_remaining, elevation_range, seed=seed)

    for idx, (azimuth, elevation) in enumerate(angles):
        location = _spherical_to_cartesian(azimuth, elevation, radius)
        euler = _look_at_euler(location, scene_center)
        poses.append({
            "view_id": idx,
            "azimuth_deg": azimuth,
            "elevation_deg": elevation,
            "distance": radius,
            "euler_xyz_deg": list(euler),
            "camera_position_xyz": list(location),
            "is_reference": False,
        })

    return poses


# ----------------------------------------------------------------------------
# 在 Blender 场景中创建/更新相机
# ----------------------------------------------------------------------------
def create_camera(pose, scene_center=(0, 0, 0)):
    """根据位姿在 Blender 场景中创建（或复用已存在的）相机对象，并令其 look-at 物体中心。

    参数：
        pose: dict，generate_camera_poses 返回的单个位姿字典
              （至少需要 camera_position_xyz 字段）。
        scene_center: 物体几何中心（世界坐标），相机始终朝向该点。

    返回：
        bpy.types.Object，创建/更新后的相机对象，并已设置为场景当前渲染相机。
    """
    camera_obj = bpy.data.objects.get("DatasetCamera")
    if camera_obj is None:
        camera_data = bpy.data.cameras.new(name="DatasetCamera")
        camera_obj = bpy.data.objects.new(name="DatasetCamera", object_data=camera_data)
        bpy.context.collection.objects.link(camera_obj)

    location = pose["camera_position_xyz"]
    camera_obj.location = location

    # 直接根据 look-at 关系计算朝向（与 generate_camera_poses 中欧拉角计算逻辑一致），
    # 而不是依赖 Track-To 约束，避免约束求值时机问题导致渲染时朝向未更新
    direction = mathutils.Vector((
        scene_center[0] - location[0],
        scene_center[1] - location[1],
        scene_center[2] - location[2],
    ))
    quat = direction.to_track_quat("-Z", "Y")
    camera_obj.rotation_euler = quat.to_euler()

    bpy.context.scene.camera = camera_obj
    bpy.context.view_layer.update()
    return camera_obj


# ----------------------------------------------------------------------------
# 独立运行：打印生成的位姿，便于调试
# ----------------------------------------------------------------------------
if __name__ == "__main__":
    sample_poses = generate_camera_poses(num_views=30, radius=5.0, elevation_range=(-30, 60), seed=42)
    for p in sample_poses:
        print(p)
