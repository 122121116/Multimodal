# -*- coding: utf-8 -*-
"""
scene_builder.py
Blender 场景搭建模块。

职责：
    - 清空默认场景（默认 cube / light / camera）
    - 搭建统一地面平面（中性灰、不反光，避免干扰主体轮廓识别）
    - 搭建统一三点光照（key/fill/back，Area light，固定强度与色温，
      保证数据集"同一场景内光照条件一致"、"跨场景光照配置可复现"的要求）
    - 设置统一纯色世界背景（简化后续抠图/对齐判断）
    - 加载目标物体（外部模型文件优先，否则优雅降级为内置几何体）
    - 将物体归一化到统一包围盒尺寸，并将几何中心对齐到世界原点
      （这是相机绕物体旋转、多视角对齐的必要前提）

本模块可被 render_pipeline.py 作为库导入使用（核心入口 build_scene），
也可通过 `blender --background --python scene_builder.py` 独立运行做场景预览。
"""

import os
import math
import random

import bpy


# ----------------------------------------------------------------------------
# 默认配置：光照 / 地面 / 背景 / 归一化尺寸 等关键参数集中定义，方便调整
# ----------------------------------------------------------------------------
DEFAULT_SCENE_CONFIG = {
    # 物体来源：外部模型文件路径（.obj/.fbx/.glb/.blend），为空则使用内置几何体
    "object_path": None,
    # 内置几何体类型（object_path 为空时使用）：
    # cube / sphere / cylinder / cone / torus / monkey
    "primitive_type": "cube",
    # 物体归一化后最大边长（Blender 单位）
    "normalize_size": 2.0,
    # 地面平面尺寸（边长，Blender 单位）
    "ground_size": 20.0,
    # 地面材质基础色（中性灰，线性空间 RGBA）
    "ground_color": (0.5, 0.5, 0.5, 1.0),
    # 地面粗糙度（越接近 1 越不反光，保证漫反射为主，不干扰轮廓识别）
    "ground_roughness": 1.0,
    # 世界背景纯色（RGBA，中性灰，与地面颜色区分以便后续抠图/分割）
    "world_background_color": (0.4, 0.4, 0.4, 1.0),
    # 三点光照强度（单位 W，Area Light）
    "key_light_power": 1000.0,
    "fill_light_power": 400.0,
    "back_light_power": 600.0,
    # 三点光照色温（单位 K，固定色温保证跨场景一致）
    "light_color_temperature": 6500.0,
    # 三点光照相对物体的角度/距离配置（球坐标，角度制）
    "key_light_azimuth": 45.0,
    "key_light_elevation": 45.0,
    "fill_light_azimuth": -45.0,
    "fill_light_elevation": 30.0,
    "back_light_azimuth": 180.0,
    "back_light_elevation": 50.0,
    "light_distance": 6.0,
    "light_size": 2.0,  # Area light 面光源尺寸
}

# 内置几何体轮换列表：无外部模型资产时，用于在多个场景间自动切换物体形态
BUILTIN_PRIMITIVES = ["cube", "sphere", "cylinder", "cone", "torus", "monkey"]


# ----------------------------------------------------------------------------
# 场景清理
# ----------------------------------------------------------------------------
def _clear_scene():
    """清空默认场景中的所有物体（默认 cube/light/camera）以及孤立数据块。"""
    bpy.ops.object.select_all(action="SELECT")
    bpy.ops.object.delete(use_global=False)

    # 清理孤立的网格/灯光/相机/材质数据块，避免多次调用 build_scene 时数据残留
    for data_block_collection in (
        bpy.data.meshes,
        bpy.data.lights,
        bpy.data.cameras,
        bpy.data.materials,
    ):
        for block in list(data_block_collection):
            if block.users == 0:
                data_block_collection.remove(block)


# ----------------------------------------------------------------------------
# 世界背景
# ----------------------------------------------------------------------------
def _setup_world_background(color):
    """设置纯色世界背景（Background 节点的强度采用默认 1.0）。"""
    world = bpy.data.worlds.get("World")
    if world is None:
        world = bpy.data.worlds.new("World")
    bpy.context.scene.world = world

    world.use_nodes = True
    bg_node = world.node_tree.nodes.get("Background")
    if bg_node is not None:
        bg_node.inputs[0].default_value = color
        bg_node.inputs[1].default_value = 1.0


# ----------------------------------------------------------------------------
# 地面平面
# ----------------------------------------------------------------------------
def _create_ground_plane(size, color, roughness):
    """创建统一的中性灰、不反光地面平面。"""
    bpy.ops.mesh.primitive_plane_add(size=size, location=(0, 0, 0))
    ground = bpy.context.active_object
    ground.name = "Ground"

    mat = bpy.data.materials.new(name="GroundMaterial")
    mat.use_nodes = True
    bsdf = mat.node_tree.nodes.get("Principled BSDF")
    if bsdf is not None:
        bsdf.inputs["Base Color"].default_value = color
        bsdf.inputs["Roughness"].default_value = roughness
        # Specular 设为 0，进一步保证地面不反光，避免高光干扰主体轮廓
        if "Specular IOR Level" in bsdf.inputs:
            bsdf.inputs["Specular IOR Level"].default_value = 0.0
        elif "Specular" in bsdf.inputs:
            bsdf.inputs["Specular"].default_value = 0.0

    ground.data.materials.append(mat)
    return ground


# ----------------------------------------------------------------------------
# 三点光照
# ----------------------------------------------------------------------------
def _spherical_to_cartesian(azimuth_deg, elevation_deg, distance):
    """球坐标（方位角/俯仰角/距离）转世界坐标系笛卡尔坐标。

    坐标系约定：Z 轴向上，azimuth 从 +X 轴绕 Z 轴逆时针旋转，
    elevation 为与 XY 平面的夹角。
    """
    az = math.radians(azimuth_deg)
    el = math.radians(elevation_deg)
    x = distance * math.cos(el) * math.cos(az)
    y = distance * math.cos(el) * math.sin(az)
    z = distance * math.sin(el)
    return (x, y, z)


def _create_area_light(name, azimuth, elevation, distance, power, color_temperature, size, target=(0, 0, 0)):
    """创建一盏 Area Light，并令其朝向物体中心（target）。"""
    location = _spherical_to_cartesian(azimuth, elevation, distance)

    light_data = bpy.data.lights.new(name=name, type="AREA")
    light_data.energy = power
    light_data.size = size
    # 使用色温换算为 RGB（Blender 灯光节点支持 Blackbody 节点，这里直接用简化色温->白光近似，
    # 保证跨场景一致性；如需精确色温可在此接入 Blackbody 节点）
    light_data.use_nodes = True
    blackbody = light_data.node_tree.nodes.new("ShaderNodeBlackbody")
    blackbody.inputs["Temperature"].default_value = color_temperature
    emission = None
    for node in light_data.node_tree.nodes:
        if node.type == "EMISSION":
            emission = node
            break
    if emission is not None:
        light_data.node_tree.links.new(blackbody.outputs["Color"], emission.inputs["Color"])

    light_obj = bpy.data.objects.new(name=name, object_data=light_data)
    bpy.context.collection.objects.link(light_obj)
    light_obj.location = location

    # 令灯光朝向物体中心：计算朝向 target 的旋转
    direction = (
        target[0] - location[0],
        target[1] - location[1],
        target[2] - location[2],
    )
    light_obj.rotation_euler = _direction_to_euler(direction)
    return light_obj


def _direction_to_euler(direction):
    """将方向向量转换为面向该方向的欧拉角（默认物体 -Z 轴朝向 target）。"""
    import mathutils

    direction_vec = mathutils.Vector(direction)
    # track_axis = -Z（灯光/相机默认朝向 -Z），up_axis = Y
    quat = direction_vec.to_track_quat("-Z", "Y")
    return quat.to_euler()


def _setup_three_point_lighting(config, target=(0, 0, 0)):
    """搭建 key / fill / back 三点光照，保证跨场景光照条件一致。"""
    lights = {}
    lights["key"] = _create_area_light(
        "KeyLight",
        config["key_light_azimuth"],
        config["key_light_elevation"],
        config["light_distance"],
        config["key_light_power"],
        config["light_color_temperature"],
        config["light_size"],
        target=target,
    )
    lights["fill"] = _create_area_light(
        "FillLight",
        config["fill_light_azimuth"],
        config["fill_light_elevation"],
        config["light_distance"],
        config["fill_light_power"],
        config["light_color_temperature"],
        config["light_size"],
        target=target,
    )
    lights["back"] = _create_area_light(
        "BackLight",
        config["back_light_azimuth"],
        config["back_light_elevation"],
        config["light_distance"],
        config["back_light_power"],
        config["light_color_temperature"],
        config["light_size"],
        target=target,
    )
    return lights


# ----------------------------------------------------------------------------
# 物体加载（外部模型 / 内置几何体降级）
# ----------------------------------------------------------------------------
def _import_external_object(object_path):
    """根据文件后缀导入外部模型文件，返回导入后新增的物体列表。"""
    ext = os.path.splitext(object_path)[1].lower()
    before = set(bpy.data.objects.keys())

    if ext == ".obj":
        bpy.ops.wm.obj_import(filepath=object_path)
    elif ext == ".fbx":
        bpy.ops.import_scene.fbx(filepath=object_path)
    elif ext == ".glb" or ext == ".gltf":
        bpy.ops.import_scene.gltf(filepath=object_path)
    elif ext == ".blend":
        # 从 .blend 文件中追加所有 Mesh 类型物体
        with bpy.data.libraries.load(object_path, link=False) as (data_from, data_to):
            data_to.objects = [name for name in data_from.objects]
        for obj in data_to.objects:
            if obj is not None:
                bpy.context.collection.objects.link(obj)
    else:
        raise ValueError("不支持的模型文件格式: {}".format(ext))

    after = set(bpy.data.objects.keys())
    new_names = after - before
    new_objects = [bpy.data.objects[name] for name in new_names]
    return new_objects


def _create_builtin_primitive(primitive_type):
    """创建内置几何体作为占位物体（无外部资产时的优雅降级方案）。"""
    if primitive_type == "cube":
        bpy.ops.mesh.primitive_cube_add(size=2.0, location=(0, 0, 0))
    elif primitive_type == "sphere":
        bpy.ops.mesh.primitive_uv_sphere_add(radius=1.0, location=(0, 0, 0))
    elif primitive_type == "cylinder":
        bpy.ops.mesh.primitive_cylinder_add(radius=1.0, depth=2.0, location=(0, 0, 0))
    elif primitive_type == "cone":
        bpy.ops.mesh.primitive_cone_add(radius1=1.0, depth=2.0, location=(0, 0, 0))
    elif primitive_type == "torus":
        bpy.ops.mesh.primitive_torus_add(major_radius=1.0, minor_radius=0.35, location=(0, 0, 0))
    elif primitive_type == "monkey":
        bpy.ops.mesh.primitive_monkey_add(size=2.0, location=(0, 0, 0))
    else:
        raise ValueError("未知的内置几何体类型: {}".format(primitive_type))

    obj = bpy.context.active_object

    # 赋予一个简单材质，避免默认灰白材质在不同物体间产生额外差异（保持视觉一致性）
    mat = bpy.data.materials.new(name="PrimitiveMaterial")
    mat.use_nodes = True
    bsdf = mat.node_tree.nodes.get("Principled BSDF")
    if bsdf is not None:
        bsdf.inputs["Base Color"].default_value = (0.7, 0.35, 0.2, 1.0)
        bsdf.inputs["Roughness"].default_value = 0.5
    obj.data.materials.append(mat)
    return [obj]


def _load_object(config):
    """加载目标物体：优先外部模型文件，否则降级为内置几何体。

    返回：(object_list, object_source) 二元组，object_source 记录物体来源
    （外部文件路径，或内置几何体名称），供后续标注使用。
    """
    object_path = config.get("object_path")
    if object_path and os.path.exists(object_path):
        objects = _import_external_object(object_path)
        if len(objects) == 0:
            raise RuntimeError("导入模型文件后未发现任何物体: {}".format(object_path))
        return objects, object_path

    # 无外部路径或路径不存在：优雅降级为内置几何体
    primitive_type = config.get("primitive_type", "cube")
    objects = _create_builtin_primitive(primitive_type)
    return objects, "builtin:{}".format(primitive_type)


# ----------------------------------------------------------------------------
# 物体归一化：统一包围盒尺寸 + 几何中心对齐世界原点
# ----------------------------------------------------------------------------
def _compute_world_bbox(objects):
    """计算一组物体在世界坐标系下的联合包围盒 (bbox_min, bbox_max)。"""
    bbox_min = [math.inf, math.inf, math.inf]
    bbox_max = [-math.inf, -math.inf, -math.inf]

    for obj in objects:
        for corner in obj.bound_box:
            world_corner = obj.matrix_world @ __import__("mathutils").Vector(corner)
            for i in range(3):
                bbox_min[i] = min(bbox_min[i], world_corner[i])
                bbox_max[i] = max(bbox_max[i], world_corner[i])

    return tuple(bbox_min), tuple(bbox_max)


def _normalize_and_center_objects(objects, target_max_size):
    """将物体组归一化到统一包围盒最大边长，并将几何中心平移到世界原点。

    这是保证"多角度相机围绕物体旋转"、"物体轮廓/纹理在不同场景间对齐"的前提：
    否则不同物体的原始尺寸/中心偏移会导致相机轨迹半径不一致、参考视角构图不一致。
    """
    bbox_min, bbox_max = _compute_world_bbox(objects)
    center = tuple((bbox_min[i] + bbox_max[i]) / 2.0 for i in range(3))
    size = tuple(bbox_max[i] - bbox_min[i] for i in range(3))
    max_size = max(size) if max(size) > 1e-8 else 1.0
    scale_factor = target_max_size / max_size

    # 将所有物体归入一个空物体（Empty）父级下，统一做缩放和平移，避免逐个物体误差累积
    empty = bpy.data.objects.new("ObjectRoot", None)
    bpy.context.collection.objects.link(empty)

    for obj in objects:
        obj.parent = empty
        obj.matrix_parent_inverse = empty.matrix_world.inverted()

    empty.location = (-center[0] * scale_factor, -center[1] * scale_factor, -center[2] * scale_factor)
    empty.scale = (scale_factor, scale_factor, scale_factor)

    bpy.context.view_layer.update()

    # 归一化后重新计算包围盒，作为最终元信息返回
    new_bbox_min, new_bbox_max = _compute_world_bbox(objects)
    new_center = tuple((new_bbox_min[i] + new_bbox_max[i]) / 2.0 for i in range(3))
    return {
        "bbox_min": new_bbox_min,
        "bbox_max": new_bbox_max,
        "center": new_center,
        "original_center": center,
        "original_size": size,
        "scale_factor": scale_factor,
    }


# ----------------------------------------------------------------------------
# 对外主入口
# ----------------------------------------------------------------------------
def build_scene(config=None):
    """搭建完整场景：清空 -> 地面 -> 世界背景 -> 三点光照 -> 加载物体 -> 归一化。

    参数：
        config: dict，场景配置，字段参见 DEFAULT_SCENE_CONFIG；
                未提供的字段使用默认值。

    返回：
        dict，场景元信息，供后续渲染/标注使用，字段包括：
            - object_center: 物体归一化后的几何中心（世界坐标，理论上为原点附近）
            - bbox_min / bbox_max: 物体归一化后的世界坐标包围盒
            - object_source: 物体来源（外部文件路径 或 "builtin:<primitive_type>"）
            - normalize_size: 归一化目标最大边长
            - objects: 物体名称列表
    """
    merged_config = dict(DEFAULT_SCENE_CONFIG)
    if config:
        merged_config.update(config)

    _clear_scene()
    _setup_world_background(merged_config["world_background_color"])
    _create_ground_plane(
        merged_config["ground_size"],
        merged_config["ground_color"],
        merged_config["ground_roughness"],
    )

    objects, object_source = _load_object(merged_config)
    norm_info = _normalize_and_center_objects(objects, merged_config["normalize_size"])

    # 光照始终朝向物体归一化后的中心，保证光照条件与物体位置解耦、跨场景一致
    _setup_three_point_lighting(merged_config, target=norm_info["center"])

    scene_info = {
        "object_center": list(norm_info["center"]),
        "bbox_min": list(norm_info["bbox_min"]),
        "bbox_max": list(norm_info["bbox_max"]),
        "object_source": object_source,
        "normalize_size": merged_config["normalize_size"],
        "objects": [obj.name for obj in objects],
    }
    return scene_info


# ----------------------------------------------------------------------------
# 独立运行：场景预览
# ----------------------------------------------------------------------------
if __name__ == "__main__":
    preview_config = dict(DEFAULT_SCENE_CONFIG)
    preview_config["primitive_type"] = random.choice(BUILTIN_PRIMITIVES)
    info = build_scene(preview_config)
    print("场景搭建完成，元信息：")
    print(info)
