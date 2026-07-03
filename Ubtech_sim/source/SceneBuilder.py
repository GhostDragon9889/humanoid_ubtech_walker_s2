import os
import random
from typing_extensions import override
import isaacsim.core.utils.stage as stage_utils
from isaacsim.core.prims import Articulation, XFormPrim
import numpy as np
from isaacsim.core.cloner import Cloner
from functools import partial
import omni.replicator.core as rep
from pxr import UsdGeom
from pxr import Gf
import omni.usd
from isaacsim.core.prims import SingleRigidPrim, RigidPrim, Articulation

from Ubtech_sim.source.coordinate_utils import CoordinateTransform
from isaacsim.core.simulation_manager import SimulationManager
import json
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Any, Optional


class SceneBuilder:
    """Configurable scene builder driven by a YAML config dict."""

    def __init__(self, cfg, data_logger):
        self.cfg = cfg
        self.root_path = cfg['root_path']
        # created object references
        self.table = None
        self.plane = None
        self.multi_parts = None
        self.box = None
        self.foam = None
        self.robot = None
        self.robot_prim_path = None
        
        # 从配置中获取目标物体列表，默认为空列表
        self.target_objects = cfg.get('target_objects', [])
        # 存储各物体的prim路径
        self.table_prim_paths = []
        self.table_root_paths = []
        self._table_world_bounds = None
        self.box_prim_paths = []
        self.foam_prim_paths = []
        self.parts_prim_paths = []
        self.part_type_by_prim_path = {}
        self.pose_logger = data_logger
        self._physics_sim_view = SimulationManager.get_physics_sim_view()
        self.coordinate_transform = None  # initialized via init_coordinate_transform()
        # SimulationManager.initialize_physics()

    def _usd_path(self, relative):
        return os.path.join(self.root_path, relative)

    @staticmethod
    def _non_negative_int(value, field_name: str) -> int:
        try:
            count = int(value)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"{field_name} must be a non-negative integer, got {value!r}") from exc
        if count < 0:
            raise ValueError(f"{field_name} must be >= 0, got {count}")
        return count

    @staticmethod
    def _asset_pool_has_entries(pool) -> bool:
        if isinstance(pool, (list, tuple)):
            return len(pool) > 0
        return bool(pool)

    @staticmethod
    def _ensure_static_mesh_colliders(root_paths, label="static"):
        """Apply CollisionAPI to mesh prims under imported static assets."""
        try:
            from pxr import Usd, UsdGeom, UsdPhysics

            stage = omni.usd.get_context().get_stage()
            if stage is None:
                return

            applied = 0
            for root_path in root_paths:
                root = stage.GetPrimAtPath(root_path)
                if not root.IsValid():
                    continue
                for prim in Usd.PrimRange(root):
                    if prim.IsA(UsdGeom.Mesh):
                        UsdPhysics.CollisionAPI.Apply(prim)
                        applied += 1
            if applied:
                print(f"[SceneBuilder] {label}: ensured CollisionAPI on {applied} mesh prims")
        except Exception as exc:
            print(f"[SceneBuilder] {label}: failed to ensure static colliders: {exc}")

    def _physics_material(self, name, cfg):
        """Create or update a reusable USD physics material."""
        try:
            from pxr import UsdGeom, UsdPhysics

            stage = omni.usd.get_context().get_stage()
            if stage is None:
                return None

            UsdGeom.Scope.Define(stage, "/Root/PhysicsMaterials")
            mat_path = f"/Root/PhysicsMaterials/{name}"
            mat_prim = stage.DefinePrim(mat_path, "Material")
            mat_api = UsdPhysics.MaterialAPI.Apply(mat_prim)
            mat_api.CreateStaticFrictionAttr(float(cfg.get("static_friction", 1.8)))
            mat_api.CreateDynamicFrictionAttr(float(cfg.get("dynamic_friction", 1.4)))
            mat_api.CreateRestitutionAttr(float(cfg.get("restitution", 0.0)))
            return mat_prim
        except Exception as exc:
            print(f"[SceneBuilder] failed to create physics material {name}: {exc}")
            return None

    def _bind_physics_material_under_paths(self, root_paths, material_prim, label="physics"):
        """Bind a physics material to Xform/Mesh prims under one or more roots."""
        if material_prim is None:
            return
        try:
            from pxr import Usd, UsdGeom, UsdShade

            stage = omni.usd.get_context().get_stage()
            if stage is None:
                return
            if isinstance(root_paths, str):
                root_paths = [root_paths]
            material = UsdShade.Material(material_prim)
            bound = 0
            for root_path in root_paths:
                root = stage.GetPrimAtPath(root_path)
                if not root.IsValid():
                    continue
                for prim in Usd.PrimRange(root):
                    if prim.IsA(UsdGeom.Xform) or prim.IsA(UsdGeom.Mesh) or prim.IsA(UsdGeom.Cube):
                        UsdShade.MaterialBindingAPI.Apply(prim).Bind(material)
                        bound += 1
            if bound:
                print(f"[SceneBuilder] {label}: bound contact material on {bound} prims")
        except Exception as exc:
            print(f"[SceneBuilder] {label}: failed to bind contact material: {exc}")

    def _apply_contact_offsets_under_path(self, root_path, cfg, label="collision"):
        """Apply small contact/rest offsets to collision meshes where PhysX supports them."""
        try:
            from pxr import Usd, UsdGeom, UsdPhysics, Sdf

            stage = omni.usd.get_context().get_stage()
            if stage is None:
                return
            root = stage.GetPrimAtPath(root_path)
            if not root.IsValid():
                return

            contact_offset = cfg.get("contact_offset", 0.006)
            rest_offset = cfg.get("rest_offset", 0.0)
            applied = 0
            for prim in Usd.PrimRange(root):
                if not prim.HasAPI(UsdPhysics.CollisionAPI):
                    continue
                prim.CreateAttribute(
                    "physxCollision:contactOffset",
                    Sdf.ValueTypeNames.Float,
                ).Set(float(contact_offset))
                prim.CreateAttribute(
                    "physxCollision:restOffset",
                    Sdf.ValueTypeNames.Float,
                ).Set(float(rest_offset))
                applied += 1
            if applied:
                print(f"[SceneBuilder] {label}: applied contact offsets on {applied} collision meshes")
        except Exception as exc:
            print(f"[SceneBuilder] {label}: failed to apply contact offsets: {exc}")

    def _apply_part_contact_physics(self, prim, prim_path):
        """Tune part contact behavior for manual dexterous grasp attempts."""
        try:
            from pxr import Sdf

            contact_cfg = self.part_cfg.get("contact", {})
            material_prim = self._physics_material("GraspPartContact", contact_cfg)
            self._bind_physics_material_under_paths(prim_path, material_prim, label="Part")
            self._apply_contact_offsets_under_path(prim_path, contact_cfg, label="Part")

            damping_cfg = contact_cfg.get("damping", {})
            linear_damping = float(damping_cfg.get("linear", 0.25))
            angular_damping = float(damping_cfg.get("angular", 0.50))
            max_linear_velocity = float(contact_cfg.get("max_linear_velocity", 2.0))
            max_angular_velocity = float(contact_cfg.get("max_angular_velocity", 8.0))

            prim.CreateAttribute(
                "physxRigidBody:linearDamping",
                Sdf.ValueTypeNames.Float,
            ).Set(linear_damping)
            prim.CreateAttribute(
                "physxRigidBody:angularDamping",
                Sdf.ValueTypeNames.Float,
            ).Set(angular_damping)
            prim.CreateAttribute(
                "physxRigidBody:maxLinearVelocity",
                Sdf.ValueTypeNames.Float,
            ).Set(max_linear_velocity)
            prim.CreateAttribute(
                "physxRigidBody:maxAngularVelocity",
                Sdf.ValueTypeNames.Float,
            ).Set(max_angular_velocity)
        except Exception as exc:
            print(f"[SceneBuilder] Part: failed to tune contact physics for {prim_path}: {exc}")

    def _create_simple_block_part(self, stage, prim_path, part_scale):
        """Create a clean box rigid body for grasp-contact calibration."""
        from pxr import Gf, Sdf, UsdGeom, UsdPhysics

        cube = UsdGeom.Cube.Define(stage, prim_path)
        prim = cube.GetPrim()
        cube.CreateSizeAttr(1.0)
        dims = np.asarray(
            self.part_cfg.get("simple_block_size", [0.055, 0.04, 0.085]),
            dtype=np.float64,
        )
        if dims.size != 3:
            raise ValueError(f"part.simple_block_size must have 3 values, got {dims}")
        dims = dims * np.asarray(part_scale, dtype=np.float64)
        cube.CreateExtentAttr([
            Gf.Vec3f(-0.5, -0.5, -0.5),
            Gf.Vec3f(0.5, 0.5, 0.5),
        ])
        UsdPhysics.CollisionAPI.Apply(prim)
        prim.CreateAttribute(
            "primvars:displayColor",
            Sdf.ValueTypeNames.Color3fArray,
        ).Set([Gf.Vec3f(0.95, 0.95, 0.90)])
        return prim, dims

    @staticmethod
    def _choose_asset_from_pool(pool):
        if isinstance(pool, (list, tuple)):
            return random.choice(pool)
        return pool

    def _resolve_asset_path(self, asset):
        """Return a usable USD path for either relative or already-resolved assets."""
        asset_path = str(asset)
        if os.path.isabs(asset_path) or asset_path.startswith(str(self.root_path)):
            return asset_path
        return self._usd_path(asset_path)

    def _part_scale_for_type(self, part_type):
        """Return the configured visual/collision scale for a Task1/Task3 part."""
        default_scale = self.part_cfg.get('scale', [1.0, 1.0, 1.0])
        type_scales = self.part_cfg.get('scales_by_type', {})
        raw_scale = type_scales.get(part_type, default_scale) if part_type else default_scale

        if isinstance(raw_scale, (int, float)):
            scale = np.array([raw_scale, raw_scale, raw_scale], dtype=np.float64)
        else:
            scale = np.array(raw_scale, dtype=np.float64).flatten()
            if scale.size == 1:
                scale = np.repeat(scale, 3)
            elif scale.size != 3:
                raise ValueError(
                    f"part scale for {part_type or 'default'} must be a scalar or 3 values, "
                    f"got {raw_scale!r}"
                )

        if np.any(scale <= 0):
            raise ValueError(f"part scale for {part_type or 'default'} must be positive, got {scale}")
        return scale

    def _get_task1_part_plan(self):
        """Return validated Task1 part creation plan as (part_type, asset_pool, count)."""
        fallback_count = self.part_cfg.get('num_parts', 2)
        num_a = self._non_negative_int(
            self.part_cfg.get('num_parts_a', fallback_count),
            'part.num_parts_a',
        )
        num_b = self._non_negative_int(
            self.part_cfg.get('num_parts_b', fallback_count),
            'part.num_parts_b',
        )
        plan = [
            ('part_a', self.part_cfg.get('part_a_assets', []), num_a),
            ('part_b', self.part_cfg.get('part_b_assets', []), num_b),
        ]

        for part_type, pool, count in plan:
            if count > 0 and not self._asset_pool_has_entries(pool):
                raise ValueError(
                    f"Task1 config requires {part_type}_assets when "
                    f"num_parts_{part_type[-1]} is {count}"
                )
        return plan

    def build_table(self):
        self.table_cfg = self.cfg['table']
        self.table = rep.create.from_usd(
            self._usd_path(self.table_cfg['table_usd']),
            position=rep.distribution.choice(self.table_cfg['table_position'], with_replacements=False),
            scale=rep.distribution.choice(self.table_cfg['table_scale'], with_replacements=False),
            count=len(self.table_cfg['table_position']),
        )
        # 保存prim路径
        if self.table is not None:
            prims_info = self.table._get_prims()
            if 'primsIn' in prims_info:
                prims_in = prims_info['primsIn']
                if not isinstance(prims_in, (list, tuple)):
                    prims_in = [prims_in]
                self.table_root_paths = [str(prim.GetPath()) for prim in prims_in]
                self.table_prim_paths = [str(prim.GetPath()) + "/Ref/material" for prim in prims_in]
                self._ensure_static_mesh_colliders(self.table_prim_paths, label="table")
                table_contact_cfg = self.cfg.get("table", {}).get("contact", {})
                table_material = self._physics_material("GraspTableContact", table_contact_cfg)
                self._bind_physics_material_under_paths(
                    self.table_prim_paths,
                    table_material,
                    label="table",
                )
        return self.table

    def get_table_world_bounds(self):
        """Return the combined table AABB as (minimum, maximum) world XYZ arrays."""
        if self._table_world_bounds is not None:
            return self._table_world_bounds

        from pxr import Usd, UsdGeom

        stage = omni.usd.get_context().get_stage()
        if stage is None or not self.table_root_paths:
            return None

        bbox_cache = UsdGeom.BBoxCache(
            Usd.TimeCode.Default(),
            [UsdGeom.Tokens.default_, UsdGeom.Tokens.render, UsdGeom.Tokens.proxy],
            useExtentsHint=True,
        )
        minimum = np.full(3, np.inf, dtype=np.float64)
        maximum = np.full(3, -np.inf, dtype=np.float64)
        for path in self.table_root_paths:
            prim = stage.GetPrimAtPath(path)
            if not prim.IsValid():
                continue
            aligned_range = bbox_cache.ComputeWorldBound(prim).ComputeAlignedRange()
            minimum = np.minimum(minimum, np.asarray(aligned_range.GetMin(), dtype=np.float64))
            maximum = np.maximum(maximum, np.asarray(aligned_range.GetMax(), dtype=np.float64))

        if not np.all(np.isfinite(minimum)) or not np.all(np.isfinite(maximum)):
            return None
        self._table_world_bounds = (minimum, maximum)
        print(f"[SceneBuilder] Table world bounds: min={minimum}, max={maximum}")
        return self._table_world_bounds

    def get_parts_outside_table(self, edge_tolerance=0.02, fall_distance=0.08):
        """Return part paths whose centers have left or fallen below the table."""
        bounds = self.get_table_world_bounds()
        if bounds is None:
            return []

        minimum, maximum = bounds
        outside = []
        for part in self.get_parts_world_poses():
            position = np.asarray(part["position"], dtype=np.float64)
            outside_xy = (
                position[0] < minimum[0] - edge_tolerance
                or position[0] > maximum[0] + edge_tolerance
                or position[1] < minimum[1] - edge_tolerance
                or position[1] > maximum[1] + edge_tolerance
            )
            below_table = position[2] < maximum[2] - fall_distance
            if outside_xy or below_table:
                outside.append(part["prim_path"])
        return outside

    def build_ConveyorBelt(self):
        self.ConveyorBelt_cfg = self.cfg['ConveyorBelt']
        self.ConveyorBelt = rep.create.from_usd(
            self._usd_path(self.ConveyorBelt_cfg['ConveyorBelt_usd']),
            position=self.ConveyorBelt_cfg['ConveyorBelt_position'],
            scale=self.ConveyorBelt_cfg['ConveyorBelt_scale'],
        )
        # 保存传送带初始位置，用于重置
        self._conveyor_initial_position = self.ConveyorBelt_cfg['ConveyorBelt_position']

        #设置传送带初始速度
        Conveyor_speed=self.ConveyorBelt_cfg['ConveyorBelt_speed']
        self.set_conveyor_speed(Conveyor_speed)

        return self.ConveyorBelt
    #设置传送带表面速度
    def set_conveyor_speed(self,speed):
        def _set_conveyor_surface_velocity(velocity_vec3)-> None:
            import omni.usd
            from pxr import PhysxSchema
            stage = omni.usd.get_context().get_stage()
            for prim in stage.Traverse():
                if prim.HasAPI(PhysxSchema.PhysxSurfaceVelocityAPI):
                            PhysxSchema.PhysxSurfaceVelocityAPI(prim).GetSurfaceVelocityAttr().Set(
                                velocity_vec3
                            )
        try:
            from pxr import Gf
            _set_conveyor_surface_velocity(Gf.Vec3f(*speed))
            print(f"[SceneBuilder] 传送带已启动，设置速度为 {speed}")

        except Exception as e:
            print(f"[SceneBuilder] 传送带启动失败: {e}")

    def build_parts(self):
        self.plane_cfg = self.cfg['plane']
        self.part_cfg = self.cfg['part']
        num_planes = len(self.plane_cfg['plane_position'])
        self.planes = []
        for i in range(num_planes):
            self.planes.append(rep.create.plane(
                position=self.plane_cfg['plane_position'][i],
                scale=self.plane_cfg['plane_scale'][i],
                visible=False,
            ))

        part_usds = [self._usd_path(v) for k, v in self.part_cfg.items()
                     if k.endswith('_usd')]

        self.parts_list = []

        if self.cfg['task_number'] == 3:
            task3_part_assets = [
                self.part_cfg.get('part_a_assets', self.part_cfg.get('part_a_usd')),
                self.part_cfg.get('part_b_assets', self.part_cfg.get('part_b_usd')),
            ]
            if self.part_cfg.get('fixed_spawn', {}).get('enabled', False):
                parts_per_group = self.part_cfg.get('num_parts', 3)
                self.parts_prim_paths = []
                for i in range(len(self.box_cfg['box_position'])):
                    current_part_asset = task3_part_assets[i]
                    part_label = "part_a" if i == 0 else "part_b"
                    group_paths = [
                        f"/Root/Task3_{part_label}_{j:02d}"
                        for j in range(parts_per_group)
                    ]
                    created_paths = self._create_parts_at_paths(
                        part_pools=[(current_part_asset, parts_per_group, part_label)],
                        target_paths=group_paths,
                        plane_index=i,
                    )
                    self.parts_prim_paths.extend(created_paths)

                self._rigid_body_paths = []
                self._parts_rigid_prims = []
                self._rigid_prims_initialized = False
                self._initial_parts_prim_paths = list(self.parts_prim_paths)
                print(f"[SceneBuilder] Task3 fixed_spawn: 初始创建 {len(self.parts_prim_paths)} 个固定点工件")
            else:
                for i in range(len(self.box_cfg['box_position'])):
                # 使用索引 i 同时指定当前的零件和对应的平面
                # 箱子 0 对应零件 A，箱子 1 对应零件 B
                    current_part_usd = self._resolve_asset_path(
                        self._choose_asset_from_pool(task3_part_assets[i])
                    )
                    target_plane = self.planes[i]

                # 每次生成 num_parts 个该种类的工件
                    new_parts = rep.create.from_usd(
                    usd=current_part_usd,
                    count=self.part_cfg.get('num_parts', 3),
                    semantics={"class": "part"}
                    )

                    self.parts_list.append(new_parts)

                    with new_parts:
                        rep.physics.rigid_body(overwrite=True)
                        rep.physics.mass(mass=0.2)

                        # 随机旋转
                        rep.modify.pose(
                            rotation=rep.distribution.uniform((-90, -90, -90), (90, 90, 90))
                        )

                        # 散布到对应的箱子平面上
                        rep.randomizer.scatter_2d(
                            surface_prims=target_plane,
                            check_for_collisions=True,
                        )

                # 保存零件 prim 路径
                self._extract_parts_prim_paths()

        elif self.cfg['task_number'] == 2:
            from isaacsim.core.prims import RigidPrim

            self.part_A = stage_utils.add_reference_to_stage(
                usd_path=part_usds[0],
                prim_path='/Root/Part_A',
            )

            self.part_B = stage_utils.add_reference_to_stage(
                usd_path=part_usds[1],
                prim_path='/Root/Part_B',
            )

            self.cloner = Cloner()
            target_paths_A = self.cloner.generate_paths("/Root/Part_A", 3)
            self.cloner.clone(
                source_prim_path="/Root/Part_A",
                prim_paths=target_paths_A
            )

            target_paths_B = self.cloner.generate_paths("/Root/Part_B", 3)
            self.cloner.clone(
                source_prim_path="/Root/Part_B",
                prim_paths=target_paths_B
            )

            self.rigid_prim = RigidPrim(
                prim_paths_expr="/Root/Part_.*",
                name="rigid_prim_view"
            )

            random_indices = np.random.permutation(np.array([0, 1, 2, 3, 4, 5, 6, 7]))

            start_position = -0.3 - 8 * self.part_cfg['part_distance']

            init_positions = np.column_stack([
                np.linspace(start_position, -0.3, 8),
                np.full(8, 0.278),
                np.full(8, 0.98),
            ])

            self.rigid_prim.set_world_poses(
                positions=init_positions,
                indices=random_indices
            )

            # 保存 Task2 初始位姿，用于 reset
            self._task2_initial_positions = init_positions.copy()
            self._task2_initial_indices = random_indices.copy()



        elif self.cfg['task_number'] == 1:
            self.parts_list = []
            self.part_type_by_prim_path = {}
            target_paths = []
            part_pools = []
            for part_type, asset_pool, count in self._get_task1_part_plan():
                count = self._non_negative_int(count, f"part.num_parts_{part_type[-1]}")
                start_idx = len(target_paths)
                target_paths.extend(
                    f"/Root/Task1_{part_type}_{start_idx + j:02d}"
                    for j in range(count)
                )
                part_pools.append((asset_pool, count, part_type))

            if target_paths:
                created_paths = self._create_parts_at_paths(
                    part_pools=part_pools,
                    target_paths=target_paths,
                    plane_index=0,
                )
                self.parts_prim_paths = created_paths
                self._initial_parts_prim_paths = list(self.parts_prim_paths)
                self._rigid_body_paths = []
                self._parts_rigid_prims = []
                self._rigid_prims_initialized = False
                print(f"[SceneBuilder] Task1 随机非重叠创建 {len(self.parts_prim_paths)} 个零件")
            else:
                self.parts_group = None
                self.parts_prim_paths = []
                self._initial_parts_prim_paths = []
                print("[SceneBuilder] Task1 配置零件总数为 0，跳过零件创建")

    def _extract_parts_prim_paths(self):
        """从 replicator 节点中提取零件的 USD prim 路径。

        流程（对齐 NVIDIA 官方推荐模式）：
        1. Replicator 负责创建物体（build_parts）
        2. 此处仅提取路径，不创建 SingleRigidPrim
        3. SingleRigidPrim 延迟到首次使用时创建（world.reset() 之后），
           避免在 world.reset() 期间 Replicator 重建 prim 导致 tensor view 失效
        """
        self.parts_prim_paths = []
        self._rigid_body_paths = []
        self._parts_rigid_prims = []
        self._rigid_prims_initialized = False
        self.part_type_by_prim_path = {}
        part_type_by_rep_id = getattr(self, '_part_type_by_rep_id', {})

        for part_rep in self.parts_list:
            try:
                prims_info = part_rep._get_prims()
                if 'primsIn' in prims_info:
                    prims_in = prims_info['primsIn']
                    if not isinstance(prims_in, (list, tuple)):
                        prims_in = [prims_in]
                    for prim in prims_in:
                        path = str(prim.GetPath())
                        self.parts_prim_paths.append(path)
                        part_type = part_type_by_rep_id.get(id(part_rep))
                        if part_type is not None:
                            self.part_type_by_prim_path[path] = part_type
            except Exception as e:
                print(f"[SceneBuilder] 提取零件路径失败: {e}")

        # 永久保存初始路径，重置时复用
        self._initial_parts_prim_paths = list(self.parts_prim_paths)
        print(f"[SceneBuilder] 提取到 {len(self.parts_prim_paths)} 个零件路径: {self.parts_prim_paths}")

    def _ensure_rigid_prims(self):
        """确保 SingleRigidPrim 缓存已创建（延迟初始化）。

        必须在 world.reset() 之后调用，否则 Replicator 重建 prim 会导致 tensor view 失效。
        """
        if getattr(self, '_rigid_prims_initialized', False):
            return
        self._rebuild_rigid_prims(self.parts_prim_paths)

    def _rebuild_rigid_prims(self, prim_paths: list):
        """根据给定路径列表，查找 RigidBodyAPI 并缓存 SingleRigidPrim。

        统一被 _ensure_rigid_prims / _randomize_task1_assets 调用。
        """
        import omni.usd
        from pxr import UsdPhysics
        stage = omni.usd.get_context().get_stage()

        self._rigid_body_paths = []
        self._parts_rigid_prims = []

        for path in prim_paths:
            rb_path = self._find_rigid_body_path(stage, path)
            self._rigid_body_paths.append(rb_path)
            if rb_path is not None:
                try:
                    self._parts_rigid_prims.append(SingleRigidPrim(prim_path=rb_path))
                except Exception as e:
                    print(f"[SceneBuilder] 创建 SingleRigidPrim 失败 {rb_path}: {e}")
                    self._parts_rigid_prims.append(None)
            else:
                self._parts_rigid_prims.append(None)

        self._rigid_prims_initialized = True
        ok = sum(1 for r in self._parts_rigid_prims if r is not None)
        print(f"[SceneBuilder] 发现 {len(prim_paths)} 个零件 ({ok} 个有 rigid body): {prim_paths}")

    @staticmethod
    def _find_rigid_body_path(stage, base_path: str):
        """在 base_path 及其直接子 prim 中查找带 UsdPhysics.RigidBodyAPI 的路径。

        返回找到的路径，找不到返回 None。
        """
        from pxr import UsdPhysics
        prim = stage.GetPrimAtPath(base_path)
        if not prim.IsValid():
            return None
        # 优先检查自身
        if prim.HasAPI(UsdPhysics.RigidBodyAPI) and SceneBuilder._is_xformable_prim(prim):
            return base_path
        # 检查直接子 prim
        for child in prim.GetChildren():
            if child.HasAPI(UsdPhysics.RigidBodyAPI) and SceneBuilder._is_xformable_prim(child):
                return str(child.GetPath())
        return None

    @staticmethod
    def _is_xformable_prim(prim) -> bool:
        if prim is None or not prim.IsValid():
            return False
        try:
            return bool(UsdGeom.Xformable(prim))
        except Exception:
            return False

    @staticmethod
    def _remove_rigid_body_api(prim) -> bool:
        try:
            from pxr import UsdPhysics
            if prim.HasAPI(UsdPhysics.RigidBodyAPI):
                prim.RemoveAPI(UsdPhysics.RigidBodyAPI)
                return True
        except Exception:
            pass
        return False

    @classmethod
    def _apply_rigid_body_to_prim(cls, prim, mass=None, enabled=True):
        """Apply physics only to xformable prims, then remove nested rigid bodies."""
        from pxr import UsdPhysics

        if not cls._is_xformable_prim(prim):
            return None

        rb_api = UsdPhysics.RigidBodyAPI.Apply(prim)
        rb_api.CreateRigidBodyEnabledAttr(enabled)
        if mass is not None:
            UsdPhysics.MassAPI.Apply(prim).CreateMassAttr(mass)
        return rb_api

    @classmethod
    def _sanitize_rigid_bodies_under_paths(cls, stage, root_paths, label="RigidBody"):
        """Remove RigidBodyAPI from non-xformable prims and nested descendants."""
        from pxr import Usd, UsdPhysics

        if isinstance(root_paths, str):
            root_paths = [root_paths]

        removed_non_xformable = 0
        removed_nested = 0
        for root_path in root_paths:
            root_prim = stage.GetPrimAtPath(root_path)
            if not root_prim.IsValid():
                continue

            for prim in Usd.PrimRange(root_prim):
                has_rigid_body = prim.HasAPI(UsdPhysics.RigidBodyAPI)
                if has_rigid_body and not cls._is_xformable_prim(prim):
                    if cls._remove_rigid_body_api(prim):
                        removed_non_xformable += 1
                    continue

                has_rigid_ancestor = False
                parent = prim.GetParent()
                while parent.IsValid():
                    if parent.HasAPI(UsdPhysics.RigidBodyAPI):
                        has_rigid_ancestor = True
                        break
                    if str(parent.GetPath()) == root_path:
                        break
                    parent = parent.GetParent()

                if has_rigid_body and has_rigid_ancestor:
                    if cls._remove_rigid_body_api(prim):
                        removed_nested += 1

        if removed_non_xformable or removed_nested:
            print(
                f"[SceneBuilder] {label}: removed invalid RigidBodyAPI "
                f"(non-xformable={removed_non_xformable}, nested={removed_nested})"
            )

    def get_parts_world_poses(self):
        """查询所有零件当前的世界坐标位姿 (通过 USD XformCache)

        Returns:
            list of dict: [{'prim_path': str,
                            'position': [x, y, z],
                            'orientation': [qx, qy, qz, qw]}, ...]
        """
        from pxr import UsdGeom, Usd
        import omni.usd
        stage = omni.usd.get_context().get_stage()
        # Create fresh XformCache with current time to get updated transforms
        xform_cache = UsdGeom.XformCache(Usd.TimeCode.Default())

        results = []
        for prim_path in self.parts_prim_paths:
            try:
                prim = stage.GetPrimAtPath(prim_path)
                if not prim.IsValid():
                    print(f"[SceneBuilder] prim 无效: {prim_path}")
                    continue
                world_tf = xform_cache.GetLocalToWorldTransform(prim)
                t = world_tf.ExtractTranslation()
                q = world_tf.ExtractRotationQuat()
                qi = q.GetImaginary()
                qr = q.GetReal()
                pos = [float(t[0]), float(t[1]), float(t[2])]
                results.append({
                    'prim_path': prim_path,
                    'position': pos,
                    'orientation': [float(qi[0]), float(qi[1]), float(qi[2]), float(qr)],
                })
                # Debug: print if position is still [0,0,0]
                if abs(pos[0]) < 1e-6 and abs(pos[1]) < 1e-6 and abs(pos[2]) < 1e-6:
                    print(f"[SceneBuilder] 警告: {prim_path} 位置仍为 [0,0,0]，可能是容器节点而非实际物体")
            except Exception as e:
                print(f"[SceneBuilder] 查询 {prim_path} 位姿失败: {e}")
        return results

    # ===================== 统一物体位姿接口（供数据采集/回放使用） =====================
    @staticmethod
    def compute_num_tracked_objects(task_cfg: dict) -> int:
        """根据任务配置计算被追踪物体的数量（不需要场景实例，可在 connect 前调用）"""
        task = task_cfg.get('task_number', 0)
        if task == 1:
            part_cfg = task_cfg.get('part', {})
            fallback_count = part_cfg.get('num_parts', 2)
            num_a = SceneBuilder._non_negative_int(
                part_cfg.get('num_parts_a', fallback_count),
                'part.num_parts_a',
            )
            num_b = SceneBuilder._non_negative_int(
                part_cfg.get('num_parts_b', fallback_count),
                'part.num_parts_b',
            )
            return num_a + num_b
        elif task == 2:
            return task_cfg.get('part', {}).get('num_parts', 5) * 2
        elif task == 3:
            num_boxes = len(task_cfg.get('box', {}).get('box_position', []))
            num_parts = task_cfg.get('part', {}).get('num_parts', 3)
            return num_boxes * num_parts
        elif task == 4:
            return 0  
        return 0

    def get_object_poses_flat(self) -> np.ndarray:
        """返回所有被追踪物体的位姿，展平为一维数组。

        每个物体 7 个值: [x, y, z, qx, qy, qz, qw]
        总维度 = num_objects * 7

        Returns:
            np.ndarray: shape (num_objects * 7,), dtype float32
        """
        task = self.cfg.get('task_number', 0)

        if task in (1, 3):
            poses = self.get_parts_world_poses()
            result = []
            for p in poses:
                result.extend(p['position'])       # [x, y, z]
                result.extend(p['orientation'])     # [qx, qy, qz, qw]
            return np.array(result, dtype=np.float32)

        elif task == 2:
            if not hasattr(self, 'rigid_prim') or self.rigid_prim is None:
                n = self.compute_num_tracked_objects(self.cfg)
                return np.zeros(n * 7, dtype=np.float32)
            positions, orientations = self.rigid_prim.get_world_poses()
            # Isaac Sim 返回 wxyz，转换为 xyzw 与 Task1/3 一致
            result = []
            for i in range(positions.shape[0]):
                result.extend(positions[i].tolist())
                w, x, y, z = orientations[i].tolist()
                result.extend([x, y, z, w])
            return np.array(result, dtype=np.float32)

        elif task == 4:
            if not hasattr(self, 'box_articulation') or self.box_articulation is None:
                return np.zeros(7, dtype=np.float32)
            pos, ori = self.box_articulation.get_world_poses()
            w, x, y, z = ori[0].tolist()
            result = list(pos[0].tolist()) + [x, y, z, w]
            return np.array(result, dtype=np.float32)

        return np.array([], dtype=np.float32)

    def set_object_poses_from_flat(self, flat_poses: np.ndarray) -> None:
        """从展平的位姿数组恢复物体位置（用于 replay 初始化场景）。

        Args:
            flat_poses: shape (num_objects * 7,)，格式 [x,y,z,qx,qy,qz,qw, ...]
        """
        from pxr import UsdGeom, Gf
        import omni.usd

        task = self.cfg.get('task_number', 0)
        num_objects = self.compute_num_tracked_objects(self.cfg)
        if flat_poses.shape[0] != num_objects * 7:
            print(f"[SceneBuilder] set_object_poses_from_flat: 维度不匹配 "
                  f"(期望 {num_objects * 7}, 得到 {flat_poses.shape[0]})")
            return

        poses_7 = flat_poses.reshape(num_objects, 7)  # (N, 7)

        if task in (1, 3):
            # 延迟初始化 SingleRigidPrim（确保在 world.reset() 之后）
            self._ensure_rigid_prims()
            rigid_prims = getattr(self, '_parts_rigid_prims', [])
            count = min(num_objects, len(self.parts_prim_paths))
            positions = poses_7[:count, :3].astype(np.float64)
            # xyzw → wxyz
            quats = poses_7[:count, 3:].astype(np.float64)
            orientations = np.column_stack([quats[:, 3], quats[:, 0], quats[:, 1], quats[:, 2]])

            restored = 0
            for i in range(count):
                if i < len(rigid_prims) and rigid_prims[i] is not None:
                    rigid_prims[i].set_world_pose(
                        position=positions[i], orientation=orientations[i])
                    rigid_prims[i].set_linear_velocity(np.zeros(3))
                    rigid_prims[i].set_angular_velocity(np.zeros(3))
                    restored += 1
                else:
                    print(f"[SceneBuilder] set_object_poses: 无 rigid body {self.parts_prim_paths[i]}")
            print(f"[SceneBuilder] 已从 flat 数据恢复 {restored}/{count} 个物体位姿")

        elif task == 2:
            # Task 2: 确保 rigid_prim 视图有效（可能在 world.reset() 后失效）
            if not hasattr(self, 'rigid_prim') or self.rigid_prim is None:
                # 尝试重新创建 rigid_prim 视图
                from isaacsim.core.prims import RigidPrim
                num_parts = self.part_cfg.get('num_parts', 5)
                clone_paths = []
                for i in range(num_parts):
                    clone_paths.append(f"/Root/Part_A{i}")
                    clone_paths.append(f"/Root/Part_B{i}")
                try:
                    self.rigid_prim = RigidPrim(
                        prim_paths_expr="/Root/Part_.*",
                        name="rigid_prim_view"
                    )
                    print(f"[SceneBuilder] Task 2: 重新创建了 rigid_prim 视图")
                except Exception as e:
                    print(f"[SceneBuilder] Task 2: 重新创建 rigid_prim 失败：{e}")
                    return
            positions = poses_7[:, :3].astype(np.float64)
            # xyzw → wxyz
            quats = poses_7[:, 3:].astype(np.float64)
            orientations = np.column_stack([quats[:, 3], quats[:, 0], quats[:, 1], quats[:, 2]])
            self.rigid_prim.set_world_poses(positions=positions, orientations=orientations)
            self.rigid_prim.set_velocities(velocities=np.zeros((num_objects, 6)))

        elif task == 4:
            if not hasattr(self, 'box_articulation') or self.box_articulation is None:
                return
            x, y, z, qx, qy, qz, qw = poses_7[0]
            pos = np.array([[x, y, z]], dtype=np.float64)
            ori = np.array([[qw, qx, qy, qz]], dtype=np.float64)  # wxyz
            self.box_articulation.set_world_poses(positions=pos, orientations=ori)
            if hasattr(self.box_articulation, 'set_velocities'):
                self.box_articulation.set_velocities(velocities=np.zeros((1, 6)))

        print(f"[SceneBuilder] 已从 flat 数据恢复 {num_objects} 个物体位姿")

    def set_object_poses_from_json(self, json_file_path: Optional[str] = None) -> None:
        """从 JSON 读取位姿，并在 world.reset() 之后用 SingleRigidPrim 恢复。

        设计对齐 _scatter_parts_direct：
        - 使用 SingleRigidPrim.set_world_pose 直接写入物理对象
        - 同步清零线速度/角速度，避免继承上一轮动力学状态

        JSON 格式：
        - prim_path: 物体路径
        - position: [x, y, z]
        - orientation: [qx, qy, qz, qw] (xyzw)
        """
        repo_root = Path(__file__).parent.parent.parent / "outputs"
        if json_file_path is None:
            pose_path = repo_root / "task1_parts_poses_init.json"
        else:
            p = Path(json_file_path).expanduser()
            pose_path = p.resolve() if p.is_absolute() else (repo_root / p).resolve()

        if not pose_path.exists():
            print(f"[SceneBuilder] set_parts_world_poses: JSON 不存在，跳过恢复: {pose_path}")
            return

        try:
            with open(pose_path, 'r', encoding='utf-8') as f:
                pose_configs = json.load(f)
        except Exception as e:
            print(f"[SceneBuilder] set_parts_world_poses: 读取 JSON 失败: {e}")
            return

        if not isinstance(pose_configs, list):
            print("[SceneBuilder] set_parts_world_poses: JSON 格式错误，期望 list")
            return

        # 先保证刚体缓存已基于当前 prim 重建（需要在 world.reset() 之后调用）
        self._ensure_rigid_prims()
        rigid_prims = getattr(self, '_parts_rigid_prims', [])

        # 使用 prim_path 作为主键匹配；同时兼容 rb_path 被记录到 JSON 的情况
        pose_by_path = {}
        for cfg in pose_configs:
            prim_path = cfg.get('prim_path')
            pos = cfg.get('position')
            orient = cfg.get('orientation')
            if (not isinstance(prim_path, str)
                    or not isinstance(pos, (list, tuple))
                    or not isinstance(orient, (list, tuple))
                    or len(pos) != 3
                    or len(orient) != 4):
                continue
            pose_by_path[prim_path] = (pos, orient)

        if not pose_by_path:
            print("[SceneBuilder] set_parts_world_poses: JSON 中没有有效位姿")
            return

        restored = 0
        attempted = 0
        for i, part_path in enumerate(self.parts_prim_paths):
            rigid = rigid_prims[i] if i < len(rigid_prims) else None
            if rigid is None:
                print(f"[SceneBuilder] set_parts_world_poses: 无缓存刚体 {part_path}")
                continue

            rb_path = None
            if hasattr(self, '_rigid_body_paths') and i < len(self._rigid_body_paths):
                rb_path = self._rigid_body_paths[i]

            pose = pose_by_path.get(part_path)
            if pose is None and rb_path is not None:
                pose = pose_by_path.get(rb_path)
            if pose is None:
                continue

            attempted += 1
            pos, orient_xyzw = pose
            try:
                position = np.array(pos, dtype=np.float64)
                qx, qy, qz, qw = orient_xyzw
                orientation = np.array([qw, qx, qy, qz], dtype=np.float64)  # wxyz
                rigid.set_world_pose(position=position, orientation=orientation)
                rigid.set_linear_velocity(np.zeros(3))
                rigid.set_angular_velocity(np.zeros(3))
                restored += 1
            except Exception as e:
                print(f"[SceneBuilder] set_parts_world_poses: 恢复 {part_path} 失败: {e}")

        print(
            f"[SceneBuilder] set_parts_world_poses: 从 {pose_path.name} 恢复 {restored}/{attempted} 个零件位姿"
        )

    def build_box(self):
        self.box_cfg = self.cfg['box']
        num_boxes = len(self.box_cfg['box_position'])

        self.box = stage_utils.add_reference_to_stage(
            usd_path=self._usd_path(self.box_cfg['box_usd']),
            prim_path='/Root/Box',
        )
        self.box_prim_paths = ['/Root/Box']

        if self.cfg['task_number'] <= 3:
            from isaacsim.core.cloner import Cloner
            self.cloner = Cloner()
            target_paths = self.cloner.generate_paths("/Root/Box", num_boxes - 1)
            self.cloner.clone(
                source_prim_path="/Root/Box",
                prim_paths=target_paths
            )
            self.box_prim_paths = ['/Root/Box'] + list(target_paths)
            stage = omni.usd.get_context().get_stage()
            self._sanitize_rigid_bodies_under_paths(stage, self.box_prim_paths, "Box")

            self.boxes = XFormPrim(
                prim_paths_expr='/Root/Box.*',
                positions=np.array(self.box_cfg['box_position']),
                scales=np.array(self.box_cfg['box_scale']),
            )

            # 箱子位置锁定：将刺有物理属性的箱子设为 kinematic
            if self.box_cfg.get('lock_boxes', False):
                self._lock_box_positions()

        elif self.cfg['task_number'] == 4:
            self.box_articulation = Articulation(
                prim_paths_expr='/Root/Box',
                positions=np.array([
                    self.box_cfg['box_position'],
                ]),
                scales=np.array([
                    self.box_cfg['box_scale'],
                ]),
                name='Box',
            )
            self.box_initial_joint_positions = self.box_articulation.get_joint_positions()
            self._box_initial_world_pos, self._box_initial_world_ori = self.box_articulation.get_world_poses()

        return self.box

    def _lock_rigid_bodies_under_path(self, path_prefix: str, label: str) -> None:
        """Set all rigid bodies under a prim path prefix to kinematic."""
        try:
            import omni.usd
            from pxr import UsdPhysics
            stage = omni.usd.get_context().get_stage()
            self._sanitize_rigid_bodies_under_paths(stage, path_prefix, label)
            locked = 0
            for prim in stage.Traverse():
                path = str(prim.GetPath())
                if not path.startswith(path_prefix):
                    continue
                if prim.HasAPI(UsdPhysics.RigidBodyAPI) and self._is_xformable_prim(prim):
                    rb_api = UsdPhysics.RigidBodyAPI(prim)
                    rb_api.CreateKinematicEnabledAttr(True)
                    locked += 1
            print(f"[SceneBuilder] {label} 锁定完成: {locked} 个 RigidBody 已设为 kinematic")
        except Exception as e:
            print(f"[SceneBuilder] {label} 锁定失败: {e}")

    def _lock_box_positions(self):
        """将 /Root/Box* 下所有 RigidBody prim 设为 kinematic，禁止物理引擎移动箱子。"""
        try:
            import omni.usd
            from pxr import UsdPhysics, Usd
            stage = omni.usd.get_context().get_stage()
            box_paths = getattr(self, 'box_prim_paths', None) or ['/Root/Box']
            self._sanitize_rigid_bodies_under_paths(stage, box_paths, "Box")
            locked = 0
            for prim in stage.Traverse():
                path = str(prim.GetPath())
                if not path.startswith("/Root/Box"):
                    continue
                if prim.HasAPI(UsdPhysics.RigidBodyAPI) and self._is_xformable_prim(prim):
                    rb_api = UsdPhysics.RigidBodyAPI(prim)
                    rb_api.CreateKinematicEnabledAttr(True)
                    locked += 1
            print(f"[SceneBuilder] 箱子锁定完成: {locked} 个 RigidBody 已设为 kinematic")
        except Exception as e:
            print(f"[SceneBuilder] 箱子锁定失败: {e}")

    def _lock_foam_positions(self):
        """Set the foam rigid bodies to kinematic so physics cannot move the foam."""
        if not self.foam_prim_paths:
            print("[SceneBuilder] Foam 锁定跳过: 未找到 foam prim 路径")
            return
        for foam_path in self.foam_prim_paths:
            self._lock_rigid_bodies_under_path(foam_path, "Foam")

    def _apply_robot_pose(self):
        """将 YAML 中 robot_position / robot_rotation 写入机器人 USD 内的
        world-anchored PhysicsFixedJoint（localPos0 / localRot0）。

        rep.create.from_usd 只设置 XForm 变换；物理引擎启动后
        FixedJoint 会用 USD 内的硬编码值覆盖该变换。
        此方法在物理启动前将关节目标坐标改写为 YAML 值，从而让
        robot_position / robot_rotation 真正生效。
        """
        import omni.usd
        import math
        from pxr import Gf, Usd

        robot_cfg = self.cfg['robot']
        pos = robot_cfg['robot_position']                    # [x, y, z]  metres
        rot_deg = robot_cfg.get('robot_rotation', [0, 0, 0]) # [roll, pitch, yaw] degrees ZYX

        # Euler ZYX (degrees) → unit quaternion (w, x, y, z)
        r_rad, p_rad, y_rad = (math.radians(a) for a in rot_deg)
        cy, sy = math.cos(y_rad * 0.5), math.sin(y_rad * 0.5)
        cp, sp = math.cos(p_rad * 0.5), math.sin(p_rad * 0.5)
        cr, sr = math.cos(r_rad * 0.5), math.sin(r_rad * 0.5)
        qw = cr*cp*cy + sr*sp*sy
        qx = sr*cp*cy - cr*sp*sy
        qy = cr*sp*cy + sr*cp*sy
        qz = cr*cp*sy - sr*sp*cy
        
        stage = omni.usd.get_context().get_stage()
        n_updated = 0
        for prim in stage.Traverse():
            if prim.GetTypeName() != "PhysicsFixedJoint":
                continue
            # 只处理 world-anchored 关节 (body0 无目标 = 锚定到世界)
            body0_rel = prim.GetRelationship("physics:body0")
            if body0_rel.IsValid() and body0_rel.GetTargets():
                continue
            # 覆盖世界侧附着点
            pos_attr = prim.GetAttribute("physics:localPos0")
            rot_attr = prim.GetAttribute("physics:localRot0")
            if pos_attr.IsValid():
                pos_attr.Set(Gf.Vec3f(float(pos[0]), float(pos[1]), float(pos[2])))
            if rot_attr.IsValid():
                rot_attr.Set(Gf.Quatf(float(qw), float(qx), float(qy), float(qz)))
            print(f"[SceneBuilder] FixedJoint 位姿已更新: {prim.GetPath()}  "
                  f"pos={[round(v, 4) for v in pos]}  rot_deg={rot_deg}")
            n_updated += 1

        if n_updated == 0:
            print("[SceneBuilder] 警告: 未找到 world-anchored PhysicsFixedJoint，"
                  "robot_position/rotation 可能无法通过 YAML 生效")
        else:
            print(f"[SceneBuilder] 已覆盖 {n_updated} 个 FixedJoint — "
                  f"robot 将在 YAML 指定位置 {pos} 生成")

    def _repo_root(self) -> Path:
        """Directory that contains WalkerS2-Model/ (zollent_technology on host)."""
        env_root = os.environ.get("ZOLLENT_REPO_ROOT")
        if env_root:
            return Path(env_root)

        docker_root = Path("/workspace")
        if (docker_root / "WalkerS2-Model").is_dir():
            return docker_root

        baseline = Path(__file__).resolve().parents[2]
        sibling = baseline.parent / "WalkerS2-Model"
        if sibling.is_dir():
            return baseline.parent

        for parent in Path(__file__).resolve().parents:
            if (parent / "WalkerS2-Model").is_dir():
                return parent

        return baseline.parent

    def _resolve_robot_urdf_path(self, urdf_cfg_value: str) -> str:
        if os.path.isabs(urdf_cfg_value) and os.path.isfile(urdf_cfg_value):
            return urdf_cfg_value

        candidates = [
            self._repo_root() / urdf_cfg_value,
            Path(self.root_path) / urdf_cfg_value,
            Path(__file__).resolve().parents[2] / urdf_cfg_value,
        ]
        tried = []
        for candidate in candidates:
            tried.append(str(candidate))
            if candidate.is_file():
                return str(candidate.resolve())

        raise FileNotFoundError(
            f"Robot URDF not found: {urdf_cfg_value!r} (tried {', '.join(tried)})"
        )

    @staticmethod
    def _robot_xform_path(stage, articulation_root: str) -> str:
        if articulation_root.endswith("/base_link"):
            parent = stage.GetPrimAtPath(articulation_root).GetParent()
            if parent.IsValid() and parent.GetName() not in ("World", ""):
                return str(parent.GetPath())
        return articulation_root

    @staticmethod
    def _apply_prim_world_pose(stage, prim_path: str, position, rotation_deg) -> None:
        from pxr import Gf, UsdGeom

        prim = stage.GetPrimAtPath(prim_path)
        if not prim.IsValid():
            raise RuntimeError(f"Robot xform prim not found: {prim_path}")

        roll_deg, pitch_deg, yaw_deg = (float(v) for v in rotation_deg)
        xformable = UsdGeom.Xformable(prim)
        xformable.ClearXformOpOrder()
        xformable.AddTranslateOp().Set(
            Gf.Vec3d(float(position[0]), float(position[1]), float(position[2]))
        )
        # USD xform rotate ops are authored in degrees. The task YAML also stores
        # robot_rotation in degrees, matching replicator's from_usd rotation API.
        xformable.AddRotateXYZOp().Set(
            Gf.Vec3f(
                roll_deg,
                pitch_deg,
                yaw_deg,
            )
        )

    @staticmethod
    def _fix_robot_base_to_world(stage, articulation_root: str) -> str:
        from pxr import Gf, Sdf, Usd, UsdGeom, UsdPhysics

        if articulation_root.endswith("/base_link"):
            base_path = articulation_root
            xform_path = SceneBuilder._robot_xform_path(stage, articulation_root)
        else:
            base_path = f"{articulation_root}/base_link"
            xform_path = articulation_root

        base_prim = stage.GetPrimAtPath(base_path)
        if not base_prim.IsValid():
            raise RuntimeError(f"Robot base_link not found: {base_path}")

        fixed_joint_path = f"{xform_path}/world_fixed_joint"
        existing = stage.GetPrimAtPath(fixed_joint_path)
        if existing.IsValid():
            stage.RemovePrim(fixed_joint_path)

        fixed_joint = UsdPhysics.FixedJoint.Define(stage, Sdf.Path(fixed_joint_path))
        fixed_joint.GetBody1Rel().SetTargets([Sdf.Path(base_path)])

        root_transform = UsdGeom.XformCache(Usd.TimeCode.Default()).GetLocalToWorldTransform(base_prim)
        root_translation = root_transform.ExtractTranslation()
        root_quat = root_transform.ExtractRotation().GetQuat()
        root_quat_imag = root_quat.GetImaginary()
        fixed_joint.CreateLocalPos0Attr(
            Gf.Vec3f(
                float(root_translation[0]),
                float(root_translation[1]),
                float(root_translation[2]),
            )
        )
        fixed_joint.CreateLocalRot0Attr(
            Gf.Quatf(
                float(root_quat.GetReal()),
                float(root_quat_imag[0]),
                float(root_quat_imag[1]),
                float(root_quat_imag[2]),
            )
        )
        fixed_joint.CreateLocalPos1Attr(Gf.Vec3f(0.0, 0.0, 0.0))
        fixed_joint.CreateLocalRot1Attr(Gf.Quatf(1.0, 0.0, 0.0, 0.0))
        print(f"[SceneBuilder] Fixed base_link to world via {fixed_joint_path}")
        return fixed_joint_path

    @staticmethod
    def _remove_stale_scene_robot_prims(stage) -> None:
        """Drop embedded robot placeholders from warehouse scenes before URDF import."""
        stale_paths = (
            "/walker_s2_official",
            "/World/walker_s2_official",
            "/Root/walker_s2_official",
        )
        removed = set()
        for path in stale_paths:
            prim = stage.GetPrimAtPath(path)
            if prim.IsValid():
                stage.RemovePrim(path)
                removed.add(path)
                print(f"[SceneBuilder] Removed stale scene robot prim: {path}")

        # Also remove any other top-level walker_s2_official xform the scene may define.
        for prim in stage.Traverse():
            if prim.GetName() != "walker_s2_official":
                continue
            path = str(prim.GetPath())
            if path in removed or path.endswith("/base_link"):
                continue
            if path.count("/") <= 2:
                stage.RemovePrim(path)
                removed.add(path)
                print(f"[SceneBuilder] Removed stale scene robot prim: {path}")

    def _build_robot_from_urdf(self):
        """Import official Walker S2 URDF at the YAML spawn pose (fixed base)."""
        import omni.kit.commands
        from isaacsim.core.utils.extensions import enable_extension
        from pxr import Usd

        robot_cfg = self.cfg["robot"]
        urdf_path = self._resolve_robot_urdf_path(robot_cfg["robot_urdf"])
        position = robot_cfg["robot_position"]
        rotation = robot_cfg.get("robot_rotation", [0, 0, 0])

        enable_extension("isaacsim.asset.importer.urdf")
        try:
            from isaacsim.asset.importer.urdf import _urdf
        except ImportError:
            import isaacsim.asset.importer.urdf as _urdf_mod

            _urdf = _urdf_mod._urdf

        stage = omni.usd.get_context().get_stage()
        # Warehouse USD already contains /walker_s2_official/* payload slots for the
        # old bundled robot; URDF import uses the same root name and crashes if kept.
        self._remove_stale_scene_robot_prims(stage)

        _, import_config = omni.kit.commands.execute("URDFCreateImportConfig")
        import_config.merge_fixed_joints = False
        # The official teleop task treats Walker S2 as a fixed-base upper-body
        # manipulation robot. Let the URDF importer anchor the articulation at
        # creation time instead of adding a separate world joint after placement.
        import_config.fix_base = True
        import_config.import_inertia_tensor = True
        import_config.self_collision = False
        import_config.make_default_prim = False
        import_config.create_physics_scene = False
        import_config.default_drive_type = _urdf.UrdfJointTargetType.JOINT_DRIVE_POSITION
        import_config.default_drive_strength = 1e4
        import_config.default_position_drive_damping = 1e3

        print(f"[SceneBuilder] Importing robot URDF: {urdf_path}")
        result = omni.kit.commands.execute(
            "URDFParseAndImportFile",
            urdf_path=urdf_path,
            import_config=import_config,
            dest_path="",
            get_articulation_root=True,
        )
        if isinstance(result, tuple):
            _, articulation_root = result
        else:
            articulation_root = result

        if not articulation_root:
            raise RuntimeError("URDF import failed (empty articulation root)")

        xform_path = self._robot_xform_path(stage, articulation_root)
        self._apply_prim_world_pose(stage, xform_path, position, rotation)

        import omni.kit.app

        for _ in range(3):
            omni.kit.app.get_app().update()

        self.robot_prim_path = articulation_root
        self.robot_xform_path = xform_path
        self.robot = None
        print(
            f"[SceneBuilder] URDF robot ready: articulation={articulation_root}, "
            f"xform={xform_path}, pos={position}, rot_deg={rotation}"
        )
        return self.robot

    def build_robot(self):
        """构建机器人并返回prim路径"""
        robot_cfg = self.cfg['robot']
        if robot_cfg.get('robot_urdf'):
            return self._build_robot_from_urdf()

        self.robot = rep.create.from_usd(
            self._usd_path(robot_cfg['robot_usd']),
            position=robot_cfg['robot_position'],
            rotation=robot_cfg['robot_rotation'],
            parent='/Root',
        )
        if self.robot is not None:
            prims_info = self.robot._get_prims()
            if 'primsIn' in prims_info and prims_info['primsIn']:
                self.robot_prim_path = str(prims_info['primsIn'][0].GetPath())
        # 将 YAML robot_position/rotation 写入 FixedJoint，使其在物理启动后生效
        self._apply_robot_pose()
        return self.robot

    def build_foam(self):
        foam_cfg = self.cfg['foam']
        foam_position = foam_cfg.get('foam_position', foam_cfg.get('robot_position'))
        if foam_position is None:
            raise KeyError("foam 配置缺少 foam_position（或兼容字段 robot_position）")

        if self.cfg.get('task_number') == 4:
            # 任务4: 箱子资产自带 foam mesh，不需要独立创建
            print(f"[SceneBuilder] Task4 箱子资产已自带 foam mesh，跳过独立创建")
        else:
            self.foam = rep.create.from_usd(
                self._usd_path(foam_cfg['foam_usd']),
                position=foam_position,
                rotation=foam_cfg.get('foam_rotation', [0, 0, 0]),
                scale=foam_cfg.get('foam_scale', [1, 1, 1]),
                parent='/Replicator',
            )
            self.foam_prim_paths = []
            try:
                prims_info = self.foam._get_prims()
                if 'primsIn' in prims_info:
                    prims_in = prims_info['primsIn']
                    if not isinstance(prims_in, (list, tuple)):
                        prims_in = [prims_in]
                    self.foam_prim_paths = [str(prim.GetPath()) for prim in prims_in]
            except Exception as e:
                print(f"[SceneBuilder] 提取 foam 路径失败: {e}")
            if foam_cfg.get('lock_foam', False):
                self._lock_foam_positions()
        return self.foam

    def sync_foam_to_box(self):
        """任务4: 箱子资产已自带 foam mesh，无需同步（保留接口兼容）。"""
        pass

    def build_all(self):
        """Build every object defined in the config."""
        if "table" in self.cfg.keys():
            self.build_table()
        if "ConveyorBelt" in self.cfg.keys():
            self.build_ConveyorBelt()
        if "box" in self.cfg.keys():
            self.build_box()
        if "foam" in self.cfg.keys():
            self.build_foam()
        if "part" in self.cfg.keys():
            self.build_parts()


    def setup_target_objects_rigidbody(self):
        """Setup rigidbody properties for target objects and get their world poses."""
        
        world_poses = {}
        # Return empty if no target_objects defined in config
        if not self.target_objects:
            return world_poses
        
        # Collect prim paths from target_objects configuration
        for target_obj in self.target_objects:
            
            if target_obj == 'table':
                table_physx_view = self._physics_sim_view.create_rigid_body_view('/Replicator/Ref_Xform/Ref/material')
                poses = table_physx_view.get_transforms()[0]
                world_poses[target_obj] = {
                    'module_name': target_obj,
                    'world_position': [poses[0], poses[1], poses[2]],
                    'world_orientation': [poses[3], poses[4], poses[5], poses[6]] # xyzw
                }
        return world_poses

    def get_target_object_transforms(self, step_size=None):
        """Get transforms of target objects and their children in world coordinate system."""
        poses_data = self.setup_target_objects_rigidbody()
        
        # Log to CSV if enabled
        self.pose_logger.log_poses(poses_data)
            
        return poses_data

    # ------------------------------------------------------------------
    # Coordinate transform & scene queries (for auto_collect)
    # ------------------------------------------------------------------

    _COORDINATE_ANCHOR_FRAMES = ("torso_link", "waist_pitch_link", "base_link")

    def init_coordinate_transform(self, ik_solver) -> None:
        """Initialise ``CoordinateTransform`` from a link present in both sim and URDF."""
        for frame_name in self._COORDINATE_ANCHOR_FRAMES:
            if not ik_solver.model.existFrame(frame_name):
                continue
            prim_path = self._robot_link_prim_path(frame_name)
            if prim_path is None:
                continue
            self.coordinate_transform = CoordinateTransform.from_anchor_frame(
                ik_solver=ik_solver,
                frame_name=frame_name,
                frame_prim_path=prim_path,
            )
            print(
                f"[SceneBuilder] CoordinateTransform anchor: {frame_name} @ {prim_path}"
            )
            return
        print(
            "[SceneBuilder] Warning: no coordinate anchor frame found; "
            "CoordinateTransform skipped"
        )

    def _robot_link_prim_path(self, link_name: str) -> str | None:
        import omni.usd

        stage = omni.usd.get_context().get_stage()
        root = self.robot_prim_path or "/Root/Ref_Xform/Ref"

        prefixes: list[str] = []
        if str(root).endswith("/base_link"):
            parent = str(root).rsplit("/", 1)[0]
            prefixes.append(parent)
            if parent.endswith("/root_joint"):
                prefixes.append(parent.rsplit("/", 1)[0])
        prefixes.append(str(root))
        if not str(root).endswith(f"/{link_name}"):
            prefixes.append(str(root))

        seen: set[str] = set()
        for prefix in prefixes:
            if prefix in seen:
                continue
            seen.add(prefix)
            candidate = f"{prefix}/{link_name}"
            if stage.GetPrimAtPath(candidate).IsValid():
                return candidate

        for prim in stage.Traverse():
            if prim.GetName() == link_name:
                return str(prim.GetPath())
        return None

    def _robot_torso_prim_path(self) -> str:
        """Legacy helper for old USD robots with torso_link."""
        path = self._robot_link_prim_path("torso_link")
        if path is not None:
            return path
        root = self.robot_prim_path or "/Root/Ref_Xform/Ref"
        if str(root).endswith("/base_link"):
            return str(root).rsplit("/", 1)[0] + "/torso_link"
        return f"{root}/torso_link"
    def get_box_positions(self) -> list:
        """返回所有箱子的世界坐标位置。

        Returns:
            list[np.ndarray]: 每个箱子的 [x, y, z] 坐标列表。
        """
        box_cfg = self.cfg.get("box", {})
        positions = box_cfg.get("box_position", [])
        if positions:
            return [np.array(p, dtype=float) for p in positions]
        return [np.array([1.2, 0.3, 1.05], dtype=float)]


    def get_robot_world_transform(self) -> np.ndarray:
        """获取机器人在世界坐标系下的变换矩阵 (4x4)。

        Returns:
            numpy.ndarray: 4x4 变换矩阵。
        """
        from pxr import UsdGeom, Usd
        import omni.usd
        stage = omni.usd.get_context().get_stage()
        robot_prim = stage.GetPrimAtPath(self.robot_prim_path)

        if not robot_prim.IsValid():
            raise ValueError(f"Robot Prim not found at path: {self.robot_prim_path}")

        time_code = Usd.TimeCode.Default()
        xform_cache = UsdGeom.XformCache(time_code)
        local_to_world_matrix = xform_cache.GetLocalToWorldTransform(robot_prim)
        mat_array = np.array(local_to_world_matrix).reshape((4, 4))
        return mat_array.T

    def world_to_robot_coords(self, world_pos) -> list:
        """将世界坐标系下的点转换为机器人基座坐标系下的点。

        Args:
            world_pos: [x, y, z] 世界坐标。

        Returns:
            [x_local, y_local, z_local] 相对于机器人的坐标。
        """
        T_robot_to_world = self.get_robot_world_transform()
        try:
            T_world_to_robot = np.linalg.inv(T_robot_to_world)
        except np.linalg.LinAlgError:
            print("[SceneBuilder] 错误：无法计算变换矩阵的逆")
            return world_pos

        pos_homo = np.append(np.array(world_pos), 1.0)
        local_pos_homo = T_world_to_robot @ pos_homo
        local_pos = local_pos_homo[:3] / local_pos_homo[3]
        return local_pos.tolist()

    @staticmethod
    def _euler_to_quat_wxyz(euler_xyz_rad):
        """欧拉角 (XYZ, 弧度) → 四元数 [w, x, y, z]"""
        cx, cy, cz = np.cos(euler_xyz_rad / 2)
        sx, sy, sz = np.sin(euler_xyz_rad / 2)
        return np.array([
            cx*cy*cz + sx*sy*sz,
            sx*cy*cz - cx*sy*sz,
            cx*sy*cz + sx*cy*sz,
            cx*cy*sz - sx*sy*cz,
        ], dtype=np.float64)

    def _sample_part_euler_xyz(self, use_fixed_task3_spawn=False, fixed_cfg=None):
        """Sample a part orientation. Task1 defaults to upright yaw-only scatter."""
        fixed_cfg = fixed_cfg or {}
        if use_fixed_task3_spawn and not fixed_cfg.get('random_rotation', True):
            return np.radians(
                np.array(fixed_cfg.get('rotation_deg', [0, 0, 0]), dtype=np.float64)
            )

        rotation_mode = str(self.part_cfg.get('rotation_mode', 'upright')).lower()
        if rotation_mode == 'upright':
            roll_pitch = np.radians(
                np.array(self.part_cfg.get('upright_rotation_deg', [0, 0]), dtype=np.float64)
            ).flatten()
            if roll_pitch.size == 1:
                roll_pitch = np.array([roll_pitch[0], 0.0], dtype=np.float64)
            roll = float(roll_pitch[0])
            pitch = float(roll_pitch[1]) if roll_pitch.size > 1 else 0.0
            yaw_range_deg = self.part_cfg.get('yaw_range_deg', [-180, 180])
            yaw_min, yaw_max = [float(v) for v in yaw_range_deg]
            yaw = np.radians(random.uniform(yaw_min, yaw_max))
            return np.array([roll, pitch, yaw], dtype=np.float64)

        return np.array([
            random.uniform(-np.pi/2, np.pi/2),
            random.uniform(-np.pi/2, np.pi/2),
            random.uniform(-np.pi/2, np.pi/2),
        ], dtype=np.float64)

    def _delete_old_parts(self):
        """删除旧零件：清除 rigid prim 缓存 → 从 USD stage 删除 prim"""
        import omni.usd
        stage = omni.usd.get_context().get_stage()

        # 先清除缓存，确保无 tensor view 引用旧 prim
        self._parts_rigid_prims = []
        self._rigid_body_paths = []
        self._rigid_prims_initialized = False

        for path in self.parts_prim_paths:
            prim = stage.GetPrimAtPath(path)
            if prim.IsValid():
                stage.RemovePrim(path)
        print(f"[SceneBuilder] 已删除 {len(self.parts_prim_paths)} 个旧零件")

    def _create_parts_at_paths(self, part_pools, target_paths, plane_index=0):
        """在指定的 prim 路径上创建新零件并随机散布。

        Args:
            part_pools: list of (pool_or_usd, count) — 每组的资产池和数量
            target_paths: 目标 prim 路径列表，新零件将创建在这些路径上
            plane_index: 散布目标平面索引
        """
        import omni.usd
        from pxr import UsdPhysics, UsdGeom, Gf
        stage = omni.usd.get_context().get_stage()

        center = np.array(self.plane_cfg['plane_position'][plane_index], dtype=np.float64)
        scale = np.array(self.plane_cfg['plane_scale'][plane_index], dtype=np.float64)
        half_x, half_y = scale[0] * 0.5, scale[1] * 0.5
        use_fixed_task3_spawn = (
            self.cfg.get('task_number', 0) == 3
            and self.part_cfg.get('fixed_spawn', {}).get('enabled', False)
        )
        use_sampled_spawn = self.cfg.get('task_number', 0) == 1 or use_fixed_task3_spawn
        sampled_xy_positions = (
            self._sample_scatter_xy_positions(center, half_x, half_y, len(target_paths))
            if use_sampled_spawn else []
        )
        fixed_cfg = self.part_cfg.get('fixed_spawn', {})
        fixed_z_offset = fixed_cfg.get('z_offset', 0.03)
        sampled_z_offset = fixed_z_offset if use_fixed_task3_spawn else self.part_cfg.get('spawn_z_offset', 0.03)

        created_paths = []
        idx = 0
        for part_spec in part_pools:
            if len(part_spec) == 2:
                pool, count = part_spec
                part_type = None
            else:
                pool, count, part_type = part_spec
            count = self._non_negative_int(count, 'part count')
            if count == 0:
                continue
            if not self._asset_pool_has_entries(pool):
                raise ValueError("Part asset pool must not be empty when count > 0")
            for _ in range(count):
                if idx >= len(target_paths):
                    print(f"[SceneBuilder] 警告: 目标路径不足，已创建 {idx} 个")
                    return created_paths
                prim_path = target_paths[idx]
                idx += 1

                part_scale = self._part_scale_for_type(part_type)
                use_simple_block = bool(self.part_cfg.get("use_simple_block", False))
                if use_simple_block:
                    prim, xform_scale = self._create_simple_block_part(
                        stage,
                        prim_path,
                        part_scale,
                    )
                else:
                    usd_file = self._resolve_asset_path(self._choose_asset_from_pool(pool))
                    stage_utils.add_reference_to_stage(usd_path=usd_file, prim_path=prim_path)
                    prim = stage.GetPrimAtPath(prim_path)
                    xform_scale = part_scale
                created_paths.append(prim_path)
                if part_type is not None:
                    self.part_type_by_prim_path[prim_path] = part_type

                self._apply_rigid_body_to_prim(
                    prim,
                    mass=self.part_cfg.get('mass', 0.2),
                    enabled=True,
                )
                self._sanitize_rigid_bodies_under_paths(stage, prim_path, "Part")
                self._apply_part_contact_physics(prim, prim_path)

                if use_sampled_spawn:
                    xy = sampled_xy_positions[idx - 1]
                    pos = Gf.Vec3d(
                        float(xy[0]),
                        float(xy[1]),
                        float(center[2] + sampled_z_offset),
                    )
                else:
                    z_offset = self.part_cfg.get('spawn_z_offset', 0.03)
                    pos = Gf.Vec3d(
                        float(center[0] + random.uniform(-half_x, half_x)),
                        float(center[1] + random.uniform(-half_y, half_y)),
                        float(center[2] + z_offset),
                    )
                euler = self._sample_part_euler_xyz(
                    use_fixed_task3_spawn=use_fixed_task3_spawn,
                    fixed_cfg=fixed_cfg,
                )
                quat = self._euler_to_quat_wxyz(euler)

                xformable = UsdGeom.Xformable(prim)
                xformable.ClearXformOpOrder()
                xformable.AddTranslateOp().Set(pos)
                xformable.AddOrientOp().Set(
                    Gf.Quatf(float(quat[0]), float(quat[1]), float(quat[2]), float(quat[3]))
                )
                xformable.AddScaleOp().Set(
                    Gf.Vec3f(
                        float(xform_scale[0]),
                        float(xform_scale[1]),
                        float(xform_scale[2]),
                    )
                )

        return created_paths

    def _randomize_task1_assets(self):
        """删除旧零件，从资产池中随机选择并在相同路径上创建新零件。"""
        saved_paths = list(self._initial_parts_prim_paths)
        self._delete_old_parts()
        self.part_type_by_prim_path = {}

        created_paths = self._create_parts_at_paths(
            part_pools=[
                (asset_pool, count, part_type)
                for part_type, asset_pool, count in self._get_task1_part_plan()
            ],
            target_paths=saved_paths,
            plane_index=0,
        )

        self.parts_prim_paths = created_paths
        self._rigid_prims_initialized = False
        print(f"[SceneBuilder] Task1 已重新创建 {len(created_paths)} 个零件 (路径不变，等待 scatter)")

    def _randomize_task3_assets(self):
        """删除旧零件，在相同路径上为每个箱子重新创建零件（仅 USD 层，位姿在 world.reset 后由 scatter_after_reset 设置）"""
        saved_paths = list(self._initial_parts_prim_paths)
        self._delete_old_parts()

        # 获取 part A / B 资产（支持 pool 或单个 usd）
        part_a = self.part_cfg.get('part_a_assets', self.part_cfg.get('part_a_usd', ''))
        part_b = self.part_cfg.get('part_b_assets', self.part_cfg.get('part_b_usd', ''))
        part_usds = [part_a, part_b]

        num_groups = len(self.box_cfg['box_position'])
        parts_per_group = self.part_cfg.get('num_parts', 3)

        for i in range(num_groups):
            asset = part_usds[i] if i < len(part_usds) else part_usds[-1]
            start = i * parts_per_group
            end = start + parts_per_group
            group_target_paths = saved_paths[start:end]
            self._create_parts_at_paths(
                part_pools=[(asset, len(group_target_paths))],
                target_paths=group_target_paths,
                plane_index=i,
            )

        self.parts_prim_paths = saved_paths
        self._rigid_prims_initialized = False
        print(f"[SceneBuilder] Task3 已重新创建 {len(saved_paths)} 个零件到 {num_groups} 个平面 (路径不变，等待 scatter)")

    def scatter_after_reset(self):
        """在 world.reset() 之后调用：用 SingleRigidPrim 设置随机位姿 + 清速度。

        此时物理视图已重建，SingleRigidPrim 可正常工作。
        """
        task = self.cfg['task_number']

        # 重建 rigid prim 缓存（world.reset 后 tensor view 已失效）
        self._rebuild_rigid_prims(self.parts_prim_paths)

        if task == 1:
            self._scatter_parts_direct(plane_index=0)
            print(f"[SceneBuilder] Task1 scatter_after_reset: 已随机散布 {len(self.parts_prim_paths)} 个零件")
        elif task == 3:
            num_groups = len(self.box_cfg['box_position'])
            parts_per_group = self.part_cfg.get('num_parts', 3)
            for i in range(num_groups):
                start = i * parts_per_group
                end = start + parts_per_group
                group_paths = self.parts_prim_paths[start:end]
                self._scatter_parts_direct(plane_index=i, prim_paths=group_paths)
            print(f"[SceneBuilder] Task3 scatter_after_reset: 已随机散布 {len(self.parts_prim_paths)} 个零件到 {num_groups} 个平面")

    def _sample_scatter_xy_positions(self, center, half_x, half_y, count):
        """Sample random XY positions away from edges with simple separation."""
        if count <= 0:
            return []

        task = self.cfg.get('task_number', 0)
        edge_margin_x = 0.0
        edge_margin_y = 0.0
        if task == 3:
            edge_margin_x = max(0.008, half_x * 0.25)
            edge_margin_y = max(0.008, half_y * 0.25)

        usable_half_x = max(half_x - edge_margin_x, half_x * 0.25)
        usable_half_y = max(half_y - edge_margin_y, half_y * 0.25)

        if task == 1:
            edge_margin_x = float(self.part_cfg.get('scatter_edge_margin_x', max(0.02, half_x * 0.15)))
            edge_margin_y = float(self.part_cfg.get('scatter_edge_margin_y', max(0.02, half_y * 0.15)))
            usable_half_x = max(half_x - edge_margin_x, half_x * 0.25)
            usable_half_y = max(half_y - edge_margin_y, half_y * 0.25)
            sample_center = np.asarray(
                self.part_cfg.get('scatter_center_xy', [center[0], center[1]]),
                dtype=np.float64,
            )
            scatter_half_size = self.part_cfg.get('scatter_half_size_xy')
            if scatter_half_size is not None:
                scatter_half_size = np.asarray(scatter_half_size, dtype=np.float64)
                usable_half_x = min(usable_half_x, max(0.0, float(scatter_half_size[0])))
                usable_half_y = min(usable_half_y, max(0.0, float(scatter_half_size[1])))
            else:
                sample_center = np.array([center[0], center[1]], dtype=np.float64)
            min_distance = float(
                self.part_cfg.get(
                    'scatter_min_distance',
                    min(0.12, max(0.08, min(half_x, half_y) * 0.75)),
                )
            )

            def task1_candidate():
                return (
                    float(sample_center[0] + random.uniform(-usable_half_x, usable_half_x)),
                    float(sample_center[1] + random.uniform(-usable_half_y, usable_half_y)),
                )

            positions = []
            for _ in range(count):
                accepted = None
                for _attempt in range(300):
                    candidate = task1_candidate()
                    if all(np.linalg.norm(np.array(candidate) - np.array(p)) >= min_distance for p in positions):
                        accepted = candidate
                        break
                if accepted is None:
                    accepted = task1_candidate()
                    print(
                        f"[SceneBuilder] Task1 scatter: 无法满足最小间距 {min_distance:.3f}m，"
                        "已使用随机候选点"
                    )
                positions.append(accepted)
            return positions

        fixed_cfg = self.part_cfg.get('fixed_spawn', {}) if task == 3 else {}
        if task == 3 and fixed_cfg.get('enabled', False):
            anchor_offsets = fixed_cfg.get('anchor_offsets')
            if anchor_offsets:
                anchors = [(float(offset[0]), float(offset[1])) for offset in anchor_offsets]
            else:
                anchor_fraction = float(fixed_cfg.get('anchor_fraction', 0.8))
                anchor_x = usable_half_x * anchor_fraction
                anchor_y = usable_half_y * anchor_fraction
                anchors = [
                    (-anchor_x, -anchor_y),
                    (-anchor_x, anchor_y),
                    (anchor_x, -anchor_y),
                    (anchor_x, anchor_y),
                ]

            if not anchors:
                return []
            if fixed_cfg.get('random_select', True):
                selected_anchors = random.sample(anchors, min(count, len(anchors)))
            else:
                selected_anchors = anchors[:count]
            while len(selected_anchors) < count:
                selected_anchors.append(random.choice(anchors))
            return [
                (float(center[0] + dx), float(center[1] + dy))
                for dx, dy in selected_anchors[:count]
            ]

        if task == 3 and count <= 4:
            anchor_x = usable_half_x * 0.8
            anchor_y = usable_half_y * 0.8
            anchors = [
                (-anchor_x, -anchor_y),
                (-anchor_x, anchor_y),
                (anchor_x, -anchor_y),
                (anchor_x, anchor_y),
            ]
            selected_anchors = random.sample(anchors, count)
            jitter_x = max(0.002, usable_half_x * 0.08)
            jitter_y = max(0.002, usable_half_y * 0.08)
            positions = []
            for dx, dy in selected_anchors:
                x = center[0] + float(np.clip(dx + random.uniform(-jitter_x, jitter_x), -usable_half_x, usable_half_x))
                y = center[1] + float(np.clip(dy + random.uniform(-jitter_y, jitter_y), -usable_half_y, usable_half_y))
                positions.append((x, y))
            return positions

        min_distance = min(
            0.045,
            max(0.025, min(usable_half_x, usable_half_y) * 0.85),
        )

        def random_candidate():
            return (
                center[0] + random.uniform(-usable_half_x, usable_half_x),
                center[1] + random.uniform(-usable_half_y, usable_half_y),
            )

        positions = []
        for _ in range(count):
            accepted = None
            for _attempt in range(100):
                candidate = random_candidate()
                if all(np.linalg.norm(np.array(candidate) - np.array(p)) >= min_distance for p in positions):
                    accepted = candidate
                    break
            if accepted is None:
                accepted = random_candidate()
            positions.append(accepted)
        return positions

    def _scatter_parts_direct(self, plane_index=0, prim_paths=None):
        """直接通过 SingleRigidPrim 随机散布零件（完全绕过 Replicator）。

        散布范围与初始化时 scatter_2d 一致：
        - Isaac Sim Z-up，平面在 XY 平面上
        - 平面基底 1×1，缩放后 X 方向范围 = scale[0]，Y 方向范围 = scale[1]
        - Z = 平面高度 + 小偏移（防止穿模）
        """
        if prim_paths is None:
            prim_paths = self.parts_prim_paths

        center = np.array(self.plane_cfg['plane_position'][plane_index], dtype=np.float64)
        scale = np.array(self.plane_cfg['plane_scale'][plane_index], dtype=np.float64)
        # 散布范围与 scatter_2d 一致：平面 X/Y 方向各 scale*0.5
        half_x = scale[0] * 0.5
        half_y = scale[1] * 0.5
        fixed_cfg = self.part_cfg.get('fixed_spawn', {})
        use_fixed_task3_spawn = (
            self.cfg.get('task_number', 0) == 3
            and fixed_cfg.get('enabled', False)
        )
        z_offset = fixed_cfg.get('z_offset', 0.03) if use_fixed_task3_spawn else self.part_cfg.get('spawn_z_offset', 0.03)

        # 延迟初始化 + 使用缓存的 rigid prim
        self._ensure_rigid_prims()
        rigid_prims = getattr(self, '_parts_rigid_prims', [])
        all_paths = self.parts_prim_paths

        xy_positions = self._sample_scatter_xy_positions(center, half_x, half_y, len(prim_paths))

        for path, xy in zip(prim_paths, xy_positions):
            # 从缓存中查找对应的 rigid prim
            try:
                idx = all_paths.index(path)
                rigid = rigid_prims[idx] if idx < len(rigid_prims) else None
            except (ValueError, IndexError):
                rigid = None
            if rigid is None:
                print(f"[SceneBuilder] _scatter_parts_direct: 无缓存刚体 {path}")
                continue

            pos = np.array([
                xy[0],
                xy[1],
                center[2] + z_offset,
            ], dtype=np.float64)
            euler = self._sample_part_euler_xyz(
                use_fixed_task3_spawn=use_fixed_task3_spawn,
                fixed_cfg=fixed_cfg,
            )
            quat = self._euler_to_quat_wxyz(euler)

            rigid.set_world_pose(position=pos, orientation=quat)
            rigid.set_linear_velocity(np.zeros(3))
            rigid.set_angular_velocity(np.zeros(3))

        print(f"[SceneBuilder] _scatter_parts_direct: 已随机散布 {len(prim_paths)} 个零件到平面 {plane_index}")

    def _clear_parts_velocities(self):
        """清零所有零件的线速度和角速度（使用缓存的 rigid prim）"""
        self._ensure_rigid_prims()
        rigid_prims = getattr(self, '_parts_rigid_prims', [])
        for i, part_path in enumerate(self.parts_prim_paths):
            if i < len(rigid_prims) and rigid_prims[i] is not None:
                try:
                    rigid_prims[i].set_linear_velocity(np.zeros(3))
                    rigid_prims[i].set_angular_velocity(np.zeros(3))
                except Exception as e:
                    print(f"[SceneBuilder] 清零 {part_path} 速度失败: {e}")
            else:
                print(f"[SceneBuilder] 清零 {part_path} 速度跳过: 无缓存 rigid body")

    def _reset_boxes(self):
        """重置 Task 1/2/3 的箱子到配置中的初始位置"""
        if hasattr(self, 'boxes') and hasattr(self, 'box_cfg'):
            self.boxes.set_world_poses(
                positions=np.array(self.box_cfg['box_position']),
            )
            print(f"[SceneBuilder] 箱子已重置到初始位置")


    def save_parts_poses(
        self,
        save_dir: Optional[Path] = None,
        episode_index: Optional[int] = None,
    ) -> Optional[Path]:
        """
        保存所有零件的世界位姿数据到 JSON 文件（适用于所有任务）

        功能说明：
            1. 调用 get_parts_world_poses 获取当前所有零件的实时世界位姿
            2. 自动生成带时间戳和任务编号的文件名，避免覆盖
            3. 将位姿数据格式化后保存为 JSON 文件
            4. 完善异常处理，保证保存失败时不崩溃并输出错误日志
            5. 返回最终保存的文件路径，方便外部调用

        参数说明：
            save_dir (Optional[Path]): 可选参数，指定保存目录
                - 不传入时：默认保存到项目根目录下的 outputs 子目录
                - 传入时：使用指定目录（自动创建不存在的目录）

        返回值：
            Optional[Path]: 成功返回保存的文件完整路径，失败返回 None
        """
        try:
            # 1. 获取零件位姿数据
            all_parts_poses: List[Dict] = self.get_parts_world_poses()

            # 空数据判断
            if not all_parts_poses:
                if self.compute_num_tracked_objects(self.cfg) == 0:
                    print("[save_parts_poses] 当前配置零件数量为 0，跳过保存")
                    return None
                print("[save_parts_poses] 错误：未获取到任何零件位姿数据，取消保存")
                return None

            # 2. 生成保存路径
            # 根据当前任务编号、episode索引和时间戳生成文件名
            task_num = self.cfg.get('task_number', 1)
            time_stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            if episode_index is not None:
                file_name = f"task{task_num}_episode{episode_index}_parts_poses_{time_stamp}.json"
            else:
                file_name = f"task{task_num}_parts_poses_{time_stamp}.json"

            # 确定保存目录
            if save_dir is None:
                # 默认路径：项目根目录下的 outputs 子目录
                save_dir = Path(__file__).parent.parent.parent / "outputs"
            else:
                # 确保传入的路径是 Path 类型
                save_dir = Path(save_dir)

            # 自动创建目录
            save_dir.mkdir(parents=True, exist_ok=True)
            save_path = save_dir / file_name

            # 3. 写入 JSON 文件（格式化输出，方便阅读）
            with open(save_path, 'w', encoding='utf-8') as f:
                json.dump(
                    all_parts_poses,
                    f,
                    ensure_ascii=False,
                    indent=4
                )

            # 4. 保存成功日志
            if episode_index is not None:
                print(f"[save_parts_poses] episode_index={episode_index}")
            print(f"[save_parts_poses] 保存成功！共 {len(all_parts_poses)} 个零件")
            print(f"[save_parts_poses] 文件路径：{save_path.absolute()}")
            return save_path

        except Exception as e:
            # 全局异常捕获，保证函数健壮性
            print(f"[save_parts_poses] 保存失败：{str(e)}")
            return None

            


    def reset(self):
        """重置场景状态，所有任务的物体恢复到初始随机化/固定位置"""
        task = self.cfg['task_number']
        print(f"[SceneBuilder] 重置任务{task}")

        # ── 任务1重置：直接通过 SingleRigidPrim 散布（完全绕过 Replicator） ──
        if task == 1:
            self.save_parts_poses()
            self._randomize_task1_assets()
            # self.set_parts_world_poses(json_file_path='task1_parts_poses_20260418_112821.json')
            self._reset_boxes()
            print("[SceneBuilder] Task1 已重置")
            return

        # ── 任务2重置：传送带零件恢复初始位置 + 重新随机排列顺序 ──
        if task == 2:

            #停止传送带
            self.set_conveyor_speed([0.0,0.0,0.0])

            # 官方标准：8个零件（5A + 5B）
            random_indices = np.random.permutation(np.array([0, 1, 2, 3, 4, 5, 6, 7]))
            # 清零速度和角速度，防止物体保留上一轮运动状态
            # 注意：rigid_prim 包含所有 /Root/Part_.* 匹配的 prim（包括原始的A/B共12个）
            # 只需要清零克隆的8个零件的速度
            self.rigid_prim.set_velocities(velocities=np.zeros((8, 6)))
            # 恢复初始位姿，随机排列顺序
            self.rigid_prim.set_world_poses(
                positions=self._task2_initial_positions,
                indices=random_indices
            )
            # 再次清零速度，确保位姿设置后速度为零
            self.rigid_prim.set_velocities(velocities=np.zeros((8, 6)))

            # 重新启动传送带（沿 X 轴正方向，速度 0.1 m/s）

            Conveyor_speed=self.ConveyorBelt_cfg['ConveyorBelt_speed']
            self.set_conveyor_speed(Conveyor_speed)

            self.save_parts_poses()
            self._reset_boxes()
            print("[SceneBuilder] Task2 已重置")
            return

        # ── 任务3重置：删除旧零件，重新创建并随机散布 ──
        if task == 3:
            self._randomize_task3_assets()
            self._reset_boxes()
            if self.box_cfg.get('lock_boxes', False):
                self._lock_box_positions()
            if self.cfg.get('foam', {}).get('lock_foam', False):
                self._lock_foam_positions()
            print("[SceneBuilder] Task3 已重置")
            return

        # ── 任务4重置：重置箱子位置 ──
        if task == 4:
            if hasattr(self, 'box_articulation') and self.box_articulation is not None:
                # 重置关节位置（仅当为 Articulation 且有关节时）
                if (self.box_initial_joint_positions is not None
                        and hasattr(self.box_articulation, 'set_joint_positions')):
                    self.box_articulation.set_joint_positions(self.box_initial_joint_positions)
                # 使用世界坐标重置箱子位置
                self.box_articulation.set_world_poses(
                    positions=self._box_initial_world_pos,
                    orientations=self._box_initial_world_ori
                )
                # 清零箱子速度（仅当物理对象支持时）
                if hasattr(self.box_articulation, 'set_velocities'):
                    self.box_articulation.set_velocities(velocities=np.zeros((1, 6)))
            print("[SceneBuilder] Task4 已重置")
            return

        print(f"[SceneBuilder] 警告: 未知的 task_number={task}，跳过重置")
