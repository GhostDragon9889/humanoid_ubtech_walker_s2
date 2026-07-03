"""
Walker S2 Isaac Sim 仿真机器人 - LeRobot 0.5.1 实现

使用用例:
    lerobot-record --robot.type=walker_s2_sim --robot.headless=false ...
    lerobot-replay --robot.type=walker_s2_sim --dataset.repo_id=...

功能:
    - 14 自由度双臂控制 (7 关节/臂)
    - 4 相机支持 (head_left, head_right, wrist_left, wrist_right)
    - 键盘遥操作 (通过 WalkerS2KeyboardTeleop)
    - ROS2 遥操作 (可选)
    - Isaac Sim 物理仿真
"""

from __future__ import annotations

import logging
import os
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional
from functools import cached_property

import numpy as np
import torch
import yaml

from src.lerobot.robots.robot import Robot
from src.lerobot.robots.config import RobotConfig
from src.lerobot.processor import RobotAction, RobotObservation

from .walkers2simConfig import WalkerS2Config
from .head_stereo_visualizer import HeadStereoVisualizer
from .isaac_sim_robot_interface import IsaacSimRobotInterface, load_config

logger = logging.getLogger(__name__)


@dataclass
class TimingMetric:
    """用于记录性能指标的数据类"""
    count: int = 0
    total_s: float = 0.0
    min_s: float = field(default_factory=lambda: float("inf"))
    max_s: float = 0.0

    def update(self, duration_s: float) -> None:
        duration_s = max(float(duration_s), 0.0)
        self.count += 1
        self.total_s += duration_s
        self.min_s = min(self.min_s, duration_s)
        self.max_s = max(self.max_s, duration_s)

    def as_dict(self) -> dict[str, float | int]:
        if self.count == 0:
            return {"count": 0, "avg_s": 0.0, "min_s": 0.0, "max_s": 0.0}
        return {
            "count": self.count,
            "avg_s": self.total_s / self.count,
            "min_s": self.min_s,
            "max_s": self.max_s,
        }





class WalkerS2sim(Robot):
    """
    Walker S2 双臂机器人 Isaac Sim 仿真实现

    功能:
        - 14 自由度双臂控制 (7 关节/臂)
        - 4 相机支持 (head_left, head_right, wrist_left, wrist_right)
        - 支持直接关节位置控制 (回放/推理)
        - 支持末端执行器增量控制 (遥操作 via IK)
        - 夹持器开/关控制
        - 环境物体状态观测 (可选)

    使用例:
        >>> robot = WalkerS2sim(config)
        >>> robot.connect()
        >>> obs = robot.get_observation()
        >>> action = {"L_shoulder_pitch_joint.pos": 0.1, ...}
        >>> robot.send_action(action)
        >>> robot.disconnect()
    """

    robot_type: str = "walker_s2_sim"
    name: str = "walkerS2"
    config_class: type[RobotConfig] = WalkerS2Config
    CAMERA_NAMES = ["head_left", "head_right", "wrist_left", "wrist_right"]

    # 状态维度：14 臂关节 + 4 手指关节 + 2 夹持器指令 = 20D
    STATE_DIM = 20
    # 动作维度：与状态相同
    ACTION_DIM = 20

    def __init__(self, config: WalkerS2Config | dict | None = None, teleop: Any = None):
        """
        初始化 Walker S2 仿真机器人

        Args:
            config: 机器人配置，可以是 WalkerS2Config 实例、字典或 None
            teleop: 可选的遥操作器实例 (WalkerS2KeyboardTeleop)
        """
        # 处理配置
        if isinstance(config, dict):
            config = WalkerS2Config(**config)
        elif config is None:
            config = WalkerS2Config()

        super().__init__(config)
        self.config = config

        # 加载任务配置
        if self.config.task_cfg_path:
            self.config.task_cfg = load_config(self.config.task_cfg_path)

        # 性能计时
        self._timing_metrics: dict[str, TimingMetric] = {
            "send_action": TimingMetric(),
            "get_observation": TimingMetric(),
            "dt_s": TimingMetric(),
        }

        # Isaac Sim 核心组件
        self._kit: Any = None
        self._world: Any = None
        self._scene_builder: Any = None
        self._robot_interface: Optional[IsaacSimRobotInterface] = None
        self._arm_joint_indices: list[int] = []

        # 遥操作器（外部注入）
        self._teleop: Any = teleop

        # ---- 回调控制相关状态 (移植自 mobile_manipulator.py) ----
        # 回调锁和注册状态
        self._callback_lock = threading.Lock()
        self._callbacks_registered = False

        # Inference 模式：send_action 写入 pending，callback 消费并执行
        self._pending_absolute_action: Optional[np.ndarray] = None
        # 相机图像缓存 (由 render callback 更新)
        self._latest_camera_rgb: dict[str, np.ndarray] = {}
        # 持久化关节目标：统一关节位置控制的核心状态
        self._hold_arm_positions: Optional[np.ndarray] = None
        self._hold_finger_positions: Optional[np.ndarray] = None
        # 夹持模式标记
        self._left_gripping: bool = False
        self._right_gripping: bool = False

        # 步数计数器
        self._send_action_step_idx: int = 0

        # 频率监测（墙钟实测）
        self._phys_cb_count: int = 0
        self._phys_cb_t0: Optional[float] = None
        self._render_cb_count: int = 0
        self._render_cb_t0: Optional[float] = None
        self.measured_physics_hz: float = 0.0
        self.measured_render_hz: float = 0.0

        # 线性插值回到初始位置
        self._go_home = False
        self._num_interpolation_steps = 200
        self._go_home_key_was_pressed = False  # 用于检测按键边缘触发
        self._last_teleop_input_log_t = 0.0
        self._teleop_ee_targets: dict[str, Optional[np.ndarray]] = {"left": None, "right": None}
        grasp_cfg = self.config.task_cfg.get("grasp", {})
        self._teleop_ik_rot_weight = float(grasp_cfg.get("teleop_ik_rot_weight", 0.1))
        self._teleop_ik_pos_tol = float(grasp_cfg.get("teleop_ik_pos_tol", 1e-5))
        self._teleop_ik_rot_tol = float(grasp_cfg.get("teleop_ik_rot_tol", 1e-5))
        self._teleop_target_max_xyz_step = 0.04
        self._teleop_target_max_rpy_step = 0.18
        self._assisted_grasp: Optional[dict[str, Any]] = None
        self._assisted_grasp_key_was_pressed = False
        self._cancel_grasp_key_was_pressed = False
        self._grasp_debug_root = "/GraspDebug"

        # 头部相机可视化
        self._head_visualizer = HeadStereoVisualizer(
            enabled=getattr(self.config, "head_viz_enabled", False),
            window_name=getattr(self.config, "head_viz_window_name", "walker_s2_cameras"),
            scale=getattr(self.config, "head_viz_scale", 1.0),
            every_n=getattr(self.config, "head_viz_every_n", 1),
            window_x=getattr(self.config, "head_viz_window_x", 40),
            window_y=getattr(self.config, "head_viz_window_y", 40),
            show_labels=getattr(self.config, "head_viz_show_labels", True),
        )

        # 环境状态缓存
        self._last_observation: Optional[RobotObservation] = None


    @property
    def cameras(self):
        """返回相机名称列表，兼容框架的 len() 检查"""
        return self.CAMERA_NAMES  # 返回6个相机名称的列表

    @property
    def is_connected(self) -> bool:
        """检查机器人是否已连接"""
        return self._robot_interface is not None

    @property
    def has_camera(self) -> bool:
        """检查是否有相机"""
        return True

    @property
    def num_cameras(self) -> int:
        """相机数量"""
        return len(self.CAMERA_NAMES)

    @property
    def available_arms(self) -> list[str]:
        """可用的机械臂列表"""
        return ["left", "right"]

    def attach_teleop(self, teleop: Any | None) -> None:
        """绑定外部 teleop，让物理回调可以直接消费键盘状态。"""
        self._teleop = teleop
        if teleop is not None and hasattr(teleop, "enable_callback_mode"):
            teleop.enable_callback_mode()
            if self._robot_interface is not None:
                self._robot_interface._smooth_alpha = 0.98
                logger.info(
                    "WalkerS2 keyboard teleop IK tuning: smooth_alpha=%.2f",
                    self._robot_interface._smooth_alpha,
                )
        elif teleop is None:
            logger.info("WalkerS2sim teleop detached")

    def _reset_teleop_ee_targets(self) -> None:
        self._teleop_ee_targets = {"left": None, "right": None}

    def _clear_grasp_debug_markers(self) -> None:
        try:
            import omni.usd
            from pxr import Sdf

            stage = omni.usd.get_context().get_stage()
            if stage is not None and stage.GetPrimAtPath(self._grasp_debug_root).IsValid():
                stage.RemovePrim(Sdf.Path(self._grasp_debug_root))
        except Exception:
            logger.exception("[assisted_grasp] Failed to clear grasp debug markers")

    @staticmethod
    def _bind_debug_material(stage, prim, material_name: str, color) -> None:
        from pxr import Gf, Sdf, UsdShade

        material_path = f"/GraspDebug/Materials/{material_name}"
        material = UsdShade.Material.Define(stage, material_path)
        shader = UsdShade.Shader.Define(stage, f"{material_path}/PreviewSurface")
        shader.CreateIdAttr("UsdPreviewSurface")
        shader.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(
            Gf.Vec3f(*color)
        )
        shader.CreateInput("emissiveColor", Sdf.ValueTypeNames.Color3f).Set(
            Gf.Vec3f(*color)
        )
        shader.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(0.25)
        material.CreateSurfaceOutput().ConnectToSource(shader.ConnectableAPI(), "surface")
        UsdShade.MaterialBindingAPI(prim).Bind(material)

    @staticmethod
    def _set_debug_sphere(stage, path: str, position, radius: float, color, material_name: str) -> None:
        from pxr import Gf, UsdGeom

        sphere = UsdGeom.Sphere.Define(stage, path)
        sphere.CreateRadiusAttr(float(radius))
        xformable = UsdGeom.Xformable(sphere.GetPrim())
        xformable.ClearXformOpOrder()
        xformable.AddTranslateOp().Set(Gf.Vec3d(*np.asarray(position, dtype=float)))
        UsdGeom.Gprim(sphere.GetPrim()).CreateDisplayColorAttr([Gf.Vec3f(*color)])
        try:
            WalkerS2sim._bind_debug_material(stage, sphere.GetPrim(), material_name, color)
        except Exception:
            logger.debug("[assisted_grasp] USD material binding failed", exc_info=True)

    @staticmethod
    def _set_debug_line(stage, path: str, start, end, color, width: float = 0.008) -> None:
        from pxr import Gf, UsdGeom

        curve = UsdGeom.BasisCurves.Define(stage, path)
        curve.CreateTypeAttr("linear")
        curve.CreateCurveVertexCountsAttr([2])
        curve.CreatePointsAttr(
            [
                Gf.Vec3f(*np.asarray(start, dtype=float)),
                Gf.Vec3f(*np.asarray(end, dtype=float)),
            ]
        )
        curve.CreateWidthsAttr([float(width), float(width)])
        UsdGeom.Gprim(curve.GetPrim()).CreateDisplayColorAttr([Gf.Vec3f(*color)])

    def _draw_debug_frame(
        self,
        stage,
        coord,
        root: str,
        origin_base: np.ndarray,
        rotation_base: np.ndarray,
        axis_length: float = 0.08,
    ) -> None:
        origin_world = coord.robot_to_world(origin_base)
        axis_specs = (
            ("x", rotation_base[:, 0], (1.0, 0.05, 0.05)),
            ("y", rotation_base[:, 1], (0.05, 0.8, 0.05)),
            ("z", rotation_base[:, 2], (0.1, 0.3, 1.0)),
        )
        for axis_name, axis_base, color in axis_specs:
            end_world = coord.robot_to_world(origin_base + axis_base * axis_length)
            self._set_debug_line(
                stage,
                f"{root}/{axis_name}_axis",
                origin_world,
                end_world,
                color,
                width=0.006,
            )

    def _draw_grasp_debug_markers(
        self,
        planner,
        pregrasp_pose: np.ndarray,
        grasp_pose: np.ndarray,
        lift_pose: np.ndarray,
    ) -> None:
        try:
            import omni.usd
            from pxr import UsdGeom

            stage = omni.usd.get_context().get_stage()
            if stage is None:
                return

            UsdGeom.Xform.Define(stage, self._grasp_debug_root)
            UsdGeom.Xform.Define(stage, f"{self._grasp_debug_root}/Materials")

            coord = self._scene_builder.coordinate_transform
            grasp_root = f"{self._grasp_debug_root}/planned_grasp"
            UsdGeom.Xform.Define(stage, grasp_root)
            tcp_base = grasp_pose[:3] + planner.tcp_offset_base
            current_ee = self._robot_interface.ik_solver.get_ee_pose(planner.grasp_arm)
            current_tcp_base = (
                np.asarray(current_ee.translation, dtype=float)
                + np.asarray(current_ee.rotation, dtype=float) @ planner.tcp_offset_local
            )

            self._set_debug_sphere(
                stage,
                f"{self._grasp_debug_root}/pregrasp",
                coord.robot_to_world(pregrasp_pose[:3]),
                0.018,
                (1.0, 0.55, 0.05),
                "pregrasp_orange",
            )
            self._set_debug_sphere(
                stage,
                f"{self._grasp_debug_root}/sixforce_target",
                coord.robot_to_world(grasp_pose[:3]),
                0.026,
                (0.05, 0.25, 1.0),
                "sixforce_blue",
            )
            self._set_debug_sphere(
                stage,
                f"{self._grasp_debug_root}/planned_tcp",
                coord.robot_to_world(tcp_base),
                0.020,
                (1.0, 0.05, 0.9),
                "planned_tcp_magenta",
            )
            self._set_debug_sphere(
                stage,
                f"{self._grasp_debug_root}/current_tcp",
                coord.robot_to_world(current_tcp_base),
                0.016,
                (0.0, 0.9, 1.0),
                "current_tcp_cyan",
            )
            self._set_debug_sphere(
                stage,
                f"{self._grasp_debug_root}/lift_target",
                coord.robot_to_world(lift_pose[:3]),
                0.016,
                (0.2, 1.0, 0.2),
                "lift_green",
            )
            self._set_debug_line(
                stage,
                f"{self._grasp_debug_root}/sixforce_to_tcp",
                coord.robot_to_world(grasp_pose[:3]),
                coord.robot_to_world(tcp_base),
                (1.0, 0.05, 0.9),
                width=0.006,
            )
            self._draw_debug_frame(
                stage,
                coord,
                grasp_root,
                grasp_pose[:3],
                planner.R_grasp,
            )
            palm_target_pos = getattr(planner, "palm_target_pos_base", None)
            palm_target_R = getattr(planner, "palm_target_R_base", None)
            if palm_target_pos is not None and palm_target_R is not None:
                palm_target_pos = np.asarray(palm_target_pos, dtype=float)
                self._set_debug_sphere(
                    stage,
                    f"{self._grasp_debug_root}/palm_target",
                    coord.robot_to_world(palm_target_pos),
                    0.018,
                    (1.0, 0.85, 0.05),
                    "palm_target_yellow",
                )
                self._draw_debug_frame(
                    stage,
                    coord,
                    f"{self._grasp_debug_root}/planned_palm",
                    palm_target_pos,
                    np.asarray(palm_target_R, dtype=float),
                    axis_length=0.06,
                )
            logger.info(
                "[assisted_grasp] Drew debug markers: blue=sixforce target, "
                "magenta=planned TCP, yellow=palm target, cyan=current TCP"
            )
        except Exception:
            logger.exception("[assisted_grasp] Failed to draw grasp debug markers")

    @staticmethod
    def _interpolate_pose(start: np.ndarray, goal: np.ndarray, alpha: float) -> np.ndarray:
        """Interpolate xyz linearly and RPY through the shortest wrapped angle."""
        alpha = float(np.clip(alpha, 0.0, 1.0))
        alpha = alpha * alpha * (3.0 - 2.0 * alpha)
        result = np.asarray(start, dtype=float).copy()
        goal = np.asarray(goal, dtype=float)
        result[:3] += alpha * (goal[:3] - result[:3])
        angle_delta = (goal[3:] - result[3:] + np.pi) % (2.0 * np.pi) - np.pi
        result[3:] += alpha * angle_delta
        return result

    def _filter_reachable_grasp_parts(
        self,
        part_poses: list[dict[str, Any]],
        coord,
        grasp_cfg: dict[str, Any],
        ee_poses: dict[str, np.ndarray],
    ) -> list[dict[str, Any]]:
        max_distance = float(grasp_cfg.get("max_grasp_distance", 0.85))
        x_bounds = np.asarray(grasp_cfg.get("workspace_x", [0.15, 0.85]), dtype=float)
        y_bounds = np.asarray(grasp_cfg.get("workspace_y", [-0.65, 0.65]), dtype=float)
        z_bounds = np.asarray(grasp_cfg.get("workspace_z", [-0.05, 0.55]), dtype=float)
        z_offset = float(grasp_cfg.get("grasp_height_offset", 0.0))
        candidates: list[tuple[float, int, dict[str, Any]]] = []

        for i, part in enumerate(part_poses):
            world = np.asarray(part.get("position", [np.nan, np.nan, np.nan]), dtype=float)
            if world.shape != (3,) or not np.all(np.isfinite(world)):
                logger.warning("[assisted_grasp] Reject part %s: invalid world pose %s", part.get("prim_path"), world)
                continue
            base = coord.world_to_robot(world + np.array([0.0, 0.0, z_offset]))
            arm = "left" if base[1] > 0.0 else "right"
            distance = float(np.linalg.norm(base - ee_poses[arm][:3]))
            in_bounds = (
                x_bounds[0] <= base[0] <= x_bounds[1]
                and y_bounds[0] <= base[1] <= y_bounds[1]
                and z_bounds[0] <= base[2] <= z_bounds[1]
            )
            if not in_bounds or distance > max_distance:
                logger.warning(
                    "[assisted_grasp] Reject part %s: base=%s arm=%s distance=%.3fm",
                    part.get("prim_path"),
                    np.array2string(base, precision=3, suppress_small=True),
                    arm,
                    distance,
                )
                continue
            priority = 0 if i == int(grasp_cfg.get("target_index", 0)) else 1
            candidates.append((priority + distance * 0.01, i, part))

        candidates.sort(key=lambda item: item[0])
        return [item[2] for item in candidates]

    def _start_assisted_grasp(self) -> bool:
        if self._scene_builder is None or self._robot_interface is None:
            logger.warning("[assisted_grasp] Scene or robot is not ready")
            return False

        try:
            self._scene_builder.init_coordinate_transform(self._robot_interface.ik_solver)
        except Exception:
            logger.exception("[assisted_grasp] Failed to refresh coordinate transform")
            return False

        coord = getattr(self._scene_builder, "coordinate_transform", None)
        if coord is None:
            logger.warning("[assisted_grasp] Coordinate transform is unavailable")
            return False

        ee_poses = self._robot_interface.get_ee_poses()
        if ee_poses is None:
            logger.warning("[assisted_grasp] End-effector poses are unavailable")
            return False

        part_poses = self._scene_builder.get_parts_world_poses()
        if not part_poses:
            logger.warning("[assisted_grasp] No parts found in the scene")
            return False

        try:
            from Ubtech_sim.source.grasp_planner import GraspPlanner

            grasp_cfg = self.config.task_cfg.get("grasp", {})
            hand_pose = str(grasp_cfg.get("hand_pose", "pinch"))
            profile_cfg = grasp_cfg.get("profiles", {}).get(hand_pose, {})
            active_grasp_cfg = {**grasp_cfg, **profile_cfg}
            reachable_parts = self._filter_reachable_grasp_parts(
                part_poses, coord, active_grasp_cfg, ee_poses
            )
            if not reachable_parts:
                logger.warning("[assisted_grasp] No reachable parts; keeping teleop running")
                return False
            planner_cfg = dict(active_grasp_cfg)
            planner_cfg["target_index"] = 0
            planner = GraspPlanner(planner_cfg, self._robot_interface, coord)
            planner.compute_grasp_target(reachable_parts)
        except Exception:
            logger.exception("[assisted_grasp] Failed to compute grasp target")
            return False

        if planner.active_grasp is None or planner.target_prim_path is None:
            logger.warning("[assisted_grasp] Planner did not produce a target")
            return False

        arm = planner.grasp_arm
        grasp_pose = np.asarray(planner.active_grasp, dtype=float)
        if str(active_grasp_cfg.get("approach_direction", "tool")).lower() == "vertical":
            approach_direction = coord.robot_world_R_inv @ np.array([0.0, 0.0, -1.0])
        else:
            approach_direction = np.asarray(planner.R_grasp[:, 2], dtype=float)
        pregrasp_pose = grasp_pose.copy()
        pregrasp_pose[:3] -= approach_direction * float(
            active_grasp_cfg.get("pregrasp_distance", 0.12)
        )
        lift_pose = grasp_pose.copy()
        lift_pose[:3] -= approach_direction * float(active_grasp_cfg.get("lift_height", 0.17))

        side = "L" if arm == "left" else "R"
        self._robot_interface.snap_dexterous_hand_pose(side, "open")
        self._reset_teleop_ee_targets()
        self._go_home = False
        self._assisted_grasp = {
            "arm": arm,
            "side": side,
            "target_path": planner.target_prim_path,
            "hand_pose": hand_pose,
            "preshape_pose": str(active_grasp_cfg.get("preshape_pose", f"{hand_pose}_pre")),
            "phase": "pregrasp",
            "phase_elapsed": 0.0,
            "phase_start": np.asarray(ee_poses[arm], dtype=float).copy(),
            "pregrasp": pregrasp_pose,
            "grasp": grasp_pose,
            "lift": lift_pose,
            "durations": {
                "pregrasp": max(float(active_grasp_cfg.get("approach_time", 3.0)), 1e-3),
                "preshape": max(float(active_grasp_cfg.get("preshape_time", 0.8)), 1e-3),
                "reach": max(float(active_grasp_cfg.get("reach_time", 2.0)), 1e-3),
                "close": max(float(active_grasp_cfg.get("close_time", 1.0)), 1e-3),
                "settle": max(float(active_grasp_cfg.get("settle_time", 1.0)), 1e-3),
                "lift": max(float(active_grasp_cfg.get("lift_time", 2.0)), 1e-3),
            },
            "ik_kwargs": {
                "rot_weight": float(active_grasp_cfg.get("ik_rot_weight", 0.35)),
                "pos_tol": float(active_grasp_cfg.get("ik_pos_tol", 0.008)),
                "rot_tol": float(active_grasp_cfg.get("ik_rot_tol", 0.08)),
                "null_weight": float(active_grasp_cfg.get("ik_null_weight", 0.02)),
                "max_iter": int(active_grasp_cfg.get("ik_max_iter", 300)),
                "damping": float(active_grasp_cfg.get("ik_damping", 1e-4)),
                "dq_max": float(active_grasp_cfg.get("ik_dq_max", 0.5)),
            },
            "waypoint_tolerance": float(active_grasp_cfg.get("waypoint_tolerance", 0.025)),
            "waypoint_timeout": float(active_grasp_cfg.get("waypoint_timeout", 2.0)),
        }
        self._draw_grasp_debug_markers(planner, pregrasp_pose, grasp_pose, lift_pose)
        logger.info(
            "[assisted_grasp] Started target=%s arm=%s pose=%s",
            planner.target_prim_path,
            arm,
            self._assisted_grasp["hand_pose"],
        )
        return True

    def _cancel_assisted_grasp(self, reason: str = "operator request") -> None:
        if self._assisted_grasp is None:
            return
        logger.info("[assisted_grasp] Cancelled: %s", reason)
        self._assisted_grasp = None
        self._reset_teleop_ee_targets()
        self._clear_grasp_debug_markers()

    def _step_assisted_grasp(self, step_size: float) -> None:
        state = self._assisted_grasp
        if state is None or self._robot_interface is None:
            return

        phase = state["phase"]
        duration = state["durations"][phase]
        state["phase_elapsed"] += max(float(step_size), 0.0)

        if phase != "pregrasp":
            self._cancel_assisted_grasp(f"unexpected assisted grasp phase: {phase}")
            return

        goal = state["pregrasp"]
        target = self._interpolate_pose(
            state["phase_start"], goal, state["phase_elapsed"] / duration
        )

        arm = state["arm"]
        ik_result = self._robot_interface.control_dual_arm_ik(
            step_size=step_size,
            left_target_xyzrpy=target if arm == "left" else None,
            right_target_xyzrpy=target if arm == "right" else None,
            **state["ik_kwargs"],
        )
        if ik_result and "smoothed_positions" in ik_result:
            positions = np.asarray(ik_result["smoothed_positions"], dtype=np.float32)
            if positions.shape[0] >= 7:
                arm_slice = slice(0, 7) if arm == "left" else slice(7, 14)
                self._hold_arm_positions[arm_slice] = positions[:7]

        if state["phase_elapsed"] < duration:
            return

        ee_poses = self._robot_interface.get_ee_poses()
        if ee_poses is None:
            self._cancel_assisted_grasp("end-effector feedback unavailable")
            return
        position_error = float(np.linalg.norm(ee_poses[arm][:3] - goal[:3]))
        if position_error > state["waypoint_tolerance"]:
            if state["phase_elapsed"] < duration + state["waypoint_timeout"]:
                return
            self._cancel_assisted_grasp(
                f"pregrasp waypoint missed by {position_error:.3f} m"
            )
            return

        logger.info(
            "[assisted_grasp] Reached pregrasp target=%s arm=%s; manual control resumed",
            state["target_path"],
            arm,
        )
        self._teleop_ee_targets[arm] = np.asarray(goal, dtype=float).copy()
        self._assisted_grasp = None

    def _accumulate_teleop_ee_target(
        self,
        side: str,
        current_pose: np.ndarray,
        delta: np.ndarray,
        step_size: float,
    ) -> np.ndarray:
        current = np.asarray(current_pose[:6], dtype=np.float64)
        base = self._teleop_ee_targets.get(side)
        if base is None:
            base = current
        else:
            base = np.asarray(base, dtype=np.float64)

        # Accumulate from the last commanded EE target. Using measured EE pose
        # every keypress bakes in arm settling drift and can make all keys sag.
        bounded_delta = np.asarray(delta, dtype=np.float64).copy()
        xyz_norm = float(np.linalg.norm(bounded_delta[:3]))
        if xyz_norm > self._teleop_target_max_xyz_step:
            bounded_delta[:3] *= self._teleop_target_max_xyz_step / xyz_norm

        bounded_delta[3:] = np.clip(
            bounded_delta[3:],
            -self._teleop_target_max_rpy_step,
            self._teleop_target_max_rpy_step,
        )

        target = base + bounded_delta
        self._teleop_ee_targets[side] = target
        return target.copy()

    @staticmethod
    def _resolve_urdf_path(urdf_path: str) -> str:
        import os

        if os.path.isabs(urdf_path) and os.path.isfile(urdf_path):
            return urdf_path

        env_root = os.environ.get("ZOLLENT_REPO_ROOT")
        baseline = Path(__file__).resolve().parents[4]

        candidates = []
        if env_root:
            candidates.append(Path(env_root) / urdf_path)
        if Path("/workspace/WalkerS2-Model").is_dir():
            candidates.append(Path("/workspace") / urdf_path)
        candidates.extend(
            [
                baseline.parent / urdf_path,
                baseline / urdf_path,
                Path(urdf_path),
            ]
        )

        tried = []
        for candidate in candidates:
            tried.append(str(candidate))
            if candidate.is_file():
                return str(candidate.resolve())

        raise FileNotFoundError(
            f"URDF not found: {urdf_path!r} (tried {', '.join(tried)})"
        )

    # --- 必须实现的抽象方法 ---
    @property
    def is_calibrated(self) -> bool:
        """仿真机器人无需标定"""
        return True

    def calibrate(self) -> None:
        """
        校准机器人

        Isaac Sim 仿真环境无需校准，此方法为空操作。
        真实机器人实现可能需要重新校准原点或传感器。
        """
        logger.info("Isaac Sim 仿真环境无需校准")
        pass

    def configure(self) -> None:
        """配置机器人 (可选)"""
        pass

    @cached_property
    def _state_features(self) -> dict[str, type]:
        """状态特征定义 (20 维：14 臂关节 + 4 手指关节 + 2 夹持器控制)"""
        return dict.fromkeys(
            (
                "L_shoulder_pitch_joint.pos", "L_shoulder_roll_joint.pos", "L_shoulder_yaw_joint.pos",
                "L_elbow_roll_joint.pos", "L_elbow_yaw_joint.pos", "L_wrist_pitch_joint.pos", "L_wrist_roll_joint.pos",
                "R_shoulder_pitch_joint.pos", "R_shoulder_roll_joint.pos", "R_shoulder_yaw_joint.pos",
                "R_elbow_roll_joint.pos", "R_elbow_yaw_joint.pos", "R_wrist_pitch_joint.pos", "R_wrist_roll_joint.pos",
                "L_finger1_joint.pos", "L_finger2_joint.pos",
                "R_finger1_joint.pos", "R_finger2_joint.pos",
                "left_gripper",  
                "right_gripper",  
            ),
            float,
        )

    @cached_property
    def _camera_features(self) -> dict[str, tuple[int, int, int]]:
        """相机特征定义 - 形状为 (H, W, 3)，与 numpy 图像格式一致"""
        return {
            name: (self.config.camera_height, self.config.camera_width, 3)
            for name in self.CAMERA_NAMES
        }

    @property
    def env_state_dim(self) -> int:
        """环境物体位姿维度 = num_objects * 7 (x, y, z, qx, qy, qz, qw)。"""
        task_cfg = getattr(self.config, "task_cfg", {})
        if not task_cfg:
            return 0

        task_number = task_cfg.get("task_number", 0)
        if task_number == 1:
            part_cfg = task_cfg.get("part", {})
            fallback_count = part_cfg.get("num_parts", 2)
            num_a = max(0, int(part_cfg.get("num_parts_a", fallback_count)))
            num_b = max(0, int(part_cfg.get("num_parts_b", fallback_count)))
            num_objects = num_a + num_b
        elif task_number == 2:
            # Task2 tracks two part types, each with num_parts instances.
            num_objects = task_cfg.get("part", {}).get("num_parts", 5) * 2
        elif task_number == 3:
            num_boxes = len(task_cfg.get('box', {}).get('box_position', []))
            num_parts = task_cfg.get('part', {}).get('num_parts', 3)
            num_objects = num_boxes * num_parts
        elif task_number == 4:
            num_objects = 0
        else:
            num_objects = 0

        return num_objects * 7

    @cached_property
    def _vel_tor_features(self) -> dict[str, type]:
        """关节速度和扭矩特征 (28 维：14 臂关节速度 + 14 臂关节扭矩)"""
        return dict.fromkeys(
            (
                "L_shoulder_pitch_joint.vel", "L_shoulder_roll_joint.vel", "L_shoulder_yaw_joint.vel",
                "L_elbow_roll_joint.vel", "L_elbow_yaw_joint.vel", "L_wrist_pitch_joint.vel", "L_wrist_roll_joint.vel",
                "R_shoulder_pitch_joint.vel", "R_shoulder_roll_joint.vel", "R_shoulder_yaw_joint.vel",
                "R_elbow_roll_joint.vel", "R_elbow_yaw_joint.vel", "R_wrist_pitch_joint.vel", "R_wrist_roll_joint.vel",
                "L_shoulder_pitch_joint.tor", "L_shoulder_roll_joint.tor", "L_shoulder_yaw_joint.tor",
                "L_elbow_roll_joint.tor", "L_elbow_yaw_joint.tor", "L_wrist_pitch_joint.tor", "L_wrist_roll_joint.tor",
                "R_shoulder_pitch_joint.tor", "R_shoulder_roll_joint.tor", "R_shoulder_yaw_joint.tor",
                "R_elbow_roll_joint.tor", "R_elbow_yaw_joint.tor", "R_wrist_pitch_joint.tor", "R_wrist_roll_joint.tor",
            ),
            float,
        )

    @cached_property
    def _env_state_features(self) -> dict[str, type]:
        """环境物体位姿特征定义，每个物体 7 个自由度：x, y, z, qx, qy, qz, qw。

        命名格式：object_1_x, object_1_y, object_1_z, object_1_qx, object_1_qy, object_1_qz, object_1_qw,
                 object_2_x, ...
        """
        env_state_dim = self.env_state_dim
        if env_state_dim == 0:
            return {}

        num_objects = env_state_dim // 7

        # 为每个物体生成 7 个特征名
        features = {}
        for i in range(1, num_objects + 1):
            for suffix in ["x", "y", "z", "qx", "qy", "qz", "qw"]:
                features[f"object_{i}_{suffix}"] = float

        return features

    @property
    def observation_features(self) -> dict[str, type | tuple]:
        """
        观测特征：状态 + 相机 + 环境物体位姿（合并到 state 中）

        Returns:
            包含 20 个机器人状态特征 (float)、4 个相机图像特征 (H, W, 3)
            以及环境物体位姿特征 (object_1_x, object_1_y, ...) 的字典
        """
        return {**self._state_features, **self._camera_features, **self._env_state_features}

    @property
    def action_features(self) -> dict[str, type]:
        """
        动作特征：与状态特征相同 (20 维：14 臂关节 + 4 手指关节 + 2 夹持器控制)

        Returns:
            包含 20 个关节位置特征 (float) 的字典
        """
        return self._state_features

    @property
    def cameras(self):
        """返回相机名称列表，兼容框架的 len() 检查"""
        return self.CAMERA_NAMES

    def record_timing(self, metric_name: str, duration_s: float) -> None:
        """记录性能指标"""
        self._timing_metrics.setdefault(metric_name, TimingMetric()).update(duration_s)

    # ---- 回调注册/注销方法 ----

    def _register_world_callbacks(self) -> None:
        """注册物理和渲染回调到 World"""
        if self._world is None or self._callbacks_registered:
            return

        self._world.add_physics_callback("robot_control", self._robot_control_callback)
        self._world.add_physics_callback("score_input_record", self._score_input_record_callback)
        self._world.add_physics_callback("foam_sync", self._foam_sync_callback)
        if self.config.enable_sim_cameras:
            self._world.add_render_callback("camera_images", self._camera_images_callback)
        self._callbacks_registered = True
        logger.info(
            "Physics callbacks registered%s",
            " with camera rendering" if self.config.enable_sim_cameras else " without sensor cameras",
        )

    def _unregister_world_callbacks(self) -> None:
        """注销所有回调"""
        if self._world is None or not self._callbacks_registered:
            return

        remove_physics = getattr(self._world, "remove_physics_callback", None)
        remove_render = getattr(self._world, "remove_render_callback", None)
        if callable(remove_physics):
            for cb_name in ["robot_control", "score_input_record", "foam_sync"]:
                try:
                    remove_physics(cb_name)
                except Exception:
                    pass
        if callable(remove_render):
            try:
                remove_render("camera_images")
            except Exception:
                pass
        self._callbacks_registered = False

    # ---- 回调实现 ----

    def _robot_control_callback(self, step_size: float) -> None:
        """每个物理步自动执行：统一关节位置控制

        控制逻辑:
        1. 初始化：首次调用时快照当前关节状态作为保持目标
        2. 推理模式：消费 _pending_absolute_action 更新保持目标
        3. 遥操作模式：调用 teleop.get_action_numpy() 获取键盘动作（帧门控在 teleop 内处理）
        4. go_home 模式：检测 toggle_go_home 按键，触发插值回到初始位置
        5. 无输入：持续发出上一帧的保持目标

        注意：帧门控 + 队列合并逻辑在 teleop.get_action_numpy() 内处理，
        因为键盘监听器在 teleop 中，_pressed_keys 和 _keyboard_cmd_queue 由 teleop 管理。
        """
        if not self.is_connected:
            return

        import time as _time
        _now = _time.perf_counter()
        if self._phys_cb_t0 is None:
            self._phys_cb_t0 = _now
        self._phys_cb_count += 1
        _elapsed = _now - self._phys_cb_t0
        if _elapsed >= 1.0:
            self.measured_physics_hz = self._phys_cb_count / _elapsed
            self._phys_cb_count = 0
            self._phys_cb_t0 = _now

        # 检查 go_home 按键（从 teleop 读取按键状态）
        if self._teleop is not None:
            keyboard_state = self._teleop.get_keyboard_state()
            if keyboard_state.get("go_home"):
                # 检测按键按下边缘（防止长按重复触发）
                if not getattr(self, '_go_home_key_was_pressed', False):
                    self._go_home_key_was_pressed = True
                    self._go_home = not self._go_home
                    if self._go_home:
                        self._cancel_assisted_grasp("go_home requested")
                        self._reset_teleop_ee_targets()
                        logger.info("[go_home] 开始插值回到初始位置...")
                    else:
                        self._reset_teleop_ee_targets()
                        logger.info("[go_home] 取消回到初始位置，恢复正常控制")
            else:
                self._go_home_key_was_pressed = False

        # 初始化保持目标 — use standing arms, never snapshot collapsed sim state.
        if self._hold_arm_positions is None:
            self._hold_arm_positions = np.array(
                self._robot_interface.arm_joint_initial_positions, dtype=np.float32
            )
            self._hold_finger_positions = np.array(
                self._robot_interface.finger_joint_initial_positions or [0.0] * 4,
                dtype=np.float32,
            )
            logger.info("[callback] Using standing pose arm targets")

        # 读取并消费 Inference mode 的 pending action
        with self._callback_lock:
            abs_action = self._pending_absolute_action
            if abs_action is not None:
                abs_action = abs_action.copy()
                self._pending_absolute_action = None

        if (abs_action is not None) and (not self._go_home):
            # ====== 推理/回放模式：直接使用记录的关节位置 ======
            self._cancel_assisted_grasp("absolute action received")
            # action 布局：[0:14]=arm, [14:18]=finger_positions, [18]=left_cmd, [19]=right_cmd
            self._reset_teleop_ee_targets()
            self._hold_arm_positions = abs_action[:14].copy()
            if abs_action.shape[0] >= 18:
                self._hold_finger_positions = np.array([self._robot_interface.gripper_open_width]*4)
                # self._hold_finger_positions = abs_action[14:18].copy()
            if abs_action.shape[0] >= 20:
                self._left_gripping = float(abs_action[18]) < -0.5
                if self._left_gripping:
                    self._hold_finger_positions[:2] = np.array([self._robot_interface.gripper_close_width]*2)
                    self._robot_interface.close_dexterous_hand("L")
                else:
                    self._robot_interface.open_dexterous_hand("L")
                self._right_gripping = float(abs_action[19]) < -0.5
                if self._right_gripping:
                    self._hold_finger_positions[2:4] = np.array([self._robot_interface.gripper_close_width]*2)
                    self._robot_interface.close_dexterous_hand("R")
                else:
                    self._robot_interface.open_dexterous_hand("R")
                self._robot_interface.apply_dexterous_hand_targets()
            print(f"[_robot_control_callback] left_gripping={self._left_gripping}, right_gripping={self._right_gripping}")
        elif not self._go_home:
            # ====== 遥操作模式：通过 teleop 读取键盘状态，计算 IK ======
            if self._teleop is not None:
                # 使用回调模式获取键盘动作
                left_delta, right_delta, left_gripper, right_gripper = self._teleop.get_action_numpy(
                    frame_id=self._send_action_step_idx
                )
                keyboard_state = self._teleop.get_keyboard_state()
                grasp_pressed = bool(keyboard_state.get("assisted_grasp"))
                cancel_grasp_pressed = bool(keyboard_state.get("cancel_assisted_grasp"))
                if cancel_grasp_pressed and not self._cancel_grasp_key_was_pressed:
                    self._cancel_assisted_grasp()
                if grasp_pressed and not self._assisted_grasp_key_was_pressed:
                    if self._assisted_grasp is None:
                        self._start_assisted_grasp()
                    else:
                        logger.info("[assisted_grasp] Sequence already running; press 'c' to cancel")
                self._assisted_grasp_key_was_pressed = grasp_pressed
                self._cancel_grasp_key_was_pressed = cancel_grasp_pressed

                has_left_input = np.linalg.norm(left_delta) > 1e-8
                has_right_input = np.linalg.norm(right_delta) > 1e-8
                has_gripper_input = abs(left_gripper) > 0.01 or abs(right_gripper) > 0.01
                ik_status_debug = ""
                ik_joint_delta_debug = ""
                ee_poses = None

                if self._assisted_grasp is not None:
                    self._step_assisted_grasp(step_size)
                elif has_left_input or has_right_input:
                    if has_left_input:
                        self._robot_interface.prepare_dexterous_hand_for_arm_motion("L")
                    if has_right_input:
                        self._robot_interface.prepare_dexterous_hand_for_arm_motion("R")
                    ee_poses = self._robot_interface.get_ee_poses()
                    if ee_poses is not None:
                        left_target = (
                            self._accumulate_teleop_ee_target("left", ee_poses["left"], left_delta, step_size)
                            if has_left_input
                            else None
                        )
                        right_target = (
                            self._accumulate_teleop_ee_target("right", ee_poses["right"], right_delta, step_size)
                            if has_right_input
                            else None
                        )

                        ik_result = self._robot_interface.control_dual_arm_ik(
                            step_size=step_size,
                            left_target_xyzrpy=left_target,
                            right_target_xyzrpy=right_target,
                            rot_weight=self._teleop_ik_rot_weight,
                            pos_tol=self._teleop_ik_pos_tol,
                            rot_tol=self._teleop_ik_rot_tol,
                        )
                        if ik_result and "smoothed_positions" in ik_result:
                            sp = ik_result["smoothed_positions"]
                            offset = 0
                            if "left_joint_positions" in ik_result:
                                left_sp = np.array(sp[offset:offset+7], dtype=np.float32)
                                ik_joint_delta_debug += (
                                    f" Ldq={float(np.linalg.norm(left_sp - self._hold_arm_positions[:7])):.4f}"
                                )
                                self._hold_arm_positions[:7] = left_sp
                                offset += 7
                            if "right_joint_positions" in ik_result:
                                right_sp = np.array(sp[offset:offset+7], dtype=np.float32)
                                ik_joint_delta_debug += (
                                    f" Rdq={float(np.linalg.norm(right_sp - self._hold_arm_positions[7:14])):.4f}"
                                )
                                self._hold_arm_positions[7:14] = right_sp
                            ik_status_debug = (
                                f" ik_ok=({ik_result.get('left_success', None)},"
                                f"{ik_result.get('right_success', None)})"
                            )

                # 夹持器控制
                gripper_step = 0.002
                g_open = self._robot_interface.gripper_open_width
                g_close = self._robot_interface.gripper_close_width
                g_lo, g_hi = min(g_open, g_close), max(g_open, g_close)
                if self._assisted_grasp is None and abs(left_gripper) > 0.01:
                    self._hold_finger_positions[:2] = np.clip(
                        self._hold_finger_positions[:2] - left_gripper * gripper_step, g_lo, g_hi
                    )
                    self._left_gripping = left_gripper < 0
                    self._robot_interface.nudge_dexterous_hand(
                        "L", -left_gripper, fraction_step=0.01
                    )
                if self._assisted_grasp is None and abs(right_gripper) > 0.01:
                    self._hold_finger_positions[2:4] = np.clip(
                        self._hold_finger_positions[2:4] - right_gripper * gripper_step, g_lo, g_hi
                    )
                    self._right_gripping = right_gripper < 0
                    self._robot_interface.nudge_dexterous_hand(
                        "R", -right_gripper, fraction_step=0.01
                    )

                pose_name = None
                if self._assisted_grasp is not None:
                    pose_name = None
                elif keyboard_state.get("hand_power"):
                    pose_name = "power"
                elif keyboard_state.get("hand_pinch"):
                    pose_name = "pinch"
                elif keyboard_state.get("hand_tripod"):
                    pose_name = "tripod"

                if pose_name is not None:
                    target_sides = []
                    current_arm = getattr(self._teleop, "current_control_arm", "left")
                    bimanual = bool(getattr(self._teleop, "bimanual_control_enabled", False))
                    if current_arm == "left" or bimanual:
                        target_sides.append("L")
                    if current_arm == "right" or bimanual:
                        target_sides.append("R")
                    for side in target_sides:
                        self._robot_interface.close_dexterous_hand(side, pose_name)
                        if side == "L":
                            self._left_gripping = True
                        else:
                            self._right_gripping = True

                if has_left_input or has_right_input or has_gripper_input or pose_name is not None:
                    log_now = _time.perf_counter()
                    if log_now - self._last_teleop_input_log_t >= 0.5:
                        self._last_teleop_input_log_t = log_now
                        ik_debug = ""
                        if has_left_input or has_right_input:
                            left_target_error = (
                                np.linalg.norm(self._teleop_ee_targets["left"][:3] - ee_poses["left"][:3])
                                if has_left_input and self._teleop_ee_targets.get("left") is not None and ee_poses is not None
                                else 0.0
                            )
                            right_target_error = (
                                np.linalg.norm(self._teleop_ee_targets["right"][:3] - ee_poses["right"][:3])
                                if has_right_input and self._teleop_ee_targets.get("right") is not None and ee_poses is not None
                                else 0.0
                            )
                            ik_debug = (
                                f" target_err=({left_target_error:.4f},{right_target_error:.4f})"
                                f"{ik_status_debug}{ik_joint_delta_debug}"
                            )
                        logger.info(
                            "[teleop] left_delta=%s right_delta=%s left_gripper=%.1f right_gripper=%.1f pose=%s%s",
                            np.array2string(left_delta, precision=3, suppress_small=True),
                            np.array2string(right_delta, precision=3, suppress_small=True),
                            left_gripper,
                            right_gripper,
                            pose_name,
                            ik_debug,
                        )

        else:
            self._robot_interface.open_dexterous_hand("L")
            self._robot_interface.open_dexterous_hand("R")
            self._robot_interface.apply_dexterous_hand_targets()
            arm_finger_indices = self._robot_interface.arm_joint_indices + self._robot_interface.finger_joint_indices
            if not self._robot_interface.joint_interpolator.interp_active:
                print('[_robot_control_callback] Starting interpolation to initial position...')
                joint_states = self._robot_interface.get_joint_states()
                if joint_states is None:
                    return
                self._robot_interface.joint_interpolator.set_target(
                    start_q=torch.tensor(joint_states['all_positions'])[arm_finger_indices],
                    target_q=torch.tensor(self._robot_interface.initial_joint_positions)[arm_finger_indices],
                    num_steps=self._num_interpolation_steps
                )
            arm_finger_positions = self._robot_interface.joint_interpolator.step()  # 执行一步插值
            if isinstance(arm_finger_positions, torch.Tensor):
                arm_finger_positions = arm_finger_positions.detach().cpu().numpy()
            else:
                arm_finger_positions = np.asarray(arm_finger_positions, dtype=np.float32)
            self._hold_arm_positions = arm_finger_positions[:14]
            self._hold_finger_positions = arm_finger_positions[14:18]
            self._left_gripping = False
            self._right_gripping = False
            if self._robot_interface.joint_interpolator.is_finished():
                self._go_home = False  
                joint_states = self._robot_interface.get_joint_states()
                if joint_states is None:
                    return
                all_positions = joint_states['all_positions']
                self._robot_interface.reset_ik(all_positions)  
                self._reset_teleop_ee_targets()
                print('[_robot_control_callback] Interpolation to initial position completed.')      
                      
        if not self._robot_interface._articulation_physics_ready():
            return

        # USD baseline: set_arm + set_body(0) (zero = standing in the asset).
        # URDF physical mode: hard-stabilize the body, but move arms through PD drives.
        if self._robot_interface.use_explicit_standing_body:
            if self._robot_interface.uses_physical_arm_control:
                self._robot_interface.hold_stabilizing_joints()
                self._robot_interface.set_arm_joint_positions_physical(
                    self._hold_arm_positions,
                    step_size=step_size,
                )
            else:
                joint_targets = self._robot_interface.build_joint_target_vector(
                    self._hold_arm_positions,
                    self._hold_finger_positions if self._robot_interface.has_old_gripper else None,
                )
                self._robot_interface.apply_all_joint_targets(joint_targets)
                self._robot_interface.set_arm_joint_positions_hard(self._hold_arm_positions)
        else:
            self._robot_interface.set_arm_joint_positions(
                target_arm_positions=self._hold_arm_positions.tolist(),
                task_num=self.config.task_cfg.get("task_number", 1),
            )
            self._robot_interface.set_body_joint_positions(
                target_body_positions=0.0,
                task_num=self.config.task_cfg.get("task_number", 1),
            )

        # Keep dexterous hand presets stable during teleop. The hand interface
        # accepts the profile argument for assisted-grasp compatibility, but
        # currently hard-holds preset joints to avoid idle finger oscillation.
        self._robot_interface.apply_dexterous_hand_targets()

        # 夹持器控制：
        #   夹持时：NaN（关闭PD）+ close_tau（纯力矩），避免位置+力矩叠加导致过夹
        #   释放时：open_width（PD开爪）+ open_tau if stuck（主动助力防卡死）
        close_tau = getattr(self._robot_interface, "gripper_close_tau", 100.0)
        open_tau = getattr(self._robot_interface, "gripper_open_tau", -100.0)
        open_width = self._robot_interface.gripper_open_width
        stuck_threshold = 0.005  # 手指实际位置超过 open_width 5mm 以上视为卡住

        # 读取实际手指位置用于卡死检测
        _states = self._robot_interface.get_joint_states()
        actual_finger_pos = (
            np.array(_states["finger_positions"], dtype=np.float32)
            if _states is not None and "finger_positions" in _states
            else np.array([], dtype=np.float32)
        )

        has_old_gripper = bool(
            getattr(self._robot_interface, "has_old_gripper", False)
            and actual_finger_pos is not None
            and actual_finger_pos.shape[0] >= 4
        )

        # The hand-version Walker S2 does not expose the old 4 gripper joints:
        # L_finger1_joint, L_finger2_joint, R_finger1_joint, R_finger2_joint.
        # In that case, skip all old gripper control but keep arm control running.
        if has_old_gripper:
            gripping = [
                self._left_gripping, self._left_gripping,
                self._right_gripping, self._right_gripping,
            ]

            finger_pos_cmd = []
            efforts = []

            for i, is_gripping in enumerate(gripping):
                if is_gripping:
                    finger_pos_cmd.append(float("nan"))  # 关闭PD，不与力矩叠加
                    efforts.append(close_tau)
                else:
                    finger_pos_cmd.append(open_width)    # PD 驱动开爪
                    if actual_finger_pos[i] > open_width + stuck_threshold:
                        efforts.append(open_tau)          # 主动开爪助力（防卡死）
                    else:
                        efforts.append(0.0)

            self._robot_interface.set_finger_positions(
                target_fingers=finger_pos_cmd,
                task_num=self.config.task_cfg.get("task_number", 1),
            )
            self._robot_interface.apply_finger_efforts(efforts)

    def _score_input_record_callback(self, step_size: float) -> None:
        """记录分数/目标物体变换"""
        if self._scene_builder is None:
            return
        get_transforms = getattr(self._scene_builder, "get_target_object_transforms", None)
        if callable(get_transforms):
            get_transforms(step_size)

    def _foam_sync_callback(self, _step_size: float) -> None:
        """task4 专用：同步泡沫到箱子"""
        if self._scene_builder is None:
            return
        sync_foam = getattr(self._scene_builder, "sync_foam_to_box", None)
        if callable(sync_foam):
            sync_foam()

    def _camera_images_callback(self, _step: float) -> None:
        """渲染回调：抓取相机图像并缓存"""
        if not self.is_connected:
            return
        import time as _time
        _now = _time.perf_counter()
        if self._render_cb_t0 is None:
            self._render_cb_t0 = _now
        self._render_cb_count += 1
        _elapsed = _now - self._render_cb_t0
        if _elapsed >= 1.0:
            self.measured_render_hz = self._render_cb_count / _elapsed
            self._render_cb_count = 0
            self._render_cb_t0 = _now

        camera_data: dict[str, np.ndarray] = {}
        for cam_name in self.CAMERA_NAMES:
            try:
                rgb = self._robot_interface.get_camera_rgb(cam_name)
                if rgb is not None:
                    camera_data[cam_name] = rgb
            except Exception:
                continue

        if camera_data:
            with self._callback_lock:
                self._latest_camera_rgb = {name: frame.copy() for name, frame in camera_data.items()}
            self._head_visualizer.update_cameras(camera_data)



    def connect(self, calibrate: bool = True) -> None:
        """
        连接机器人并初始化 Isaac Sim 仿真环境

        Args:
            calibrate (bool): 是否自动标定 (仿真环境忽略)

        连接流程:
            1. 创建 SimulationApp
            2. 加载场景 USD
            3. 创建并初始化 World
            4. SceneBuilder 构建场景
            5. 创建机器人接口并初始化
        """
        if self.is_connected:
            logger.info("已经连接")
            return

        if not self.config.task_cfg_path:
            raise ValueError("必须提供 task_cfg_path 以加载场景")

        # 步骤 1: 创建 SimulationApp
        from isaacsim import SimulationApp
        logger.info("步骤 1: 创建 SimulationApp...")
        self._kit = SimulationApp({
            "width": self.config.sim_width,
            "height": self.config.sim_height,
            "headless": self.config.headless,
        })
        logger.info("SimulationApp 创建成功")

        # Older graph schemas embedded in the scene assets trigger the same
        # non-actionable file-format-upgrade callback error for every graph.
        # Silence that channel only while assets are being upgraded, then
        # restore its previous threshold before normal simulation begins.
        import carb.settings

        carb_settings = carb.settings.get_settings()
        omnigraph_log_key = "/log/channels/omni.graph"
        previous_omnigraph_log_level = carb_settings.get(omnigraph_log_key)
        carb_settings.set(omnigraph_log_key, "fatal")

        # 步骤 2: 加载场景 USD（关键！之前缺少这一步）
        from isaacsim.core.api import World
        import omni.usd as omni_usd


        logger.info("步骤 2: 加载场景 USD...")
        import os
        scene_path = os.path.join(self.config.task_cfg.get("root_path", ""), self.config.task_cfg.get("scene_usd", ""))
        logger.info(f"场景路径: {scene_path}")
        
        if not os.path.exists(scene_path):
            raise FileNotFoundError(f"场景 USD 文件不存在: {scene_path}")
        
        omni_usd.get_context().open_stage(scene_path)
        logger.info("场景 USD 加载成功")

        # 步骤 3: 创建 World（现在 World 会基于已加载的场景）
        logger.info("步骤 3: 创建 World...")
        if World is None:
            raise ImportError("isaacsim.core.api.World 不可用")
        
        self._world = World(
            stage_units_in_meters=1.0,
            physics_dt=self.config.physics_dt,
            rendering_dt=self.config.rendering_dt,
        )
        self._world.initialize_physics()
        logger.info("World 初始化完成")

        # 步骤 4: SceneBuilder 构建场景（添加桌子、零件、箱子等）
        logger.info("步骤 4: SceneBuilder 构建场景...")
        try:
            # 导入 SceneBuilder 和 DataLogger
            # 添加项目根目录到 sys.path，使得 lerobot.Ubtech_sim 可导入
            project_root = Path(__file__).parent.parent.parent.parent.parent
            if str(project_root) not in os.sys.path:
                os.sys.path.append(str(project_root))
                logger.info(f"已将 {project_root} 添加到 sys.path")
            from Ubtech_sim.source.SceneBuilder import SceneBuilder
            from Ubtech_sim.source.DataLogger import DataLogger
            
            # 创建 DataLogger（禁用文件记录）
            data_logger = DataLogger(
                enabled=False,
                csv_path="",
                camera_enabled=False,
                camera_hdf5_path="",
            )
            print("步骤 4: SceneBuilder 构建场景...",self.config.task_cfg)
            self._scene_builder = SceneBuilder(self.config.task_cfg, data_logger=data_logger)
            uses_urdf = bool(self.config.task_cfg.get("robot", {}).get("robot_urdf"))
            settle_time = float(
                self.config.task_cfg.get("grasp", {}).get("settle_time", 2.0)
            )

            self._scene_builder.build_all()

            # Official main.py for both USD and URDF: settle scene, pause, then add robot.
            self._world.play()
            settle_steps = max(1, int(settle_time / self._world.get_physics_dt()))
            for _ in range(settle_steps):
                self._world.step(render=False)
            logger.info(f"Scene settled ({settle_time}s, {settle_steps} steps)")

            self._world.pause()
            self._scene_builder.build_robot()
            logger.info("Robot built (physics paused)")

            if uses_urdf:
                logger.info("Rebuilding physics view after URDF import...")
                self._world.reset()
                if self._world.is_playing():
                    self._world.pause()

            logger.info("SceneBuilder 场景构建完成")
        except ImportError as e:
            carb_settings.set(
                omnigraph_log_key,
                previous_omnigraph_log_level or "error",
            )
            logger.error(f"无法导入 SceneBuilder: {e}")
            raise
        except Exception as e:
            carb_settings.set(
                omnigraph_log_key,
                previous_omnigraph_log_level or "error",
            )
            logger.error(f"场景构建失败: {e}")
            import traceback
            logger.error(traceback.format_exc())
            raise

        carb_settings.set(
            omnigraph_log_key,
            previous_omnigraph_log_level or "error",
        )

        # 步骤 5: 创建机器人接口（连接到 SceneBuilder 创建的机器人）
        logger.info("步骤 5: 创建机器人接口...")

        actual_prim_path = self._scene_builder.robot_prim_path or self.config.prim_path
        self.config.prim_path = actual_prim_path
        logger.info(f"Robot articulation prim: {actual_prim_path}")

        urdf_path = self._resolve_urdf_path(self.config.urdf_path)
        uses_urdf = bool(self.config.task_cfg.get("robot", {}).get("robot_urdf"))

        self._robot_interface = IsaacSimRobotInterface(
            prim_path=actual_prim_path,
            name=self.config.robot_name,
            world=self._world,
            urdf_path=urdf_path,
            use_explicit_standing_body=uses_urdf,
            arm_control_cfg=self.config.task_cfg.get("robot", {}).get("arm_control", {}),
            enable_cameras=self.config.enable_sim_cameras,
        )

        logger.info("初始化 Articulation（物理暂停中）...")
        self._robot_interface.initialize()

        if uses_urdf:
            import omni.kit.app

            for _ in range(5):
                omni.kit.app.get_app().update()

        if self._scene_builder is not None:
            self._scene_builder.init_coordinate_transform(
                self._robot_interface.ik_solver
            )
            logger.info("Coordinate transform initialized")

        self._hold_arm_positions = np.array(
            self._robot_interface.arm_joint_initial_positions, dtype=np.float32
        )
        self._hold_finger_positions = np.array(
            self._robot_interface.finger_joint_initial_positions or [0.0] * 4,
            dtype=np.float32,
        )
        logger.info("Using standing pose as initial hold targets")

        if uses_urdf:
            self._robot_interface.apply_all_joint_targets(
                self._robot_interface.standing_joint_positions
            )

        self._register_world_callbacks()

        self._world.play()
        logger.info("World started")

        if uses_urdf:
            # play() resets sim state — re-apply standing before any physics step.
            self._robot_interface.apply_standing_pose_after_play()
            self._robot_interface.apply_all_joint_targets(
                self._robot_interface.standing_joint_positions
            )

        settle_time = float(
            self.config.task_cfg.get("grasp", {}).get("settle_time", 2.0)
        )
        if uses_urdf:
            settle_steps = max(1, int(settle_time / self._world.get_physics_dt()))
            for _ in range(settle_steps):
                self._robot_interface.apply_all_joint_targets(
                    self._robot_interface.standing_joint_positions
                )
                self._world.step(render=False)
            logger.info(f"Scene settled with robot ({settle_time}s, {settle_steps} steps)")
        else:
            for _ in range(10):
                self._world.step(render=False)
            logger.info("Physics warmup complete (10 steps)")

        logger.info(f"连接成功！正在控制 {len(self._robot_interface.arm_joint_indices)} 个手臂关节")

    def sim_is_running(self) -> bool:
        """Return False when the Isaac Sim app has exited (window closed or crash)."""
        if self._kit is None:
            return self.is_connected
        if hasattr(self._kit, "is_running"):
            return bool(self._kit.is_running())
        return self.is_connected

    def pump_simulation(self, render: bool = True) -> bool:
        """Advance Isaac Sim during blocking waits (Enter prompt, etc.).

        Uses world.step() so physics callbacks keep holding the robot pose.
        Returns False if the SimulationApp is no longer running.
        """
        if not self.is_connected:
            return False
        if not self.sim_is_running():
            logger.error("Isaac Sim exited unexpectedly (kit.is_running()=False)")
            return False
        try:
            if self._world is not None:
                self._world.step(render=render)
            elif self._kit is not None:
                self._kit.update()
        except Exception:
            logger.exception("pump_simulation failed")
            raise
        if not self.sim_is_running():
            logger.error("Isaac Sim stopped during pump_simulation")
            return False
        return True

    def disconnect(self) -> None:
        """断开连接并清理资源

        清理流程:
            1. 注销回调
            2. 清理机器人接口
            3. 停止 World
            4. 关闭 SimulationApp
        """
        if not self.is_connected:
            return

        # 1. 注销回调
        self._unregister_world_callbacks()

        # 2. 清理机器人接口
        if self._robot_interface:
            self._robot_interface.cleanup()
            self._robot_interface = None

        # 4. 停止 World
        if self._world:
            try:
                self._world.stop()
            except Exception:
                pass
            self._world = None

        # 5. 关闭 SimulationApp
        if self._kit:
            try:
                self._kit.close()
            except Exception:
                pass
            self._kit = None

        logger.info("已断开连接")

    def send_action(self, action: RobotAction | None = None) -> RobotAction:
        """
        发送动作指令到机器人

        Args:
            action (RobotAction | None): 动作字典或 None
                - not None: 推理/回放模式，写入 pending，由 callback 消费执行
                - None: 遥操作模式，callback 直接读取键盘状态完成控制

        Returns:
            RobotAction: 实际执行的动作字典（用于记录），包含 20 个键：
                - 14 臂关节位置
                - 4 手指关节位置
                - 2 夹持器控制指令 (left_gripper, right_gripper)

        控制模式:
            1. 推理/回放模式 (action is not None):
               - 写入 _pending_absolute_action
               - callback 在下一物理步消费并执行
            2. 遥操作模式 (action is None):
               - callback 直接读取键盘状态完成控制
               - 这里仅构建 action 字典用于数据记录
        """
        send_action_start_t = time.perf_counter()
        try:
            if not self.is_connected:
                raise RuntimeError("机器人未连接")

            if action is not None:
                # ====== 模式 A: 推理/回放（统一 20D） ======
                # 写入 pending（callback 将在下一物理步消费）
                # 解析动作字典为 numpy 数组用于验证
                arm_positions = np.array([
                    action[f"L_{j}.pos"] for j in
                    ["shoulder_pitch_joint", "shoulder_roll_joint", "shoulder_yaw_joint", "elbow_roll_joint", "elbow_yaw_joint", "wrist_pitch_joint", "wrist_roll_joint"]
                ] + [
                    action[f"R_{j}.pos"] for j in
                    ["shoulder_pitch_joint", "shoulder_roll_joint", "shoulder_yaw_joint", "elbow_roll_joint", "elbow_yaw_joint", "wrist_pitch_joint", "wrist_roll_joint"]
                ], dtype=np.float32)

                finger_positions = np.array([
                    action.get("L_finger1_joint.pos", 0.0),
                    action.get("L_finger2_joint.pos", 0.0),
                    action.get("R_finger1_joint.pos", 0.0),
                    action.get("R_finger2_joint.pos", 0.0),
                ], dtype=np.float32)

                left_gripper = action.get("left_gripper", 0.0)
                right_gripper = action.get("right_gripper", 0.0)
                print(f"[send_action] left_gripper={left_gripper}, right_gripper={right_gripper}")
                # 构建 20D action 数组用于验证
                action_np = np.concatenate([
                    arm_positions,
                    finger_positions,
                    np.array([left_gripper, right_gripper], dtype=np.float32)
                ])

                if action_np.shape[0] != self.ACTION_DIM:
                    raise ValueError(f"推理动作 Dimension error: Expected {self.ACTION_DIM}, got {action_np.shape[0]}")

                # 写入 pending（callback 将在下一物理步消费）
                with self._callback_lock:
                    self._pending_absolute_action = action_np.copy()

                # 执行一步物理仿真让 callback 消费 pending action
                self.step(render=True)

                return action

            else:
                # ====== 模式 B: 遥操作（仅构建 action 字典用于记录） ======
                # 实际控制由 callback 通过读取键盘状态完成
                joints_states = self._robot_interface.get_joint_states()
                if joints_states and "arm_positions" in joints_states:
                    arm_pos = joints_states["arm_positions"]
                    finger_pos = joints_states.get("finger_positions", [0.0] * 4)

                    # 构建 action 字典
                    action_dict: RobotAction = {
                        f"L_shoulder_pitch_joint.pos": arm_pos[0],
                        f"L_shoulder_roll_joint.pos": arm_pos[1],
                        f"L_shoulder_yaw_joint.pos": arm_pos[2],
                        f"L_elbow_roll_joint.pos": arm_pos[3],
                        f"L_elbow_yaw_joint.pos": arm_pos[4],
                        f"L_wrist_pitch_joint.pos": arm_pos[5],
                        f"L_wrist_roll_joint.pos": arm_pos[6],
                        f"R_shoulder_pitch_joint.pos": arm_pos[7],
                        f"R_shoulder_roll_joint.pos": arm_pos[8],
                        f"R_shoulder_yaw_joint.pos": arm_pos[9],
                        f"R_elbow_roll_joint.pos": arm_pos[10],
                        f"R_elbow_yaw_joint.pos": arm_pos[11],
                        f"R_wrist_pitch_joint.pos": arm_pos[12],
                        f"R_wrist_roll_joint.pos": arm_pos[13],
                        "L_finger1_joint.pos": finger_pos[0],
                        "L_finger2_joint.pos": finger_pos[1],
                        "R_finger1_joint.pos": finger_pos[2],
                        "R_finger2_joint.pos": finger_pos[3],
                        "left_gripper": -1.0 if self._left_gripping else 1.0,
                        "right_gripper": -1.0 if self._right_gripping else 1.0,
                    }
                else:
                    raise RuntimeError("无法获取关节状态以构建 action 字典")
                # 执行一步物理仿真
                self.step(render=True)

                return action_dict

        finally:
            duration_s = time.perf_counter() - send_action_start_t
            self.record_timing("send_action", duration_s)

    def step(self, render: bool = True) -> None:
        """推进仿真一步

        Args:
            render: 是否渲染图像，默认 True
        """
        if self._world:
            self._world.step(render=render)
            self._send_action_step_idx += 1

    def get_observation(self) -> RobotObservation:
        """
        获取机器人观测 (关节状态 + 相机 RGB + 环境状态)

        Returns:
            RobotObservation: 扁平字典，key 与 observation_features 完全匹配:
                - 14 臂关节位置 (float): L_shoulder_pitch_joint.pos, ..., R_wrist_roll_joint.pos
                - 4 手指关节位置 (float): L_finger1_joint.pos, L_finger2_joint.pos, R_finger1_joint.pos, R_finger2_joint.pos
                - 2 夹持器控制 (float): left_gripper, right_gripper
                - 4 相机 RGB 图像 (H, W, 3): head_left, head_right, wrist_left, wrist_right
                - 可选环境状态向量 (N,): observation.environment_state
        """
        start_t = time.perf_counter()
        try:
            if not self.is_connected:
                raise RuntimeError("机器人未连接")

            obs: RobotObservation = {}

            # 获取关节状态 - 扁平 key 与 observation_features 匹配
            joints_states = self._robot_interface.get_joint_states()
            if joints_states and 'arm_positions' in joints_states:
                arm_pos = joints_states['arm_positions']
                finger_pos = joints_states.get('finger_positions', [0.0] * 4)

                # 14 臂关节
                arm_joint_names = [
                    "L_shoulder_pitch_joint", "L_shoulder_roll_joint", "L_shoulder_yaw_joint",
                    "L_elbow_roll_joint", "L_elbow_yaw_joint", "L_wrist_pitch_joint", "L_wrist_roll_joint",
                    "R_shoulder_pitch_joint", "R_shoulder_roll_joint", "R_shoulder_yaw_joint",
                    "R_elbow_roll_joint", "R_elbow_yaw_joint", "R_wrist_pitch_joint", "R_wrist_roll_joint",
                ]
                for i, joint_name in enumerate(arm_joint_names):
                    obs[f"{joint_name}.pos"] = torch.tensor(arm_pos[i], dtype=torch.float32)

                # 14 臂关节速度和扭矩（侧录用，不进 observation.state）
                arm_vel = joints_states.get('arm_velocities', [0.0] * 14)
                arm_tau = joints_states.get('arm_torques', [0.0] * 14)
                for i, joint_name in enumerate(arm_joint_names):
                    obs[f"_vel_{joint_name}"] = torch.tensor(arm_vel[i], dtype=torch.float32)
                    obs[f"_tor_{joint_name}"] = torch.tensor(arm_tau[i], dtype=torch.float32)

                # 4 手指关节
                finger_joint_names = ["L_finger1_joint", "L_finger2_joint", "R_finger1_joint", "R_finger2_joint"]
                for i, joint_name in enumerate(finger_joint_names):
                    obs[f"{joint_name}.pos"] = torch.tensor(finger_pos[i], dtype=torch.float32)

                # 2 夹持器控制
                obs["left_gripper"] = torch.tensor(-1.0 if self._left_gripping else 1.0, dtype=torch.float32)
                obs["right_gripper"] = torch.tensor(-1.0 if self._right_gripping else 1.0, dtype=torch.float32)
            else:
                raise RuntimeError("无法获取关节状态")

            # 获取相机图像 - key 与 observation_features 匹配，形状为 (H, W, 3)
            for cam_name in self.CAMERA_NAMES:
                try:
                    rgbd = self._robot_interface.get_camera_rgbd(cam_name)
                    if rgbd and rgbd.get("rgb") is not None:
                        # 保持 (H, W, 3) 格式，不做 permute
                        img = torch.from_numpy(rgbd["rgb"]).float() / 255.0
                        obs[cam_name] = img
                    else:
                        logger.warning(f"相机 {cam_name} 无法获取 RGB 图像")
                        h, w = self.config.camera_height, self.config.camera_width
                        obs[cam_name] = torch.zeros(h, w, 3, dtype=torch.float32)
                except Exception as e:
                    logger.warning(f"获取相机 {cam_name} 图像失败：{e}")
                    h, w = self.config.camera_height, self.config.camera_width
                    obs[cam_name] = torch.zeros(h, w, 3, dtype=torch.float32)

            # 获取环境物体位姿 - 作为独立的 object_1_x, object_1_y, ... 键添加到 obs 中
            # 这些特征会被合并到 observation.state 中
            env_state_dim = self.env_state_dim
            if env_state_dim > 0:
                try:
                    if self._scene_builder is None:
                        raise RuntimeError("SceneBuilder 未初始化")

                    env_state_np = np.asarray(self._scene_builder.get_object_poses_flat(), dtype=np.float32).reshape(-1)
                    if env_state_np.shape[0] != env_state_dim:
                        raise RuntimeError(
                            f"环境状态维度不匹配：期望 {env_state_dim}, 实际 {env_state_np.shape[0]}"
                        )

                    # 将扁平向量分解为独立的 object_i_x, object_i_y, ... 键
                    # 格式：[obj1_x, obj1_y, obj1_z, obj1_qx, obj1_qy, obj1_qz, obj1_qw, obj2_x, ...]
                    num_objects = env_state_dim // 7
                    for i in range(1, num_objects + 1):
                        base_idx = (i - 1) * 7
                        obs[f"object_{i}_x"] = torch.tensor(env_state_np[base_idx], dtype=torch.float32)
                        obs[f"object_{i}_y"] = torch.tensor(env_state_np[base_idx + 1], dtype=torch.float32)
                        obs[f"object_{i}_z"] = torch.tensor(env_state_np[base_idx + 2], dtype=torch.float32)
                        obs[f"object_{i}_qx"] = torch.tensor(env_state_np[base_idx + 3], dtype=torch.float32)
                        obs[f"object_{i}_qy"] = torch.tensor(env_state_np[base_idx + 4], dtype=torch.float32)
                        obs[f"object_{i}_qz"] = torch.tensor(env_state_np[base_idx + 5], dtype=torch.float32)
                        obs[f"object_{i}_qw"] = torch.tensor(env_state_np[base_idx + 6], dtype=torch.float32)
                except Exception as e:
                    logger.warning(f"获取环境物体位姿失败：{e}")
                    # 创建零值特征
                    num_objects = env_state_dim // 7
                    for i in range(1, num_objects + 1):
                        obs[f"object_{i}_x"] = torch.tensor(0.0, dtype=torch.float32)
                        obs[f"object_{i}_y"] = torch.tensor(0.0, dtype=torch.float32)
                        obs[f"object_{i}_z"] = torch.tensor(0.0, dtype=torch.float32)
                        obs[f"object_{i}_qx"] = torch.tensor(0.0, dtype=torch.float32)
                        obs[f"object_{i}_qy"] = torch.tensor(0.0, dtype=torch.float32)
                        obs[f"object_{i}_qz"] = torch.tensor(0.0, dtype=torch.float32)
                        obs[f"object_{i}_qw"] = torch.tensor(0.0, dtype=torch.float32)

            self._last_observation = obs
            return obs

        finally:
            self._timing_metrics["get_observation"].update(time.perf_counter() - start_t)



    def log_ee_poses(self) -> None:
        """打印双臂末端姿态 (xyzrpy)，用于遥操作实时监控"""
        if not self.is_connected or self._robot_interface is None:
            return
        try:
            ee_poses = self._robot_interface.get_ee_poses()
            if ee_poses is None:
                return
            left = ee_poses.get("left")
            right = ee_poses.get("right")
            if left is not None and right is not None:
                print(
                    f"[EE] L: xyz=({left[0]:.4f}, {left[1]:.4f}, {left[2]:.4f}) "
                    f"rpy=({left[3]:.4f}, {left[4]:.4f}, {left[5]:.4f}) | "
                    f"R: xyz=({right[0]:.4f}, {right[1]:.4f}, {right[2]:.4f}) "
                    f"rpy=({right[3]:.4f}, {right[4]:.4f}, {right[5]:.4f})"
                )
        except Exception as e:
            logger.warning(f"log_ee_poses failed: {e}")

    def print_logs(self) -> None:
        """打印机器人当前状态信息"""
        if not self.is_connected:
            print("未连接")
            return

        print(f"当前控制臂：{getattr(self, 'current_control_arm', 'N/A')}")
        print(f"双臂同步模式：{getattr(self, 'bimanual_control_enabled', False)}")
        if self._hold_arm_positions is not None:
            print(f"保持位置：{self._hold_arm_positions.tolist()[:6]}...")  # 只显示前 6 个

    def reset(self) -> None:
        """重置环境：场景物体恢复初始 Pose/随机化，机器人恢复初始关节，控制接口保持不变。

        重置流程:
            1. 取消注册回调（防止 reset 过程中 callback 干扰）
            2. 重置场景（SceneBuilder.reset()）
            3. 任务 1/3 需要 world.reset() 重建物理视图
            4. 重置机器人关节到初始 Pose
            5. 推进物理仿真让新 Pose 生效 + 物理稳定
            6. 重置步数计数器
            7. 清空 pending 控制状态和键盘信号队列
            8. 重新快照 joint states 作为保持目标
            9. 重新注册回调
        """
        if self._scene_builder is None:
            logger.warning("SceneBuilder 未初始化，无法重置")
            return

        # 1. 取消注册回调
        self._unregister_world_callbacks()

        # 2. 重置场景
        self._scene_builder.reset()

        # 3. 任务 1/3 删除了旧 prim 并创建新 prim，物理视图失效，需要 world.reset() 重建
        task_num = self.config.task_cfg.get("task_number", 0)
        if task_num in (1, 3):
            logger.info("[reset] 重新初始化物理仿真 (due to prim deletion/creation)...")
            if self._world is not None:
                self._world.reset()
                scatter_after_reset = getattr(self._scene_builder, "scatter_after_reset", None)
                if callable(scatter_after_reset):
                    scatter_after_reset()

        # 4. 重置机器人关节到初始 Pose
        if self._robot_interface is not None:
            self._robot_interface.reset()

        # 5. 推进物理仿真，让新 Pose 生效 + 物理稳定
        settle_steps = 5
        for _ in range(settle_steps):
            if self._robot_interface is not None and self._robot_interface._world is not None:
                self._robot_interface._world.step(render=True)

        # 6. 重置步数计数器
        self._send_action_step_idx = 0

        # 7. 清空 pending 控制状态
        with self._callback_lock:
            self._pending_absolute_action = None
            self._latest_camera_rgb = {}

        # 重置 teleop 键盘状态
        if self._teleop is not None:
            self._teleop.reset()

        self._left_gripping = False
        self._right_gripping = False
        self._go_home = False  # 重置回家标志
        self._go_home_key_was_pressed = False  # 重置回家按键状态
        self._assisted_grasp = None
        self._assisted_grasp_key_was_pressed = False
        self._cancel_grasp_key_was_pressed = False
        self._reset_teleop_ee_targets()
        self._clear_grasp_debug_markers()

        # 8. 重新快照 joint states 作为保持目标
        states = self._robot_interface.get_joint_states()
        if states:
            self._hold_arm_positions = np.array(self._robot_interface.arm_joint_initial_positions, dtype=np.float32)
            self._hold_finger_positions = np.array(self._robot_interface.finger_joint_initial_positions, dtype=np.float32)
        else:
            self._hold_arm_positions = None
            self._hold_finger_positions = None

        # 9. 重新注册回调
        self._register_world_callbacks()

        logger.info("[WalkerS2sim] Environment reset complete")

    def set_environment_state(self, env_state: np.ndarray | torch.Tensor | list[float]) -> None:
        """按扁平环境状态向量恢复仿真物体位姿。"""
        expected_dim = self.env_state_dim
        env_state_np = np.asarray(env_state, dtype=np.float32).reshape(-1)
        if env_state_np.shape[0] != expected_dim:
            raise ValueError(f"环境状态维度不匹配: 期望 {expected_dim}, 实际 {env_state_np.shape[0]}")

        callbacks_registered = self._callbacks_registered
        if callbacks_registered:
            self._unregister_world_callbacks()

        try:
            self._scene_builder.set_object_poses_from_flat(env_state_np)
            if self._world is not None:
                settle_steps = max(1, int(0.2 / self.config.physics_dt))
                for _ in range(settle_steps):
                    self._world.step(render=False)
        finally:
            if callbacks_registered:
                self._register_world_callbacks()

        self._last_observation = None
        logger.info("[WalkerS2sim] Environment state restored from SceneBuilder")
