"""Isaac Sim robot interface: Direct connection to Articulation created by SceneBuilder.

Contains:
- load_config — Load task YAML and resolve root_path to absolute path
- IsaacSimRobotInterface — Articulation wrapper, IK, camera, gripper control
"""

from __future__ import annotations

import logging
import os
from typing import Any, Optional

import numpy as np
import torch
import yaml
from scipy.spatial.transform import Rotation, Slerp

logger = logging.getLogger(__name__)

# Standing pose for the teleop robot.
#
# The official challenge robot is an upper-body teleop asset: the task state/action
# space does not control the legs. In the USD asset, body zero is already the
# standing lower-body pose. Keep the URDF lower body at that same zero pose and
# only apply non-zero defaults to the arms/head.
STANDING_JOINT_POSE: dict[str, float] = {
    "L_hip_roll_joint": 0.0,
    "L_hip_yaw_joint": 0.0,
    "L_hip_pitch_joint": 0.0,
    "L_knee_pitch_joint": 0.0,
    "L_ankle_pitch_joint": 0.0,
    "L_ankle_roll_joint": 0.0,
    "R_hip_roll_joint": 0.0,
    "R_hip_yaw_joint": 0.0,
    "R_hip_pitch_joint": 0.0,
    "R_knee_pitch_joint": 0.0,
    "R_ankle_pitch_joint": 0.0,
    "R_ankle_roll_joint": 0.0,
    "waist_yaw_joint": 0.0,
    "waist_pitch_joint": 0.0,
    "L_elbow_roll_joint": -1.8963565338596158,
    "L_elbow_yaw_joint": 1.4000461262831179,
    "L_shoulder_pitch_joint": 0.09322471888572098,
    "L_shoulder_roll_joint": -0.5933223843430208,
    "L_shoulder_yaw_joint": -1.595878574835185,
    "L_wrist_pitch_joint": -0.00048740902645395785,
    "L_wrist_roll_joint": 0.0998718010009366,
    "R_elbow_roll_joint": -1.8963607249359917,
    "R_elbow_yaw_joint": -1.4000874256427638,
    "R_shoulder_pitch_joint": -0.09321727661087699,
    "R_shoulder_roll_joint": -0.5933455607833843,
    "R_shoulder_yaw_joint": 1.595869459316937,
    "R_wrist_pitch_joint": 0.00048144049606466176,
    "R_wrist_roll_joint": 0.09985407619802703,
    "head_pitch_joint": -0.785398163,
    "head_yaw_joint": 1.9677590016147396e-07,
}

HIGH_GAIN_JOINT_KEYWORDS: tuple[str, ...] = ("hip", "knee", "ankle", "waist")

UBT_HAND_JOINT_NAMES: dict[str, tuple[str, ...]] = {
    "L": (
        "L_thumb_cmp_joint",
        "L_thumb_mpp_joint",
        "L_thumb_ip_joint",
        "L_index_mpp_joint",
        "L_index_ip_joint",
        "L_middle_mpp_joint",
        "L_middle_ip_joint",
        "L_ring_mpp_joint",
        "L_ring_ip_joint",
        "L_little_mpp_joint",
        "L_little_ip_joint",
    ),
    "R": (
        "R_thumb_cmp_joint",
        "R_thumb_mpp_joint",
        "R_thumb_ip_joint",
        "R_index_mpp_joint",
        "R_index_ip_joint",
        "R_middle_mpp_joint",
        "R_middle_ip_joint",
        "R_ring_mpp_joint",
        "R_ring_ip_joint",
        "R_little_mpp_joint",
        "R_little_ip_joint",
    ),
}

UBT_HAND_JOINT_LIMITS: dict[str, tuple[float, float]] = {
    "L_thumb_cmp_joint": (-0.96, 0.0),
    "L_thumb_mpp_joint": (0.0, 1.04),
    "L_thumb_ip_joint": (0.0, 1.05),
    "L_index_mpp_joint": (0.0, 1.46),
    "L_index_ip_joint": (0.0, 1.62),
    "L_middle_mpp_joint": (0.0, 1.46),
    "L_middle_ip_joint": (0.0, 1.62),
    "L_ring_mpp_joint": (0.0, 1.46),
    "L_ring_ip_joint": (0.0, 1.62),
    "L_little_mpp_joint": (0.0, 1.46),
    "L_little_ip_joint": (0.0, 1.62),
    "R_thumb_cmp_joint": (0.0, 0.96),
    "R_thumb_mpp_joint": (0.0, 1.04),
    "R_thumb_ip_joint": (0.0, 1.05),
    "R_index_mpp_joint": (0.0, 1.46),
    "R_index_ip_joint": (0.0, 1.62),
    "R_middle_mpp_joint": (0.0, 1.46),
    "R_middle_ip_joint": (0.0, 1.62),
    "R_ring_mpp_joint": (0.0, 1.46),
    "R_ring_ip_joint": (0.0, 1.62),
    "R_little_mpp_joint": (0.0, 1.46),
    "R_little_ip_joint": (0.0, 1.62),
}

UBT_HAND_POSE_FRACTIONS: dict[str, tuple[float, ...]] = {
    # Each value is a normalized curl amount for UBT_HAND_JOINT_NAMES order.
    # Sign is handled by the joint limit. Keep the physical open pose slightly
    # inside every hard stop; commanding exactly 0 makes the drive and limit
    # solver fight each other and causes idle finger chatter.
    "open": (0.03, 0.03, 0.03, 0.03, 0.03, 0.03, 0.03, 0.03, 0.03, 0.03, 0.03),
    # Keep the thumb opposed and the fingers slightly flexed while the arm moves.
    # A fully open hand leaves the long thumb links free to strike the table.
    "travel": (0.35, 0.22, 0.14, 0.12, 0.10, 0.10, 0.08, 0.10, 0.08, 0.10, 0.08),
    "pinch_pre": (0.64, 0.42, 0.30, 0.24, 0.18, 0.10, 0.08, 0.08, 0.06, 0.08, 0.06),
    "tripod_pre": (0.64, 0.44, 0.32, 0.26, 0.20, 0.26, 0.20, 0.10, 0.08, 0.10, 0.08),
    "power_pre": (0.52, 0.38, 0.30, 0.34, 0.30, 0.36, 0.32, 0.36, 0.32, 0.34, 0.30),
    "power": (0.78, 0.78, 0.72, 0.78, 0.78, 0.82, 0.82, 0.82, 0.82, 0.76, 0.76),
    "pinch": (0.82, 0.64, 0.54, 0.56, 0.48, 0.14, 0.10, 0.05, 0.05, 0.05, 0.05),
    "tripod": (0.82, 0.66, 0.56, 0.60, 0.54, 0.58, 0.50, 0.18, 0.14, 0.12, 0.10),
}

# Maximum principal inertia of each official actuated child link, in
# UBT_HAND_JOINT_NAMES order. The right-hand values are symmetric.
UBT_HAND_JOINT_INERTIAS = (
    1.31111335e-6,
    3.32254706e-5,
    1.87688540e-6,
    3.57000000e-6,
    2.65758516e-6,
    3.44034879e-6,
    3.48784866e-6,
    3.44034879e-6,
    2.86472779e-6,
    3.44034879e-6,
    1.82560831e-6,
)
UBT_HAND_DRIVE_PROFILES = {
    "firm": {"frequency": 45.0, "min_kp": 0.008, "max_effort": 0.05},
    "contact": {"frequency": 35.0, "min_kp": 0.005, "max_effort": 0.03},
}

# =========================================================================
# YAML Task configuration loading
# =========================================================================


def load_config(config_path: str) -> dict:
    """
    Load YAML task config file and resolve relative paths.

    Args:
        config_path: Absolute or relative path to YAML config file.

    Returns:
        Parsed config dict with ``root_path`` resolved to absolute path.
    """
    config_path = os.path.abspath(config_path)

    with open(config_path, "r") as f:
        cfg = yaml.safe_load(f)

    yaml_dir = os.path.dirname(config_path)

    if "root_path" in cfg:
        cfg["root_path"] = os.path.abspath(
            os.path.join(yaml_dir, cfg["root_path"])
        )
    return cfg


# =========================================================================
# Joint interpolator interface: Linear interpolation for joint position commands
# =========================================================================


class JointInterpolator:
    """关节角线性插值器（LERP），适用于 Isaac Lab 仿真逐帧执行"""

    def __init__(self):
        self.reset()

    def reset(self):
        """重置插值状态"""
        self.current_step = 0
        self.total_steps = 0
        self.start_q = None
        self.target_q = None
        self.interp_active = False

    def set_target(self, start_q: torch.Tensor, target_q: torch.Tensor, num_steps: int = 10):
        """
        设置插值起点、终点、总步数

        Args:
            start_q: 初始关节角 (N, num_dofs)
            target_q: 目标关节角 (N, num_dofs)
            num_steps: 插值总步数（多少步到达目标）
        """
        if num_steps <= 0:
            self.interp_active = False
            return

        self.start_q = start_q.clone()
        self.target_q = target_q.clone()
        self.total_steps = num_steps
        self.current_step = 0
        self.interp_active = True

    def step(self) -> torch.Tensor:
        """
        执行一帧插值，返回当前应该到达的关节角

        Returns:
            插值后的关节角 (N, num_dofs)
        """
        if self.start_q is None or self.target_q is None:
            raise RuntimeError("JointInterpolator: 请先调用 set_target() 设置起始和目标关节角")

        if not self.interp_active:
            return self.target_q.clone()

        alpha = float(self.current_step) / max(self.total_steps - 1, 1)
        alpha = float(max(0.0, min(1.0, alpha)))

        q_interp = self.start_q + alpha * (self.target_q - self.start_q)

        self.current_step += 1
        if self.current_step >= self.total_steps:
            self.interp_active = False

        return q_interp

    def is_finished(self) -> bool:
        """判断插值是否完成"""
        return not self.interp_active


# =========================================================================
# 笛卡尔空间轨迹规划：位置LERP + 姿态Slerp
# =========================================================================


class CartesianTrajectoryPlanner:
    """
    笛卡尔空间轨迹规划
    - 位置：线性插值 LERP
    - 姿态：RPY → Rotation → scipy Slerp → RPY
    """

    def __init__(self):
        self.reset()

    def reset(self):
        self.current_step = 0
        self.total_steps = 0
        self.start_pose = None
        self.target_pose = None
        self.slerp = None
        self.traj_active = False

    def set_target(
        self,
        start_pose: np.ndarray,
        target_pose: np.ndarray,
        num_steps: int = 50,
    ):
        if num_steps <= 0:
            self.traj_active = False
            return

        self.start_pose = np.array(start_pose, dtype=np.float32)
        self.target_pose = np.array(target_pose, dtype=np.float32)
        self.total_steps = num_steps
        self.current_step = 0

        rpy0 = self.start_pose[3:]
        rpy1 = self.target_pose[3:]
        rot0 = Rotation.from_euler("xyz", rpy0)
        rot1 = Rotation.from_euler("xyz", rpy1)

        self.slerp = Slerp([0.0, 1.0], Rotation.concatenate([rot0, rot1]))
        self.traj_active = True

    def step(self) -> np.ndarray:
        if not self.traj_active or self.start_pose is None or self.target_pose is None:
            raise RuntimeError("先调用 set_target() 设置起始和目标姿态")

        alpha = float(self.current_step) / max(self.total_steps - 1, 1)
        alpha = np.clip(alpha, 0.0, 1.0)

        pos_interp = self.start_pose[:3] + alpha * (self.target_pose[:3] - self.start_pose[:3])

        rot_interp = self.slerp(alpha)
        rpy_interp = rot_interp.as_euler("xyz")

        current_pose = np.concatenate([pos_interp, rpy_interp])

        self.current_step += 1
        if self.current_step >= self.total_steps:
            self.traj_active = False

        return current_pose

    def is_finished(self) -> bool:
        return not self.traj_active


# =========================================================================
# Robot interface (simplified version, directly uses robot created by SceneBuilder)
# =========================================================================


class IsaacSimRobotInterface:
    """
    Robot interface, directly connects to robot created by SceneBuilder.
    """

    arm_joint_names: list[str] = [
        "L_shoulder_pitch_joint",
        "L_shoulder_roll_joint",
        "L_shoulder_yaw_joint",
        "L_elbow_roll_joint",
        "L_elbow_yaw_joint",
        "L_wrist_pitch_joint",
        "L_wrist_roll_joint",
        "R_shoulder_pitch_joint",
        "R_shoulder_roll_joint",
        "R_shoulder_yaw_joint",
        "R_elbow_roll_joint",
        "R_elbow_yaw_joint",
        "R_wrist_pitch_joint",
        "R_wrist_roll_joint",
    ]

    # Old gripper joints from the original challenge model.
    # These may not exist in the hand-version robot.
    finger_joint_names: list[str] = [
        "L_finger1_joint",
        "L_finger2_joint",
        "R_finger1_joint",
        "R_finger2_joint",
    ]

    head_joint_names: list[str] = [
        "head_pitch_joint",
        "head_yaw_joint",
    ]

    dexterous_hand_joint_keywords: tuple[str, ...] = (
        "thumb",
        "index",
        "middle",
        "ring",
        "little",
    )

    gripper_open_width = -0.0215
    gripper_close_width = 0.01
    gripper_open_tau = -100.0
    gripper_close_tau = 100.0

    sixforce_joint_names: list[str] = [
        "L_sixforce_joint",
        "R_sixforce_joint",
    ]

    def __init__(
        self,
        prim_path: str,
        name: str = "walkerS2",
        world: Any = None,
        urdf_path: Optional[str] = None,
        use_explicit_standing_body: bool = False,
        arm_control_cfg: Optional[dict[str, Any]] = None,
        enable_cameras: bool = True,
    ):
        self.prim_path = prim_path
        self.name = name
        self._world = world
        self._articulation = None
        self.time = 0.0
        self.urdf_path = urdf_path
        self.use_explicit_standing_body = use_explicit_standing_body
        self.enable_cameras = bool(enable_cameras)
        arm_control_cfg = arm_control_cfg or {}
        self.arm_control_mode = str(arm_control_cfg.get("mode", "kinematic")).lower()
        if self.arm_control_mode not in ("physical", "kinematic"):
            raise ValueError(
                f"Unsupported arm control mode {self.arm_control_mode!r}; "
                "expected 'physical' or 'kinematic'"
            )
        self.arm_drive_stiffness = float(arm_control_cfg.get("stiffness", 180.0))
        self.arm_drive_damping = float(arm_control_cfg.get("damping", 25.0))
        self.arm_shoulder_effort_limit = float(
            arm_control_cfg.get("shoulder_effort_limit", 80.0)
        )
        self.arm_elbow_effort_limit = float(
            arm_control_cfg.get("elbow_effort_limit", 65.0)
        )
        self.arm_wrist_effort_limit = float(
            arm_control_cfg.get("wrist_effort_limit", 25.0)
        )
        self.arm_velocity_limit = float(arm_control_cfg.get("velocity_limit", 1.2))
        self.arm_acceleration_limit = float(
            arm_control_cfg.get("acceleration_limit", 3.0)
        )

        self.arm_joint_indices: list[int] = []
        self.finger_joint_indices: list[int] = []
        self.has_old_gripper: bool = False
        self.dexterous_hand_joint_indices: dict[str, list[int]] = {"L": [], "R": []}
        self.dexterous_hand_current_positions: dict[str, list[float]] = {"L": [], "R": []}
        self.dexterous_hand_target_positions: dict[str, list[float]] = {"L": [], "R": []}
        self.dexterous_hand_active_pose: dict[str, str] = {"L": "open", "R": "open"}
        self.dexterous_hand_close_pose: dict[str, str] = {"L": "power", "R": "power"}
        self.dexterous_hand_max_step = 0.004
        self._dexterous_hand_all_indices: list[int] = []
        self._dexterous_hand_drive_profile: Optional[str] = None
        self._dexterous_hand_drive_sides: tuple[str, ...] = ()
        self._standing_control_indices: list[int] = []
        self.body_joint_indices: list[int] = []
        self.head_joint_indices: list[int] = []

        self.sixforce_joint_names = ["L_sixforce_joint", "R_sixforce_joint"]
        self.cameras = {}

        self._camera_prim_paths = {
            "head_left": f"{prim_path}/head_pitch_link/head_stereo_left/head_stereo_left_Camera_01",
            "head_right": f"{prim_path}/head_pitch_link/head_stereo_right/head_stereo_right_Camera_01",
            "wrist_left": f"{prim_path}/L_camera_link/L_camera_link/L_wrist_camera/L_wrist_Camera",
            "wrist_right": f"{prim_path}/R_camera_link/R_camera_link/R_wrist_camera/R_wrist_Camera",
        }

        self.initial_joint_positions = None
        self.standing_joint_positions: list[float] = []
        self.ik_solver: Optional[Any] = None
        self._left_arm_isaac_indices = None
        self._right_arm_isaac_indices = None
        self._waist_isaac_indices = None
        self._waist_init_positions = None
        self._ik_warn_counter = 0
        self._smooth_alpha = 0.3
        self._last_arm_positions = {}
        self._arm_hard_follow_positions: Optional[torch.Tensor] = None
        self._arm_drive_command_positions: Optional[torch.Tensor] = None
        self._arm_drive_command_velocities: Optional[torch.Tensor] = None
        self._joint_value_map = {
            "L_elbow_roll_joint": -1.8963565338596158,
            "L_elbow_yaw_joint": 1.4000461262831179,
            "L_shoulder_pitch_joint": 0.09322471888572098,
            "L_shoulder_roll_joint": -0.5933223843430208,
            "L_shoulder_yaw_joint": -1.595878574835185,
            "L_wrist_pitch_joint": -0.00048740902645395785,
            "L_wrist_roll_joint": 0.0998718010009366,
            "R_elbow_roll_joint": -1.8963607249359917,
            "R_elbow_yaw_joint": -1.4000874256427638,
            "R_shoulder_pitch_joint": -0.09321727661087699,
            "R_shoulder_roll_joint": -0.5933455607833843,
            "R_shoulder_yaw_joint": 1.595869459316937,
            "R_wrist_pitch_joint": 0.00048144049606466176,
            "R_wrist_roll_joint": 0.09985407619802703,
            "head_pitch_joint": -0.785398163,
            "head_yaw_joint": 1.9677590016147396e-07,
        }

        self.joint_interpolator = JointInterpolator()
        self.cartesian_planner = CartesianTrajectoryPlanner()

    @staticmethod
    def _joint_limit_target(joint_name: str, fraction: float) -> float:
        low, high = UBT_HAND_JOINT_LIMITS[joint_name]
        target_limit = low if abs(low) > abs(high) else high
        value = float(fraction) * target_limit
        return float(np.clip(value, low, high))

    def _pose_values_for_side(self, side: str, pose_name: str) -> list[float]:
        pose_key = pose_name if pose_name in UBT_HAND_POSE_FRACTIONS else "open"
        fractions = UBT_HAND_POSE_FRACTIONS[pose_key]
        return [
            self._joint_limit_target(joint_name, fractions[i])
            for i, joint_name in enumerate(UBT_HAND_JOINT_NAMES[side])
        ]

    def _articulation_physics_ready(self) -> bool:
        return self._articulation is not None and hasattr(self._articulation, "_physics_view")

    def _initialize_dexterous_hand_joints(self, all_joint_names: list[str]) -> None:
        self.dexterous_hand_joint_indices = {"L": [], "R": []}
        self.dexterous_hand_current_positions = {"L": [], "R": []}
        self.dexterous_hand_target_positions = {"L": [], "R": []}
        self.dexterous_hand_active_pose = {"L": "open", "R": "open"}
        self.dexterous_hand_close_pose = {"L": "power", "R": "power"}

        for side in ("L", "R"):
            missing = []
            for joint_name in UBT_HAND_JOINT_NAMES[side]:
                if joint_name in all_joint_names:
                    self.dexterous_hand_joint_indices[side].append(
                        all_joint_names.index(joint_name)
                    )
                else:
                    missing.append(joint_name)

            if missing:
                logger.warning(
                    "Dexterous %s hand incomplete; preset control disabled for that side. Missing: %s",
                    side,
                    missing,
                )
                self.dexterous_hand_joint_indices[side] = []
                continue

            open_pose = self._pose_values_for_side(side, "open")
            self.dexterous_hand_current_positions[side] = list(open_pose)
            self.dexterous_hand_target_positions[side] = list(open_pose)
            logger.info(
                "Dexterous %s hand preset control enabled (%s joints).",
                side,
                len(self.dexterous_hand_joint_indices[side]),
            )

    @property
    def has_dexterous_hand(self) -> bool:
        return bool(
            self.dexterous_hand_joint_indices.get("L")
            or self.dexterous_hand_joint_indices.get("R")
        )

    def set_dexterous_hand_pose(self, side: str, pose_name: str) -> None:
        if side not in ("L", "R"):
            return
        if not self.dexterous_hand_joint_indices.get(side):
            return
        pose_key = pose_name if pose_name in UBT_HAND_POSE_FRACTIONS else "open"
        self.dexterous_hand_target_positions[side] = self._pose_values_for_side(side, pose_key)
        self.dexterous_hand_active_pose[side] = pose_key
        if pose_key in ("power", "pinch", "tripod"):
            self.dexterous_hand_close_pose[side] = pose_key

    def snap_dexterous_hand_pose(self, side: str, pose_name: str) -> None:
        self.set_dexterous_hand_pose(side, pose_name)
        target = self.dexterous_hand_target_positions.get(side)
        if target:
            self.dexterous_hand_current_positions[side] = list(target)

    def open_dexterous_hand(self, side: str) -> None:
        self.set_dexterous_hand_pose(side, "open")

    def travel_dexterous_hand(self, side: str) -> None:
        self.snap_dexterous_hand_pose(side, "travel")

    def prepare_dexterous_hand_for_arm_motion(self, side: str) -> None:
        # Arm and hand targets are independent.
        return

    def close_dexterous_hand(self, side: str, pose_name: Optional[str] = None) -> None:
        pose = pose_name or self.dexterous_hand_close_pose.get(side, "power")
        self.set_dexterous_hand_pose(side, pose)

    def nudge_dexterous_hand(
        self,
        side: str,
        direction: float,
        pose_name: Optional[str] = None,
        fraction_step: float = 0.08,
    ) -> None:
        if side not in ("L", "R"):
            return
        if not self.dexterous_hand_joint_indices.get(side):
            return

        if direction < 0.0:
            # Opening moves away from contact, so return directly to the stable
            # free-space hold instead of using the contact controller.
            self.open_dexterous_hand(side)
            return

        close_pose = pose_name or self.dexterous_hand_close_pose.get(side, "power")
        open_values = np.asarray(self._pose_values_for_side(side, "open"), dtype=np.float32)
        close_values = np.asarray(self._pose_values_for_side(side, close_pose), dtype=np.float32)
        raw_target = self.dexterous_hand_target_positions.get(side)
        target = np.asarray(
            open_values if raw_target is None else raw_target, dtype=np.float32
        )
        if target.shape != open_values.shape:
            target = open_values.copy()

        delta = float(np.sign(direction)) * float(fraction_step) * (close_values - open_values)
        lo = np.minimum(open_values, close_values)
        hi = np.maximum(open_values, close_values)
        next_target = np.clip(target + delta, lo, hi)
        self.dexterous_hand_target_positions[side] = next_target.astype(np.float32).tolist()
        self.dexterous_hand_active_pose[side] = "manual"
        self.dexterous_hand_close_pose[side] = close_pose

    def preshape_dexterous_hand(self, side: str, pose_name: str) -> None:
        preshape = f"{pose_name}_pre"
        if preshape not in UBT_HAND_POSE_FRACTIONS:
            preshape = "travel"
        self.set_dexterous_hand_pose(side, preshape)

    def step_dexterous_hands(self) -> None:
        for side in ("L", "R"):
            if not self.dexterous_hand_joint_indices.get(side):
                continue
            current = np.asarray(self.dexterous_hand_current_positions[side], dtype=np.float32)
            target = np.asarray(self.dexterous_hand_target_positions[side], dtype=np.float32)
            if current.shape != target.shape:
                current = target.copy()
            delta = np.clip(target - current, -self.dexterous_hand_max_step, self.dexterous_hand_max_step)
            self.dexterous_hand_current_positions[side] = (current + delta).astype(np.float32).tolist()

    def _apply_dexterous_hand_targets_to_vector(self, targets: list[float]) -> None:
        self.step_dexterous_hands()
        for side in ("L", "R"):
            values = self.dexterous_hand_current_positions.get(side) or []
            for local_i, global_idx in enumerate(self.dexterous_hand_joint_indices.get(side, [])):
                if local_i < len(values):
                    targets[global_idx] = float(values[local_i])

    def initialize(self):
        """Initialize articulation."""
        from isaacsim.core.prims import Articulation

        if Articulation is None:
            raise ImportError("isaacsim.core.prims.Articulation 不可用")

        logger.info(f"Connecting to Articulation: {self.prim_path}")

        self._articulation = Articulation(
            prim_paths_expr=self.prim_path,
            name=self.name,
        )
        self._articulation.initialize()

        all_joint_names = self._articulation.dof_names
        logger.info(f"Total robot joints: {len(all_joint_names)}")
        logger.info(f"All joints: {all_joint_names}")

        self.arm_joint_indices = []
        self.finger_joint_indices = []
        self.body_joint_indices = []

        arm_missing_joints = []
        finger_missing_joints = []

        # Required arm joints.
        for arm_joint in self.arm_joint_names:
            if arm_joint in all_joint_names:
                idx = all_joint_names.index(arm_joint)
                self.arm_joint_indices.append(idx)
                logger.info(f"  [{len(self.arm_joint_indices)-1}] {arm_joint} -> global index {idx}")
            else:
                arm_missing_joints.append(arm_joint)
                logger.error(f"  Joint not found: {arm_joint}")

        if arm_missing_joints:
            raise RuntimeError(
                f"Missing {len(arm_missing_joints)} required arm joints: {arm_missing_joints}"
            )

        if len(self.arm_joint_indices) != 14:
            raise RuntimeError(
                f"Arm joint count error: expected 14, got {len(self.arm_joint_indices)}"
            )

        # Optional old gripper joints.
        # The original challenge model has:
        #   L_finger1_joint, L_finger2_joint, R_finger1_joint, R_finger2_joint
        # The hand-version robot may not have these old gripper joints.
        for finger_joint in self.finger_joint_names:
            if finger_joint in all_joint_names:
                idx = all_joint_names.index(finger_joint)
                self.finger_joint_indices.append(idx)
                logger.info(f"  [{len(self.finger_joint_indices)-1}] {finger_joint} -> global index {idx}")
            else:
                finger_missing_joints.append(finger_joint)
                logger.warning(f"未找到旧 Gripper/Finger joint: {finger_joint}")

        self.has_old_gripper = len(self.finger_joint_indices) == 4

        if not self.has_old_gripper:
            logger.warning(
                "Old gripper joints not found or incomplete. Continuing without old gripper control. "
                "Expected 4 old gripper joints, got %s. Missing: %s",
                len(self.finger_joint_indices),
                finger_missing_joints,
            )
            self.finger_joint_indices = []
        else:
            logger.info("Old gripper joints found. Old gripper control enabled.")

        self.head_joint_indices = [
            all_joint_names.index(joint_name)
            for joint_name in self.head_joint_names
            if joint_name in all_joint_names
        ]

        self._initialize_dexterous_hand_joints(all_joint_names)
        dexterous_hand_joint_indices = (
            self.dexterous_hand_joint_indices["L"]
            + self.dexterous_hand_joint_indices["R"]
        )
        self._dexterous_hand_all_indices = list(dexterous_hand_joint_indices)

        used_indices = set(
            self.arm_joint_indices
            + self.finger_joint_indices
            + self.head_joint_indices
            + dexterous_hand_joint_indices
        )
        self.body_joint_indices = [
            idx for idx in range(len(all_joint_names)) if idx not in used_indices
        ]
        logger.info(f"Body joints (exclude arm+old_finger+head+dex_hand): {len(self.body_joint_indices)}")

        arm_initial_values = [
            0.09322471888572098,
            -0.5933223843430208,
            -1.595878574835185,
            -1.8963565338596158,
            1.4000461262831179,
            -0.00048740902645395785,
            0.0998718010009366,
            -0.09321727661087699,
            -0.5933455607833843,
            1.595869459316937,
            -1.8963607249359917,
            -1.4000874256427638,
            0.00048144049606466176,
            0.09985407619802703,
        ]
        head_defaults = {
            "head_pitch_joint": -0.785398163,
            "head_yaw_joint": 1.9677590016147396e-07,
        }

        if self.use_explicit_standing_body:
            # Official URDF: joint zero != standing. Use full standing table.
            self.standing_joint_positions = [
                float(STANDING_JOINT_POSE.get(name, 0.0)) for name in all_joint_names
            ]
            self.initial_joint_positions = list(self.standing_joint_positions)
            leg_debug = {
                name: self.standing_joint_positions[all_joint_names.index(name)]
                for name in (
                    "L_hip_pitch_joint",
                    "L_knee_pitch_joint",
                    "R_hip_pitch_joint",
                    "R_knee_pitch_joint",
                )
                if name in all_joint_names
            }
            logger.info("URDF standing leg joints (rad): %s", leg_debug)
        else:
            # Official baseline USD: body joints stay at 0 (standing is baked into the asset).
            self.initial_joint_positions = [0.0] * len(all_joint_names)
            for i, global_idx in enumerate(self.arm_joint_indices):
                self.initial_joint_positions[global_idx] = arm_initial_values[i]
            for joint_name, value in head_defaults.items():
                if joint_name in all_joint_names:
                    self.initial_joint_positions[all_joint_names.index(joint_name)] = value
            self.standing_joint_positions = list(self.initial_joint_positions)

        # The URDF's open position is also the lower/upper hard stop for every
        # actuated hand joint. Initialize and reset to the guarded open pose so
        # physics never starts with the finger drives pressing into those stops.
        for side in ("L", "R"):
            open_pose = self.dexterous_hand_current_positions.get(side) or []
            for local_i, global_idx in enumerate(self.dexterous_hand_joint_indices.get(side, [])):
                if local_i < len(open_pose):
                    value = float(open_pose[local_i])
                    self.initial_joint_positions[global_idx] = value
                    self.standing_joint_positions[global_idx] = value

        self.finger_joint_initial_positions = [
            self.gripper_open_width
        ] * len(self.finger_joint_indices)
        self.arm_joint_initial_positions = [
            self.initial_joint_positions[idx] for idx in self.arm_joint_indices
        ]

        if self.has_old_gripper:
            for global_idx in self.finger_joint_indices:
                self.initial_joint_positions[global_idx] = self.gripper_open_width
                self.standing_joint_positions[global_idx] = self.gripper_open_width

        all_indices = list(range(len(all_joint_names)))
        self._all_dof_indices = all_indices
        dexterous_index_set = set(self._dexterous_hand_all_indices)
        self._standing_control_indices = [
            idx for idx in all_indices if idx not in dexterous_index_set
        ]

        if self._world is not None and self._world.is_playing():
            self._world.pause()

        self._articulation.set_joint_positions(
            torch.tensor(self.initial_joint_positions, dtype=torch.float32),
            joint_indices=torch.tensor(all_indices, dtype=torch.int32),
        )
        self._articulation.set_joint_velocities(
            torch.zeros(len(all_joint_names), dtype=torch.float32),
            joint_indices=torch.tensor(all_indices, dtype=torch.int32),
        )
        if self.use_explicit_standing_body:
            self._apply_standing_gains(all_joint_names)
            self.apply_all_joint_targets(self.standing_joint_positions)
            actual = self._articulation.get_joint_positions().flatten().tolist()
            for jname in ("L_hip_pitch_joint", "L_knee_pitch_joint", "L_hip_roll_joint"):
                if jname in all_joint_names:
                    idx = all_joint_names.index(jname)
                    logger.info(
                        "Joint %s: target=%.3f actual=%.3f",
                        jname,
                        self.standing_joint_positions[idx],
                        actual[idx],
                    )

        self._setup_cameras()

        self.initialize_ik(urdf_path=self.urdf_path)

        logger.info(
            f"Robot initialization complete, controlling {len(self.arm_joint_indices)} arm joints. "
            f"Old gripper enabled: {self.has_old_gripper}"
        )

    def apply_standing_pose_after_play(self) -> None:
        """Re-apply standing pose and PD gains after World.play()/reset()."""
        if self._articulation is None or not self.standing_joint_positions:
            return

        all_joint_names = list(self._articulation.dof_names)
        all_indices = torch.tensor(self._all_dof_indices, dtype=torch.int32)
        positions = torch.tensor(self.standing_joint_positions, dtype=torch.float32)

        self._articulation.set_joint_positions(positions, joint_indices=all_indices)
        self._articulation.set_joint_velocities(
            torch.zeros(len(self._all_dof_indices), dtype=torch.float32),
            joint_indices=all_indices,
        )
        self._apply_standing_gains(all_joint_names)
        self.apply_all_joint_targets(self.standing_joint_positions)
        logger.info("Re-applied standing pose and PD gains after physics start.")

    def _apply_standing_gains(self, all_joint_names: list[str]) -> None:
        hand_joint_names = {
            joint_name
            for side_names in UBT_HAND_JOINT_NAMES.values()
            for joint_name in side_names
        }
        kps = []
        kds = []
        max_efforts = []
        for name in all_joint_names:
            if name in hand_joint_names:
                # Dexterous hands use deterministic interpolated state control;
                # disable imported drives so they cannot fight that controller.
                kps.append(0.0)
                kds.append(0.0)
                max_efforts.append(0.0)
            elif name in self.arm_joint_names:
                kps.append(self.arm_drive_stiffness)
                kds.append(self.arm_drive_damping)
                if "wrist" in name:
                    max_efforts.append(self.arm_wrist_effort_limit)
                elif "elbow" in name:
                    max_efforts.append(self.arm_elbow_effort_limit)
                else:
                    max_efforts.append(self.arm_shoulder_effort_limit)
            elif name in self.head_joint_names:
                kps.append(450.0)
                kds.append(45.0)
                max_efforts.append(80.0)
            elif any(k in name for k in HIGH_GAIN_JOINT_KEYWORDS):
                kps.append(900.0)
                kds.append(70.0)
                max_efforts.append(350.0)
            else:
                kps.append(180.0)
                kds.append(18.0)
                max_efforts.append(120.0)
        self._articulation.set_gains(
            kps=torch.tensor([kps], dtype=torch.float32),
            kds=torch.tensor([kds], dtype=torch.float32),
        )
        self._dexterous_hand_drive_profile = None
        self._dexterous_hand_drive_sides = ()
        if hasattr(self._articulation, "set_max_efforts"):
            self._articulation.set_max_efforts(
                torch.tensor([max_efforts], dtype=torch.float32)
            )
        logger.info(
            "Arm controller: mode=%s kp=%.1f kd=%.1f velocity=%.2f acceleration=%.2f",
            self.arm_control_mode,
            self.arm_drive_stiffness,
            self.arm_drive_damping,
            self.arm_velocity_limit,
            self.arm_acceleration_limit,
        )

    def apply_all_joint_targets(self, positions, include_dexterous_hands: bool = False) -> None:
        """Apply one PD target vector to the stable standing/arm DOFs.

        The dexterous hand fingers are intentionally excluded during idle holds.
        Their small links can jitter if we re-command every finger at physics rate
        while the imported hand is also settling against its own drive/collision
        setup. Hand pose commands use ``apply_dexterous_hand_targets`` instead.
        """
        from isaacsim.core.utils.types import ArticulationActions

        if self._articulation is None:
            return

        if not isinstance(positions, torch.Tensor):
            positions = torch.tensor(positions, dtype=torch.float32)
        if positions.ndim == 1:
            positions = positions.unsqueeze(0)

        if include_dexterous_hands:
            control_indices = list(self._all_dof_indices)
        else:
            control_indices = list(self._standing_control_indices or self._all_dof_indices)

        target_positions = positions[:, control_indices]

        self._articulation.apply_action(
            ArticulationActions(
                joint_positions=target_positions,
                joint_indices=torch.tensor(control_indices, dtype=torch.int32),
            )
        )
        if self.use_explicit_standing_body:
            self._lock_body_joints_to_standing()

    def _configure_dexterous_hand_drives(
        self,
        profile: str,
        physical_sides: Optional[set[str]] = None,
    ) -> None:
        active_sides = physical_sides or {"L", "R"}
        active_sides_key = tuple(sorted(active_sides))
        if (
            profile == self._dexterous_hand_drive_profile
            and active_sides_key == self._dexterous_hand_drive_sides
        ):
            return
        cfg = UBT_HAND_DRIVE_PROFILES[profile]
        frequency = float(cfg["frequency"])
        min_kp = float(cfg["min_kp"])
        max_effort = float(cfg["max_effort"])
        kps = []
        kds = []
        efforts = []
        for side in ("L", "R"):
            for inertia in UBT_HAND_JOINT_INERTIAS:
                if side in active_sides:
                    kp = max(min_kp, inertia * frequency**2)
                    kd = 2.0 * np.sqrt(kp * inertia)
                    effort = min(max_effort, max(0.008, 2.0 * kp))
                else:
                    kp = 0.0
                    kd = 0.0
                    effort = 0.0
                kps.append(kp)
                kds.append(kd)
                efforts.append(effort)

        joint_indices = torch.tensor(self._dexterous_hand_all_indices, dtype=torch.int32)
        self._articulation.set_gains(
            kps=torch.tensor([kps], dtype=torch.float32),
            kds=torch.tensor([kds], dtype=torch.float32),
            joint_indices=joint_indices,
        )
        self._articulation.set_max_efforts(
            torch.tensor([efforts], dtype=torch.float32),
            joint_indices=joint_indices,
        )
        self._dexterous_hand_drive_profile = profile
        self._dexterous_hand_drive_sides = active_sides_key
        logger.info(
            "Dexterous physical drives: %s sides=%s kp=[%.4f, %.4f] effort=[%.4f, %.4f]",
            profile,
            ",".join(sorted(active_sides)),
            min(kps),
            max(kps),
            min(efforts),
            max(efforts),
        )

    def _disable_dexterous_hand_drives(self) -> None:
        if self._articulation is None or not self._dexterous_hand_all_indices:
            return
        if self._dexterous_hand_drive_profile is None:
            return

        joint_indices = torch.tensor(self._dexterous_hand_all_indices, dtype=torch.int32)
        count = len(self._dexterous_hand_all_indices)
        self._articulation.set_gains(
            kps=torch.zeros((1, count), dtype=torch.float32),
            kds=torch.zeros((1, count), dtype=torch.float32),
            joint_indices=joint_indices,
        )
        self._articulation.set_max_efforts(
            torch.zeros((1, count), dtype=torch.float32),
            joint_indices=joint_indices,
        )
        self._dexterous_hand_drive_profile = None
        self._dexterous_hand_drive_sides = ()

    def _hard_hold_dexterous_hand_targets(self) -> None:
        """Servo-hold free-space hand poses without injecting contact forces.

        The official UBT hand links are very light. Leaving them under a weak
        PhysX drive while idle makes the distal links chatter. For open/travel
        poses there is no grasp contact to model, so use a deterministic servo
        hold and reserve physical drives for closing/contact poses.
        """
        self._disable_dexterous_hand_drives()
        self.step_dexterous_hands()

        hand_indices = []
        hand_values = []
        for side in ("L", "R"):
            values = self.dexterous_hand_current_positions.get(side) or []
            for local_i, global_idx in enumerate(
                self.dexterous_hand_joint_indices.get(side, [])
            ):
                if local_i < len(values):
                    hand_indices.append(global_idx)
                    hand_values.append(float(values[local_i]))

        if not hand_indices:
            return

        joint_indices = torch.tensor(hand_indices, dtype=torch.int32)
        self._articulation.set_joint_positions(
            torch.tensor(hand_values, dtype=torch.float32),
            joint_indices=joint_indices,
        )
        self._articulation.set_joint_velocities(
            torch.zeros(len(hand_indices), dtype=torch.float32),
            joint_indices=joint_indices,
        )

    def apply_dexterous_hand_targets(self, control_profile: Optional[str] = None) -> None:
        """Track interpolated hand targets.

        Open/travel poses are servo-held for stability. Closing/contact poses
        use bounded physical PhysX drives so object interaction remains
        compliant instead of teleporting through contacts.
        """
        if self._articulation is None or not self._dexterous_hand_all_indices:
            return

        if control_profile is None:
            contact_poses = {
                "manual",
                "power",
                "pinch",
                "tripod",
                "power_pre",
                "pinch_pre",
                "tripod_pre",
            }
            physical_sides = {
                side
                for side in ("L", "R")
                if self.dexterous_hand_active_pose.get(side) in contact_poses
            }
            control_profile = (
                "contact"
                if physical_sides
                else "firm"
            )
        else:
            physical_sides = {
                side
                for side in ("L", "R")
                if self.dexterous_hand_active_pose.get(side)
                not in {"open", "travel"}
            } or {"L", "R"}

        if control_profile == "firm":
            self._hard_hold_dexterous_hand_targets()
            return

        self._configure_dexterous_hand_drives(control_profile, physical_sides)
        self.step_dexterous_hands()
        physical_indices = []
        physical_values = []
        hold_indices = []
        hold_values = []
        for side in ("L", "R"):
            values = self.dexterous_hand_current_positions.get(side) or []
            for local_i, global_idx in enumerate(
                self.dexterous_hand_joint_indices.get(side, [])
            ):
                if local_i < len(values):
                    if side in physical_sides:
                        physical_indices.append(global_idx)
                        physical_values.append(float(values[local_i]))
                    else:
                        hold_indices.append(global_idx)
                        hold_values.append(float(values[local_i]))

        if hold_indices:
            hold_joint_indices = torch.tensor(hold_indices, dtype=torch.int32)
            self._articulation.set_joint_positions(
                torch.tensor(hold_values, dtype=torch.float32),
                joint_indices=hold_joint_indices,
            )
            self._articulation.set_joint_velocities(
                torch.zeros(len(hold_indices), dtype=torch.float32),
                joint_indices=hold_joint_indices,
            )

        if physical_indices:
            physical_joint_indices = torch.tensor(physical_indices, dtype=torch.int32)
            self._articulation.set_joint_position_targets(
                torch.tensor([physical_values], dtype=torch.float32),
                joint_indices=physical_joint_indices,
            )

    def _lock_body_joints_to_standing(self) -> None:
        """Hard-lock non-teleop body/head DOFs to the official standing pose."""
        if (
            self._articulation is None
            or not self._articulation_physics_ready()
            or not (self.body_joint_indices or self.head_joint_indices)
            or not self.standing_joint_positions
        ):
            return

        stabilizing_indices = list(self.body_joint_indices) + list(self.head_joint_indices)

        self._articulation.set_joint_positions(
            torch.tensor(
                [self.standing_joint_positions[idx] for idx in stabilizing_indices],
                dtype=torch.float32,
            ),
            joint_indices=torch.tensor(stabilizing_indices, dtype=torch.int32),
        )
        self._articulation.set_joint_velocities(
            torch.zeros(len(stabilizing_indices), dtype=torch.float32),
            joint_indices=torch.tensor(stabilizing_indices, dtype=torch.int32),
        )

    def build_joint_target_vector(
        self,
        arm_positions,
        finger_positions=None,
    ) -> list[float]:
        """Merge teleop arm (and optional gripper) targets into the standing pose vector."""
        targets = list(self.standing_joint_positions)
        arm_values = np.asarray(arm_positions, dtype=np.float32).flatten()
        for arm_i, global_idx in enumerate(self.arm_joint_indices):
            targets[global_idx] = float(arm_values[arm_i])
        if self.has_old_gripper and self.finger_joint_indices and finger_positions is not None:
            finger_values = np.asarray(finger_positions, dtype=np.float32).flatten()
            for finger_i, global_idx in enumerate(self.finger_joint_indices):
                if finger_i < len(finger_values):
                    targets[global_idx] = float(finger_values[finger_i])
        return targets

    def apply_full_standing_hold(self) -> None:
        targets = list(self.standing_joint_positions)
        self.apply_all_joint_targets(targets)

    def hold_stabilizing_joints(self) -> None:
        """Hold body/head while leaving arm drives under trajectory control."""
        self._lock_body_joints_to_standing()

    def _setup_cameras(self):
        """Setup cameras."""
        if not self.enable_cameras:
            logger.info("Isaac sensor cameras disabled; using the main viewport only")
            return

        from isaacsim.sensors.camera import Camera

        if Camera is None:
            logger.warning("isaacsim.sensors.camera.Camera unavailable, camera function disabled")
            return

        for cam_name, prim_path in self._camera_prim_paths.items():
            try:
                self.cameras[cam_name] = Camera(prim_path=prim_path, resolution=(640, 480))
                self.cameras[cam_name].initialize()
                self.cameras[cam_name].add_distance_to_image_plane_to_frame()
                logger.info(f"Camera {cam_name} initialized successfully")
            except Exception as e:
                logger.warning(f"Camera {cam_name} 初始化failed: {e}")

        self.cameras["dummy_camera_top"] = Camera(
            prim_path="/Root/Ref_Xform/Ref/head_pitch_link/head_stereo_left/dummy_camera_top",
            translation=[-2.54145, -0.06363, 2.4821],
            orientation=[0.942732, -0.008441, 0.333388, 0.006151],
        )
        self.cameras["dummy_camera_top"].initialize()
        self.cameras["dummy_camera_top"].add_distance_to_image_plane_to_frame()

        self.cameras["dummy_camera_side"] = Camera(
            prim_path="/Replicator/Ref_Xform/Ref/dummy_camera_side",
            translation=[2.06555, -0.02631, 0.95453],
            orientation=[-5.94300e-03, -3.24760e-02, -2.01000e-04, 9.99455e-01],
        )
        self.cameras["dummy_camera_side"].initialize()
        self.cameras["dummy_camera_side"].add_distance_to_image_plane_to_frame()

    def reinitialize_articulation(self):
        """Rebuild Articulation object after scene reset to restore _physics_view.

        In Isaac Sim, once USD stage changes, the old Articulation object's _physics_view
        is destroyed and cannot be restored by calling initialize() again.
        Must create a brand new Articulation wrapper.
        """
        from isaacsim.core.prims import Articulation

        logger.info("[reinitialize_articulation] Rebuilding Articulation ...")
        self._articulation = Articulation(
            prim_paths_expr=self.prim_path,
            name=self.name,
        )
        self._articulation.initialize()
        logger.info("[reinitialize_articulation] Articulation rebuilt complete")

    def reset_ik(self, current_joint_positions: Optional[list[float]] = None):
        self._ik_warn_counter = 0
        self._last_arm_positions.clear()
        self.reset_arm_command_state()

        if self.ik_solver is not None:
            self.ik_solver.reset_runtime_state()
            self.ik_solver.sync_joint_positions(
                self._articulation.dof_names,
                current_joint_positions,
            )
            self.ik_solver.save_initial_q()

    def reset(self):
        """将机器人重置到初始化时的关节初始Pose，并同步重置IK相关状态。"""
        self.open_dexterous_hand("L")
        self.open_dexterous_hand("R")
        for side in ("L", "R"):
            if self.dexterous_hand_target_positions.get(side):
                self.dexterous_hand_current_positions[side] = list(
                    self.dexterous_hand_target_positions[side]
                )

        initial_positions = np.asarray(self.standing_joint_positions, dtype=np.float32)
        reset_targets = list(initial_positions.tolist())
        self._apply_dexterous_hand_targets_to_vector(reset_targets)
        initial_positions = np.asarray(reset_targets, dtype=np.float32)
        all_indices = np.asarray(self._all_dof_indices, dtype=np.int32)

        self._articulation.set_joint_positions(
            torch.tensor(initial_positions, dtype=torch.float32),
            joint_indices=torch.tensor(all_indices, dtype=torch.int32),
        )
        self._articulation.set_joint_velocities(
            torch.zeros(len(all_indices), dtype=torch.float32),
            joint_indices=torch.tensor(all_indices, dtype=torch.int32),
        )

        self.time = 0.0
        self._ik_warn_counter = 0
        self._last_arm_positions.clear()
        self.reset_arm_command_state()

        if self.ik_solver is not None:
            self.ik_solver.reset_runtime_state()
            self.ik_solver.sync_joint_positions(
                self._articulation.dof_names,
                initial_positions.tolist(),
            )
            self.ik_solver.save_initial_q()

        print("[RobotArticulation] Robot reset to initial pose complete")

    def cleanup(self):
        """Cleanup resources."""
        if self._articulation is not None:
            try:
                if self.initial_joint_positions is not None:
                    all_joint_names = self._articulation.dof_names
                    all_indices = list(range(len(all_joint_names)))
                    self._articulation.set_joint_positions(
                        torch.tensor(self.initial_joint_positions),
                        joint_indices=torch.tensor(all_indices),
                    )
            except Exception:
                pass
            self._articulation = None

        self.cameras.clear()

    def get_joint_states(self):
        """获取joint states."""
        if self._articulation is None or not self._articulation_physics_ready():
            return None

        try:
            all_names = self._articulation.dof_names
            all_joint_positions = self._articulation.get_joint_positions()
            all_joint_velocities = self._articulation.get_joint_velocities()
            all_joint_efforts = self._articulation.get_measured_joint_efforts()

            if hasattr(all_joint_positions, "shape") and len(all_joint_positions.shape) > 1:
                all_joint_positions = all_joint_positions.flatten()
            if hasattr(all_joint_velocities, "shape") and len(all_joint_velocities.shape) > 1:
                all_joint_velocities = all_joint_velocities.flatten()
            if hasattr(all_joint_efforts, "shape") and len(all_joint_efforts.shape) > 1:
                all_joint_efforts = all_joint_efforts.flatten()

            if isinstance(all_joint_positions, np.ndarray):
                all_joint_positions = all_joint_positions.flatten()
            if isinstance(all_joint_velocities, np.ndarray):
                all_joint_velocities = all_joint_velocities.flatten()
            if isinstance(all_joint_efforts, np.ndarray):
                all_joint_efforts = all_joint_efforts.flatten()

            if not isinstance(all_joint_positions, torch.Tensor):
                all_joint_positions = torch.tensor(all_joint_positions, dtype=torch.float32)
            if not isinstance(all_joint_velocities, torch.Tensor):
                all_joint_velocities = torch.tensor(all_joint_velocities, dtype=torch.float32)
            if not isinstance(all_joint_efforts, torch.Tensor):
                all_joint_efforts = torch.tensor(all_joint_efforts, dtype=torch.float32)

            all_joint_positions = all_joint_positions.flatten()
            all_joint_velocities = all_joint_velocities.flatten()
            all_joint_efforts = all_joint_efforts.flatten()

            if len(all_joint_positions) != len(all_names):
                return None

            arm_indices = torch.tensor(self.arm_joint_indices, dtype=torch.long)
            arm_positions = all_joint_positions[arm_indices]
            arm_vel = all_joint_velocities[arm_indices]
            arm_tau = all_joint_efforts[arm_indices]

            if self.finger_joint_indices:
                finger_indices = torch.tensor(self.finger_joint_indices, dtype=torch.long)
                finger_positions = all_joint_positions[finger_indices]
                finger_tau = all_joint_efforts[finger_indices]
                finger_vel = all_joint_velocities[finger_indices]
            else:
                # Keep the official 20D dataset schema stable for the hand-v3 URDF.
                # The dexterous hand does not expose the old four gripper joints, but
                # record/replay code still expects these legacy slots to exist.
                finger_positions = torch.zeros(4, dtype=torch.float32)
                finger_tau = torch.zeros(4, dtype=torch.float32)
                finger_vel = torch.zeros(4, dtype=torch.float32)

            return {
                "all_names": all_names,
                "all_positions": all_joint_positions.tolist(),
                "arm_names": self.arm_joint_names,
                "arm_positions": arm_positions.tolist(),
                "arm_indices": self.arm_joint_indices,
                "arm_velocities": arm_vel.tolist(),
                "arm_torques": arm_tau.tolist(),
                "finger_names": self.finger_joint_names,
                "finger_positions": finger_positions.tolist(),
                "finger_indices": self.finger_joint_indices,
                "finger_velocities": finger_vel.tolist(),
                "finger_torques": finger_tau.tolist(),
            }

        except Exception as e:
            logger.error(f"[get_joint_states] Exception: {e}")
            import traceback

            logger.error(traceback.format_exc())
            return None

    def set_arm_joint_positions(self, target_arm_positions, task_num: int = None):
        """设置14arm joints的目标位置（使用物理驱动，避免穿模）"""
        from isaacsim.core.utils.types import ArticulationActions

        if self._articulation is None:
            raise RuntimeError("Articulation uninitialized")

        if not isinstance(target_arm_positions, torch.Tensor):
            target_arm_positions = torch.tensor(target_arm_positions, dtype=torch.float32)

        if target_arm_positions.shape[0] != 14:
            raise ValueError(f"Expected14joint positions，got {target_arm_positions.shape[0]} ")

        if target_arm_positions.ndim == 1:
            target_arm_positions = target_arm_positions.unsqueeze(0)

        joint_indices = torch.tensor(self.arm_joint_indices, dtype=torch.int32)

        self._articulation.apply_action(
            ArticulationActions(
                joint_positions=target_arm_positions,
                joint_indices=joint_indices,
            )
        )

    @property
    def uses_physical_arm_control(self) -> bool:
        return self.arm_control_mode == "physical"

    def reset_arm_command_state(self) -> None:
        """Re-seed the arm trajectory generator from measured simulation state."""
        self._arm_drive_command_positions = None
        self._arm_drive_command_velocities = None
        self._arm_hard_follow_positions = None

    @staticmethod
    def _advance_arm_trajectory(
        command_positions: torch.Tensor,
        command_velocities: torch.Tensor,
        target: torch.Tensor,
        dt: float,
        velocity_limit: float,
        acceleration_limit: float,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Advance one stopping-distance-aware trapezoidal trajectory step."""
        error = target - command_positions
        at_target = torch.abs(error) <= 1e-7
        stopping_velocity = torch.sqrt(2.0 * acceleration_limit * torch.abs(error))
        desired_velocity = torch.sign(error) * torch.minimum(
            stopping_velocity,
            torch.full_like(stopping_velocity, velocity_limit),
        )
        velocity_delta = torch.clamp(
            desired_velocity - command_velocities,
            min=-acceleration_limit * dt,
            max=acceleration_limit * dt,
        )
        next_velocity = command_velocities + velocity_delta
        position_step = next_velocity * dt
        would_overshoot = (position_step * error > 0.0) & (
            torch.abs(position_step) >= torch.abs(error)
        )
        position_step = torch.where(would_overshoot, error, position_step)
        next_position = command_positions + position_step
        next_position = torch.where(at_target, target, next_position)
        return next_position, next_velocity

    def set_arm_joint_positions_physical(self, target_arm_positions, step_size: float) -> None:
        """Send acceleration-limited position targets through Isaac articulation drives."""
        from isaacsim.core.utils.types import ArticulationActions

        if self._articulation is None or not self._articulation_physics_ready():
            return

        target = torch.as_tensor(target_arm_positions, dtype=torch.float32).flatten()
        if target.shape[0] != 14:
            raise ValueError(f"Expected 14 arm joint positions, got {target.shape[0]}")

        dt = max(float(step_size), 1e-4)
        joint_indices = torch.tensor(self.arm_joint_indices, dtype=torch.int32)
        if self._arm_drive_command_positions is None:
            measured = self._articulation.get_joint_positions()
            if measured is None:
                command_positions = target.clone()
            else:
                measured = torch.as_tensor(measured, dtype=torch.float32).flatten()
                command_positions = measured[
                    torch.tensor(self.arm_joint_indices, dtype=torch.long)
                ].clone()
            self._arm_drive_command_positions = command_positions
            self._arm_drive_command_velocities = torch.zeros_like(command_positions)

        acceleration = max(self.arm_acceleration_limit, 1e-4)
        velocity_limit = max(self.arm_velocity_limit, 1e-4)
        next_position, next_velocity = self._advance_arm_trajectory(
            self._arm_drive_command_positions,
            self._arm_drive_command_velocities,
            target,
            dt,
            velocity_limit,
            acceleration,
        )

        self._arm_drive_command_positions = next_position
        self._arm_drive_command_velocities = next_velocity
        self._articulation.apply_action(
            ArticulationActions(
                joint_positions=next_position.unsqueeze(0),
                joint_indices=joint_indices,
            )
        )

    def set_arm_joint_positions_hard(self, target_arm_positions) -> None:
        """Smoothly follow arm targets for imported URDF teleop stability.

        This is a kinematic actuator shim for the imported URDF only. The IK
        target generation remains the official pipeline; this filtered command
        prevents the visual snap caused by directly setting large IK jumps.
        """
        if self._articulation is None or not self._articulation_physics_ready():
            return

        if not isinstance(target_arm_positions, torch.Tensor):
            target_arm_positions = torch.tensor(target_arm_positions, dtype=torch.float32)

        target_arm_positions = target_arm_positions.flatten()
        if target_arm_positions.shape[0] != 14:
            raise ValueError(
                f"Expected 14 arm joint positions, got {target_arm_positions.shape[0]}"
            )

        joint_indices = torch.tensor(self.arm_joint_indices, dtype=torch.int32)
        if self._arm_hard_follow_positions is None:
            current = self._articulation.get_joint_positions()
            if current is not None:
                if not isinstance(current, torch.Tensor):
                    current = torch.tensor(current, dtype=torch.float32)
                current = current.flatten()
                self._arm_hard_follow_positions = current[
                    torch.tensor(self.arm_joint_indices, dtype=torch.long)
                ].clone()
            else:
                self._arm_hard_follow_positions = target_arm_positions.clone()

        max_step = 0.006
        self._arm_hard_follow_positions = self._arm_hard_follow_positions + torch.clamp(
            target_arm_positions - self._arm_hard_follow_positions,
            min=-max_step,
            max=max_step,
        )

        self._articulation.set_joint_positions(
            self._arm_hard_follow_positions,
            joint_indices=joint_indices,
        )
        self._articulation.set_joint_velocities(
            torch.zeros(len(self.arm_joint_indices), dtype=torch.float32),
            joint_indices=joint_indices,
        )

    def set_body_joint_positions(self, target_body_positions, task_num: int = None):
        """Hold body DOFs. USD baseline passes 0 (asset zero = standing). URDF uses standing table."""
        from isaacsim.core.utils.types import ArticulationActions

        if self._articulation is None:
            raise RuntimeError("Articulation uninitialized")

        if not self.body_joint_indices:
            return

        if isinstance(target_body_positions, (int, float)) and float(target_body_positions) == 0.0:
            if self.use_explicit_standing_body:
                target_body_positions = [
                    self.standing_joint_positions[idx]
                    for idx in self.body_joint_indices
                ]
            else:
                target_body_positions = [0.0] * len(self.body_joint_indices)

        if isinstance(target_body_positions, (int, float)):
            target_body_positions = [float(target_body_positions)] * len(self.body_joint_indices)

        if not isinstance(target_body_positions, torch.Tensor):
            target_body_positions = torch.tensor(target_body_positions, dtype=torch.float32)

        target_body_positions = target_body_positions.flatten()

        if target_body_positions.shape[0] != len(self.body_joint_indices):
            raise ValueError(
                f"Body joints expected {len(self.body_joint_indices)} positions, "
                f"got {target_body_positions.shape[0]}"
            )

        self._articulation.apply_action(
            ArticulationActions(
                joint_positions=target_body_positions.unsqueeze(0),
                joint_indices=torch.tensor(self.body_joint_indices, dtype=torch.int32),
            )
        )

    def set_finger_positions(self, target_fingers, side: Optional[str] = None, task_num: int = None):
        """Set old gripper target positions.

        This controls the original 4-joint gripper model only.
        If the hand-version robot does not expose old gripper joints, this function safely does nothing.
        """
        import math
        from isaacsim.core.utils.types import ArticulationActions

        if self._articulation is None:
            raise RuntimeError("Articulation uninitialized")

        if not self.finger_joint_indices:
            logger.debug("No old gripper joints available; skipping set_finger_positions.")
            return

        if not isinstance(target_fingers, torch.Tensor):
            target_fingers = torch.tensor(target_fingers, dtype=torch.float32)

        target_fingers = target_fingers.flatten()

        if side == "left":
            if target_fingers.shape[0] != 2:
                raise ValueError(f"left GripperExpected2joint positions，got {target_fingers.shape[0]} ")
            all_indices = self.finger_joint_indices[:2]
        elif side == "right":
            if target_fingers.shape[0] != 2:
                raise ValueError(f"right GripperExpected2joint positions，got {target_fingers.shape[0]} ")
            all_indices = self.finger_joint_indices[2:4]
        else:
            if target_fingers.shape[0] != 4:
                raise ValueError(f"机器人接口Expected4Finger joint位置，got {target_fingers.shape[0]} ")
            all_indices = self.finger_joint_indices

        valid_pos = []
        valid_idx = []

        for pos, idx in zip(target_fingers.tolist(), all_indices):
            if not math.isnan(pos):
                valid_pos.append(pos)
                valid_idx.append(idx)

        if not valid_pos:
            return

        self._articulation.apply_action(
            ArticulationActions(
                joint_positions=torch.tensor([valid_pos], dtype=torch.float32),
                joint_indices=torch.tensor(valid_idx, dtype=torch.int32),
            )
        )

    def move_home(self):
        """将机器人重置到初始化时的关节初始Pose，并同步重置IK相关状态。"""
        self.open_dexterous_hand("L")
        self.open_dexterous_hand("R")
        for side in ("L", "R"):
            if self.dexterous_hand_target_positions.get(side):
                self.dexterous_hand_current_positions[side] = list(
                    self.dexterous_hand_target_positions[side]
                )

        initial_positions = np.asarray(self.initial_joint_positions, dtype=np.float32)
        home_targets = list(initial_positions.tolist())
        self._apply_dexterous_hand_targets_to_vector(home_targets)
        initial_positions = np.asarray(home_targets, dtype=np.float32)
        all_indices = np.asarray(self._all_dof_indices, dtype=np.int32)

        self._articulation.set_joint_positions(
            torch.tensor(initial_positions, dtype=torch.float32),
            joint_indices=torch.tensor(all_indices, dtype=torch.int32),
        )
        self._articulation.set_joint_velocities(
            torch.zeros(len(all_indices), dtype=torch.float32),
            joint_indices=torch.tensor(all_indices, dtype=torch.int32),
        )

        self.time = 0.0
        self._ik_warn_counter = 0
        self._last_arm_positions.clear()
        self.reset_arm_command_state()

        if self.ik_solver is not None:
            self.ik_solver.reset_runtime_state()
            self.ik_solver.sync_joint_positions(
                self._articulation.dof_names,
                initial_positions.tolist(),
            )
            self.ik_solver.save_initial_q()

    def apply_finger_efforts(self, efforts: list[float]) -> None:
        """Apply efforts to old 4-joint gripper.

        If old gripper joints do not exist, safely do nothing.
        """
        from isaacsim.core.utils.types import ArticulationActions

        if self._articulation is None:
            return

        if not self.finger_joint_indices:
            logger.debug("No old gripper joints available; skipping apply_finger_efforts.")
            return

        t = torch.tensor([efforts], dtype=torch.float32)
        idx = torch.tensor(self.finger_joint_indices, dtype=torch.int32)

        self._articulation.apply_action(
            ArticulationActions(joint_efforts=t, joint_indices=idx)
        )

    def close_gripper(self, side: Optional[str] = None, task_name: Optional[str] = None):
        """Close old gripper on specified side.

        If old gripper joints do not exist, safely do nothing.
        """
        from isaacsim.core.utils.types import ArticulationActions

        if self._articulation is None:
            return

        if not self.finger_joint_indices:
            logger.debug("No old gripper joints available; skipping close_gripper.")
            return

        target_pos = [self.gripper_close_width] * 2

        if side == "left":
            control_finger_indices = torch.tensor(self.finger_joint_indices[:2], dtype=torch.int32)
        elif side == "right":
            control_finger_indices = torch.tensor(self.finger_joint_indices[2:4], dtype=torch.int32)
        else:
            control_finger_indices = torch.tensor(self.finger_joint_indices, dtype=torch.int32)
            target_pos = target_pos * 2

        self._articulation.apply_action(
            ArticulationActions(
                joint_positions=torch.tensor([target_pos], dtype=torch.float32),
                joint_indices=control_finger_indices,
            )
        )

    def open_gripper(self, side: Optional[str] = None, task_name: Optional[str] = None):
        """Open old gripper on specified side.

        If old gripper joints do not exist, safely do nothing.
        """
        from isaacsim.core.utils.types import ArticulationActions

        if self._articulation is None:
            return

        if not self.finger_joint_indices:
            logger.debug("No old gripper joints available; skipping open_gripper.")
            return

        target_pos = [self.gripper_open_width] * 2

        if side == "left":
            control_finger_indices = torch.tensor(self.finger_joint_indices[:2], dtype=torch.int32)
        elif side == "right":
            control_finger_indices = torch.tensor(self.finger_joint_indices[2:4], dtype=torch.int32)
        else:
            control_finger_indices = torch.tensor(self.finger_joint_indices, dtype=torch.int32)
            target_pos = target_pos * 2

        self._articulation.apply_action(
            ArticulationActions(
                joint_positions=torch.tensor([target_pos], dtype=torch.float32),
                joint_indices=control_finger_indices,
            )
        )

    def get_camera_rgb(self, camera_name):
        """Get camera RGB."""
        if camera_name not in self.cameras:
            return None

        try:
            return self.cameras[camera_name].get_rgb()
        except Exception:
            return None

    def get_camera_depth(self, camera_name):
        """Get camera depth."""
        if camera_name not in self.cameras:
            return None

        try:
            return self.cameras[camera_name].get_depth()
        except Exception:
            return None

    def get_camera_rgbd(self, camera_name):
        """Get camera RGB-D."""
        return {
            "rgb": self.get_camera_rgb(camera_name),
            "depth": self.get_camera_depth(camera_name),
            "camera_name": camera_name,
        }

    def get_sixforce(self):
        wrench_data_list = []
        sensor_joint_forces = self._articulation.get_measured_joint_forces()[0]

        for joint_name in self.sixforce_joint_names:
            joint_index = self._articulation.get_joint_index(joint_name)
            sixforce_data = sensor_joint_forces[joint_index + 1]
            wrench_data = {
                "frame_id": joint_name.replace("_joint", "_frame"),
                "force": sixforce_data[:3].tolist(),
                "torque": sixforce_data[3:].tolist(),
            }
            wrench_data_list.append(wrench_data)

        return wrench_data_list

    def initialize_ik(self, urdf_path: str):
        """Initialize Pinocchio IK solver and build joint index mapping."""
        from Ubtech_sim.source.DualArmIK import DualArmIK

        logger.info(f"Initializing DualArmIK, using URDF: {urdf_path}")
        self.ik_solver = DualArmIK(urdf_path)
        logger.info("DualArmIK initialized successfully, building joint index mapping...")

        dof_names = self._articulation.dof_names

        self._left_arm_isaac_indices = []
        for jname in DualArmIK.LEFT_ARM_JOINTS:
            if jname in dof_names:
                self._left_arm_isaac_indices.append(self._articulation.get_dof_index(jname))

        self._right_arm_isaac_indices = []
        for jname in DualArmIK.RIGHT_ARM_JOINTS:
            if jname in dof_names:
                self._right_arm_isaac_indices.append(self._articulation.get_dof_index(jname))

        WAIST_JOINTS = ["waist_yaw_joint", "waist_pitch_joint"]
        self._waist_isaac_indices = []
        self._waist_init_positions = []

        for jname in WAIST_JOINTS:
            if jname in dof_names:
                idx = self._articulation.get_dof_index(jname)
                self._waist_isaac_indices.append(idx)
                self._waist_init_positions.append(
                    float(STANDING_JOINT_POSE.get(jname, 0.0))
                )

        if self._waist_isaac_indices:
            self._articulation.set_joint_positions(
                torch.tensor(self._waist_init_positions, dtype=torch.float32),
                joint_indices=torch.tensor(self._waist_isaac_indices, dtype=torch.int32),
            )

        joints = self.get_joint_states()

        if joints is not None:
            self.ik_solver.sync_joint_positions(
                joints["all_names"],
                joints["all_positions"],
            )

        self.ik_solver.save_initial_q()

        left_neutral = [self._joint_value_map.get(j, 0.0) for j in DualArmIK.LEFT_ARM_JOINTS]
        right_neutral = [self._joint_value_map.get(j, 0.0) for j in DualArmIK.RIGHT_ARM_JOINTS]

        self.ik_solver.set_neutral_config(left_neutral, right_neutral)

        print(
            f"[DualArmIK] Initialization complete  Left arm {len(self._left_arm_isaac_indices)} DOF, "
            f"Right arm {len(self._right_arm_isaac_indices)} DOF, "
            f"Waist locked {len(self._waist_isaac_indices)} DOF"
        )

    def get_ee_poses(self):
        """Get current dual-arm end-effector xyzrpy."""
        joints = self.get_joint_states()

        if joints is None or self.ik_solver is None:
            return None

        self.ik_solver.sync_joint_positions(
            joints["all_names"],
            joints["all_positions"],
        )

        return self.ik_solver.get_both_ee_poses()

    def control_dual_arm_ik(
        self,
        step_size: float,
        left_target_xyzrpy=None,
        right_target_xyzrpy=None,
        **ik_kwargs,
    ):
        """Dual-arm IK control."""
        from isaacsim.core.utils.types import ArticulationActions

        self.time += step_size

        if self.ik_solver is None:
            print("[DualArmIK] IK solver not initialized, please call initialize_ik() first")
            return

        joints = self.get_joint_states()

        if joints is None:
            return

        isaac_names = joints["all_names"]
        isaac_positions = joints["all_positions"]

        ik_result = self.ik_solver.solve_dual_arm(
            left_target_xyzrpy=left_target_xyzrpy,
            right_target_xyzrpy=right_target_xyzrpy,
            isaac_joint_names=isaac_names,
            isaac_joint_positions=isaac_positions,
            **ik_kwargs,
        )

        all_indices = []
        all_positions = []
        warn_msg = []

        if "left_joint_positions" in ik_result:
            smoothed = self._smooth_joints("left", ik_result["left_joint_positions"])
            all_indices.extend(self._left_arm_isaac_indices)
            all_positions.extend(smoothed.tolist())

            if not ik_result["left_success"]:
                warn_msg.append("left arm")

        if "right_joint_positions" in ik_result:
            smoothed = self._smooth_joints("right", ik_result["right_joint_positions"])
            all_indices.extend(self._right_arm_isaac_indices)
            all_positions.extend(smoothed.tolist())

            if not ik_result["right_success"]:
                warn_msg.append("right arm")

        if warn_msg:
            self._ik_warn_counter += 1

            if self._ik_warn_counter >= 200:
                diag = ""

                if hasattr(self.ik_solver, "_last_fail_info"):
                    info = self.ik_solver._last_fail_info
                    diag = (
                        f" | pos_err={info['pos_err']:.4f}m"
                        f" rot_err={info['rot_err']:.4f}rad"
                        f" rot_tol={info['effective_rot_tol']:.4f}"
                    )

                print(f"[DualArmIK] Warning: {', '.join(warn_msg)} IK not converged{diag}")
                self._ik_warn_counter = 0
        else:
            self._ik_warn_counter = 0

        if self._waist_isaac_indices:
            self._articulation.set_joint_positions(
                torch.tensor(self._waist_init_positions, dtype=torch.float32),
                joint_indices=torch.tensor(self._waist_isaac_indices, dtype=torch.int32),
            )

        # In physical mode the callback-owned trajectory generator is the only
        # writer of arm drive targets. Kinematic mode keeps the legacy behavior.
        if len(all_indices) > 0 and not self.uses_physical_arm_control:
            self._articulation.apply_action(
                ArticulationActions(
                    joint_positions=torch.tensor([all_positions], dtype=torch.float32),
                    joint_indices=torch.tensor(all_indices, dtype=torch.int32),
                )
            )

        ik_result["smoothed_positions"] = all_positions
        return ik_result

    def _smooth_joints(self, side: str, ik_positions: np.ndarray) -> np.ndarray:
        """EMA smoothing of IK joint output to reduce jitter."""
        ik_positions = np.asarray(ik_positions, dtype=float)

        if side not in self._last_arm_positions:
            self._last_arm_positions[side] = ik_positions.copy()
            return ik_positions

        prev = self._last_arm_positions[side]
        alpha = self._smooth_alpha
        smoothed = prev + alpha * (ik_positions - prev)
        self._last_arm_positions[side] = smoothed.copy()

        return smoothed
