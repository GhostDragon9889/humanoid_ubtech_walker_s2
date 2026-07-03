#!/usr/bin/env python

# Copyright 2024 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Grasp target computation and real-time update module

Core functions:
1. Compute robot grasp target (6D pose: x/y/z + roll/pitch/yaw) from scattered part poses
2. Automatically select optimal grasp arm (left/right)
3. Real-time update of grasp targets (adapt to part position changes)
4. Periodically output debug information to monitor grasp status and errors

Dependencies:
- numpy: numerical computation
- pinocchio: robot kinematics (RPY to rotation matrix conversion)
- pxr/omni.usd: read real-time part poses from USD scene
- RobotArticulation/CoordinateTransform: custom classes for robot control and coordinate transformation
"""

import numpy as np
import pinocchio as pin


def rotation_between_vectors(v_from: np.ndarray, v_to: np.ndarray) -> np.ndarray:
    """
    Compute minimal rotation matrix (3x3) from source vector to target vector based on Rodrigues formula.
    Core: Find the shortest path rotation matrix that rotates v_from to v_to.

    Args:
        v_from: np.ndarray - Source vector (3D), works even if not normalized
        v_to: np.ndarray - Target vector (3D), works even if not normalized

    Returns:
        np.ndarray - 3x3 rotation matrix satisfying R @ v_from = v_to (after normalization)

    Boundary handling:
    - Vectors almost aligned (dot product > 0.9999): Returns identity matrix
    - Vectors almost opposite (dot product < -0.9999): Rotate 180 degrees around perpendicular axis
    """
    # Normalize input vectors
    v_from = v_from / np.linalg.norm(v_from)
    v_to = v_to / np.linalg.norm(v_to)
    # Calculate cross product (rotation axis) and dot product (cosine of rotation angle)
    cross = np.cross(v_from, v_to)
    dot = np.dot(v_from, v_to)

    # Vectors almost aligned: no rotation needed
    if dot > 0.9999:
        return np.eye(3)
    # Vectors almost opposite: rotate 180 degrees around perpendicular axis
    if dot < -0.9999:
        # Choose reference vector not collinear with v_from, compute perpendicular axis
        perp = np.array([1, 0, 0]) if abs(v_from[0]) < 0.9 else np.array([0, 1, 0])
        axis = np.cross(v_from, perp)
        axis /= np.linalg.norm(axis)
        return 2 * np.outer(axis, axis) - np.eye(3)

    # Build skew-symmetric matrix (matrix form of cross product)
    skew = np.array([
        [0,        -cross[2],  cross[1]],
        [cross[2],  0,        -cross[0]],
        [-cross[1], cross[0],  0       ],
    ])
    # Rodrigues rotation formula: R = I + sin(theta)*skew + (1-cos(theta))*skew^2
    return np.eye(3) + skew + skew @ skew / (1 + dot)


class GraspPlanner:
    """
    Grasp target planner for dual-arm manipulation

    Core capabilities:
    1. Compute optimal grasp pose from scene part poses
    2. Automatically select left/right arm for grasping
    3. Real-time update of grasp targets (adapt to part movement)
    4. Output grasp status debug information

    Attributes:
        grasp_arm: str - Current selected grasp arm ("left"/"right")
        active_grasp: np.ndarray - Current 6D grasp target [x,y,z,roll,pitch,yaw] (robot base frame)
        left_init: np.ndarray - Left arm end-effector initial 6D pose
        right_init: np.ndarray - Right arm end-effector initial 6D pose
        target_index: int - Target part index in part_poses list
        tcp_offset_local: np.ndarray - Tool Center Point (TCP) offset relative to end-effector in local frame (3D)
        ik_rot_weight: float - Rotation weight in IK solving (smaller = prioritize position accuracy)
        target_prim_path: str - Target part USD path in scene
        tcp_offset_base: np.ndarray - TCP offset transformed to robot base frame (3D)
        R_grasp: np.ndarray - Grasp target rotation matrix (3x3, base coordinate system)
    """

    def __init__(self, grasp_cfg: dict, robot, coord_transform):
        """
        Initialize grasp planner

        Args:
            grasp_cfg: dict - Grasp configuration dictionary (from task YAML "grasp" field), containing:
                - target_index: Target part index
                - tcp_offset: TCP local offset [x,y,z]
                - ik_rot_weight: IK rotation weight
                - debug_interval: Debug info output interval (steps)
            robot: RobotArticulation - Initialized robot instance (with IK solver)
            coord_transform: CoordinateTransform - Coordinate transformation instance (world/robot base conversion)

        Returns:
            None
        """
        self.grasp_cfg = grasp_cfg
        self.robot = robot  # Robot instance (with IK solve, end-effector pose retrieval interfaces)
        self.coord = coord_transform  # Coordinate transformation tool (world <-> base)

        # Initialize configuration parameters
        self.target_index = grasp_cfg.get("target_index", 0)  # Default: grasp part at index 0
        self.tcp_offset_local = np.array(
            grasp_cfg.get("tcp_offset", [0.0, 0.0, 0.0]), dtype=float
        )  # TCP local offset (end-effector -> grasp point)
        self.ik_rot_weight = grasp_cfg.get("ik_rot_weight", 0.1)  # IK rotation weight

        # Initialize runtime variables
        self.target_prim_path = None  # Target part USD path
        self.tcp_offset_base = np.zeros(3, dtype=float)  # TCP offset (base frame)
        self.R_grasp = np.eye(3)  # Grasp rotation matrix (initial identity)
        self.palm_target_pos_base = None
        self.palm_target_R_base = None
        self.grasp_arm = "left"  # Default left arm
        self.active_grasp = None  # Current grasp target (6D)
        self.left_init = None  # Left arm initial pose
        self.right_init = None  # Right arm initial pose
        self._debug_counter = 0  # Debug counter
        self._debug_interval = grasp_cfg.get("debug_interval", 200)  # Debug output interval

    def compute_grasp_target(self, part_poses: list) -> None:
        """
        Core function: Compute grasp target from scattered part poses (arm selection, pose, rotation)

        Args:
            part_poses: list[dict] - Scene part pose list (from SceneBuilder.get_parts_world_poses()), each element contains:
                - position: Part world coordinates [x,y,z]
                - prim_path: Part USD path

        Returns:
            None - Results stored in self.active_grasp and other attributes

        Core logic:
        1. Cache manipulator arm initial poses
        2. Validate target part index
        3. Transform part coordinates to robot base frame
        4. Auto-select grasp arm based on part y coordinate (right half -> left arm, left half -> right arm)
        5. Compute grasp rotation (ensure end-effector Z-axis points down, X-axis points to part)
        6. Compute TCP offset and get final grasp target pose
        """
        # Cache initial end-effector poses
        ee_poses = self.robot.get_ee_poses()
        self.left_init = ee_poses["left"].copy()
        self.right_init = ee_poses["right"].copy()
        print(f"[IK] Left arm initial xyzrpy: {self.left_init}")
        print(f"[IK] Right arm initial xyzrpy: {self.right_init}")

        # Validate target part index
        if self.target_index >= len(part_poses):
            print(
                f"[Grasp] Error: target_index={self.target_index} "
                f"out of range (num_parts={len(part_poses)})"
            )
            self.active_grasp = self.left_init.copy()  # Fallback to left arm initial pose
            return

        # Get target part information
        target_info = part_poses[self.target_index]
        target_world = np.array(target_info["position"], dtype=float)
        target_world[2] += float(self.grasp_cfg.get("grasp_height_offset", 0.0))
        self.target_prim_path = target_info["prim_path"]  # Part USD path
        print(f"[Grasp] Target part: {target_info['prim_path']}")
        print(f"[Grasp] World position: {target_world}")

        # Transform part coordinates to robot base coordinate system
        obj_robot = self.coord.world_to_robot(target_world)
        print(f"[Grasp] Object base position: {obj_robot}")

        # Auto-select arm: based on part y coordinate (base frame), right half -> left arm, left half -> right arm
        if obj_robot[1] > 0:
            self.grasp_arm = "left"
            active_init = self.left_init.copy()
        else:
            self.grasp_arm = "right"
            active_init = self.right_init.copy()
        print(f"[Grasp] Auto select arm: {self.grasp_arm} (object y={obj_robot[1]:.3f})")

        # Build the palm target in the robot base frame. For a side grasp, the
        # palm normal points toward the object and finger direction is configurable.
        world_down = np.array([0.0, 0.0, -1.0])  # World frame down direction
        base_down = self.coord.robot_world_R_inv @ world_down  # Transform to base frame
        print(f"[Grasp] World down (base frame): {base_down}")

        # Get initial end-effector pose of selected arm
        init_se3 = self.robot.ik_solver.get_ee_pose(self.grasp_arm)
        ee_pos_base = np.array(init_se3.translation)

        # Compute end-effector -> part direction as a fallback and for top grasps.
        reach_dir = obj_robot - ee_pos_base  # Direction from end-effector to part
        reach_dir = reach_dir - np.dot(reach_dir, base_down) * base_down  # Project to horizontal plane
        # Boundary handling: direction vector near zero
        if np.linalg.norm(reach_dir) < 1e-6:
            cand = np.array([1, 0, 0]) if abs(base_down[0]) < 0.9 else np.array([0, 1, 0])
            reach_dir = cand - np.dot(cand, base_down) * base_down
        reach_dir /= np.linalg.norm(reach_dir)
        grasp_orientation = str(self.grasp_cfg.get("grasp_orientation", "top")).lower()
        if grasp_orientation == "side":
            side_approach_mode = str(
                self.grasp_cfg.get("side_approach_mode", "arm_lateral")
            ).lower()
            if side_approach_mode == "from_hand":
                z_grasp = reach_dir
            else:
                # Tienkung-style side grasp: keep the hand vertical and approach
                # from the active arm's outside lateral side, not from the current
                # wrist-to-object diagonal.
                lateral_axis = np.array([0.0, 1.0, 0.0], dtype=float)
                lateral_axis = lateral_axis - np.dot(lateral_axis, base_down) * base_down
                if np.linalg.norm(lateral_axis) < 1e-6:
                    lateral_axis = np.array([0.0, 0.0, 1.0], dtype=float)
                    lateral_axis = lateral_axis - np.dot(lateral_axis, base_down) * base_down
                lateral_axis /= np.linalg.norm(lateral_axis)

                # Base +Y is the robot's left side in the imported Walker frame.
                # Palm +Z points from the palm toward the object.
                z_grasp = -lateral_axis if self.grasp_arm == "left" else lateral_axis

                forward_bias = float(self.grasp_cfg.get("side_forward_bias", 0.0))
                if abs(forward_bias) > 1e-6:
                    forward_axis = np.array([1.0, 0.0, 0.0], dtype=float)
                    forward_axis = forward_axis - np.dot(forward_axis, base_down) * base_down
                    forward_axis /= np.linalg.norm(forward_axis)
                    z_grasp = z_grasp + forward_bias * forward_axis
                    z_grasp /= np.linalg.norm(z_grasp)

            finger_direction = str(
                self.grasp_cfg.get("side_finger_direction", "forward")
            ).lower()
            if finger_direction == "down":
                x_grasp = base_down
            elif finger_direction == "up":
                x_grasp = -base_down
            else:
                forward_axis = np.array([1.0, 0.0, 0.0], dtype=float)
                forward_axis = forward_axis - np.dot(forward_axis, base_down) * base_down
                if np.linalg.norm(forward_axis) < 1e-6:
                    forward_axis = reach_dir
                forward_axis /= np.linalg.norm(forward_axis)
                x_grasp = -forward_axis if finger_direction == "backward" else forward_axis

            x_grasp = x_grasp - np.dot(x_grasp, z_grasp) * z_grasp
            if np.linalg.norm(x_grasp) < 1e-6:
                x_grasp = reach_dir - np.dot(reach_dir, z_grasp) * z_grasp
            x_grasp /= np.linalg.norm(x_grasp)
            y_grasp = np.cross(z_grasp, x_grasp)
            y_grasp /= np.linalg.norm(y_grasp)
            x_grasp = np.cross(y_grasp, z_grasp)
            x_grasp /= np.linalg.norm(x_grasp)
            desired_palm_R = np.column_stack([x_grasp, y_grasp, z_grasp])
            print(f"[Grasp] Side finger direction: {finger_direction}")
            print(f"[Grasp] Side approach mode: {side_approach_mode}")
        else:
            x_grasp = reach_dir
            y_grasp = np.cross(base_down, x_grasp)
            y_grasp /= np.linalg.norm(y_grasp)
            desired_palm_R = np.column_stack([x_grasp, y_grasp, base_down])
        print(f"[Grasp] Orientation mode: {grasp_orientation}")
        grasp_frame = str(self.grasp_cfg.get("grasp_frame", "sixforce")).lower()

        if grasp_frame == "palm":
            palm_frame = f"{'L' if self.grasp_arm == 'left' else 'R'}_palm_link"
            palm_frame_id = self.robot.ik_solver.model.getFrameId(palm_frame)
            palm_se3 = self.robot.ik_solver.data.oMf[palm_frame_id].copy()
            sixforce_to_palm = init_se3.inverse() * palm_se3
            palm_tcp = np.asarray(
                self.grasp_cfg.get("palm_tcp_offset", [0.09, 0.0, -0.01]),
                dtype=float,
            )
            if bool(self.grasp_cfg.get("mirror_palm_tcp_lateral", True)):
                lateral = float(self.grasp_cfg.get("palm_tcp_lateral_offset", abs(palm_tcp[1])))
                if lateral > 0.0:
                    palm_tcp[1] = lateral if self.grasp_arm == "left" else -lateral

            # Build the desired palm pose first, then recover the corresponding
            # sixforce pose expected by the arm IK solver.
            desired_palm_pos = obj_robot - desired_palm_R @ palm_tcp
            self.palm_target_pos_base = desired_palm_pos.copy()
            self.palm_target_R_base = desired_palm_R.copy()
            self.R_grasp = desired_palm_R @ sixforce_to_palm.rotation.T
            grasp_pos = desired_palm_pos - self.R_grasp @ sixforce_to_palm.translation

            # Cache the complete sixforce -> pinch-center offset for tracking.
            self.tcp_offset_local = (
                sixforce_to_palm.translation + sixforce_to_palm.rotation @ palm_tcp
            )
            print(f"[Grasp] Using palm frame: {palm_frame}")
            print(f"[Grasp] Palm TCP local: {palm_tcp}")
            print(f"[Grasp] Palm +X (finger direction): {desired_palm_R[:, 0]}")
            print(f"[Grasp] Palm +Z (surface normal): {desired_palm_R[:, 2]}")
        else:
            self.R_grasp = desired_palm_R
            self.palm_target_pos_base = None
            self.palm_target_R_base = None
            grasp_pos = obj_robot - self.R_grasp @ self.tcp_offset_local

        grasp_rpy = pin.rpy.matrixToRpy(self.R_grasp)
        self.tcp_offset_base = self.R_grasp @ self.tcp_offset_local
        self.active_grasp = np.concatenate([grasp_pos, grasp_rpy])  # Combine 6D pose

        # Verify TCP offset computation correctness
        tcp_pred = grasp_pos + self.tcp_offset_base
        tcp_residual = tcp_pred - obj_robot
        print(
            f"[Grasp] TCP check: tcp_pred={tcp_pred}, residual={tcp_residual}, "
            f"|res|={np.linalg.norm(tcp_residual):.6f}m"
        )
        # Warning: TCP offset too large (may exceed manipulator workspace)
        if np.linalg.norm(self.tcp_offset_local) > 0.5:
            print(f"[Grasp] Warning: tcp_offset too large: {self.tcp_offset_local}")

        # Output grasp target summary
        delta = grasp_pos - active_init[:3]
        print(f"[Grasp] === Target Summary ===")
        print(f"  Object (base):       {obj_robot}")
        print(f"  TCP offset (base):   {self.tcp_offset_base}")
        print(f"  sixforce target:     {grasp_pos}")
        print(f"  Initial sixforce:    {active_init[:3]}")
        print(f"  sixforce displacement:    {delta}  (norm={np.linalg.norm(delta):.4f}m)")
        print(f"  grasp rpy:         {grasp_rpy}")
        print(f"  rot_weight:        {self.ik_rot_weight}")

    def update_active_target(self) -> None:
        """
        Real-time update of grasp target position (adapt to part movement)

        Core logic: Read part current pose, recompute grasp target position (keep rotation unchanged)

        Args:
            None

        Returns:
            None

        Note:
        - Only updates position (first 3 dimensions), rotation (last 3 dimensions) remains unchanged
        - Uses pre-computed R_grasp (avoids feedback errors from real-time rotation computation)
        """
        if self.target_prim_path is None:
            return
        # Get part current world coordinates
        current_world = self._get_prim_world_position(self.target_prim_path)
        print(f">>> [GraspPlanner] update_active_target: Current part world position: {current_world}")
        current_world += 0.15
        if current_world is None:
            return
        # Transform to base coordinate system
        obj_robot_now = self.coord.world_to_robot(current_world)
        # Update grasp target position (keep rotation unchanged)
        self.active_grasp[:3] = obj_robot_now - self.tcp_offset_base
        # Debug: for easier subsequent grasping
        print(f">>> [GraspPlanner] update_active_target: Updated grasp target: {self.active_grasp}")

    def get_control_targets(self):
        """
        Get target poses for robot control

        Args:
            None

        Returns:
            tuple: (left_target, right_target, ik_rot_weight)
                - left_target: Left arm 6D control target (active_grasp if grasp arm is left, otherwise initial pose)
                - right_target: Right arm 6D control target (active_grasp if grasp arm is right, otherwise initial pose)
                - ik_rot_weight: IK solver rotation weight
        """
        if self.grasp_arm == "left":
            return self.active_grasp, self.right_init, self.ik_rot_weight
        else:
            return self.left_init, self.active_grasp, self.ik_rot_weight

    def log_debug(self) -> None:
        """
        Periodically output grasp status debug information (every debug_interval steps)

        Monitor content:
        1. End-effector/TCP/target/part world coordinates
        2. TCP to part distance
        3. End-effector pose error, Z-axis direction error

        Args:
            None

        Returns:
            None
        """
        self._debug_counter += 1
        if self._debug_counter < self._debug_interval:
            return
        self._debug_counter = 0  # Reset counter

        # Skip if no valid grasp target
        if self.active_grasp is None or self.target_prim_path is None:
            return

        # Get current end-effector pose (base frame)
        ee_se3 = self.robot.ik_solver.get_ee_pose(self.grasp_arm)
        ee_pos_base = np.array(ee_se3.translation)
        ee_R = np.array(ee_se3.rotation)
        ee_z_base = ee_R[:, 2]  # End-effector Z-axis direction

        # Grasp target pose (base frame)
        tgt_pos_base = self.active_grasp[:3]
        tgt_rpy = self.active_grasp[3:]

        # Compute position error (end-effector -> target)
        pos_err = np.linalg.norm(ee_pos_base - tgt_pos_base)

        # Compute Z-axis direction error (world frame, should point to [0,0,-1])
        ee_z_world = self.coord.robot_world_R @ ee_z_base
        z_angle_err = np.arccos(np.clip(np.dot(ee_z_world, [0, 0, -1]), -1, 1))

        # Compute current TCP position (base frame, using actual end-effector rotation)
        tcp_base = ee_pos_base + ee_R @ self.tcp_offset_local

        # Transform to world frame for comparison
        ee_world = self.coord.robot_to_world(ee_pos_base)
        tcp_world = self.coord.robot_to_world(tcp_base)
        tgt_world = self.coord.robot_to_world(tgt_pos_base)
        obj_world = self._get_prim_world_position(self.target_prim_path)
        tcp_obj_dist = np.linalg.norm(tcp_world - obj_world) if obj_world is not None else -1

        # Output debug information
        print(f"[Debug] -- Grasp Status ({self.grasp_arm} arm) --")
        print(f"  EE (world):        {ee_world}")
        print(f"  TCP (world):       {tcp_world}")
        print(f"  sixforce target (world): {tgt_world}")
        print(f"  Object (world):    {obj_world}")
        print(f"  TCP<->Object distance:    {tcp_obj_dist:.4f}m")
        print(f"  EE pos_err:        {pos_err:.4f}m  (sixforce->target)")
        print(f"  EE Z-axis (world): {ee_z_world}  (should be [0,0,-1])")
        print(f"  Z-axis angle (world):  {np.degrees(z_angle_err):.2f}deg")
        print(f"  Target rpy:        {tgt_rpy}")

    @staticmethod
    def _get_prim_world_position(prim_path: str):
        """
        Static helper: Read part world coordinates from USD scene given path

        Args:
            prim_path: str - Part path in USD scene (e.g., "/World/Part01")

        Returns:
            np.ndarray | None: Part world coordinates (3D), None if part is invalid
        """
        from pxr import UsdGeom
        import omni.usd

        # Get USD stage
        stage = omni.usd.get_context().get_stage()
        prim = stage.GetPrimAtPath(prim_path)
        # Validate part
        if not prim.IsValid():
            return None
        # Read part world transform
        xform_cache = UsdGeom.XformCache()
        world_tf = xform_cache.GetLocalToWorldTransform(prim)
        # Extract translation component (world coordinates)
        return np.array(world_tf.ExtractTranslation(), dtype=float)
