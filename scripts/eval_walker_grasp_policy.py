#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
from pathlib import Path

import numpy as np

import isaac_walker_fixed_grasp_demo as demo


def parse_args():
    parser = argparse.ArgumentParser(
        description="Roll out a trained LeRobot ACT policy in the Walker S2 grasp scene."
    )
    parser.add_argument("--urdf", required=True, help="Absolute or relative path to the Walker S2 URDF")
    parser.add_argument(
        "--policy-path",
        required=True,
        help="Path to a LeRobot pretrained_model checkpoint directory",
    )
    parser.add_argument("--headless", action="store_true", help="Run Isaac Sim headless")
    parser.add_argument("--robot-z", type=float, default=0.86)
    parser.add_argument("--cube-center", type=float, nargs=3, default=(0.92, 0.20, 1.105))
    parser.add_argument("--randomize-cube", action="store_true")
    parser.add_argument("--cube-x-range", type=float, nargs=2, default=(0.88, 0.96))
    parser.add_argument("--cube-y-range", type=float, nargs=2, default=(0.16, 0.24))
    parser.add_argument("--random-seed", type=int, default=None)
    parser.add_argument("--container-center", type=float, nargs=2, default=(0.62, 0.24))
    parser.add_argument("--container-size", type=float, nargs=2, default=(0.28, 0.20))
    parser.add_argument("--disable-container", action="store_true")
    parser.add_argument("--show-hand-colliders", action="store_true")
    parser.add_argument("--debug-collider-visual-scale", type=float, default=1.0)
    parser.add_argument(
        "--enable-camera-view-windows",
        dest="camera_view_windows",
        action="store_true",
        default=True,
        help="Open the draggable stereo camera viewer window",
    )
    parser.add_argument(
        "--disable-camera-view-windows",
        dest="camera_view_windows",
        action="store_false",
        help="Do not open the draggable stereo camera viewer window",
    )
    parser.add_argument("--camera-viewer-scale", type=float, default=1.0)
    parser.add_argument("--warmup-steps", type=int, default=180, help="Settle scene before policy rollout")
    parser.add_argument("--rollout-frames", type=int, default=450, help="Number of policy frames to run")
    parser.add_argument(
        "--policy-step-stride",
        type=int,
        default=4,
        help="Number of sim steps to hold each policy action; 4 matches 60 Hz sim to 15 FPS data",
    )
    parser.add_argument("--action-blend", type=float, default=1.0, help="Blend from previous command to policy action")
    parser.add_argument(
        "--max-action-delta",
        type=float,
        default=0.0,
        help="Optional per-policy-frame joint delta clamp in radians; 0 disables",
    )
    parser.add_argument("--hold-final-steps", type=int, default=240)
    parser.add_argument("--pause-before-policy", action="store_true")
    parser.add_argument("--no-preprocess-hand-collisions", action="store_true")
    parser.add_argument("--save-stage", default="")
    return parser.parse_args()


def set_named(q, name_to_i, values):
    q = np.array(q, dtype=float).copy()
    missing = []
    for name, value in values.items():
        if name in name_to_i:
            q[name_to_i[name]] = float(value)
        else:
            missing.append(name)
    if missing:
        print(f"[WARN] Missing joints, skipped: {missing}")
    return q


def make_ready_pose(q_home, name_to_i):
    ready_arm_pose = {
        "L_shoulder_pitch_joint": 0.09322471888572098,
        "L_shoulder_roll_joint": -0.5933223843430208,
        "L_shoulder_yaw_joint": -1.595878574835185,
        "L_elbow_roll_joint": -1.8963565338596158,
        "L_elbow_yaw_joint": 1.4000461262831179,
        "L_wrist_pitch_joint": -0.00048740902645395785,
        "L_wrist_roll_joint": 0.0998718010009366,
        "R_shoulder_pitch_joint": -0.09321727661087699,
        "R_shoulder_roll_joint": -0.5933455607833843,
        "R_shoulder_yaw_joint": 1.595869459316937,
        "R_elbow_roll_joint": -1.8963607249359917,
        "R_elbow_yaw_joint": -1.4000874256427638,
        "R_wrist_pitch_joint": 0.00048144049606466176,
        "R_wrist_roll_joint": 0.09985407619802703,
        "head_pitch_joint": -0.785398163,
        "head_yaw_joint": 1.9677590016147396e-07,
    }
    right_hand_open = {
        "hand3_v1_right_R_thumb_cmp_joint": 0.03,
        "hand3_v1_right_R_thumb_mpp_joint": 0.03,
        "hand3_v1_right_R_thumb_ip_joint": 0.03,
        "hand3_v1_right_R_index_mpp_joint": 0.03,
        "hand3_v1_right_R_index_ip_joint": 0.03,
        "hand3_v1_right_R_middle_mpp_joint": 0.03,
        "hand3_v1_right_R_middle_ip_joint": 0.03,
        "hand3_v1_right_R_ring_mpp_joint": 0.03,
        "hand3_v1_right_R_ring_ip_joint": 0.03,
        "hand3_v1_right_R_little_mpp_joint": 0.03,
        "hand3_v1_right_R_little_ip_joint": 0.03,
    }
    left_hand_open = {
        name.replace("hand3_v1_right_R_", "hand3_v1_left_L_"): (
            -value if "thumb_cmp_joint" in name else value
        )
        for name, value in right_hand_open.items()
    }
    q_ready = set_named(q_home, name_to_i, ready_arm_pose)
    return set_named(set_named(q_ready, name_to_i, left_hand_open), name_to_i, right_hand_open)


def load_policy(policy_path, baseline_dir):
    os.environ.setdefault("HF_HOME", str(baseline_dir / ".hf_cache"))
    os.environ.setdefault("HF_DATASETS_CACHE", str(baseline_dir / ".hf_cache" / "datasets"))

    import torch
    from src.lerobot.configs.policies import PreTrainedConfig
    from src.lerobot.policies.factory import get_policy_class, make_pre_post_processors

    policy_path = Path(policy_path).expanduser().resolve()
    cfg = PreTrainedConfig.from_pretrained(policy_path)
    cfg.pretrained_path = str(policy_path)
    policy_cls = get_policy_class(cfg.type)
    policy = policy_cls.from_pretrained(pretrained_name_or_path=policy_path, config=cfg)
    policy.eval()
    if hasattr(policy, "reset"):
        policy.reset()

    device = str(next(policy.parameters()).device)
    preprocessor, postprocessor = make_pre_post_processors(
        policy_cfg=cfg,
        pretrained_path=str(policy_path),
        preprocessor_overrides={
            "device_processor": {"device": device},
            "normalizer_processor": {"device": device},
        },
        postprocessor_overrides={
            "device_processor": {"device": "cpu"},
        },
    )
    return torch, policy, preprocessor, postprocessor


def move_to_device(value, device):
    if hasattr(value, "to"):
        return value.to(device=device)
    if isinstance(value, dict):
        return {key: move_to_device(item, device) for key, item in value.items()}
    if isinstance(value, list):
        return [move_to_device(item, device) for item in value]
    if isinstance(value, tuple):
        return tuple(move_to_device(item, device) for item in value)
    return value


def frame_to_policy_image(frame):
    if frame is None:
        image = np.zeros((demo.HEAD_CAMERA_HEIGHT, demo.HEAD_CAMERA_WIDTH, 3), dtype=np.float32)
    else:
        image = np.asarray(frame)
        if image.ndim != 3:
            raise ValueError(f"Camera frame must be HWC, got shape {image.shape}")
        if image.shape[2] > 3:
            image = image[:, :, :3]
        if image.dtype != np.float32:
            image = image.astype(np.float32)
        if image.size and float(np.nanmax(image)) > 1.0:
            image = image / 255.0
        image = np.clip(image, 0.0, 1.0)
        if image.shape[0] != demo.HEAD_CAMERA_HEIGHT or image.shape[1] != demo.HEAD_CAMERA_WIDTH:
            try:
                import cv2

                image = cv2.resize(
                    image,
                    (demo.HEAD_CAMERA_WIDTH, demo.HEAD_CAMERA_HEIGHT),
                    interpolation=cv2.INTER_AREA,
                )
            except Exception:
                image = image[: demo.HEAD_CAMERA_HEIGHT, : demo.HEAD_CAMERA_WIDTH, :3]
    return np.ascontiguousarray(np.transpose(image[:, :, :3], (2, 0, 1)), dtype=np.float32)


def get_cube_position(cube):
    try:
        pose = cube.get_world_pose()
        return np.asarray(pose[0], dtype=float)
    except Exception:
        return None


def main():
    args = parse_args()
    baseline_dir = demo.ensure_repo_python_paths()
    input_urdf = Path(args.urdf).expanduser().resolve()
    policy_path = Path(args.policy_path).expanduser().resolve()
    if not input_urdf.exists():
        raise FileNotFoundError(input_urdf)
    if not policy_path.exists():
        raise FileNotFoundError(policy_path)

    if args.no_preprocess_hand_collisions:
        urdf_for_import = input_urdf
    else:
        urdf_for_import = demo.simplify_hand_collisions(
            input_urdf,
            show_debug=args.show_hand_colliders,
        )
        print(f"[INFO] Wrote simplified-hand-collision URDF: {urdf_for_import}")

    (
        sim_app,
        _urdf,
        World,
        DynamicCuboid,
        FixedCuboid,
        ArticulationWrapper,
        ArticulationAction,
        omni,
        Gf,
        UsdGeom,
    ) = demo.import_isaac_modules(args.headless)

    torch, policy, preprocessor, postprocessor = load_policy(policy_path, baseline_dir)
    policy_device = next(policy.parameters()).device
    print(f"[INFO] Loaded policy: {policy_path}")
    print(f"[INFO] Policy inference device: {policy_device}")

    world = World(stage_units_in_meters=1.0)
    world.scene.add_default_ground_plane()
    stage = omni.usd.get_context().get_stage()

    cfg = demo.make_import_config(_urdf)
    prim_path = demo.import_robot(urdf_for_import, cfg, omni)
    print(f"[INFO] Imported robot at prim: {prim_path}")

    robot_xyz = (0.75, -0.2, args.robot_z)
    robot_yaw_deg = 90.0
    demo.set_prim_pose(stage, UsdGeom, Gf, prim_path, robot_xyz, (0.0, 0.0, robot_yaw_deg))

    camera_paths = demo.create_robot_cameras(stage, UsdGeom, Gf, prim_path)
    for camera_name, camera_path in camera_paths.items():
        print(f"[INFO] Robot camera {camera_name}: {camera_path}")

    if args.show_hand_colliders and not args.no_preprocess_hand_collisions:
        debug_count = demo.draw_usd_hand_collider_debug_boxes(
            stage,
            UsdGeom,
            Gf,
            prim_path,
            args.debug_collider_visual_scale,
        )
        print(f"[INFO] Drew USD hand collider debug boxes: {debug_count}")

    table_center = np.array([0.75, 0.3, 1.02])
    table_size = np.array([1.2, 0.65, 0.04])
    table_top_z = float(table_center[2] + table_size[2] * 0.5)
    if args.randomize_cube:
        rng = np.random.default_rng(args.random_seed)
        cube_center = np.asarray(args.cube_center, dtype=float).copy()
        cube_center[0] = rng.uniform(float(args.cube_x_range[0]), float(args.cube_x_range[1]))
        cube_center[1] = rng.uniform(float(args.cube_y_range[0]), float(args.cube_y_range[1]))
    else:
        cube_center = np.asarray(args.cube_center, dtype=float)

    world.scene.add(
        FixedCuboid(
            prim_path="/World/table",
            name="table",
            position=table_center,
            scale=table_size,
        )
    )
    container_info = None
    if not args.disable_container:
        container_info = demo.add_pick_place_container(
            world,
            stage,
            UsdGeom,
            Gf,
            FixedCuboid,
            args.container_center,
            table_top_z,
            args.container_size,
        )

    cube = world.scene.add(
        DynamicCuboid(
            prim_path="/World/grasp_cube",
            name="grasp_cube",
            position=cube_center,
            scale=np.array([0.035, 0.035, 0.13]),
            mass=0.05,
        )
    )

    if args.save_stage:
        omni.usd.get_context().save_as_stage(str(Path(args.save_stage).expanduser().resolve()))
        print(f"[INFO] Saved stage to: {args.save_stage}")

    try:
        robot = world.scene.add(ArticulationWrapper(prim_path=prim_path, name="walker_s2"))
    except TypeError:
        robot = world.scene.add(ArticulationWrapper(prim_path=prim_path))

    world.reset()
    try:
        robot.initialize()
    except Exception:
        pass

    dof_names = demo.get_dof_names(robot)
    name_to_i = {name: i for i, name in enumerate(dof_names)}
    q_home = demo.get_joint_positions(robot, len(dof_names))
    if len(q_home) != len(dof_names):
        q_home = np.zeros(len(dof_names), dtype=float)
    q_ready_open = make_ready_pose(q_home, name_to_i)

    if hasattr(robot, "set_joints_default_state"):
        try:
            robot.set_joints_default_state(positions=q_ready_open)
            world.reset()
            try:
                robot.initialize()
            except Exception:
                pass
        except Exception:
            pass
    demo.set_joint_positions(robot, q_ready_open)
    demo.set_object_position(cube, stage, UsdGeom, Gf, "/World/grasp_cube", cube_center)

    camera_sensors = demo.create_head_camera_sensors(camera_paths, world=world)
    camera_viewer = None
    if not args.headless and args.camera_view_windows:
        camera_viewer = demo.HeadCameraCvViewer(scale=args.camera_viewer_scale)
        if camera_viewer.started:
            print("[INFO] Opened draggable head-camera viewer: walker_s2_cameras")

    camera_frames = {}
    warned_blank = set()

    def read_camera_frames():
        for name, sensor in camera_sensors.items():
            try:
                frame = sensor.get_rgb()
                if frame is not None:
                    camera_frames[name] = np.asarray(frame)
            except Exception as exc:
                print(f"[WARN] Could not read camera sensor {name}: {exc}")
        if camera_viewer is not None:
            camera_viewer.update(dict(camera_frames))

    def close_resources():
        for sensor in camera_sensors.values():
            sensor.destroy()
        if camera_viewer is not None:
            camera_viewer.close()

    def apply_full_body(q):
        robot.apply_action(ArticulationAction(joint_positions=np.asarray(q, dtype=float)))

    def step_world():
        world.step(render=True)
        read_camera_frames()

    print(f"[INFO] Cube start center: {cube_center.tolist()}")
    if container_info is not None:
        print(
            "[INFO] Pick-place container center/top: "
            f"{container_info['center'].tolist()}, size_xy={container_info['size'].tolist()}"
        )
    print("[INFO] Settling ready pose before policy rollout.")
    q_command = q_ready_open.copy()
    for _ in range(max(0, int(args.warmup_steps))):
        apply_full_body(q_command)
        step_world()

    missing_cameras = [name for name in ("head_left", "head_right") if name not in camera_frames]
    if missing_cameras:
        print(
            "[WARN] Missing live camera frames before rollout: "
            f"{missing_cameras}. Blank frames will be used for those cameras."
        )

    if args.pause_before_policy and not args.headless:
        input("[INFO] Press Enter to start trained-policy rollout...")

    print(
        "[INFO] Starting policy rollout: "
        f"frames={args.rollout_frames}, stride={args.policy_step_stride}, action_blend={args.action_blend}"
    )
    for frame_i in range(max(0, int(args.rollout_frames))):
        observation_q = demo.get_joint_positions(robot, len(dof_names)).astype(np.float32)
        observation = {
            "observation.state": torch.from_numpy(observation_q),
            "observation.images.head_left": torch.from_numpy(
                frame_to_policy_image(camera_frames.get("head_left"))
            ),
            "observation.images.head_right": torch.from_numpy(
                frame_to_policy_image(camera_frames.get("head_right"))
            ),
        }
        for name in ("head_left", "head_right"):
            if name not in camera_frames and name not in warned_blank:
                warned_blank.add(name)
                print(f"[WARN] Policy is receiving blank image for missing camera: {name}")

        with torch.inference_mode():
            batch = preprocessor(observation)
            batch = move_to_device(batch, policy_device)
            action = policy.select_action(batch)
            action = postprocessor(action)

        q_policy = action.detach().cpu().numpy().reshape(-1).astype(float)
        if len(q_policy) != len(dof_names):
            raise ValueError(f"Policy action has {len(q_policy)} DOFs, robot has {len(dof_names)}")

        blend = float(np.clip(args.action_blend, 0.0, 1.0))
        q_next = (1.0 - blend) * q_command + blend * q_policy
        if args.max_action_delta and args.max_action_delta > 0.0:
            delta = np.clip(q_next - q_command, -args.max_action_delta, args.max_action_delta)
            q_next = q_command + delta
        q_command = q_next

        for _ in range(max(1, int(args.policy_step_stride))):
            apply_full_body(q_command)
            step_world()

        if frame_i % 30 == 0:
            cube_pos = get_cube_position(cube)
            cube_text = "unknown" if cube_pos is None else np.round(cube_pos, 4).tolist()
            print(f"[INFO] Policy frame {frame_i:04d}, cube={cube_text}")

    print(f"[INFO] Holding final policy command for {args.hold_final_steps} sim steps.")
    for _ in range(max(0, int(args.hold_final_steps))):
        apply_full_body(q_command)
        step_world()

    cube_pos = get_cube_position(cube)
    if cube_pos is not None:
        print(f"[INFO] Final cube position: {cube_pos.tolist()}")
        if container_info is not None:
            center = container_info["center"]
            size = container_info["size"]
            in_xy = (
                abs(float(cube_pos[0] - center[0])) <= float(size[0]) * 0.5
                and abs(float(cube_pos[1] - center[1])) <= float(size[1]) * 0.5
            )
            above_table = float(cube_pos[2]) > float(container_info["table_top_z"])
            print(f"[INFO] Container XY success check: {bool(in_xy)}, above table: {bool(above_table)}")

    close_resources()
    sim_app.close()


if __name__ == "__main__":
    main()
