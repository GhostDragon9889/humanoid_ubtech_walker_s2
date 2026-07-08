#!/usr/bin/env python3
"""
Standalone Isaac Sim demo for UBTECH Walker S2 URDF:
- preprocesses only the hand collision meshes into simple box collisions
- imports URDF with fixed base and self-collision disabled
- creates a table + graspable cube
- solves a right-arm IK grasp pose and closes the right hand on the cube

Run with Isaac Sim, for example:
  /isaac-sim/python.sh isaac_walker_fixed_grasp_demo.py \
    --urdf /absolute/path/to/walker_s2_description_hand3_v1_left_hand3_v1_right.urdf

This script is intentionally small but reuses the baseline DualArmIK solver.
"""

import argparse
import sys
import os
import math
import queue
import time
import shutil
import struct
import subprocess
import threading
import textwrap
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np


HEAD_CAMERA_WIDTH = 640
HEAD_CAMERA_HEIGHT = 480


def find_challenge_repo_root() -> Path:
    here = Path(__file__).resolve()
    for parent in (here.parent, *here.parents):
        if (parent / "Ubtech_sim").is_dir() and (parent / "assets/resources").is_dir():
            return parent
        candidate = parent / "GlobalHumanoidRobotChallenge_2026_Baseline"
        if (candidate / "Ubtech_sim").is_dir() and (candidate / "assets/resources").is_dir():
            return candidate
    raise RuntimeError("Could not locate GlobalHumanoidRobotChallenge_2026_Baseline repo root")


def ensure_repo_python_paths() -> Path:
    baseline_dir = find_challenge_repo_root()
    src_dir = baseline_dir / "src"
    src_s = str(src_dir)
    if src_s in sys.path:
        sys.path.remove(src_s)
    sys.path.insert(0, src_s)

    # Keep the repo root available for `src.*` and `Ubtech_sim.*` imports, but
    # do not let local folders such as `datasets/` shadow installed packages.
    baseline_s = str(baseline_dir)
    if baseline_s in sys.path:
        sys.path.remove(baseline_s)
    sys.path.append(baseline_s)
    return baseline_dir


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--urdf", required=True, help="Absolute or relative path to the Walker S2 URDF")
    p.add_argument("--headless", action="store_true", help="Run Isaac Sim headless")
    p.add_argument("--robot-z", type=float, default=0.86, help="Lift base_link so feet are near ground")
    p.add_argument("--duration-after", type=int, default=2400, help="Extra sim steps to hold the closed grasp")
    p.add_argument(
        "--cube-center",
        type=float,
        nargs=3,
        default=(0.92, 0.20, 1.105),
        help="Cube center in world coordinates.",
    )
    p.add_argument(
        "--randomize-cube",
        action="store_true",
        help="Randomize the cube XY position within --cube-x-range/--cube-y-range",
    )
    p.add_argument(
        "--cube-x-range",
        type=float,
        nargs=2,
        default=(0.88, 0.96),
        help="World X range used when --randomize-cube is enabled",
    )
    p.add_argument(
        "--cube-y-range",
        type=float,
        nargs=2,
        default=(0.16, 0.24),
        help="World Y range used when --randomize-cube is enabled",
    )
    p.add_argument("--random-seed", type=int, default=None, help="Optional seed for randomized scene setup")
    p.add_argument(
        "--container-center",
        type=float,
        nargs=2,
        default=(0.62, 0.24),
        help="Container/tray center XY position on the table",
    )
    p.add_argument(
        "--container-size",
        type=float,
        nargs=2,
        default=(0.28, 0.20),
        help="Container/tray outer XY size",
    )
    p.add_argument(
        "--disable-container",
        action="store_true",
        help="Do not create the pick-and-place container/tray",
    )
    p.add_argument(
        "--palm-tcp-offset",
        type=float,
        nargs=3,
        default=(0.005, -0.018, 0.025),
        help="Object center in right palm_link coordinates",
    )
    p.add_argument(
        "--palm-world-nudge",
        type=float,
        nargs=3,
        default=(0.0, 0.0, 0.0),
        help="Extra world-frame offset applied to the IK palm target",
    )
    p.add_argument(
        "--object-palm-offset",
        type=float,
        nargs=3,
        default=None,
        help="Optional final cube center in the solved actual right palm frame; the third value follows the blue palm axis",
    )
    p.add_argument(
        "--debug-collider-visual-scale",
        type=float,
        default=1.0,
        help="Visual-only scale for red USD hand collider debug boxes",
    )
    p.add_argument("--show-hand-colliders", action="store_true", help="Show red hand collider debug boxes")
    p.add_argument(
        "--enable-robot-cameras",
        dest="enable_robot_cameras",
        action="store_true",
        default=True,
        help="Create the robot stereo head cameras (default)",
    )
    p.add_argument(
        "--disable-robot-cameras",
        dest="enable_robot_cameras",
        action="store_false",
        help="Do not create robot cameras",
    )
    p.add_argument(
        "--enable-camera-view-windows",
        dest="camera_view_windows",
        action="store_true",
        default=True,
        help="Open the auxiliary draggable head-camera viewer window (default)",
    )
    p.add_argument(
        "--disable-camera-view-windows",
        dest="camera_view_windows",
        action="store_false",
        help="Do not open the auxiliary draggable head-camera viewer window",
    )
    p.add_argument(
        "--camera-viewer-scale",
        type=float,
        default=1.0,
        help="Display scale for the draggable head-camera viewer",
    )
    p.add_argument("--lift-height", type=float, default=0.08, help="World-Z lift height after closing the grasp")
    p.add_argument("--grasp-clearance", type=float, default=0.0, help="Optional retreat along the palm normal")
    p.add_argument("--pregrasp-distance", type=float, default=0.08, help="Palm-normal clearance before the grasp pose")
    p.add_argument("--pregrasp-steps", type=int, default=240, help="Number of sim steps from ready pose to pregrasp")
    p.add_argument("--approach-steps", type=int, default=180, help="Number of sim steps from pregrasp to grasp pose")
    p.add_argument("--lift-steps", type=int, default=180, help="Number of sim steps for the post-grasp lift")
    p.add_argument("--record-dataset-root", default="", help="Record one episode to this local LeRobot dataset root")
    p.add_argument("--record-repo-id", default="walker_s2_grasp", help="LeRobot repo_id stored in dataset metadata")
    p.add_argument("--record-task", default="pick the block and place it in the tray", help="Task string saved with recorded frames")
    p.add_argument("--record-fps", type=int, default=15, help="FPS metadata for recorded LeRobot episodes")
    p.add_argument(
        "--record-step-stride",
        type=int,
        default=4,
        help="Record one frame every N sim steps; default assumes roughly 60 Hz sim -> 15 FPS recording",
    )
    p.add_argument(
        "--record-start",
        choices=("task", "launch"),
        default="task",
        help="When to start saving frames. 'task' starts at the automatic grasp trigger; 'launch' records teleop waiting too.",
    )
    p.add_argument("--replay-dataset-root", default="", help="Replay actions from this local LeRobot dataset root")
    p.add_argument("--replay-repo-id", default="walker_s2_grasp", help="LeRobot repo_id for replay metadata")
    p.add_argument("--replay-episode", type=int, default=0, help="Episode index to replay from the dataset")
    p.add_argument("--no-preprocess-hand-collisions", action="store_true")
    p.add_argument("--save-stage", default="", help="Optional output USD path after import")
    return p.parse_args()


def is_hand_link_name(name: str) -> bool:
    return name.startswith("hand3_v1_left") or name.startswith("hand3_v1_right")


def hand_collider_box_size(name: str) -> str:
    # Conservative simple colliders. These are intentionally smaller than the visual meshes.
    if "palm" in name:
        return "0.048 0.052 0.016"
    if name.endswith("hand3_v1_left") or name.endswith("hand3_v1_right"):
        return "0.055 0.045 0.045"
    if "thumb_cmp" in name:
        return "0.027 0.020 0.018"
    if "thumb_mpp" in name:
        return "0.071 0.020 0.020"
    if "thumb_ip" in name:
        return "0.031 0.019 0.018"
    if "mpp" in name:
        return "0.053 0.015 0.014"
    if "middle_ip" in name:
        return "0.048 0.014 0.012"
    if "little_ip" in name:
        return "0.038 0.014 0.012"
    if "ip" in name:
        return "0.045 0.014 0.012"
    return "0.040 0.020 0.020"


def hand_collider_origin(name: str):
    if "palm" in name:
        return (0.0, 0.0, -0.004)
    mirrored_y = 1.0 if name.startswith("hand3_v1_left") else -1.0
    if "thumb_cmp" in name:
        return (0.0045, mirrored_y * 0.001, -0.007)
    if "thumb_mpp" in name:
        return (0.0425, mirrored_y * 0.0055, 0.0)
    if "thumb_ip" in name:
        return (0.0145, mirrored_y * 0.0055, 0.0)
    if "mpp" in name:
        return (0.0165, mirrored_y * 0.0015, 0.0)
    if "middle_ip" in name:
        return (0.0235, mirrored_y * 0.0005, 0.0)
    if "little_ip" in name:
        return (0.0175, mirrored_y * 0.0008, 0.0)
    if "ip" in name:
        return (0.022, mirrored_y * 0.0006, 0.0)
    return (0.0, 0.0, 0.0)


def simplify_hand_collisions(input_urdf: Path, show_debug: bool = False) -> Path:
    """Create a URDF copy with visual meshes unchanged but hand collision meshes replaced by boxes."""
    tree = ET.parse(str(input_urdf))
    root = tree.getroot()

    # urdfdom/Pinocchio logs an error for inline visual materials without names.
    # Add stable names to the temporary URDF only; visual colors stay unchanged.
    material_i = 0
    for link in root.findall("link"):
        link_name = link.attrib.get("name", "link")
        for visual in link.findall("visual"):
            material = visual.find("material")
            if material is not None and not material.attrib.get("name"):
                material.set("name", f"{link_name}_visual_material_{material_i}")
                material_i += 1

    # Replace only collision geometry of hand links. Keep visual STL meshes.
    for link in root.findall("link"):
        name = link.attrib.get("name", "")
        if not is_hand_link_name(name):
            continue
        if show_debug:
            for visual in link.findall("visual"):
                material = visual.find("material")
                if material is None:
                    material = ET.SubElement(visual, "material", name=f"{name}_debug_faded_visual")
                color = material.find("color")
                if color is None:
                    color = ET.SubElement(material, "color")
                rgba = color.attrib.get("rgba", "0.75 0.75 0.75 1").split()
                while len(rgba) < 4:
                    rgba.append("1")
                rgba[3] = "0.25"
                color.set("rgba", " ".join(rgba[:4]))
        for c in list(link.findall("collision")):
            link.remove(c)
        size = hand_collider_box_size(name)
        origin = hand_collider_origin(name)
        origin_xyz = f"{origin[0]} {origin[1]} {origin[2]}"
        collision = ET.SubElement(link, "collision")
        ET.SubElement(collision, "origin", xyz=origin_xyz, rpy="0 0 0")
        geom = ET.SubElement(collision, "geometry")
        ET.SubElement(geom, "box", size=size)

        if show_debug:
            visual = ET.SubElement(link, "visual")
            ET.SubElement(visual, "origin", xyz=origin_xyz, rpy="0 0 0")
            vgeom = ET.SubElement(visual, "geometry")
            ET.SubElement(vgeom, "box", size=size)
            mat = ET.SubElement(visual, "material", name=f"{name}_debug_collider_red")
            ET.SubElement(mat, "color", rgba="1 0 0 1")

    # Add/overwrite damping/friction. Isaac import can also override these, but this helps stability.
    for joint in root.findall("joint"):
        if joint.attrib.get("type") not in ("revolute", "continuous", "prismatic"):
            continue
        name = joint.attrib.get("name", "")
        dyn = joint.find("dynamics")
        if dyn is None:
            dyn = ET.SubElement(joint, "dynamics")
        if "hand3" in name:
            dyn.set("damping", "0.08")
            dyn.set("friction", "0.01")
        elif name.startswith("R_") or name.startswith("L_"):
            dyn.set("damping", "2.0")
            dyn.set("friction", "0.02")
        else:
            dyn.set("damping", "5.0")
            dyn.set("friction", "0.05")

    out = input_urdf.with_name(input_urdf.stem + "_isaac_simple_hand_collision.urdf")
    tree.write(str(out), encoding="utf-8", xml_declaration=True)
    return out


def import_isaac_modules(headless: bool):
    # Support both newer isaacsim.* and older omni.isaac.* module names.
    try:
        from isaacsim import SimulationApp
    except Exception:
        from omni.isaac.kit import SimulationApp

    sim_app = SimulationApp({"headless": headless})

    try:
        from isaacsim.core.utils.extensions import enable_extension
    except Exception:
        from omni.isaac.core.utils.extensions import enable_extension

    # Extension name changed across Isaac Sim versions. Try both.
    for ext in ("isaacsim.asset.importer.urdf", "omni.isaac.urdf"):
        try:
            enable_extension(ext)
        except Exception:
            pass
    sim_app.update()

    try:
        from isaacsim.asset.importer.urdf import _urdf
    except Exception:
        from omni.isaac.urdf import _urdf

    try:
        from isaacsim.core.api import World
        from isaacsim.core.api.objects import DynamicCuboid, FixedCuboid
    except Exception:
        from omni.isaac.core import World
        from omni.isaac.core.objects import DynamicCuboid, FixedCuboid

    try:
        from isaacsim.core.prims import SingleArticulation as ArticulationWrapper
    except Exception:
        try:
            from omni.isaac.core.articulations import Articulation as ArticulationWrapper
        except Exception:
            from omni.isaac.core.robots import Robot as ArticulationWrapper

    try:
        from isaacsim.core.utils.types import ArticulationAction
    except Exception:
        from omni.isaac.core.utils.types import ArticulationAction

    import omni.kit.commands
    import omni.usd
    from pxr import Gf, UsdGeom

    return sim_app, _urdf, World, DynamicCuboid, FixedCuboid, ArticulationWrapper, ArticulationAction, omni, Gf, UsdGeom


def make_import_config(_urdf):
    cfg = _urdf.ImportConfig()
    # Key safety settings for this humanoid test.
    cfg.fix_base = True
    cfg.self_collision = False
    cfg.make_default_prim = True
    cfg.distance_scale = 1.0
    cfg.density = 0.0

    # Preserve fixed hand attachment links instead of merging everything aggressively.
    for attr, value in {
        "merge_fixed_joints": False,
        "import_inertia_tensor": True,
        "convex_decomp": False,
        "parse_mimic": False,
        "replace_cylinders_with_capsules": True,
        "override_joint_dynamics": False,
    }.items():
        if hasattr(cfg, attr):
            try:
                setattr(cfg, attr, value)
            except Exception:
                pass
    return cfg


def tune_urdf_model_drives(robot_model):
    """Set drive gains before importing. Values are conservative for hand stability."""
    if not hasattr(robot_model, "joints"):
        return
    for joint_name in robot_model.joints:
        joint = robot_model.joints[joint_name]
        if not hasattr(joint, "drive"):
            continue
        try:
            if "hand3" in joint_name:
                joint.drive.strength = 12.0
                joint.drive.damping = 1.2
            elif joint_name.startswith("R_") or joint_name.startswith("L_"):
                joint.drive.strength = 900.0
                joint.drive.damping = 90.0
            else:
                joint.drive.strength = 1500.0
                joint.drive.damping = 150.0
        except Exception:
            pass


def import_robot(urdf_path: Path, cfg, omni):
    result, robot_model = omni.kit.commands.execute(
        "URDFParseFile", urdf_path=str(urdf_path), import_config=cfg
    )
    if not result:
        raise RuntimeError(f"URDFParseFile failed: {urdf_path}")

    tune_urdf_model_drives(robot_model)

    result, prim_path = omni.kit.commands.execute(
        "URDFImportRobot", urdf_robot=robot_model, import_config=cfg
    )
    if not result:
        raise RuntimeError("URDFImportRobot failed")
    return str(prim_path)


def set_prim_pose(stage, UsdGeom, Gf, prim_path: str, xyz, rpy_deg):
    prim = stage.GetPrimAtPath(prim_path)
    if not prim.IsValid():
        raise RuntimeError(f"Imported robot prim not found: {prim_path}")
    xform = UsdGeom.Xformable(prim)
    xform.ClearXformOpOrder()
    xform.AddTranslateOp().Set(Gf.Vec3d(float(xyz[0]), float(xyz[1]), float(xyz[2])))
    xform.AddRotateXYZOp().Set(Gf.Vec3f(float(rpy_deg[0]), float(rpy_deg[1]), float(rpy_deg[2])))


def draw_usd_hand_collider_debug_boxes(stage, UsdGeom, Gf, robot_prim_path: str, visual_scale: float = 1.0):
    root = stage.GetPrimAtPath(robot_prim_path)
    if not root.IsValid():
        return 0

    count = 0
    root_prefix = str(root.GetPath())
    for prim in stage.Traverse():
        prim_path = str(prim.GetPath())
        name = prim.GetName()
        if not prim_path.startswith(root_prefix) or not is_hand_link_name(name):
            continue

        size = np.fromstring(hand_collider_box_size(name), sep=" ", dtype=float)
        if size.size != 3:
            continue
        origin = hand_collider_origin(name)
        box_path = f"{prim_path}/debug_collider_box"
        box = UsdGeom.Cube.Define(stage, box_path)
        box.CreateSizeAttr(1.0)
        xform = UsdGeom.Xformable(box.GetPrim())
        xform.ClearXformOpOrder()
        xform.AddTranslateOp().Set(Gf.Vec3d(float(origin[0]), float(origin[1]), float(origin[2])))
        xform.AddScaleOp().Set(
            Gf.Vec3f(
                float(size[0] * visual_scale),
                float(size[1] * visual_scale),
                float(size[2] * visual_scale),
            )
        )
        box.CreateDisplayColorAttr([Gf.Vec3f(1.0, 0.0, 0.0)])
        box.CreateDisplayOpacityAttr([0.85])
        count += 1
    return count


def create_robot_cameras(stage, UsdGeom, Gf, robot_prim_path: str):
    camera_specs = (
        # The challenge USD optical center uses x=0.10704. This URDF's head mesh
        # blocks that pose, so keep the official orientation but move the mount
        # slightly outward.
        ("head_left", "head_stereo_left", (0.125, 0.13247, -0.03199)),
        ("head_right", "head_stereo_right", (0.125, 0.13247, 0.03199)),
    )

    camera_paths = {}
    head_path = f"{robot_prim_path}/head_pitch_link"
    if not stage.GetPrimAtPath(head_path).IsValid():
        print(f"[WARN] Camera parent link not found: {head_path}")
        return camera_paths

    for name, mount_name, translation in camera_specs:
        mount_path = f"{head_path}/{mount_name}"
        mount = UsdGeom.Xform.Define(stage, mount_path)
        mount_xform = UsdGeom.Xformable(mount.GetPrim())
        mount_xform.ClearXformOpOrder()
        mount_xform.AddTranslateOp().Set(Gf.Vec3d(*translation))
        mount_xform.AddOrientOp(UsdGeom.XformOp.PrecisionDouble).Set(
            Gf.Quatd(0.7071067811865476, Gf.Vec3d(-0.7071067811865475, 0.0, 0.0))
        )

        camera_path = f"{mount_path}/{mount_name}_Camera_01"
        camera = UsdGeom.Camera.Define(stage, camera_path)
        camera_xform = UsdGeom.Xformable(camera.GetPrim())
        camera_xform.ClearXformOpOrder()
        camera_xform.AddTranslateOp().Set(Gf.Vec3d(0.0, 0.0, 0.0))
        camera_xform.AddOrientOp(UsdGeom.XformOp.PrecisionDouble).Set(
            Gf.Quatd(0.5, Gf.Vec3d(0.5, -0.5, -0.5))
        )
        camera.CreateProjectionAttr("perspective")
        # Same FOV ratio as the challenge stereo cameras, scaled up to avoid
        # the tiny exported USD values that can break RGB render-product setup.
        camera.CreateFocalLengthAttr(23.316089063882828)
        camera.CreateHorizontalApertureAttr(57.5999990105629)
        camera.CreateVerticalApertureAttr(46.08000069856644)
        camera.CreateClippingRangeAttr(Gf.Vec2f(0.05, 100000.0))
        camera.GetFStopAttr().Set(0.0)
        camera.GetFocusDistanceAttr().Set(400.0)
        camera_paths[name] = camera_path

    return camera_paths


def create_camera_capture_viewports(camera_paths):
    from omni.kit.widget.viewport import ViewportWidget

    viewports = {}
    for name, camera_path in camera_paths.items():
        try:
            widget = ViewportWidget(
                camera_path=camera_path,
                resolution=(HEAD_CAMERA_WIDTH, HEAD_CAMERA_HEIGHT),
                viewport_api=f"walker_{name}_capture_viewport",
            )
            viewports[name] = widget
        except Exception as exc:
            print(f"[WARN] Could not create capture viewport for {name}: {exc}")
    return viewports


class DirectRgbCameraSensor:
    """RGB readback without Camera.initialize(), which attaches ReferenceTime."""

    def __init__(self, name, camera_path):
        import omni.replicator.core as rep

        self._annotator = None
        self._render_product = None
        try:
            self._render_product = rep.create.render_product(
                camera_path,
                resolution=(HEAD_CAMERA_WIDTH, HEAD_CAMERA_HEIGHT),
                name=f"walker_{name}_rgb",
            )
            self._annotator = rep.AnnotatorRegistry.get_annotator("rgb", do_array_copy=True)
            self._annotator.attach([self._render_product.path])
        except Exception:
            self.destroy()
            raise

    def get_rgb(self):
        data = self._annotator.get_data()
        if data is None or getattr(data, "size", 0) == 0:
            return None
        return np.asarray(data)[:, :, :3]

    def destroy(self):
        if self._annotator is not None and self._render_product is not None:
            try:
                self._annotator.detach([self._render_product.path])
            except Exception:
                pass
        self._annotator = None
        if self._render_product is not None:
            try:
                self._render_product.destroy()
            except Exception:
                pass
        self._render_product = None


def create_head_camera_sensors(camera_paths, world=None):
    sensors = {}
    for name, camera_path in camera_paths.items():
        try:
            sensors[name] = DirectRgbCameraSensor(name, camera_path)
            if world is not None:
                world.step(render=True)
                world.step(render=True)
            print(f"[INFO] Initialized direct RGB head camera: {name}")
        except Exception as exc:
            print(f"[WARN] Direct RGB camera unavailable for {name}: {exc}")
    return sensors


def schedule_viewport_capture(name, viewport_api, state):
    from omni.kit.viewport.utility import capture_viewport_to_buffer

    if name in state["pending"]:
        return

    def on_capture(buffer, buffer_size, width, height, byte_format):
        try:
            import omni.kit.renderer_capture

            raw = omni.kit.renderer_capture.convert_raw_bytes_to_list(
                buffer,
                buffer_size,
                width,
                height,
                byte_format,
            )
            arr = np.asarray(raw, dtype=np.uint8)
            pixel_count = int(width) * int(height)
            if pixel_count <= 0:
                return
            channels = int(arr.size) // pixel_count
            if channels < 3:
                return
            arr = arr[: pixel_count * channels].reshape((int(height), int(width), channels))
            state["frames"][name] = arr[:, :, :3].copy()
            if not state["logged"]:
                print(
                    "[INFO] Head camera viewport capture: "
                    f"{name} shape={state['frames'][name].shape}, format={byte_format}"
                )
                state["logged"] = True
        except Exception as exc:
            print(f"[WARN] Could not decode camera viewport capture {name}: {exc}")
        finally:
            state["pending"].discard(name)

    state["pending"].add(name)
    try:
        capture = capture_viewport_to_buffer(viewport_api, on_capture, is_hdr=False)
        if capture is not None:
            state["captures"].append(capture)
    except Exception as exc:
        state["pending"].discard(name)
        print(f"[WARN] Could not schedule camera viewport capture {name}: {exc}")


class HeadCameraCvViewer:
    def __init__(self, window_name="walker_s2_cameras", scale=1.0, window_x=40, window_y=40):
        self.window_name = window_name
        self.scale = max(float(scale), 0.1)
        self.window_x = int(window_x)
        self.window_y = int(window_y)
        self.proc = None
        self.started = False
        self._stop_event = threading.Event()
        self._send_queue = queue.Queue(maxsize=2)
        self._sender_thread = None

        try:
            self.proc = self._start_viewer_process()
            self.started = self.proc is not None and self.proc.poll() is None
            if self.started:
                self._sender_thread = threading.Thread(target=self._send_loop, daemon=True)
                self._sender_thread.start()
            self.update({})
        except Exception as exc:
            print(f"[WARN] Could not open camera viewer process: {exc}")

    def _start_viewer_process(self):
        python, env = self._find_viewer_python()
        if not python:
            print("[WARN] Could not find a system Python with PySide2/PyQt5 for the draggable camera viewer")
            return None

        viewer_code = r'''
import struct
import sys
import threading

try:
    from PySide2 import QtCore, QtGui, QtWidgets
    Signal = QtCore.Signal
except Exception:
    from PyQt5 import QtCore, QtGui, QtWidgets
    Signal = QtCore.pyqtSignal


def read_exact(stream, n):
    chunks = []
    remaining = n
    while remaining > 0:
        chunk = stream.read(remaining)
        if not chunk:
            return None
        chunks.append(chunk)
        remaining -= len(chunk)
    return b"".join(chunks)


class Receiver(QtCore.QObject):
    frame = Signal(bytes, int, int)
    closed = Signal()


class Viewer(QtWidgets.QWidget):
    def __init__(self, window_name, width, height, x, y):
        super().__init__()
        self.setWindowTitle(window_name)
        self.image_label = QtWidgets.QLabel()
        self.image_label.setAlignment(QtCore.Qt.AlignCenter)
        self.image_label.setMinimumSize(width, height)
        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self.image_label)
        self.resize(width, height)
        self.move(x, y)

    def set_frame(self, data, width, height):
        image = QtGui.QImage(data, width, height, width * 3, QtGui.QImage.Format_RGB888).copy()
        self.image_label.setPixmap(QtGui.QPixmap.fromImage(image))


def main():
    window_name = sys.argv[1]
    width = int(sys.argv[2])
    height = int(sys.argv[3])
    x = int(sys.argv[4])
    y = int(sys.argv[5])

    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    viewer = Viewer(window_name, width, height, x, y)
    receiver = Receiver()
    receiver.frame.connect(viewer.set_frame)
    receiver.closed.connect(app.quit)
    viewer.show()

    def reader():
        stream = sys.stdin.buffer
        while True:
            header = read_exact(stream, 16)
            if header is None:
                break
            magic, frame_width, frame_height, byte_count = struct.unpack("<4sIII", header)
            if magic != b"FRAM":
                break
            data = read_exact(stream, byte_count)
            if data is None:
                break
            receiver.frame.emit(data, frame_width, frame_height)
        receiver.closed.emit()

    threading.Thread(target=reader, daemon=True).start()
    app.exec_()


if __name__ == "__main__":
    main()
'''
        return subprocess.Popen(
            [
                python,
                "-u",
                "-c",
                viewer_code,
                self.window_name,
                str(int(2 * HEAD_CAMERA_WIDTH * self.scale)),
                str(int(HEAD_CAMERA_HEIGHT * self.scale)),
                str(self.window_x),
                str(self.window_y),
            ],
            stdin=subprocess.PIPE,
            env=env,
        )

    def _find_viewer_python(self):
        env = os.environ.copy()
        env.pop("PYTHONHOME", None)
        env.pop("PYTHONPATH", None)

        candidates = []
        env_python = os.environ.get("WALKER_CAMERA_VIEWER_PYTHON")
        if env_python:
            candidates.append(env_python)
        candidates.extend(("/usr/bin/python3", "/bin/python3", shutil.which("python3")))

        seen = set()
        for python in candidates:
            if not python or python in seen:
                continue
            seen.add(python)
            try:
                result = subprocess.run(
                    [
                        python,
                        "-c",
                        "try:\n import PySide2\nexcept Exception:\n import PyQt5\n",
                    ],
                    env=env,
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                    timeout=3.0,
                    check=False,
                )
            except Exception:
                continue
            if result.returncode == 0:
                return python, env
        return None, env

    def update(self, camera_frames):
        if not self.started or self.proc is None or self.proc.stdin is None or self.proc.poll() is not None:
            return

        tiles = []
        for name in ("head_left", "head_right"):
            rgb = camera_frames.get(name)
            if rgb is None:
                tile = np.zeros((HEAD_CAMERA_HEIGHT, HEAD_CAMERA_WIDTH, 3), dtype=np.uint8)
            else:
                tile = np.asarray(rgb)
                if tile.ndim == 3 and tile.shape[2] > 3:
                    tile = tile[:, :, :3]
                if tile.dtype != np.uint8:
                    if tile.size and float(np.nanmax(tile)) <= 1.0:
                        tile = np.clip(tile * 255.0, 0, 255).astype(np.uint8)
                    else:
                        tile = np.clip(tile, 0, 255).astype(np.uint8)
                if tile.shape[0] != HEAD_CAMERA_HEIGHT or tile.shape[1] != HEAD_CAMERA_WIDTH:
                    try:
                        import cv2

                        tile = cv2.resize(
                            tile,
                            (HEAD_CAMERA_WIDTH, HEAD_CAMERA_HEIGHT),
                            interpolation=cv2.INTER_AREA,
                        )
                    except Exception:
                        tile = np.ascontiguousarray(tile[:HEAD_CAMERA_HEIGHT, :HEAD_CAMERA_WIDTH, :3])

            tile = np.ascontiguousarray(tile[:, :, :3])
            try:
                import cv2

                cv2.putText(
                    tile,
                    name,
                    (12, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (0, 255, 0),
                    2,
                    cv2.LINE_AA,
                )
            except Exception:
                pass
            tiles.append(tile)

        canvas = np.ascontiguousarray(np.hstack(tiles))
        if self.scale != 1.0:
            try:
                import cv2

                canvas = cv2.resize(
                    canvas,
                    None,
                    fx=self.scale,
                    fy=self.scale,
                    interpolation=cv2.INTER_AREA,
                )
            except Exception:
                pass

        height, width = int(canvas.shape[0]), int(canvas.shape[1])
        payload = canvas.tobytes()
        try:
            if self._send_queue.full():
                self._send_queue.get_nowait()
            self._send_queue.put_nowait((width, height, payload))
        except queue.Full:
            pass

    def _send_loop(self):
        while not self._stop_event.is_set():
            try:
                width, height, payload = self._send_queue.get(timeout=0.05)
            except queue.Empty:
                continue
            try:
                self.proc.stdin.write(struct.pack("<4sIII", b"FRAM", width, height, len(payload)))
                self.proc.stdin.write(payload)
                self.proc.stdin.flush()
            except Exception as exc:
                print(f"[WARN] Camera viewer process stopped: {exc}")
                self.started = False
                return

    def close(self):
        if self.proc is None:
            return
        self._stop_event.set()
        if self._sender_thread is not None:
            self._sender_thread.join(timeout=1.0)
            self._sender_thread = None
        try:
            if self.proc.stdin is not None:
                self.proc.stdin.close()
            self.proc.terminate()
        except Exception:
            pass


def get_dof_names(robot):
    for attr in ("dof_names", "joint_names"):
        if hasattr(robot, attr):
            names = getattr(robot, attr)
            if names:
                return list(names)
    if hasattr(robot, "get_dof_names"):
        return list(robot.get_dof_names())
    raise RuntimeError("Could not read articulation DOF names from imported robot")


def get_joint_positions(robot, n):
    for method in ("get_joint_positions", "get_dof_positions"):
        if hasattr(robot, method):
            q = getattr(robot, method)()
            if q is not None:
                return np.array(q, dtype=float)
    return np.zeros(n, dtype=float)


def set_joint_positions(robot, q):
    for method in ("set_joint_positions", "set_dof_positions"):
        if hasattr(robot, method):
            getattr(robot, method)(np.array(q, dtype=float))
            return
    raise RuntimeError("Could not set articulation joint positions")


def set_object_position(obj, stage, UsdGeom, Gf, prim_path: str, xyz):
    try:
        obj.set_world_pose(position=np.array(xyz, dtype=float))
        return
    except Exception:
        pass
    prim = stage.GetPrimAtPath(prim_path)
    if not prim.IsValid():
        raise RuntimeError(f"Object prim not found: {prim_path}")
    xform = UsdGeom.Xformable(prim)
    translate_ops = [op for op in xform.GetOrderedXformOps() if op.GetOpType() == UsdGeom.XformOp.TypeTranslate]
    if translate_ops:
        translate_ops[0].Set(Gf.Vec3d(float(xyz[0]), float(xyz[1]), float(xyz[2])))
    else:
        xform.AddTranslateOp().Set(Gf.Vec3d(float(xyz[0]), float(xyz[1]), float(xyz[2])))


def set_prim_display_color(stage, UsdGeom, Gf, prim_path: str, color, opacity=1.0):
    prim = stage.GetPrimAtPath(prim_path)
    if not prim.IsValid():
        return
    try:
        gprim = UsdGeom.Gprim(prim)
        gprim.CreateDisplayColorAttr([Gf.Vec3f(float(color[0]), float(color[1]), float(color[2]))])
        gprim.CreateDisplayOpacityAttr([float(opacity)])
    except Exception:
        pass


def add_pick_place_container(
    world,
    stage,
    UsdGeom,
    Gf,
    FixedCuboid,
    center_xy,
    table_top_z,
    size_xy,
):
    cx, cy = float(center_xy[0]), float(center_xy[1])
    sx, sy = float(size_xy[0]), float(size_xy[1])
    UsdGeom.Xform.Define(stage, "/World/place_container")
    wall_t = 0.015
    bottom_t = 0.012
    wall_h = 0.08
    bottom_z = float(table_top_z) + bottom_t * 0.5
    wall_z = float(table_top_z) + bottom_t + wall_h * 0.5
    pieces = (
        ("bottom", (cx, cy, bottom_z), (sx, sy, bottom_t)),
        ("left_wall", (cx - sx * 0.5 + wall_t * 0.5, cy, wall_z), (wall_t, sy, wall_h)),
        ("right_wall", (cx + sx * 0.5 - wall_t * 0.5, cy, wall_z), (wall_t, sy, wall_h)),
        ("front_wall", (cx, cy + sy * 0.5 - wall_t * 0.5, wall_z), (sx, wall_t, wall_h)),
        ("back_wall", (cx, cy - sy * 0.5 + wall_t * 0.5, wall_z), (sx, wall_t, wall_h)),
    )
    for name, pos, scale in pieces:
        prim_path = f"/World/place_container/{name}"
        world.scene.add(
            FixedCuboid(
                prim_path=prim_path,
                name=f"place_container_{name}",
                position=np.array(pos, dtype=float),
                scale=np.array(scale, dtype=float),
            )
        )
        set_prim_display_color(stage, UsdGeom, Gf, prim_path, (0.05, 0.25, 0.9), opacity=0.9)

    return {
        "center": np.array([cx, cy, float(table_top_z) + bottom_t], dtype=float),
        "size": np.array([sx, sy], dtype=float),
        "table_top_z": float(table_top_z),
    }


def rotation_z(theta):
    c = math.cos(theta)
    s = math.sin(theta)
    return np.array(
        [
            [c, -s, 0.0],
            [s, c, 0.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=float,
    )


def world_to_robot_base(xyz_world, robot_xyz, robot_yaw_deg):
    R_world_from_base = rotation_z(math.radians(robot_yaw_deg))
    return R_world_from_base.T @ (np.asarray(xyz_world, dtype=float) - np.asarray(robot_xyz, dtype=float))


def robot_base_pose_to_world(pos_base, R_base, robot_xyz, robot_yaw_deg):
    R_world_from_base = rotation_z(math.radians(robot_yaw_deg))
    pos_world = np.asarray(robot_xyz, dtype=float) + R_world_from_base @ np.asarray(pos_base, dtype=float)
    R_world = R_world_from_base @ np.asarray(R_base, dtype=float)
    return pos_world, R_world


def palm_pose_to_object_world(palm_pos_world, palm_R_world, object_palm_offset):
    return np.asarray(palm_pos_world, dtype=float) + np.asarray(palm_R_world, dtype=float) @ np.asarray(
        object_palm_offset, dtype=float
    )


def draw_pose_marker(stage, UsdGeom, Gf, root_path, pos, R, center_color, axis_length=0.12):
    UsdGeom.Xform.Define(stage, root_path)
    pos = np.asarray(pos, dtype=float)
    R = np.asarray(R, dtype=float)

    sphere = UsdGeom.Sphere.Define(stage, f"{root_path}/origin")
    sphere.CreateRadiusAttr(0.018)
    sphere.AddTranslateOp().Set(Gf.Vec3d(float(pos[0]), float(pos[1]), float(pos[2])))
    sphere.CreateDisplayColorAttr([Gf.Vec3f(*center_color)])

    for name, color, axis in (
        ("x", (1.0, 0.0, 0.0), R[:, 0]),
        ("y", (0.0, 1.0, 0.0), R[:, 1]),
        ("z", (0.0, 0.2, 1.0), R[:, 2]),
    ):
        end = pos + axis_length * axis
        curve = UsdGeom.BasisCurves.Define(stage, f"{root_path}/{name}_axis")
        curve.CreateTypeAttr("linear")
        curve.CreateCurveVertexCountsAttr([2])
        curve.CreatePointsAttr(
            [
                Gf.Vec3f(float(pos[0]), float(pos[1]), float(pos[2])),
                Gf.Vec3f(float(end[0]), float(end[1]), float(end[2])),
            ]
        )
        curve.CreateWidthsAttr([0.01, 0.01])
        curve.CreateDisplayColorAttr([Gf.Vec3f(*color)])


def solve_right_arm_to_cube(
    urdf_path,
    dof_names,
    q_seed,
    cube_world,
    robot_xyz,
    robot_yaw_deg,
    palm_tcp_offset,
    palm_world_nudge,
):
    baseline_dir = find_challenge_repo_root()
    if str(baseline_dir) not in sys.path:
        sys.path.append(str(baseline_dir))

    import pinocchio as pin
    from Ubtech_sim.source.DualArmIK import DualArmIK

    ik = DualArmIK(str(urdf_path))
    ik.sync_joint_positions(dof_names, q_seed)
    ik.save_initial_q()
    ik.set_neutral_config(
        [q_seed[dof_names.index(n)] for n in DualArmIK.LEFT_ARM_JOINTS],
        [q_seed[dof_names.index(n)] for n in DualArmIK.RIGHT_ARM_JOINTS],
    )

    cube_base = world_to_robot_base(cube_world, robot_xyz, robot_yaw_deg)
    palm_nudge_base = world_to_robot_base(
        np.asarray(robot_xyz, dtype=float) + np.asarray(palm_world_nudge, dtype=float),
        robot_xyz,
        robot_yaw_deg,
    )
    init_ee = ik.get_ee_pose("right")
    palm_frame = next(
        (
            name
            for name in ("R_palm_link", "hand3_v1_right_R_palm_link", "hand3_v1_right")
            if ik.model.existFrame(name)
        ),
        None,
    )
    if palm_frame is None:
        raise RuntimeError("Could not find right palm frame for IK target")
    palm_frame_id = ik.model.getFrameId(palm_frame)
    palm_se3 = ik.data.oMf[palm_frame_id].copy()
    ee_to_palm = init_ee.inverse() * palm_se3

    x_grasp = np.array([1.0, 0.0, 0.0], dtype=float)
    z_grasp = np.array([0.0, 1.0, 0.0], dtype=float)
    y_grasp = np.cross(z_grasp, x_grasp)
    y_grasp /= np.linalg.norm(y_grasp)
    desired_palm_R = np.column_stack([x_grasp, y_grasp, z_grasp])
    desired_palm_pos = cube_base - desired_palm_R @ np.asarray(palm_tcp_offset, dtype=float) + palm_nudge_base

    target_ee_R = desired_palm_R @ ee_to_palm.rotation.T
    target_ee_pos = desired_palm_pos - target_ee_R @ ee_to_palm.translation
    target_xyzrpy = np.concatenate([target_ee_pos, pin.rpy.matrixToRpy(target_ee_R)])

    result = ik.solve_dual_arm(
        right_target_xyzrpy=target_xyzrpy,
        isaac_joint_names=dof_names,
        isaac_joint_positions=q_seed,
        max_iter=400,
        pos_tol=0.006,
        rot_tol=0.08,
        rot_weight=0.35,
        null_weight=0.02,
        dq_max=0.5,
    )
    print(f"[INFO] IK cube base position: {cube_base.tolist()}")
    print(f"[INFO] IK palm world nudge: {np.asarray(palm_world_nudge, dtype=float).tolist()}")
    print(
        "[INFO] Palm local axes in world: "
        f"+X={robot_base_pose_to_world(np.zeros(3), desired_palm_R, robot_xyz, robot_yaw_deg)[1][:, 0].tolist()}, "
        f"+Y={robot_base_pose_to_world(np.zeros(3), desired_palm_R, robot_xyz, robot_yaw_deg)[1][:, 1].tolist()}, "
        f"+Z={robot_base_pose_to_world(np.zeros(3), desired_palm_R, robot_xyz, robot_yaw_deg)[1][:, 2].tolist()}"
    )
    print(f"[INFO] IK palm frame: {palm_frame}, success={result.get('right_success')}")
    if not result.get("right_success"):
        print("[WARN] IK did not fully converge; using best returned right-arm pose")

    q = np.array(q_seed, dtype=float).copy()
    for name, value in zip(result["right_joint_names"], result["right_joint_positions"]):
        if name in dof_names:
            q[dof_names.index(name)] = float(value)

    ik.sync_joint_positions(dof_names, q)
    ik.get_ee_pose("right")
    actual_palm_se3 = ik.data.oMf[palm_frame_id].copy()

    target_pos_world, target_R_world = robot_base_pose_to_world(
        desired_palm_pos, desired_palm_R, robot_xyz, robot_yaw_deg
    )
    actual_pos_world, actual_R_world = robot_base_pose_to_world(
        actual_palm_se3.translation, actual_palm_se3.rotation, robot_xyz, robot_yaw_deg
    )
    return q, {
        "target_pos_world": target_pos_world,
        "target_R_world": target_R_world,
        "actual_pos_world": actual_pos_world,
        "actual_R_world": actual_R_world,
    }


def main():
    args = parse_args()
    baseline_dir = ensure_repo_python_paths()
    record_enabled = bool(args.record_dataset_root)
    replay_enabled = bool(args.replay_dataset_root)
    if record_enabled and replay_enabled:
        raise ValueError("Use either --record-dataset-root or --replay-dataset-root, not both")
    input_urdf = Path(args.urdf).expanduser().resolve()
    if not input_urdf.exists():
        raise FileNotFoundError(input_urdf)

    if args.no_preprocess_hand_collisions:
        urdf_for_import = input_urdf
    else:
        urdf_for_import = simplify_hand_collisions(input_urdf, show_debug=args.show_hand_colliders)
        print(f"[INFO] Wrote simplified-hand-collision URDF: {urdf_for_import}")

    sim_app, _urdf, World, DynamicCuboid, FixedCuboid, ArticulationWrapper, ArticulationAction, omni, Gf, UsdGeom = import_isaac_modules(args.headless)

    world = World(stage_units_in_meters=1.0)
    world.scene.add_default_ground_plane()
    stage = omni.usd.get_context().get_stage()

    cfg = make_import_config(_urdf)
    prim_path = import_robot(urdf_for_import, cfg, omni)
    print(f"[INFO] Imported robot at prim: {prim_path}")

    # Put pelvis/base_link around 0.86 m above world, so feet are approximately on the ground.
    robot_xyz = (0.75, -0.2, args.robot_z)
    robot_yaw_deg = 90.0
    set_prim_pose(stage, UsdGeom, Gf, prim_path, robot_xyz, (0.0, 0.0, robot_yaw_deg))
    camera_paths = {}
    if args.enable_robot_cameras:
        camera_paths = create_robot_cameras(stage, UsdGeom, Gf, prim_path)
        for camera_name, camera_path in camera_paths.items():
            print(f"[INFO] Robot camera {camera_name}: {camera_path}")
    if args.show_hand_colliders and not args.no_preprocess_hand_collisions:
        debug_count = draw_usd_hand_collider_debug_boxes(
            stage,
            UsdGeom,
            Gf,
            prim_path,
            args.debug_collider_visual_scale,
        )
        print(
            "[INFO] Drew USD hand collider debug boxes: "
            f"{debug_count} boxes, visual scale={args.debug_collider_visual_scale}"
        )

    # Table uses the old Part_Sorting task layout.
    table_center = np.array([0.75, 0.3, 1.02])
    table_size = np.array([1.2, 0.65, 0.04])
    table_top_z = float(table_center[2] + table_size[2] * 0.5)
    if args.randomize_cube:
        rng = np.random.default_rng(args.random_seed)
        task_cube_center = np.asarray(args.cube_center, dtype=float).copy()
        task_cube_center[0] = rng.uniform(float(args.cube_x_range[0]), float(args.cube_x_range[1]))
        task_cube_center[1] = rng.uniform(float(args.cube_y_range[0]), float(args.cube_y_range[1]))
    else:
        task_cube_center = np.asarray(args.cube_center, dtype=float)
    fixed_arm_reference_center = task_cube_center.copy()

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
        container_info = add_pick_place_container(
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
            position=task_cube_center,
            scale=np.array([0.035, 0.035, 0.13]),
            mass=0.05,
        )
    )

    if args.save_stage:
        omni.usd.get_context().save_as_stage(str(Path(args.save_stage).expanduser().resolve()))
        print(f"[INFO] Saved stage to: {args.save_stage}")

    # Wrap articulation after robot exists in the stage.
    try:
        robot = world.scene.add(ArticulationWrapper(prim_path=prim_path, name="walker_s2"))
    except TypeError:
        robot = world.scene.add(ArticulationWrapper(prim_path=prim_path))

    world.reset()
    try:
        robot.initialize()
    except Exception:
        pass

    dof_names = get_dof_names(robot)
    print("[INFO] DOFs:")
    for i, n in enumerate(dof_names):
        print(f"  {i:02d}: {n}")

    name_to_i = {n: i for i, n in enumerate(dof_names)}
    q_home = get_joint_positions(robot, len(dof_names))
    if len(q_home) != len(dof_names):
        q_home = np.zeros(len(dof_names), dtype=float)

    def set_named(q, values):
        q = np.array(q, dtype=float).copy()
        missing = []
        for n, v in values.items():
            if n in name_to_i:
                q[name_to_i[n]] = float(v)
            else:
                missing.append(n)
        if missing:
            print(f"[WARN] Missing joints, skipped: {missing}")
        return q

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
    right_hand_close = {
        "hand3_v1_right_R_thumb_cmp_joint": 0.95,
        "hand3_v1_right_R_thumb_mpp_joint": 1.03,
        "hand3_v1_right_R_thumb_ip_joint": 1.02,
        "hand3_v1_right_R_index_mpp_joint": 1.20,
        "hand3_v1_right_R_index_ip_joint": 1.30,
        "hand3_v1_right_R_middle_mpp_joint": 1.20,
        "hand3_v1_right_R_middle_ip_joint": 1.30,
        "hand3_v1_right_R_ring_mpp_joint": 1.20,
        "hand3_v1_right_R_ring_ip_joint": 1.30,
        "hand3_v1_right_R_little_mpp_joint": 1.15,
        "hand3_v1_right_R_little_ip_joint": 1.25,
    }

    left_hand_open = {
        name.replace("hand3_v1_right_R_", "hand3_v1_left_L_"): (
            -value if "thumb_cmp_joint" in name else value
        )
        for name, value in right_hand_open.items()
    }
    left_hand_close = {
        name.replace("hand3_v1_right_R_", "hand3_v1_left_L_"): (
            -value if "thumb_cmp_joint" in name else value
        )
        for name, value in right_hand_close.items()
    }

    q_ready = set_named(q_home, ready_arm_pose)
    q_ready_open = set_named(set_named(q_ready, left_hand_open), right_hand_open)
    palm_normal_world = rotation_z(math.radians(robot_yaw_deg)) @ np.array([0.0, 1.0, 0.0])
    grasp_world_nudge = (
        np.asarray(args.palm_world_nudge, dtype=float)
        - palm_normal_world * args.grasp_clearance
    )
    q_grasp_open, palm_debug = solve_right_arm_to_cube(
        urdf_for_import,
        dof_names,
        q_ready_open,
        fixed_arm_reference_center,
        robot_xyz,
        robot_yaw_deg,
        args.palm_tcp_offset,
        grasp_world_nudge,
    )
    pregrasp_world_nudge = (
        grasp_world_nudge
        - palm_debug["target_R_world"][:, 2] * args.pregrasp_distance
    )
    q_pregrasp_open, pregrasp_debug = solve_right_arm_to_cube(
        urdf_for_import,
        dof_names,
        q_ready_open,
        fixed_arm_reference_center,
        robot_xyz,
        robot_yaw_deg,
        args.palm_tcp_offset,
        pregrasp_world_nudge,
    )
    q_lift_open, _ = solve_right_arm_to_cube(
        urdf_for_import,
        dof_names,
        q_grasp_open,
        fixed_arm_reference_center,
        robot_xyz,
        robot_yaw_deg,
        args.palm_tcp_offset,
        grasp_world_nudge + np.array([0.0, 0.0, args.lift_height], dtype=float),
    )
    draw_pose_marker(
        stage,
        UsdGeom,
        Gf,
        "/World/PoseDebug/target_palm",
        palm_debug["target_pos_world"],
        palm_debug["target_R_world"],
        (1.0, 0.9, 0.0),
        axis_length=0.14,
    )
    draw_pose_marker(
        stage,
        UsdGeom,
        Gf,
        "/World/PoseDebug/actual_palm",
        palm_debug["actual_pos_world"],
        palm_debug["actual_R_world"],
        (1.0, 0.0, 1.0),
        axis_length=0.10,
    )
    right_hand_joint_names = list(right_hand_open.keys())
    right_hand_indices = np.array([name_to_i[n] for n in right_hand_joint_names], dtype=int)
    right_hand_open_pos = np.array([right_hand_open[n] for n in right_hand_joint_names], dtype=float)
    right_hand_close_pos = np.array([right_hand_close[n] for n in right_hand_joint_names], dtype=float)
    left_hand_joint_names = list(left_hand_open.keys())
    left_hand_indices = np.array([name_to_i[n] for n in left_hand_joint_names], dtype=int)
    left_hand_open_pos = np.array([left_hand_open[n] for n in left_hand_joint_names], dtype=float)
    left_hand_close_pos = np.array([left_hand_close[n] for n in left_hand_joint_names], dtype=float)
    hand_control = {
        "left": (left_hand_indices, left_hand_open_pos, left_hand_close_pos),
        "right": (right_hand_indices, right_hand_open_pos, right_hand_close_pos),
    }
    q_grasp_closed = q_grasp_open.copy()
    q_grasp_closed[right_hand_indices] = right_hand_close_pos
    q_lift_closed = q_lift_open.copy()
    q_lift_closed[right_hand_indices] = right_hand_close_pos

    def full_body_with_right_hand(hand_pos):
        q = q_grasp_open.copy()
        q[right_hand_indices] = np.array(hand_pos, dtype=float)
        return q

    def apply_full_body_with_hand(hand_pos):
        apply_full_body(full_body_with_right_hand(hand_pos))

    def apply_full_body(q):
        robot.apply_action(ArticulationAction(joint_positions=np.array(q, dtype=float)))

    # Spawn/reset at the ready pose. The grasp pose is reached only after the G key trigger.
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
    set_joint_positions(robot, q_ready_open)

    if args.object_palm_offset is not None:
        object_center = palm_pose_to_object_world(
            palm_debug["actual_pos_world"],
            palm_debug["actual_R_world"],
            args.object_palm_offset,
        )
    else:
        object_center = task_cube_center.copy()
    set_object_position(cube, stage, UsdGeom, Gf, "/World/grasp_cube", object_center)

    camera_capture_viewports = {}
    camera_capture_state = {"frames": {}, "pending": set(), "captures": [], "logged": False}
    camera_sensors = {}
    camera_viewer = None
    if args.enable_robot_cameras and camera_paths and not args.headless and args.camera_view_windows:
        camera_viewer = HeadCameraCvViewer(scale=args.camera_viewer_scale)
        if camera_viewer.started:
            print("[INFO] Opened draggable head-camera viewer: walker_s2_cameras")
            camera_sensors = create_head_camera_sensors(camera_paths, world=world)
            fallback_paths = {
                name: path for name, path in camera_paths.items() if name not in camera_sensors
            }
            if fallback_paths:
                camera_capture_viewports = create_camera_capture_viewports(fallback_paths)
                print(
                    "[INFO] Created fallback camera capture viewports: "
                    f"{len(camera_capture_viewports)}/{len(fallback_paths)}"
                )

    def update_camera_viewer():
        if camera_viewer is None:
            return
        update_camera_viewer.frame_count += 1
        if update_camera_viewer.frame_count < 2 or update_camera_viewer.frame_count % 2 != 0:
            return
        for name, sensor in camera_sensors.items():
            try:
                frame = sensor.get_rgb()
                if frame is not None:
                    camera_capture_state["frames"][name] = np.asarray(frame)
            except Exception as exc:
                if name not in update_camera_viewer.failed_sensors:
                    update_camera_viewer.failed_sensors.add(name)
                    print(f"[WARN] Could not read direct camera sensor {name}: {exc}")
        for name, widget in camera_capture_viewports.items():
            try:
                schedule_viewport_capture(name, widget.viewport_api, camera_capture_state)
            except Exception as exc:
                print(f"[WARN] Could not request camera viewport capture {name}: {exc}")
        if len(camera_capture_state["captures"]) > 20:
            camera_capture_state["captures"] = camera_capture_state["captures"][-20:]
        frames = dict(camera_capture_state["frames"])
        if frames and not update_camera_viewer.logged_frame_shapes:
            shapes = {
                name: None if frame is None else tuple(np.asarray(frame).shape)
                for name, frame in frames.items()
            }
            stats = {}
            for name, frame in frames.items():
                arr = np.asarray(frame)
                if arr.size:
                    stats[name] = {
                        "min": float(np.nanmin(arr)),
                        "max": float(np.nanmax(arr)),
                        "mean": float(np.nanmean(arr)),
                    }
            print(f"[INFO] Head camera frame shapes: {shapes}")
            print(f"[INFO] Head camera frame stats: {stats}")
            update_camera_viewer.logged_frame_shapes = True
        camera_viewer.update(frames)
    update_camera_viewer.frame_count = 0
    update_camera_viewer.logged_frame_shapes = False
    update_camera_viewer.failed_sensors = set()

    def close_camera_resources():
        for sensor in camera_sensors.values():
            sensor.destroy()
        if camera_viewer is not None:
            camera_viewer.close()

    def step_world(render=True):
        world.step(render=render)
        update_camera_viewer()

    recorder = None
    record_step_counter = 0
    recording_active = args.record_start == "launch"
    if record_enabled:
        from walker_s2_grasp_sim import WalkerS2LeRobotRecorder

        try:
            recorder = WalkerS2LeRobotRecorder(
                repo_id=args.record_repo_id,
                root=args.record_dataset_root,
                fps=args.record_fps,
                dof_names=dof_names,
                image_shape=(HEAD_CAMERA_HEIGHT, HEAD_CAMERA_WIDTH, 3),
                task=args.record_task,
            )
        except ModuleNotFoundError as exc:
            missing = exc.name or str(exc)
            raise RuntimeError(
                "Recording needs the LeRobot dataset dependencies inside Isaac Sim's Python. "
                f"Missing module: {missing}. Install them with Isaac's python.sh before running "
                "record_walker_grasp.sh."
            ) from exc
        print(f"[INFO] Recording LeRobot episode to: {recorder.root}")

    def maybe_record(action_q):
        nonlocal record_step_counter
        if recorder is None or not recording_active:
            return
        record_step_counter += 1
        if record_step_counter % max(1, int(args.record_step_stride)) != 0:
            return
        observation_q = get_joint_positions(robot, len(dof_names))
        recorder.add_frame(
            observation_state=observation_q,
            action=action_q,
            camera_frames=dict(camera_capture_state["frames"]),
        )

    def finish_recording():
        if recorder is not None:
            recorder.save()

    def start_recording(reason):
        nonlocal recording_active, record_step_counter
        if recorder is None or recording_active:
            return
        recording_active = True
        record_step_counter = 0
        print(f"[INFO] Started recording: {reason}")

    print(f"[INFO] Fixed arm reference center: {fixed_arm_reference_center.tolist()}")
    print(f"[INFO] Placed cube at {object_center.tolist()}")
    if args.randomize_cube:
        print(
            "[INFO] Randomized cube XY from "
            f"x={tuple(args.cube_x_range)}, y={tuple(args.cube_y_range)}, seed={args.random_seed}"
        )
    if container_info is not None:
        print(
            "[INFO] Pick-place container center/top: "
            f"{container_info['center'].tolist()}, size_xy={container_info['size'].tolist()}"
        )
    print(f"[INFO] Grasp palm-normal clearance: {args.grasp_clearance:.4f} m")
    if args.object_palm_offset is not None:
        print(f"[INFO] Object palm offset: {np.asarray(args.object_palm_offset, dtype=float).tolist()}")
    print("[INFO] Pose markers: yellow origin = target palm, magenta origin = IK actual palm; RGB axes show pose orientation")
    print(f"[INFO] Target palm world: {palm_debug['target_pos_world'].tolist()}")
    print(f"[INFO] Actual palm world: {palm_debug['actual_pos_world'].tolist()}")
    print(f"[INFO] Pregrasp palm world: {pregrasp_debug['actual_pos_world'].tolist()}")
    print(
        "[INFO] Palm target/actual position error: "
        f"{float(np.linalg.norm(palm_debug['target_pos_world'] - palm_debug['actual_pos_world'])):.4f} m"
    )

    render = not args.headless
    print("[INFO] Settling the ready pose and object before the grasp trigger.")
    for _ in range(180):
        apply_full_body(q_ready_open)
        step_world(render=render)

    if replay_enabled:
        from walker_s2_grasp_sim import WalkerS2LeRobotReplay

        replay = WalkerS2LeRobotReplay(
            repo_id=args.replay_repo_id,
            root=args.replay_dataset_root,
        )
        print(
            "[INFO] Replaying LeRobot episode "
            f"{args.replay_episode} from: {Path(args.replay_dataset_root).expanduser()}"
        )
        for q_replay in replay.iter_actions(args.replay_episode):
            if len(q_replay) != len(dof_names):
                raise ValueError(
                    f"Replay action has {len(q_replay)} DOFs, current robot has {len(dof_names)}"
                )
            apply_full_body(q_replay)
            step_world(render=render)
        close_camera_resources()
        sim_app.close()
        return

    q_teleop = q_ready_open.copy()
    if render:
        from walker_s2_grasp_sim import WalkerS2CartesianController, WalkerS2GraspKeyboard

        cartesian_controller = WalkerS2CartesianController(
            urdf_for_import,
            dof_names,
            q_ready_open,
        )
        teleop = WalkerS2GraspKeyboard()
        teleop.connect()
        print("[TELEOP] W/S: X  A/D: Y  R/F: Z")
        print("[TELEOP] Y/U: roll  V/B: pitch  N/M: yaw")
        print("[TELEOP] O: switch arm  0: bimanual  K/L: open/close hand")
        print("[TELEOP] +/-: speed  H: home  G: automatic grasp and lift  Q: quit")

        def run_manual_teleop(q_start, allow_grasp):
            q_command = np.asarray(q_start, dtype=float).copy()
            grasp_requested = False
            quit_requested = False
            while sim_app.is_running() and not grasp_requested and not quit_requested:
                command = teleop.sample()
                grasp_requested = command.assisted_grasp and allow_grasp
                quit_requested = command.quit
                if command.assisted_grasp and not allow_grasp:
                    print("[TELEOP] The object is already lifted; press H to return home or Q to quit")

                if command.go_home:
                    q_home_start = q_command.copy()
                    for i in range(120):
                        a = (i + 1) / 120.0
                        s = a * a * (3.0 - 2.0 * a)
                        q_command = (1.0 - s) * q_home_start + s * q_ready_open
                        apply_full_body(q_command)
                        step_world(render=True)
                        maybe_record(q_command)
                    cartesian_controller.reset(q_command)
                    print("[TELEOP] Returned to ready pose")
                    continue

                q_command, ik_status = cartesian_controller.step(q_command, command.arm_deltas)
                if ik_status and not all(ik_status.values()):
                    now = time.monotonic()
                    if now - run_manual_teleop.last_ik_warning_time > 1.0:
                        run_manual_teleop.last_ik_warning_time = now
                        print(f"[TELEOP] IK did not fully converge: {ik_status}")
                if abs(command.hand_delta) > 0.0:
                    for side in command.target_sides:
                        indices, open_pos, close_pos = hand_control[side]
                        q_command[indices] = np.clip(
                            q_command[indices] + command.hand_delta * (close_pos - open_pos),
                            np.minimum(open_pos, close_pos),
                            np.maximum(open_pos, close_pos),
                        )

                apply_full_body(q_command)
                step_world(render=True)
                maybe_record(q_command)
            return q_command, grasp_requested, quit_requested
        run_manual_teleop.last_ik_warning_time = 0.0

        q_teleop, grasp_requested, quit_requested = run_manual_teleop(
            q_teleop,
            allow_grasp=True,
        )

        if quit_requested or not grasp_requested:
            teleop.close()
            finish_recording()
            close_camera_resources()
            sim_app.close()
            return
    else:
        print("[INFO] Headless mode: starting the automatic grasp immediately.")

    start_recording("automatic grasp motion")

    print(f"[INFO] Moving from ready pose to pregrasp ({args.pregrasp_distance:.3f} m clearance).")
    for i in range(args.pregrasp_steps):
        a = (i + 1) / float(args.pregrasp_steps)
        s = a * a * (3.0 - 2.0 * a)
        q = (1.0 - s) * q_teleop + s * q_pregrasp_open
        apply_full_body(q)
        step_world(render=render)
        maybe_record(q)

    print("[INFO] Approaching from pregrasp to grasp pose.")
    for i in range(args.approach_steps):
        a = (i + 1) / float(args.approach_steps)
        s = a * a * (3.0 - 2.0 * a)
        q = (1.0 - s) * q_pregrasp_open + s * q_grasp_open
        apply_full_body(q)
        step_world(render=render)
        maybe_record(q)

    print("[INFO] Closing hand.")
    for i in range(240):
        a = (i + 1) / 240.0
        s = a * a * (3.0 - 2.0 * a)
        hand_q = (1.0 - s) * right_hand_open_pos + s * right_hand_close_pos
        q = full_body_with_right_hand(hand_q)
        apply_full_body(q)
        step_world(render=render)
        maybe_record(q)

    print(f"[INFO] Lifting closed grasp by {args.lift_height:.3f} m.")
    for i in range(args.lift_steps):
        a = (i + 1) / float(args.lift_steps)
        s = a * a * (3.0 - 2.0 * a)
        q = (1.0 - s) * q_grasp_closed + s * q_lift_closed
        apply_full_body(q)
        step_world(render=render)
        maybe_record(q)

    if render:
        q_teleop = q_lift_closed.copy()
        cartesian_controller.reset(q_teleop)
        print("[TELEOP] Grasp complete. Manual keyboard control remains active; press Q to quit.")
        run_manual_teleop(q_teleop, allow_grasp=False)
        teleop.close()
    else:
        for _ in range(args.duration_after):
            apply_full_body(q_lift_closed)
            step_world(render=False)
            maybe_record(q_lift_closed)

    print("[INFO] Done. Tune --object-palm-offset if the cube is not in the palm contact area.")
    finish_recording()
    close_camera_resources()
    sim_app.close()


if __name__ == "__main__":
    main()
