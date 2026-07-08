#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import numpy as np


def parse_args():
    parser = argparse.ArgumentParser(description="Trim leading idle frames from a Walker S2 LeRobot dataset.")
    parser.add_argument("--input-root", default="recordings/walker_s2_grasp_train")
    parser.add_argument("--output-root", default="recordings/walker_s2_grasp_train_trimmed")
    parser.add_argument("--repo-id", default="walker_s2_grasp")
    parser.add_argument(
        "--motion-threshold",
        type=float,
        default=0.02,
        help="First frame whose max action delta from episode frame 0 exceeds this value is considered motion.",
    )
    parser.add_argument(
        "--keep-before-motion",
        type=int,
        default=2,
        help="Keep this many frames before the detected motion frame.",
    )
    return parser.parse_args()


def setup_local_hf_cache(repo_root: Path):
    cache_root = repo_root / ".hf_cache"
    os.environ.setdefault("HF_HOME", str(cache_root))
    os.environ.setdefault("HF_DATASETS_CACHE", str(cache_root / "datasets"))
    os.environ.setdefault("HF_HUB_OFFLINE", "1")


def to_numpy(value):
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    return np.asarray(value)


def to_image_hwc_uint8(value):
    image = to_numpy(value)
    if image.ndim == 3 and image.shape[0] in (1, 3, 4):
        image = np.transpose(image, (1, 2, 0))
    if image.ndim == 3 and image.shape[2] > 3:
        image = image[:, :, :3]
    if image.dtype != np.uint8:
        if image.size and float(np.nanmax(image)) <= 1.0:
            image = image * 255.0
        image = np.clip(image, 0, 255).astype(np.uint8)
    return np.ascontiguousarray(image)


def make_unique_output_root(root: Path) -> Path:
    if not root.exists():
        return root
    from datetime import datetime

    suffix = datetime.now().strftime("%Y%m%d_%H%M%S")
    return root.parent / f"{root.name}_{suffix}"


def find_motion_start(dataset, start: int, end: int, threshold: float) -> int:
    first_action = dataset[start]["action"]
    for index in range(start, end):
        action = dataset[index]["action"]
        if float((action - first_action).abs().max()) > threshold:
            return index
    return start


def main():
    args = parse_args()
    repo_root = Path(__file__).resolve().parents[1]
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))
    setup_local_hf_cache(repo_root)

    from src.lerobot.datasets.lerobot_dataset import LeRobotDataset

    input_root = Path(args.input_root).expanduser()
    source = LeRobotDataset(args.repo_id, root=input_root)
    source_features = dict(source.meta.features)
    data_features = {
        key: value
        for key, value in source_features.items()
        if key
        not in {
            "timestamp",
            "frame_index",
            "episode_index",
            "index",
            "task_index",
        }
    }
    output_root = make_unique_output_root(Path(args.output_root).expanduser())
    trimmed = LeRobotDataset.create(
        repo_id=args.repo_id,
        fps=int(source.meta.fps),
        root=output_root,
        robot_type=source.meta.robot_type,
        features=data_features,
        use_videos=False,
        image_writer_threads=4,
    )
    image_keys = [key for key in data_features if key.startswith("observation.images.")]

    total_in = 0
    total_out = 0
    try:
        for episode_index, episode in enumerate(source.meta.episodes):
            start = int(episode["dataset_from_index"])
            end = int(episode["dataset_to_index"])
            motion_start = find_motion_start(source, start, end, float(args.motion_threshold))
            trim_start = max(start, motion_start - max(0, int(args.keep_before_motion)))
            print(
                "[INFO] Episode "
                f"{episode_index}: len={end - start}, motion_start={motion_start - start}, "
                f"trimmed={trim_start - start}"
            )
            for index in range(trim_start, end):
                item = source[index]
                frame = {
                    "observation.state": to_numpy(item["observation.state"]).astype(np.float32),
                    "action": to_numpy(item["action"]).astype(np.float32),
                    "task": item.get("task", "pick the block and place it in the tray"),
                }
                for key in image_keys:
                    frame[key] = to_image_hwc_uint8(item[key])
                trimmed.add_frame(frame)
                total_out += 1
            total_in += end - start
            trimmed.save_episode()
    finally:
        trimmed.finalize()

    print(f"[INFO] Wrote trimmed dataset: {trimmed.root}")
    print(f"[INFO] Input frames: {total_in}")
    print(f"[INFO] Output frames: {total_out}")


if __name__ == "__main__":
    main()
