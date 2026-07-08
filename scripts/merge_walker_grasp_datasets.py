#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import numpy as np


def parse_args():
    parser = argparse.ArgumentParser(description="Merge one-episode Walker S2 LeRobot datasets.")
    parser.add_argument(
        "--source-glob",
        default="recordings/walker_s2_grasp_20260708_*",
        help="Glob for source dataset roots",
    )
    parser.add_argument(
        "--output-root",
        default="recordings/walker_s2_grasp_train",
        help="New merged dataset root. Existing roots are not modified; a timestamp suffix is added if needed.",
    )
    parser.add_argument("--repo-id", default="walker_s2_grasp", help="LeRobot repo_id for source and target")
    parser.add_argument("--fps", type=int, default=None, help="Override target FPS; default reads the first source")
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


def main():
    args = parse_args()
    repo_root = Path(__file__).resolve().parents[1]
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))
    setup_local_hf_cache(repo_root)

    from src.lerobot.datasets.lerobot_dataset import LeRobotDataset

    source_roots = sorted(Path(".").glob(args.source_glob))
    source_roots = [p for p in source_roots if p.is_dir() and (p / "meta" / "info.json").exists()]
    if not source_roots:
        raise FileNotFoundError(f"No source datasets matched: {args.source_glob}")

    first = LeRobotDataset(args.repo_id, root=source_roots[0])
    source_features = dict(first.meta.features)
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
    fps = int(args.fps or first.meta.fps)
    output_root = make_unique_output_root(Path(args.output_root))

    merged = LeRobotDataset.create(
        repo_id=args.repo_id,
        fps=fps,
        root=output_root,
        robot_type=first.meta.robot_type,
        features=data_features,
        use_videos=False,
        image_writer_threads=4,
    )

    image_keys = [key for key in data_features if key.startswith("observation.images.")]
    total_frames = 0
    try:
        for episode_index, source_root in enumerate(source_roots):
            ds = LeRobotDataset(args.repo_id, root=source_root)
            print(f"[INFO] Merging episode {episode_index}: {source_root} ({len(ds)} frames)")
            for i in range(len(ds)):
                item = ds[i]
                frame = {
                    "observation.state": to_numpy(item["observation.state"]).astype(np.float32),
                    "action": to_numpy(item["action"]).astype(np.float32),
                    "task": item.get("task", "pick the block and place it in the tray"),
                }
                for key in image_keys:
                    frame[key] = to_image_hwc_uint8(item[key])
                merged.add_frame(frame)
                total_frames += 1
            merged.save_episode()
    finally:
        merged.finalize()

    print(f"[INFO] Wrote merged dataset: {merged.root}")
    print(f"[INFO] Episodes: {len(source_roots)}")
    print(f"[INFO] Frames: {total_frames}")


if __name__ == "__main__":
    main()
