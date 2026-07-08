#!/usr/bin/env python
from __future__ import annotations

import argparse
import os
from pathlib import Path
import sys

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

os.environ.setdefault("HF_HOME", str(REPO_ROOT / ".hf_cache"))
os.environ.setdefault("HF_DATASETS_CACHE", str(REPO_ROOT / ".hf_cache" / "datasets"))

from src.lerobot.configs.policies import PreTrainedConfig
from src.lerobot.datasets.lerobot_dataset import LeRobotDataset
from src.lerobot.policies.factory import make_policy, make_pre_post_processors


def parse_args():
    parser = argparse.ArgumentParser(
        description="Offline smoke-test a trained Walker S2 ACT checkpoint on one recorded frame."
    )
    parser.add_argument(
        "--dataset-root",
        default="recordings/walker_s2_grasp_train",
        help="Local LeRobot dataset root used for the test observation.",
    )
    parser.add_argument("--repo-id", default="walker_s2_grasp", help="LeRobot repo_id for dataset metadata.")
    parser.add_argument(
        "--policy-path",
        default="recordings/train/walker_s2_grasp_act_smoke/checkpoints/001000/pretrained_model",
        help="Path to the saved pretrained_model checkpoint directory.",
    )
    parser.add_argument("--frame-index", type=int, default=0, help="Dataset frame index to run inference on.")
    return parser.parse_args()


def main():
    args = parse_args()
    dataset_root = Path(args.dataset_root).expanduser()
    policy_path = Path(args.policy_path).expanduser()

    dataset = LeRobotDataset(repo_id=args.repo_id, root=dataset_root)
    if args.frame_index < 0 or args.frame_index >= len(dataset):
        raise IndexError(f"--frame-index must be in [0, {len(dataset) - 1}], got {args.frame_index}")

    policy_cfg = PreTrainedConfig.from_pretrained(policy_path)
    policy_cfg.pretrained_path = str(policy_path)
    policy = make_policy(cfg=policy_cfg, ds_meta=dataset.meta)
    policy.eval()

    preprocessor, postprocessor = make_pre_post_processors(
        policy_cfg=policy_cfg,
        pretrained_path=str(policy_path),
        preprocessor_overrides={
            "device_processor": {"device": str(policy.config.device)},
            "normalizer_processor": {"device": str(policy.config.device)},
        },
        postprocessor_overrides={
            "device_processor": {"device": "cpu"},
        },
    )

    sample = dataset[args.frame_index]
    observation = {
        "observation.state": sample["observation.state"],
        "observation.images.head_left": sample["observation.images.head_left"],
        "observation.images.head_right": sample["observation.images.head_right"],
    }

    with torch.inference_mode():
        batch = preprocessor(observation)
        action = policy.select_action(batch)
        action = postprocessor(action)

    action = action.detach().cpu().squeeze(0)
    recorded = sample["action"].detach().cpu()
    diff = action - recorded

    print(f"[INFO] Loaded dataset: {dataset_root}")
    print(f"[INFO] Loaded policy: {policy_path}")
    print(f"[INFO] Frame index: {args.frame_index}")
    print(f"[INFO] Predicted action shape: {tuple(action.shape)}")
    print(f"[INFO] Recorded action shape: {tuple(recorded.shape)}")
    print(f"[INFO] Predicted action min/max: {action.min().item():.4f} / {action.max().item():.4f}")
    print(f"[INFO] Recorded action min/max: {recorded.min().item():.4f} / {recorded.max().item():.4f}")
    print(f"[INFO] Mean absolute difference to recorded action: {diff.abs().mean().item():.4f}")
    print(f"[INFO] Max absolute difference to recorded action: {diff.abs().max().item():.4f}")


if __name__ == "__main__":
    main()
