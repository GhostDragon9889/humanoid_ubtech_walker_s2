from __future__ import annotations

from pathlib import Path

import numpy as np


class WalkerS2LeRobotRecorder:
    """Small LeRobot v3 episode writer for the standalone Walker grasp sim."""

    def __init__(
        self,
        repo_id: str,
        root: str | Path,
        fps: int,
        dof_names: list[str],
        image_shape: tuple[int, int, int],
        task: str,
        image_keys: tuple[str, ...] = ("head_left", "head_right"),
    ):
        from src.lerobot.datasets.lerobot_dataset import LeRobotDataset

        self.repo_id = repo_id
        self.root = Path(root).expanduser()
        self.fps = int(fps)
        self.dof_names = list(dof_names)
        self.image_shape = tuple(int(v) for v in image_shape)
        self.task = task
        self.image_keys = tuple(image_keys)
        self.frame_count = 0
        self._warned_missing_images: set[str] = set()

        features = {
            "observation.state": {
                "dtype": "float32",
                "shape": (len(self.dof_names),),
                "names": self.dof_names,
            },
            "action": {
                "dtype": "float32",
                "shape": (len(self.dof_names),),
                "names": self.dof_names,
            },
        }
        for key in self.image_keys:
            features[f"observation.images.{key}"] = {
                "dtype": "image",
                "shape": self.image_shape,
                "names": ["height", "width", "channels"],
            }

        if (self.root / "meta" / "info.json").exists():
            self.dataset = LeRobotDataset(
                repo_id=self.repo_id,
                root=self.root,
            )
            if int(self.dataset.fps) != self.fps:
                raise ValueError(
                    f"Existing dataset FPS is {self.dataset.fps}, requested {self.fps}"
                )
            print(
                "[INFO] Appending to existing LeRobot dataset: "
                f"{self.root} (episodes={self.dataset.meta.total_episodes})"
            )
        else:
            if self.root.exists() and not any(self.root.iterdir()):
                self.root.rmdir()
            self.dataset = LeRobotDataset.create(
                repo_id=self.repo_id,
                fps=self.fps,
                root=self.root,
                robot_type="walker_s2_grasp_sim",
                features=features,
                use_videos=False,
                image_writer_threads=max(1, 2 * len(self.image_keys)),
            )
            print(f"[INFO] Created new LeRobot dataset: {self.dataset.root}")
        self.root = self.dataset.root

    def _image_or_blank(self, frames: dict[str, np.ndarray], key: str) -> np.ndarray:
        frame = frames.get(key)
        if frame is None:
            if key not in self._warned_missing_images:
                self._warned_missing_images.add(key)
                print(f"[WARN] Recording blank frames until camera image is available: {key}")
            return np.zeros(self.image_shape, dtype=np.uint8)

        image = np.asarray(frame)
        if image.ndim == 3 and image.shape[2] > 3:
            image = image[:, :, :3]
        if image.dtype != np.uint8:
            if image.size and float(np.nanmax(image)) <= 1.0:
                image = image * 255.0
            image = np.clip(image, 0, 255).astype(np.uint8)
        if tuple(image.shape) != self.image_shape:
            raise ValueError(
                f"Camera frame {key} has shape {image.shape}, expected {self.image_shape}"
            )
        return np.ascontiguousarray(image)

    def add_frame(self, observation_state, action, camera_frames: dict[str, np.ndarray]) -> None:
        frame = {
            "observation.state": np.asarray(observation_state, dtype=np.float32),
            "action": np.asarray(action, dtype=np.float32),
            "task": self.task,
        }
        for key in self.image_keys:
            frame[f"observation.images.{key}"] = self._image_or_blank(camera_frames, key)
        self.dataset.add_frame(frame)
        self.frame_count += 1

    def save(self) -> None:
        if self.frame_count <= 0:
            print("[WARN] No recorded frames; skipping dataset save.")
            self.dataset.finalize()
            return
        self.dataset.save_episode()
        self.dataset.finalize()
        print(f"[INFO] Saved LeRobot episode with {self.frame_count} frames to: {self.root}")


class WalkerS2LeRobotReplay:
    """Read action vectors from a local LeRobot dataset episode."""

    def __init__(self, repo_id: str, root: str | Path):
        from src.lerobot.datasets.lerobot_dataset import LeRobotDataset

        self.dataset = LeRobotDataset(repo_id=repo_id, root=Path(root).expanduser())

    def iter_actions(self, episode_index: int = 0):
        episodes = self.dataset.meta.episodes
        if episodes is None or len(episodes) <= int(episode_index):
            raise IndexError(f"Episode {episode_index} does not exist in dataset")
        episode = episodes[int(episode_index)]
        start = int(episode["dataset_from_index"])
        end = int(episode["dataset_to_index"])
        for index in range(start, end):
            item = self.dataset[index]
            action = item["action"]
            if hasattr(action, "detach"):
                action = action.detach().cpu().numpy()
            yield np.asarray(action, dtype=float)
