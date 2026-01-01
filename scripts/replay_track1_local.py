"""Minimal Track1 trajectory replay using the project's wrappers."""
from pathlib import Path
import json
import h5py
import numpy as np
import torch
from omegaconf import OmegaConf

from scripts.training.env_utils import make_env


def main(traj_path: str):
    traj_path = Path(traj_path)
    json_path = traj_path.with_suffix(".json")
    data = json.loads(json_path.read_text())

    env_cfg_dict = data["env_info"]["env_kwargs"]["cfg"]
    cfg = OmegaConf.create(env_cfg_dict)

    # Ensure we capture video/trajectory on replay too
    cfg.capture_video = True
    cfg.recording.save_trajectory = True
    cfg.recording.save_video = True
    cfg.recording.save_env_state = True

    video_dir = traj_path.parent

    env = make_env(cfg, num_envs=1, for_eval=True, video_dir=str(video_dir))
    episode = data["episodes"][0]
    seed = np.array([episode.get("episode_seed", 0)], dtype=np.int64)
    env.reset(seed=seed)

    with h5py.File(traj_path, "r") as f:
        actions = np.array(f["traj_0"]["actions"])

    device = env.unwrapped.device if hasattr(env, "unwrapped") else torch.device("cuda" if torch.cuda.is_available() else "cpu")

    for act in actions:
        act_tensor = torch.tensor(act, device=device, dtype=torch.float32)[None, :]
        obs, reward, term, trunc, info = env.step(act_tensor)
        if term.any() or trunc.any():
            break

    env.close()
    print(f"Replay finished. Video/trajectory written to {video_dir}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("traj_path", help="Path to recorded .h5 trajectory")
    args = parser.parse_args()
    main(args.traj_path)
