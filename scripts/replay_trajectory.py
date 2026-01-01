"""Convenience wrapper to replay ManiSkill trajectories with Track1 env registration."""
import scripts.envs.track1_env  # Registers Track1-v0 with gym/ManiSkill
from mani_skill.trajectory.replay_trajectory import main as ms_main, parse_args


if __name__ == "__main__":
    args = parse_args()
    ms_main(args)
