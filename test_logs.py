from scripts.training.runner_utils import compute_reward_logs
import numpy as np

def test_reward_logs():
    # Simulate episode_metrics with the new prefixed keys
    metrics = {
        "reward/approach": [0.1, 0.2, 0.3],
        "reward/grasp": [1.0, 1.0, 1.0],
        "success": [0, 1, 0]
    }
    
    logs = compute_reward_logs(metrics)
    print("Generated Logs:")
    for k, v in logs.items():
        print(f"  {k}: {v}")

    # Check for double prefix
    if "reward/reward/approach" in logs:
        print("\n[DETECTED BUG]: Found double prefix 'reward/reward/xxx'")
    elif "reward/approach" in logs:
        print("\n[SUCCESS]: Keys are correctly formatted.")

if __name__ == "__main__":
    test_reward_logs()
