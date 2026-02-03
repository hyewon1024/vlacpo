import gym
import safety_gym.envs
import numpy as np
import pickle
import os
from tqdm import trange

SAVE_ROOT = "./safetygym_rollouts"
os.makedirs(SAVE_ROOT, exist_ok=True)

ROBOTS = ["Point", "Car", "Doggo"]
TASKS = ["Push", "Goal", "Button"]
LEVELS = [2]

EPISODES_PER_ENV = 500

for robot in ROBOTS:
    for task in TASKS:
        for level in LEVELS:
            env_name = f"Safexp-{robot}{task}{level}-v0"
            print(f"\n===== Collecting {env_name} =====")

            env = gym.make(env_name)
            save_dir = os.path.join(SAVE_ROOT, env_name)
            os.makedirs(save_dir, exist_ok=True)

            for ep in trange(EPISODES_PER_ENV, desc=env_name):
                obs = env.reset()
                done = False

                features = []
                actions = []
                constraint_costs = []

                while not done:
                    action = env.action_space.sample()

                    next_obs, reward, done, info = env.step(action)

                    features.append(obs.copy())
                    actions.append(action.copy())

                    cost = info.get("cost", 0.0)
                    constraint_costs.append(cost)

                    obs = next_obs

                features = np.array(features)
                actions = np.array(actions)
                constraint_costs = np.array(constraint_costs)

                failure_labels = (constraint_costs.cumsum() > 0).astype(np.float32)
                episode_success = int(constraint_costs.sum() == 0)

                data = {
                    "features": features,
                    "actions": actions,
                    "failure_labels": failure_labels,
                    "episode_success": episode_success,
                    "episode_idx": ep,
                    "env_name": env_name,
                }

                save_path = os.path.join(save_dir, f"rollout_{ep:04d}.pkl")
                with open(save_path, "wb") as f:
                    pickle.dump(data, f)

            env.close()

print("\n All Safety Gym rollouts collected!")
