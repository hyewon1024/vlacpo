import glob
import json
import os
import pickle
import re

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

from failure_prob.conf import Config

from .utils import Rollout, set_task_min_step, split_rollouts_by_seen_unseen, process_tensor_idx_rel

def compute_hand_crafted_metrics(
    df: pd.DataFrame,
) -> pd.DataFrame:
    metrics_raw = {}
    
    # Compute the token-level uncertainty metrics
    if 'action/token_0_prob' in df.columns:
        token_prob = df[[
            'action/token_0_prob', 'action/token_1_prob', 'action/token_2_prob', 'action/token_3_prob',
            'action/token_4_prob', 'action/token_5_prob', 'action/token_6_prob'
        ]].values # (n_step, n_token)
        token_entropy = df[[
            'action/token_0_entropy', 'action/token_1_entropy', 'action/token_2_entropy', 'action/token_3_entropy',
            'action/token_4_entropy', 'action/token_5_entropy', 'action/token_6_entropy'
        ]].values # (n_step, n_token)
        max_token_prob = (- np.log(token_prob)).max(axis=-1) # (n_step, )
        avg_token_prob = (- np.log(token_prob)).mean(axis=-1) # (n_step, )
        max_token_entropy = token_entropy.max(axis=-1) # (n_step, )
        avg_token_entropy = token_entropy.mean(axis=-1) # (n_step, )
        metrics_raw.update({
            "max_token_prob": max_token_prob,
            "avg_token_prob": avg_token_prob,
            "max_token_entropy": max_token_entropy,
            "avg_token_entropy": avg_token_entropy,
        })
    
    # Extract the sample-level uncertianty metrics
    for k in [
        "total_var", "general_var", "pos_var", "rot_var", "gripper_var", "entropy_linkage.01", "entropy_linkage.05"
    ]:
        if f"action/{k}" in df.columns:
            values = df[f"action/{k}"].values # (n_step, )
            metrics_raw[k] = np.asarray(values) # (n_steps, )

    # Compute the cumulative version of the metrics (running mean)
    metrics = {}
    for k, v in metrics_raw.items():
        metrics[k] = np.asarray(v) # (n_steps, )
        metrics[f"{k}_rmean"] = np.cumsum(np.asarray(v)) / (np.arange(len(v)) + 1) # (n_steps, )

        # # In LIBERO, the rollouts terminates early when the task is successfully completed. 
        # # Cumsum metrics gives advantages for failure detection to cheat and thus not proper to use. 
        # metrics[f"{k}_csum"] = np.cumsum(np.asarray(v)) # (n_steps, )
        
    df_metrics = pd.DataFrame(metrics)
    
    return df_metrics


def extract_info_from_path(filename):
    # Define the regex pattern
    pattern = r"task(\d+)--ep(\d+)--succ(\d+)\.csv"
    
    # Match the pattern
    match = re.match(pattern, filename)
    
    if match:
        # Extract and convert values
        task_id = int(match.group(1))
        episode_id = int(match.group(2))
        success = bool(int(match.group(3)))  # Convert 1/0 to True/False
        return task_id, episode_id, success
    else:
        raise ValueError("Filename format is incorrect")
    

def load_rollouts(cfg: Config) -> list[Rollout]:
    all_pkl = glob.glob(os.path.join(cfg.dataset.data_path, "**", "*.pkl"), recursive=True)
    
    unique_envs = sorted(list(set([os.path.basename(os.path.dirname(p)) for p in all_pkl])))
    env_to_id = {name: i for i, name in enumerate(unique_envs)}
    
    print(f"Detected {len(unique_envs)} unique tasks: {unique_envs}")
    
    all_rollouts = []
    for pkl_path in tqdm(all_pkl, desc="Loading SafetyGym rollouts"):
        with open(pkl_path, "rb") as f:
            data = pickle.load(f)

        env_folder_name = os.path.basename(os.path.dirname(pkl_path))
        task_id = env_to_id[env_folder_name]

        features = torch.tensor(data["features"], dtype=torch.float32)
        action_vectors = torch.tensor(data["actions"], dtype=torch.float32)

        episode_success = int(data["episode_success"])
        episode_idx = data["episode_idx"]

        r = Rollout(
            hidden_states=features,
            task_suite_name="safetygym",
            task_id=task_id, 
            task_description=env_folder_name,
            episode_idx=episode_idx,
            episode_success=episode_success,
            mp4_path=None,
            logs=None,
            action_vectors=action_vectors,
        )
        all_rollouts.append(r)

    print(f"Loaded {len(all_rollouts)} SafetyGym rollouts")
    return all_rollouts 

def split_rollouts(cfg: Config, all_rollouts: list[Rollout]) -> dict[str, list[Rollout]]:
    task_ids = list(set([r.task_id for r in all_rollouts]))
    print(len(task_ids))
    n_unseen = round(cfg.dataset.unseen_task_ratio * len(task_ids))
    n_seen = len(task_ids) - n_unseen
    
    np.random.shuffle(task_ids)
    seen_task_ids = task_ids[:n_seen]
    unseen_task_ids = task_ids[n_seen:]
    
    rollouts_by_split_name = split_rollouts_by_seen_unseen(
        cfg, all_rollouts, seen_task_ids, unseen_task_ids
    )

    return rollouts_by_split_name