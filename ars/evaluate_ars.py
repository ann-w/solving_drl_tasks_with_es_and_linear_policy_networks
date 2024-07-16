import os
import csv
import pickle
import torch
import gymnasium as gym
import numpy as np
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import (
    DummyVecEnv,
    VecNormalize,
)

import multiprocessing
from custom_ars import CustomARS
        

def evaluate_single_weight(args):
    folder_path, filename, method, n_episodes = args
    env_name = folder_path.split('/')[-2]
    timesteps = int(filename.rsplit("_", 1)[1].split(".")[0])
    
    rms_file = os.path.join(folder_path, f"rms_weights_{timesteps}.pkl")
    with open(rms_file, "rb") as f:
        rms = pickle.load(f) 
    try:
        env = DummyVecEnv([lambda: Monitor(gym.make(env_name))])
    except:
        env_name = f"ALE/{env_name}"
        env = DummyVecEnv([lambda: Monitor(gym.make(env_name))])
        
    env = VecNormalize(
        env,
        norm_obs=True,
        norm_reward=False,
    )
    env.obs_rms = rms
    
    model = CustomARS("LinearPolicy", env, training=False, verbose=0)
    file_path = os.path.join(folder_path, filename)
    timesteps = filename.split('_')[-1].split('.')[0]
    weights = torch.load(file_path)
    model.policy.load_state_dict(weights)
    episode_rewards, episode_lengths = evaluate_policy(model.policy, env, n_eval_episodes=n_episodes, return_episode_rewards=True)
    reward_mean = np.mean(episode_rewards)
    reward_median = np.median(episode_rewards)
    reward_std = np.std(episode_rewards)
    return [timesteps, reward_mean, reward_median, reward_std]

def evaluate_weights(folder_path, method, n_episodes, n_processes):
    filenames = sorted([f for f in os.listdir(folder_path) if f.startswith(method)], key=lambda x: int(x.split('_')[-1].split('.')[0]))
    with multiprocessing.Pool(n_processes) as pool:
        results = pool.imap(evaluate_single_weight, [(folder_path, f, method, n_episodes) for f in filenames])
        results = list(results)
    results_file = os.path.join(folder_path, 'rewards.csv')
    with open(results_file, 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['timesteps', 'test_reward_mean', 'test_reward_median', 'test_reward_std'])
        writer.writerows(results)
    print(f"Results saved to {results_file}")

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('folder_path', type=str, help='Path to the folder containing the weights')
    parser.add_argument('--method', default='best', type=str, help='Method to evaluate', choices=['best', 'mean'])
    parser.add_argument('--n_episodes', default=10, type=int, help='Number of episodes to evaluate')
    parser.add_argument('--n_processes', default=multiprocessing.cpu_count(), type=int, help='Number of processes to use')
    args = parser.parse_args()

    evaluate_weights(args.folder_path, args.method, args.n_episodes, args.n_processes)