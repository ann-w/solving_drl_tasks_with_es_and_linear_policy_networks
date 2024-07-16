import pickle
import os

import numpy as np
import datetime
import torch as th
from sb3_contrib.ars import ARS
from stable_baselines3.common.callbacks import BaseCallback
from typing import Optional, Dict
from stable_baselines3.common.vec_env import (
    DummyVecEnv,
    VecNormalize,
)
from sb3_contrib.common.vec_env import AsyncEval
import argparse
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.utils import set_random_seed
import gymnasium as gym

ALIVE_BONUS_OFFSET_MAPPING = {
    "Hopper-v4": -1,
    "Walker2d-v4": -1,
    "Ant-v4": -1,
    "Humanoid-v4": -5,
}


def get_name(env):
    if isinstance(env, str):
        return env
    return env.venv.envs[0].unwrapped.spec.id


class CustomARS(ARS):
    def __init__(
        self,
        policy,
        env,
        *args,
        base_save_path="./logs/",
        training=True,
        **kwargs,
    ):
        env_name = get_name(env)
        alive_bonus_offset = 0
        if (
            training and ALIVE_BONUS_OFFSET_MAPPING.get(env_name)
        ):
            alive_bonus_offset = ALIVE_BONUS_OFFSET_MAPPING.get(env_name)
        super().__init__(
            policy, env, *args, alive_bonus_offset=alive_bonus_offset, **kwargs
        )

        # Directory for saving model weights
        algorithm_name = type(self).__name__
        datetime_stamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        self.full_save_path = os.path.join(
            base_save_path, algorithm_name, env_name, datetime_stamp
        )
        os.makedirs(self.full_save_path, exist_ok=True)

        if self.verbose:
            print(f"Model and logs will be saved in: {self.full_save_path}")

    def evaluate_candidates(
        self,
        candidate_weights: th.Tensor,
        callback: BaseCallback,
        async_eval: Optional[AsyncEval],
    ) -> th.Tensor:
        candidate_returns = super().evaluate_candidates(
            candidate_weights, callback, async_eval
        )
        best_idx = th.argmax(candidate_returns)
        self.best_weights = candidate_weights[best_idx].detach().cpu().clone()

        mean_weights = th.mean(candidate_weights, dim=0)
        self.mean_weights = mean_weights.detach().cpu().clone()

        return candidate_returns

    def _do_one_update(
        self, callback: BaseCallback, async_eval: Optional[AsyncEval]
    ) -> None:
        super()._do_one_update(callback, async_eval)
        self._save_weights(self.policy.state_dict(), "best")
        # self._save_weights(self.policy.state_dict(), "mean")

        rms_file = os.path.join(self.full_save_path, f"rms_weights_{self.num_timesteps}.pkl")
        with open(rms_file, "wb") as f:
            pickle.dump(self.env.obs_rms, f)
            if self.verbose:
                print(
                    f"Saved rms at timestep {self.num_timesteps} to {rms_file}"
                )

    def _save_weights(self, state_dict: Dict[str, th.Tensor], descriptor: str):
        # Construct the full file path
        file_path = os.path.join(
            self.full_save_path, f"{descriptor}_weights_{self.num_timesteps}.pth"
        )
        # Save the state dictionary to file
        th.save(state_dict, file_path)
        if self.verbose:
            print(
                f"Saved {descriptor} weights at timestep {self.num_timesteps} to {file_path}"
            )


class ClipRewardVecNormalize(VecNormalize):
    def __init__(self, *args, clip=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.clip = clip

    def normalize_reward(self, reward: np.ndarray) -> np.ndarray:
        if self.clip:
            return np.clip(reward, -self.clip, self.clip)
        return reward


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-n", "--n-timesteps", help="Number of timesteps", default=-1, type=int
    )
    parser.add_argument("--env", type=str, default="CartPole-v1", help="environment ID")
    parser.add_argument(
        "--n-eval-envs",
        help="Number of environments for evaluation in parallel",
        default=1,
        type=int,
    )
    parser.add_argument("--seed", help="Random generator seed", type=int, default=-1)
    parser.add_argument(
        "--n_delta",
        help="How many random perturbations of the policy at each update step",
        type=int,
        default=8,
    )
    parser.add_argument(
        "--n_top",
        help="How many of the top delta to use in each update step, default is n_delta",
        type=int,
    )
    parser.add_argument(
        "--learning_rate",
        help="float or schedule for the step size",
        type=float,
        default=0.02,
    )
    parser.add_argument(
        "--delta_std",
        help="Float or schedule for the exploration noise",
        type=float,
        default=0.05,
    )

    args = parser.parse_args()

    if args.seed < 0:
        # Seed but with a random one
        args.seed = np.random.randint(2**32 - 1, dtype="int64").item()

    set_random_seed(args.seed)

    def make_env():
        env = DummyVecEnv([lambda: gym.make(args.env)])
        env = ClipRewardVecNormalize(
            env,
            norm_obs=True,
            norm_reward=False,
            clip=int(args.env == "BipedalWalker-v3"),
        )
        return env

    model = CustomARS(
        "LinearPolicy",
        env=make_env(),
        verbose=1,
        n_delta=args.n_delta,
        n_top=args.n_top,
        learning_rate=args.learning_rate,
        delta_std=args.delta_std,
    )

    # Create env for asynchronous evaluation (run in different processes)
    async_eval = AsyncEval([make_env for _ in range(args.n_eval_envs)], model.policy)
    model.learn(total_timesteps=args.n_timesteps, async_eval=async_eval)
