import os
import io
import time
from itertools import chain
from dataclasses import dataclass
from contextlib import redirect_stdout

import numpy as np
# Patch for NumPy 2.0 compatibility
if not hasattr(np, "float_"):
    np.float_ = np.float64

import gymnasium as gym
from gymnasium.utils.save_video import save_video

from skimage.measure import block_reduce

from .setting import EnvSetting

from .network import Network


class Normalizer:
    def __init__(self, nb_inputs):
        self.mean = np.zeros(nb_inputs)
        self.var = np.ones(nb_inputs)
        self.std = np.ones(nb_inputs)

    def observe(self, _):
        pass

    def __call__(self, x):
        return (x - self.mean) / self.std


class UpdatingNormalizer(Normalizer):
    def __init__(self, nb_inputs):
        super().__init__(nb_inputs)
        self.k = nb_inputs
        self.s = np.zeros(nb_inputs)

    def observe(self, X):
        for x in X:
            self.k += 1
            delta = x - self.mean.copy()
            self.mean += delta / self.k
            self.s += delta * (x - self.mean)

        self.var = self.s / (self.k - 1)
        self.std = np.sqrt(self.var)
        self.std[self.std < 1e-7] = np.inf


@dataclass
class Objective:
    setting: EnvSetting
    n_episodes: int = 5
    n_timesteps: int = 100
    n_eval_timesteps: int = 100
    n_hidden: int = 8
    n_layers: int = 3
    net: Network = None
    n: int = None
    parallel: bool = True
    n_test_episodes: int = 10
    normalized: bool = False
    bias: bool = False
    eval_total_timesteps: bool = True
    store_video: bool = True
    aggregator: callable = np.mean
    n_train_timesteps: int = 0
    n_train_episodes: int = 0
    n_test_evals: int = 0
    n_evals: int = 0
    data_folder: str = None
    seed_train_envs: int = None
    penalize_inactivity: bool = False
    break_timesteps: bool = True
    max_parallel: int = 128

    def __post_init__(self):
        if self.normalized:
            self.normalizer = UpdatingNormalizer(self.setting.state_size)
        else:
            self.normalizer = Normalizer(self.setting.state_size)

        self.net = Network(
            self.setting.state_size,
            self.setting.action_size,
            self.n_hidden,
            self.n_layers,
            self.setting.last_activation,
            self.bias,
        )
        self.n = self.net.n_weights
        self.nets = []

    def open(self):
        self.train_writer = open(
            os.path.join(self.data_folder, "train_evals.csv"), "a+"
        )
        self.test_writer = open(os.path.join(self.data_folder, "test_evals.csv"), "a+")
        # header = ", ".join([f"w{i}" for i in range(self.n)])
        header = f"evals, fitness\n"  # , {header}\n"
        self.train_writer.write(header)
        self.test_writer.write(header)

    def __call__(self, x):
        if self.parallel:
            # x shape is (n_weights, population_size), split along population axis
            f = np.array(
                list(
                    chain.from_iterable(
                        [
                            self.eval_parallel(split)
                            for split in np.array_split(
                                x, np.ceil(x.shape[1] / self.max_parallel), axis=1
                            )
                            if split.shape[1] > 0  # Skip empty splits
                        ]
                    )
                )
            )

        else:
            f = np.array([self.eval_sequential(xi) for xi in x.T])

        for y, xi in zip(f, x.T):
            self.n_evals += 1
            if hasattr(self, "train_writer"):
                self.train_writer.write(f"{self.n_evals}, {y}\n")
        return f

    def should_stop(self):
        if self.break_timesteps:
            return self.n_train_timesteps >= self.setting.max_train_timesteps
        return self.n_evals >= self.setting.max_train_episodes

    def reset_envs(self, envs):
        seeds = None
        if self.seed_train_envs is not None:
            seeds = [self.seed_train_envs * 7 * i for i in range(1, 1 + envs.num_envs)]

        for _ in range(5):
            try:
                observations, *_ = envs.reset(seed=seeds)
                break
            except:
                time.sleep(1)
        return observations

    def eval_sequential(self, x, train: bool = True, shaped: bool = True):
        envs = gym.vector.AsyncVectorEnv(
            [lambda: self.setting.make() for _ in range(self.n_episodes)]
        )
        observations = self.reset_envs(envs)

        self.net.set_weights(x)

        n_timesteps = self.n_timesteps
        if not train:
            n_timesteps = self.n_eval_timesteps

        data_over_time = np.zeros((n_timesteps, 2, self.n_episodes))
        for t in range(n_timesteps):
            observations = self.setting.obs_mapper(observations)
            actions = self.net(self.normalizer(observations))
            self.normalizer.observe(observations)
            observations, rewards, dones, trunc, *_ = envs.step(actions)

            if train or shaped:
                rewards = self.setting.reward_shaping(rewards)

            data_over_time[t] = np.vstack([rewards, np.logical_or(dones, trunc)])
            if not self.eval_total_timesteps and np.logical_or(dones, trunc):
                break

        returns = []
        for i in range(self.n_episodes):
            ret, n_eps, n_timesteps = self.calculate_returns(data_over_time[:, :, i])
            if train:
                self.n_train_timesteps += n_timesteps
                self.n_train_episodes += n_eps
            returns.extend(ret)

        y = -self.aggregator(returns)
        return y

    def eval_parallel(self, x, train: bool = True, shaped: bool = True):
        n = x.shape[1]
        if n != len(self.nets):
            self.nets = [
                Network(
                    self.setting.state_size,
                    self.setting.action_size,
                    self.n_hidden,
                    self.n_layers,
                    self.setting.last_activation,
                    self.bias,
                )
                for _ in range(n)
            ]

        last_error = None
        for _ in range(5):
            try:
                self.envs = gym.vector.AsyncVectorEnv(
                    [lambda: self.setting.make() for _ in range(self.n_episodes * n)]
                )
                break
            except Exception as e:
                last_error = e
                time.sleep(1)
        else:
            raise RuntimeError(f"Failed to create environments after multiple retries: {last_error}")

        for net, w in zip(self.nets, x.T):
            net.set_weights(w)

        observations = self.reset_envs(self.envs)

        n_total_episodes = action_shape = self.n_episodes * n

        actions = np.ones(action_shape, dtype=int)
        if not self.setting.is_discrete:
            action_shape = (action_shape, self.setting.action_size)
            actions = np.ones(action_shape, dtype=float)

        n_timesteps = self.n_timesteps
        if not train:
            n_timesteps = self.n_eval_timesteps

        data_over_time = np.zeros((n_timesteps, 2, n_total_episodes))
        actions_over_time = []
        for t in range(n_timesteps):
            observations = self.setting.obs_mapper(observations)
            for i, net in enumerate(self.nets):
                idx = i * self.n_episodes
                obs = observations[idx : idx + self.n_episodes, :]
                actions[idx : idx + self.n_episodes] = net(self.normalizer(obs))

                if train:
                    self.normalizer.observe(obs)

            observations, rewards, dones, trunc, info = self.envs.step(actions)
            actions_over_time.append(actions)

            if train or shaped:
                rewards = self.setting.reward_shaping(rewards)
            data_over_time[t] = np.vstack([rewards, np.logical_or(dones, trunc)])

            first_ep_done = data_over_time[:, 1, :].sum(axis=0) >= 1
            first_ep_all_done = first_ep_done.all()

            if not self.eval_total_timesteps and first_ep_all_done:
                break
        aggregated_returns = np.empty(n)
        for k, j in enumerate(range(0, n_total_episodes, self.n_episodes)):
            returns = []
            for i in range(self.n_episodes):
                ret, n_eps, n_timesteps = self.calculate_returns(
                    data_over_time[:, :, j + i]
                )
                if train:
                    self.n_train_timesteps += n_timesteps
                    self.n_train_episodes += n_eps
                returns.extend(ret)
            aggregated_returns[k] = self.aggregator(returns)

        y = -aggregated_returns

        if train and self.penalize_inactivity:
            if not self.setting.is_discrete:
                raise NotImplementedError

            actions_over_time = np.array(actions_over_time)
            inactivity_penalty = (actions_over_time.std(axis=0) > 0.1) * 5
            y += inactivity_penalty
        return y

    def calculate_returns(self, Y):
        _, idx = np.unique(np.cumsum(Y[:, 1]) - Y[:, 1], return_index=True)
        episodes = np.split(Y[:, 0], idx)[1:]
        if len(episodes) > 1:
            episodes = episodes[:-1]

        returns_ = [x.sum() for x in episodes]
        n_timesteps = len(Y)
        if not self.eval_total_timesteps:
            returns_ = returns_[:1]
            n_timesteps = len(episodes[0])

        # TODO: we can remove incomplete episodes from the last optionally
        return returns_, len(returns_), n_timesteps

    def test_policy(self, x, name: str = None, save: bool = True):
        x = x.reshape(-1, 1)
        X = np.tile(x, (1, self.n_test_episodes))
        returns = self.eval_parallel(X, False, False)
        if save:
            os.makedirs(f"{self.data_folder}/policies", exist_ok=True)
            loc = f"{self.data_folder}/policies/{name}"

            np.save(loc, x)
            np.save(f"{loc}-norm-std", self.normalizer.std)
            np.save(f"{loc}-norm-mean", self.normalizer.mean)

            for ret in returns:
                self.n_test_evals += 1
                self.test_writer.write(f"{self.n_test_evals}, {ret}\n")

            self.play_check(loc, "rgb_array_list", name)

        return np.mean(returns), np.median(returns), np.std(returns)

    def load_network(self, loc: str):
        net = Network(
            self.setting.state_size,
            self.setting.action_size,
            self.n_hidden,
            self.n_layers,
            self.setting.last_activation,
            self.bias,
        )
        net.set_weights(np.load(f"{loc}.npy"))
        normalizer = Normalizer(self.setting.state_size)
        if os.path.isfile(f"{loc}-norm-std.npy"):
            normalizer.std = np.load(f"{loc}-norm-std.npy")
            normalizer.mean = np.load(f"{loc}-norm-mean.npy")
        return net, normalizer

    def play_check(
        self,
        location,
        render_mode=None,
        name=None,
        n_reps: int = 1,
        render_fps: int = 50,
    ):
        if not self.store_video and render_mode == "rgb_array_list":
            return

        net, normalizer = self.load_network(location)
        returns = []
        try:
            for episode_index in range(n_reps):
                env = self.setting.make(render_mode=render_mode)
                if render_mode == "rgb_array_list":
                    env.metadata["render_fps"] = max(
                        env.metadata.get("render_fps") or render_fps, render_fps
                    )

                observation, *_ = env.reset()

                if render_mode == "human":
                    env.render()
                done = False
                ret = 0
                step_index = 0
                while not done:
                    observation = self.setting.obs_mapper(observation)
                    obs = normalizer(observation.reshape(1, -1))
                    action, *_ = net(obs)
                    observation, reward, terminated, truncated, *_ = env.step(action)
                    done = terminated or truncated

                    ret += reward
                    if render_mode == "human":
                        print(
                            f"step {step_index}, return {ret: .3f} {' ' * 25}", end="\r"
                        )
                    step_index += 1
                returns.append(ret)
                if render_mode == "human":
                    print()
                if render_mode == "rgb_array_list" and episode_index == 0:
                    if self.store_video:
                        os.makedirs(f"{self.data_folder}/videos", exist_ok=True)
                        with redirect_stdout(io.StringIO()):
                            save_video(
                                env.render(),
                                f"{self.data_folder}/videos",
                                fps=env.metadata.get("render_fps"),
                                step_starting_index=0,
                                episode_index=0,
                                name_prefix=name,
                            )
                    render_mode = None

        except KeyboardInterrupt:
            pass
        finally:
            env.close()
        return returns
