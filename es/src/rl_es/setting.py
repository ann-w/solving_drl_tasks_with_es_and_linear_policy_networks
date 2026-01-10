from dataclasses import dataclass

import numpy as np
import gymnasium as gym

from .utils import identity, argmax, softmax, uint8tofloat


@dataclass
class EnvSetting:
    name: str
    max_train_episodes: int
    max_train_timesteps: int = 1e7
    reward_shaping: callable = identity
    action_size: int = None
    state_size: int = None
    max_episode_steps: int = None
    reward_threshold: float = None
    last_activation: callable = identity
    is_discrete: bool = True
    env_kwargs: dict = None
    obs_mapper: callable = identity
    wrapper: object = None

    def __post_init__(self):
        if self.env_kwargs is None:
            self.env_kwargs = dict()

        env = gym.make(self.name, **self.env_kwargs)
        self.state_size = self.state_size or np.prod(env.observation_space.shape)
        self.is_discrete = isinstance(env.action_space, gym.spaces.discrete.Discrete)
        self.max_episode_steps = self.max_episode_steps or env.spec.max_episode_steps
        self.reward_threshold = env.spec.reward_threshold
        if self.is_discrete:
            self.action_size = env.action_space.n
            self.last_activation = argmax
        else:
            self.action_size = env.action_space.shape[0]

        if self.max_episode_steps is None:
            self.max_episode_steps = int(
                env.spec.kwargs["max_num_frames_per_episode"]
                / env.spec.kwargs["frameskip"]
            )

    def make(self, **kwargs):
        env = gym.make(self.name, **{**self.env_kwargs, **kwargs})
        if self.wrapper is not None:
            env = self.wrapper(env)
        return env

    @property
    def n(self):
        return self.state_size * self.action_size


ATARI_SETTINGS = dict(
    env_kwargs={
        "obs_type": "ram",
        "frameskip": 4,
        "repeat_action_probability": 0.0,
    },
    obs_mapper=uint8tofloat,
    max_train_episodes=20_000,
    max_train_timesteps=2e7,
)


ENVIRONMENTS = {
    "CartPole-v1": EnvSetting("CartPole-v1", max_train_episodes=1000, max_train_timesteps=5e5),
    "Acrobot-v1": EnvSetting("Acrobot-v1", max_train_episodes=5_000, max_train_timesteps=5e5),
    "Pendulum-v1": EnvSetting(
        "Pendulum-v1", max_train_episodes=5_000, max_train_timesteps=5e5, last_activation=lambda x: 2 * np.tanh(x)
    ),
    "MountainCar-v0": EnvSetting("MountainCar-v0", max_train_episodes=5_000, max_train_timesteps=5e5),
    "LunarLander-v2": EnvSetting("LunarLander-v2", max_train_episodes=10_000, max_train_timesteps=5e5),
    "BipedalWalker-v3": EnvSetting(
        "BipedalWalker-v3", max_train_episodes=20_000, max_train_timesteps=2e6, reward_shaping=lambda x: np.clip(x, -1, 1)
    ),
    "Swimmer-v4": EnvSetting("Swimmer-v4", max_train_episodes=2_000, max_train_timesteps=5e5, last_activation=np.tanh),
    "Reacher-v4": EnvSetting("Reacher-v4", max_train_episodes=20_000, max_train_timesteps=1e6, last_activation=np.tanh),
    "InvertedPendulum-v4": EnvSetting(
        "InvertedPendulum-v4", max_train_episodes=5_000, max_train_timesteps=5e5, last_activation=lambda x: 3 * np.tanh(x)
    ),
    "Hopper-v4": EnvSetting(
        "Hopper-v4", max_train_episodes=20_000, max_train_timesteps=1e6, reward_shaping=lambda x: x - 1, last_activation=np.tanh
    ),
    "HalfCheetah-v4": EnvSetting("HalfCheetah-v4", max_train_episodes=10_000, max_train_timesteps=3e6, last_activation=np.tanh),
    "Walker2d-v4": EnvSetting(
        "Walker2d-v4", max_train_episodes=50_000, max_train_timesteps=2e6, reward_shaping=lambda x: x - 1, last_activation=np.tanh
    ),
    "Ant-v4": EnvSetting(
        "Ant-v4",
        max_train_episodes=50_000,
        max_train_timesteps=1e7,
        reward_shaping=lambda x: x - 1,
        last_activation=np.tanh,
    ),
    "Humanoid-v4": EnvSetting(
        "Humanoid-v4",
        max_train_episodes=500_000,
        max_train_timesteps=1e7,
        reward_shaping=lambda x: x - 5,
        last_activation=lambda x: 0.4 * np.tanh(x),
    ),
    "HumanoidStandup-v4": EnvSetting(
        "HumanoidStandup-v4",
        max_train_episodes=500_000,
        max_train_timesteps=1e7,
        last_activation=lambda x: 0.4 * np.tanh(x),
    ),
    "Assault-v5": EnvSetting("ALE/Assault-v5", **ATARI_SETTINGS),
    "Atlantis-v5": EnvSetting("ALE/Atlantis-v5", **ATARI_SETTINGS),
    "BeamRider-v5": EnvSetting("ALE/BeamRider-v5", **ATARI_SETTINGS),
    "Boxing-v5": EnvSetting("ALE/Boxing-v5", **ATARI_SETTINGS),
    "CrazyClimber-v5": EnvSetting("ALE/CrazyClimber-v5", **ATARI_SETTINGS),
    "Enduro-v5": EnvSetting("ALE/Enduro-v5", **ATARI_SETTINGS),
    "Pong-v5": EnvSetting("ALE/Pong-v5", **ATARI_SETTINGS),
    "Qbert-v5": EnvSetting("ALE/Qbert-v5", **ATARI_SETTINGS),
    "SpaceInvaders-v5": EnvSetting("ALE/SpaceInvaders-v5", **ATARI_SETTINGS),
    "Seaquest-v5": EnvSetting("ALE/Seaquest-v5", **ATARI_SETTINGS),
}
