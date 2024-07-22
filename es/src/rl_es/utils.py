import numpy as np
import gymnasium as gym
from skimage.measure import block_reduce


def softmax(x):
    x_exp = np.exp(x - np.max(x))
    return x_exp / x_exp.sum()


def argmax(x):
    return np.argmax(x, axis=1)


def identity(x):
    return x


def clip(lb, ub):
    def inner(x):
        return np.clip(x, lb, ub)

    return inner


def uint8tofloat(obs):
    return ((obs.astype(float) / 255) * 2) - 1
