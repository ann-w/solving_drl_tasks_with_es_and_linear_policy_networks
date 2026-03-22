import numpy as np
from dataclasses import dataclass, field
from .utils import softmax, argmax, identity, clip


@dataclass
class Layer:
    n: int
    m: int
    activation: callable = np.tanh
    bias: bool = False
    w: np.ndarray = field(default=None, repr=None)
    b: np.ndarray = field(default=None, repr=None)

    def __post_init__(self):
        self.w_size = self.n * self.m
        self.b_size = self.m * self.bias
        self.size = self.w_size + self.b_size

    def set_weights(self, w, idx):
        self.w = w[idx : idx + self.w_size].reshape(self.n, self.m)
        if self.bias:
            self.b = w[idx + self.w_size : idx + self.w_size + self.b_size].ravel()

    def __call__(self, x):
        output = np.matmul(x, self.w)
        if self.bias:
            output = output + self.b
        return self.activation(output)


@dataclass
class Network:
    input_size: int
    output_size: int
    hidden_size: int = 16
    n_layers: int = 1
    last_activation: callable = argmax
    bias: bool = True
    w: np.ndarray = None

    def __post_init__(self):
        if self.n_layers == 1:
            self.layers = [
                Layer(
                    self.input_size,
                    self.output_size,
                    self.last_activation,
                    self.bias,
                )
            ]
        else:
            self.layers = (
                [Layer(self.input_size, self.hidden_size, bias=self.bias)]
                + [
                    Layer(self.hidden_size, self.hidden_size, bias=self.bias)
                    for _ in range(self.n_layers - 2)
                ]
                + [
                    Layer(
                        self.hidden_size,
                        self.output_size,
                        self.last_activation,
                        self.bias,
                    )
                ]
            )

        self.n_weights = sum(layer.size for layer in self.layers)

        if self.w is None:
            self.w = np.zeros(self.n_weights)
        self.set_weight_views()

    def set_weight_views(self):
        idx = 0
        for layer in self.layers:
            layer.set_weights(self.w, idx)
            idx += layer.size

    def set_weights(self, w):
        self.w = w.copy()
        self.set_weight_views()

    def __call__(self, x):
        for layer in self.layers:
            x = layer(x)
        return x


@dataclass
class MinMaxNormalizer:
    lb: np.ndarray
    ub: np.ndarray

    def __post_init__(self):
        self.db = self.ub - self.lb

    def __call__(self, x):
        return (x - self.lb) / self.db
