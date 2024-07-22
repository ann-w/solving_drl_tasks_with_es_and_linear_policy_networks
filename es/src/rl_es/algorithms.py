import warnings
import time
import os
from typing import Any
from dataclasses import dataclass, field


import numpy as np
from scipy.stats import qmc

from .objective import Objective


SIGMA_MAX = 1e3


@dataclass
class Solution:
    y: float = float("inf")
    x: np.ndarray = field(default=None, repr=None)


@dataclass
class Logger:
    folder_name: str

    def __post_init__(self):
        if self.folder_name is not None:
            self.writer = open(os.path.join(self.folder_name, "stats.csv"), "w")
            self.columns = (
                "generation",
                "dt",
                "n_evals",
                "n_train_episodes",
                "n_train_timesteps",
                "best",
                "current",
                "sigma",
                "best_test",
                "current_test",
                "population_mean",
                "population_std",
                "best_median",
                "best_std",
                "current_median",
                "current_std",
            )
            self.writer.write(f'{",".join(self.columns)}\n')

    def write(self, x) -> Any:
        if self.folder_name is not None:
            assert len(x) == len(self.columns)
            self.writer.write(f'{",".join(map(str, x))}\n')
            self.writer.flush()

    def close(self):
        if self.folder_name is not None:
            self.writer.close()


class State:
    def __init__(
        self, name, data_folder, test_gen, lamb, revaluate_best_after: int = 5
    ):
        self.counter = 0
        self.best = Solution()
        self.mean = Solution()
        self.best_test = float("inf")
        self.mean_test = float("inf")
        self.tic = time.perf_counter()
        self.data_folder = data_folder
        self.logger = Logger(data_folder)
        self.test_gen: int = test_gen
        self.name = name
        self.lamb: int = lamb
        self.mean_test = None
        self.best_test = None
        self.best_median = None
        self.best_std = None
        self.mean_median = None
        self.mean_std = None
        self.time_since_best_update = 0
        self.revaluate_best_every = revaluate_best_after

    def update(
        self,
        problem: Objective,
        best_offspring: Solution,
        mean: Solution,
        sigma: float,
        f: np.ndarray,
    ) -> None:
        self.counter += 1
        self.time_since_best_update += 1

        if (
            self.revaluate_best_every is not None
            and self.revaluate_best_every < self.time_since_best_update
        ):
            self.time_since_best_update = 0
            best_old = self.best.y
            self.best.y = problem.eval_sequential(self.best.x, False, True)
            # got_test, got_mean, got_std = problem.test_policy(
            #     self.best.x,
            #     name=f"t-{self.counter}-best",
            #     save=False
            # )
            # print("reevaluating best, expected: ", -best_old, -self.best.y, -got_test, -got_mean, got_std)

        toc = time.perf_counter()
        dt = toc - self.tic
        self.tic = toc

        if best_offspring.y < self.best.y:
            self.best = best_offspring
            self.time_since_best_update = 0
        self.mean = mean

        print(
            f"{self.name}, counter: {self.counter}, dt: {dt:.3f} n_timesteps {problem.n_train_timesteps:.2e}, n_episodes: {problem.n_train_episodes} "
            f"best (train): {-self.best.y}, mean (train): {-mean.y}, sigma: {sigma} "
            f"best (test): {self.best_test}, mean (test): {self.mean_test}"
        )

        if self.counter % self.test_gen == 0:
            self.best_test, self.best_median, self.best_std = problem.test_policy(
                self.best.x, name=f"t-{self.counter}-best", save=True
            )
            self.best_test, self.best_median = -self.best_test, -self.best_median
            print("Test with best x (max):", self.best_test)
            self.mean_test, self.mean_median, self.mean_std = problem.test_policy(
                self.mean.x, name=f"t-{self.counter}-mean", save=True
            )
            self.mean_test, self.mean_median = -self.mean_test, -self.mean_median
            print("Test with mean x (max):", self.mean_test)

        self.logger.write(
            (
                self.counter,
                dt,
                problem.n_evals,
                problem.n_train_episodes,
                problem.n_train_timesteps,
                -self.best.y,
                -self.mean.y,
                sigma,
                self.best_test,
                self.mean_test,
                np.mean(-f),
                np.std(-f),
                self.best_median,
                self.best_std,
                self.mean_median,
                self.mean_std,
            )
        )


@dataclass
class Weights:
    mu: int
    lambda_: int
    n: int
    method: str = "log"

    def __post_init__(self):
        self.set_weights()
        self.normalize_weights()

    def set_weights(self):
        if self.method == "log":
            self.wi_raw = np.log(self.lambda_ / 2 + 0.5) - np.log(
                np.arange(1, self.mu + 1)
            )
        elif self.method == "linear":
            self.wi_raw = np.arange(1, self.mu + 1)[::-1]
        elif self.method == "equal":
            self.wi_raw = np.ones(self.mu)

    def normalize_weights(self):
        self.w = self.wi_raw / np.sum(self.wi_raw)
        self.w_all = np.r_[self.w, -self.w[::-1]]

    @property
    def mueff(self):
        return 1 / np.sum(np.power(self.w, 2))

    @property
    def c_s(self):
        return (self.mueff + 2) / (self.n + self.mueff + 5)

    @property
    def d_s(self):
        return 1 + self.c_s + 2 * max(0, np.sqrt((self.mueff - 1) / (self.n + 1)) - 1)

    @property
    def sqrt_s(self):
        return np.sqrt(self.c_s * (2 - self.c_s) * self.mueff)


def init_lambda(n, method="n/2"):
    """
    range:      2*mu < lambda < 2*n + 10
    default:    4 + floor(3 * ln(n))
    """

    if method == "default":
        return (4 + np.floor(3 * np.log(n))).astype(int)

    elif method == "n/2":
        return min(128, max(32, np.floor(n / 2).astype(int)))
    elif isinstance(method, int):
        return method
    else:
        raise ValueError()


@dataclass
class Initializer:
    n: int
    lb: float = -0.1
    ub: float = 0.1
    method: str = "lhs"
    fallback: str = "zero"
    n_evals: int = 0
    max_evals: int = 32 * 5
    max_observed: float = -np.inf
    min_observed: float = np.inf
    aggregate: bool = False

    def __post_init__(self):
        self.sampler = qmc.LatinHypercube(self.n)

    def static_init(self, method):
        if method == "zero":
            return np.zeros((self.n, 1))
        elif method == "uniform":
            return np.random.uniform(self.lb, self.ub, size=(self.n, 1))
        elif method == "gauss":
            return np.random.normal(size=(self.n, 1))
        raise ValueError()

    def get_x_prime(self, problem, samples_per_trial: int = 128) -> np.ndarray:
        if self.method != "lhs":
            return self.static_init(self.method)

        samples = None
        sample_values = np.array([])
        f = np.array([0])
        while self.n_evals < self.max_evals:
            X = qmc.scale(self.sampler.random(samples_per_trial), self.lb, self.ub).T
            f = problem(X)
            self.n_evals += samples_per_trial
            self.max_observed = max(self.max_observed, f.max())
            self.min_observed = min(self.min_observed, f.min())

            if f.std() > 0:
                idx = f != self.max_observed
                if samples is None:
                    samples = X[:, idx]
                else:
                    samples = np.c_[samples, X[:, idx]]
                sample_values = np.r_[sample_values, f[idx]]

        if not any(sample_values):
            warnings.warn(
                f"DOE did not find any variation after max_evals={self.max_evals}"
                f", using fallback {self.fallback} intialization."
            )
            return self.static_init(self.fallback)

        idx = np.argsort(sample_values)
        if self.aggregate:
            w = np.log(len(sample_values) + 0.5) - np.log(
                np.arange(1, len(sample_values) + 1)
            )
            w = w / w.sum()
            x_prime = np.sum(w * samples[:, idx], axis=1, keepdims=True)
        else:
            x_prime = samples[:, idx[0]].reshape(-1, 1)

        print("lhs:", problem.n_evals, self.min_observed, self.max_observed)
        return x_prime


@dataclass
class CSA:
    n: int

    lambda_: int = None
    mu: float = None
    sigma0: float = 0.5
    verbose: bool = True
    test_gen: int = 25
    initialization: str = "zero"
    data_folder: str = None
    uncertainty_handling: bool = False
    mirrored: bool = True
    revaluate_best_after: int = None

    def __post_init__(self):
        self.lambda_ = self.lambda_ or init_lambda(self.n)
        if self.lambda_ % 2 != 0:
            self.lambda_ += 1
        self.mu = self.mu or self.lambda_ // 2
        print(f"n: {self.n}, lambda: {self.lambda_}, mu: {self.mu}")

    def __call__(self, problem: Objective):
        weights = Weights(self.mu, self.lambda_, self.n)

        echi = np.sqrt(self.n) * (1 - (1 / self.n / 4) - (1 / self.n / self.n / 21))

        init = Initializer(self.n, method=self.initialization, max_evals=500)
        x_prime = init.get_x_prime(problem)

        sigma = self.sigma0
        s = np.ones((self.n, 1))

        state = State("CSA", self.data_folder, self.test_gen, self.lambda_)
        n_samples = self.lambda_ if not self.mirrored else self.lambda_ // 2
        try:
            while not problem.should_stop():
                Z = np.random.normal(size=(self.n, n_samples))
                if self.mirrored:
                    Z = np.hstack([Z, -Z])
                X = x_prime + (sigma * Z)
                f = problem(X)
                idx = np.argsort(f)

                mu_best = idx[: self.mu]
                idx_min = idx[0]

                z_prime = np.sum(weights.w * Z[:, mu_best], axis=1, keepdims=True)
                x_prime = x_prime + (sigma * z_prime)
                s = ((1 - weights.c_s) * s) + (weights.sqrt_s * z_prime)
                sigma = sigma * np.exp(
                    weights.c_s / weights.d_s * (np.linalg.norm(s) / echi - 1)
                )

                state.update(
                    problem,
                    Solution(f[idx_min], X[:, idx_min].copy()),
                    Solution((weights.w * f[mu_best]).sum(), x_prime.copy()),
                    np.mean(sigma),
                    f,
                )

        except KeyboardInterrupt:
            pass
        finally:
            state.logger.close()
        return state.best, state.mean


@dataclass
class CMAES:
    n: int

    data_folder: str = None
    test_gen: int = 25
    sigma0: float = 0.02
    lambda_: int = 16
    mu: int = None
    initialization: str = "zero"
    sep: bool = False
    # tpa better with ineffective axis
    tpa: bool = False
    active: bool = False

    def __post_init__(self):
        self.lambda_ = self.lambda_ or init_lambda(self.n)
        if self.sep:
            # self.tpa = True
            self.lambda_ = max(4, self.lambda_)

        if self.lambda_ % 2 != 0:
            self.lambda_ += 1
        self.mu = self.lambda_ // 2

        print(self.n, self.lambda_, self.mu, self.sigma0)

    def __call__(self, problem: Objective):
        init = Initializer(self.n, method=self.initialization, max_evals=500)
        m = init.get_x_prime(problem)

        w = np.log((self.lambda_ + 1) / 2) - np.log(np.arange(1, self.lambda_ + 1))
        w = w[: self.mu]
        mueff = w.sum() ** 2 / (w**2).sum()
        w = w / w.sum()
        w_active = np.r_[w, -1 * w[::-1]]

        # Learning rates
        n = self.n
        c1 = 2 / ((n + 1.3) ** 2 + mueff)

        cmu = 2 * (mueff - 2 + 1 / mueff) / ((n + 2) ** 2 + 2 * mueff / 2)
        if self.sep:
            cmu *= (n + 2) / 3
        cc = (4 + (mueff / n)) / (n + 4 + (2 * mueff / n))

        cs = (mueff + 2) / (n + mueff + 5)
        damps = 1.0 + (2.0 * max(0.0, np.sqrt((mueff - 1) / (n + 1)) - 1) + cs)
        chiN = n**0.5 * (1 - 1 / (4 * n) + 1 / (21 * n**2))

        # dynamic parameters
        dm = np.zeros((n, 1))
        pc = np.zeros((n, 1))
        ps = np.zeros((n, 1))
        B = np.eye(n)
        C = np.eye(n)
        D = np.ones((n, 1))
        invC = np.eye(n)

        state = State(
            f"{'sep-' if self.sep else ''}{'a' if self.active else ''}CMA-ES",
            self.data_folder,
            self.test_gen,
            self.lambda_,
        )
        sigma = self.sigma0

        if self.tpa:
            cs = 0.3
        s = 0
        hs = True
        z_exponent = 0.5
        damp = n**0.5

        try:
            while not problem.should_stop():
                active_tpa = self.tpa and state.counter != 0
                n_offspring = self.lambda_ - (2 * active_tpa)
                Z = np.random.normal(0, 1, (n, n_offspring))
                Y = np.dot(B, D * Z)
                if active_tpa:
                    Y = np.c_[-dm, dm, Y]

                X = m + (sigma * Y)
                f = np.array(problem(X))

                # select
                fidx = np.argsort(f)
                mu_best = fidx[: self.mu]

                # recombine
                m_old = m.copy()
                m = m_old + (1 * ((X[:, mu_best] - m_old) @ w).reshape(-1, 1))

                # adapt
                dm = (m - m_old) / sigma
                ps = (1 - cs) * ps + (np.sqrt(cs * (2 - cs) * mueff) * invC @ dm)
                hs = (
                    np.linalg.norm(ps)
                    / np.sqrt(
                        1 - np.power(1 - cs, 2 * (problem.n_evals / self.lambda_))
                    )
                ) < (1.4 + (2 / (n + 1))) * chiN

                if not self.tpa:
                    sigma *= np.exp((cs / damps) * ((np.linalg.norm(ps) / chiN) - 1))
                elif state.counter != 0:
                    z = (fidx[0] - fidx[1]) / (self.lambda_ - 1)
                    s = (1 - cs) * s + cs * np.sign(z) * pow(np.abs(z), z_exponent)
                    sigma *= np.exp(s / damp)

                dhs = (1 - hs) * cc * (2 - cc)
                pc = (1 - cc) * pc + (hs * np.sqrt(cc * (2 - cc) * mueff)) * dm

                old_C = (1 - (c1 * dhs) - c1 - (cmu * w.sum())) * C
                rank_one = c1 * pc * pc.T
                if self.active:
                    rank_mu = cmu * (w_active * Y[:, fidx] @ Y[:, fidx].T)
                else:
                    rank_mu = cmu * (w * Y[:, mu_best] @ Y[:, mu_best].T)

                C = old_C + rank_one + rank_mu

                if np.isinf(C).any() or np.isnan(C).any() or (not 1e-16 < sigma < 1e6):
                    sigma = self.sigma0
                    pc = np.zeros((n, 1))
                    ps = np.zeros((n, 1))
                    C = np.eye(n)
                    B = np.eye(n)
                    D = np.ones((n, 1))
                    invC = np.eye(n)
                else:
                    C = np.triu(C) + np.triu(C, 1).T
                    if not self.sep:
                        D, B = np.linalg.eigh(C)
                    else:
                        D = np.diag(C)

                D = np.sqrt(D).reshape(-1, 1)
                invC = np.dot(B, D**-1 * B.T)

                best_idx = mu_best[0]
                state.update(
                    problem,
                    Solution(f[best_idx], X[:, best_idx].copy()),
                    Solution(np.mean(f), m.copy()),
                    sigma,
                    f,
                )
        except KeyboardInterrupt:
            pass
        finally:
            state.logger.close()
        return state.best, state.mean
