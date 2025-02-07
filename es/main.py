import os
import time
import argparse
import json
import numpy as np

from rl_es.algorithms import (
    CSA,
    CMAES,
    ARS,
    ARS_OPTIMAL_PARAMETERS,
)
from rl_es.objective import Objective
from rl_es.setting import ENVIRONMENTS

DATA = os.path.join(os.path.realpath(os.path.dirname(__file__)), "data")
STRATEGIES = (
    "csa",
    "cma-es",
    "sep-cma-es",
    "ars",
    "ars-v2",
)

def run_optimizer(args, obj):
    obj.open()
    if args.strategy == "csa":
        optimizer = CSA(
            obj.n,
            sigma0=args.sigma0,
            mu=args.mu,
            lambda_=args.lamb,
            data_folder=data_folder,
            initialization=args.initialization,
            uncertainty_handling=args.uncertainty_handled,
            test_gen=args.test_every_nth_iteration,
            mirrored=args.mirrored,
        )
    elif args.strategy.endswith("cma-es"):
        optimizer = CMAES(
            obj.n,
            data_folder=data_folder,
            test_gen=args.test_every_nth_iteration,
            sigma0=args.sigma0,
            lambda_=args.lamb,
            mu=args.mu,
            initialization=args.initialization,
            sep="sep" in args.strategy,
            active="active" in args.strategy
        )
    elif args.strategy == "ars" or args.strategy == "ars-v2":
        params = ARS_OPTIMAL_PARAMETERS.get(args.env_name)
        if args.ars_optimal and params:
            args.alpha = params.alpha
            args.sigma0 = params.sigma
            args.lamb = params.lambda0
            args.mu = params.mu

        optimizer = ARS(
            obj.n,
            sigma0=args.sigma0,
            alpha=args.alpha,
            data_folder=data_folder,
            test_gen=args.test_every_nth_iteration,
            mu=args.mu,
            lambda_=args.lamb,
            initialization=args.initialization,
        )
    else:
        raise ValueError(f"{args.strategy} is not implemented")

    args.sigma0 = optimizer.sigma0
    args.mu = int(optimizer.mu)
    args.lambda_ = int(optimizer.lambda_)
    args.n = int(obj.n)

    with open(os.path.join(data_folder, "settings.json"), "w+") as f:
        json.dump(vars(args), f)

    best, mean = optimizer(obj)
    np.save(f"{data_folder}/best.npy", best.x)
    np.save(f"{data_folder}/mean.npy", mean.x)
    best, mean = best.x, mean.x

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--seed", default=42, help="Set seed for reproducibility", type=int
    )
    parser.add_argument(
        "--n_episodes",
        help="Number of episodes to play in the fitness function",
        type=int,
        default=1,
    )
    parser.add_argument(
        "--n_test_episodes",
        help="Number of episodes to play in the fitness function",
        type=int,
        default=10,
    )
    parser.add_argument(
        "--test_every_nth_iteration",
        help="Number of episodes to play in the fitness function",
        type=int,
        default=5,
    )
    parser.add_argument(
        "--n_timesteps",
        help="Number of timesteps for each training episode",
        type=int,
        default=None,
    )
    parser.add_argument(
        "--mu",
        type=int,
        default=None,
    )
    parser.add_argument(
        "--max_parallel",
        type=int,
        default=128,
    )
    parser.add_argument(
        "--lamb",
        type=int,
        default=None,
    )
    parser.add_argument(
        "--sigma0",
        type=float,
        default=None,
    )
    parser.add_argument(
        "--alpha",
        type=float, 
        default=0.02,
    )
    parser.add_argument(
        "--n_hidden",
        help="Number of hidden units in layer of the neural network (only used if n_layers > 2)",
        type=int,
        default=8,
    )
    parser.add_argument(
        "--n_layers", help="Number of layers in the controller", type=int, default=1
    )
    parser.add_argument("--with_bias", action="store_true")
    parser.add_argument("--normalized", action="store_true")
    parser.add_argument("--mirrored", action="store_true")
    parser.add_argument("--uncertainty_handled", action="store_true")
    parser.add_argument("--store_videos", action="store_true")
    parser.add_argument("--ars_optimal", action="store_true")
    parser.add_argument("--seed_train_envs", action="store_true")
    parser.add_argument("--scale_by_std", action="store_true")
    parser.add_argument("--break_timesteps", action="store_true")

    parser.add_argument(
        "--initialization",
        type=str,
        choices=("lhs", "zero", "uniform", "gauss"),
        default="zero",
    )

    parser.add_argument("--strategy", type=str, choices=STRATEGIES, default="r-cma-es")
    parser.add_argument(
        "--env_name", type=str, default="LunarLander-v2", choices=ENVIRONMENTS.keys()
    )
    parser.add_argument("--eval_total_timesteps", action="store_true")
    parser.add_argument("--penalize_inactivity", action="store_true")

    parser.add_argument(
        "--play",
        type=str,
        default=None,
    )
    parser.add_argument(
        "--revaluate_best_after",
        type=int,
        default=None,
    )   

    args = parser.parse_args()

    t = time.time()
    env_setting = ENVIRONMENTS[args.env_name]

    if not os.path.isdir(DATA):
        os.makedirs(DATA)

    if args.n_timesteps is None:
        args.n_timesteps = env_setting.max_episode_steps


    print(args)
    print(env_setting)

    np.random.seed(args.seed)
    plot = True
    strategy_name = args.strategy
    if not args.strategy.startswith("ars"):
        if args.uncertainty_handled:
            strategy_name = f"UH-{strategy_name}"
        if args.mirrored:
            strategy_name = f"{strategy_name}-mirrored"
        if args.normalized:
            strategy_name = f"{strategy_name}-norm"
        if args.sigma0 is None:
            strategy_name = f"{strategy_name}-sigma-default"
            n = env_setting.action_size * env_setting.state_size
            args.sigma0 = min(0.01, max(1 / np.sqrt(n), 0.005))
            print("using sigma0", args.sigma0)
        else:
            strategy_name = f"{strategy_name}-sigma-{args.sigma0:.2e}"

        if args.scale_by_std:
            strategy_name = f"{strategy_name}-std"

        if args.lamb is not None:
            strategy_name = f"{strategy_name}-lambda-{args.lamb}"

    data_folder = f"{DATA}/{args.env_name}/{strategy_name}/{t}"
    if args.play is None:
        os.makedirs(data_folder, exist_ok=True)

    if args.strategy == "ars-v2":
        args.normalized = True
    elif args.strategy == "ars":
        args.normalized = False
    
    obj = Objective(
        env_setting,
        args.n_episodes,
        args.n_timesteps,
        env_setting.max_episode_steps,
        args.n_hidden,
        args.n_layers,
        normalized=args.normalized,
        bias=args.with_bias,
        eval_total_timesteps=args.eval_total_timesteps,
        n_test_episodes=args.n_test_episodes,
        store_video=args.store_videos,
        data_folder=data_folder,
        seed_train_envs=args.seed if args.seed_train_envs else None,
        penalize_inactivity=args.penalize_inactivity,
        break_timesteps=args.break_timesteps,
        max_parallel=args.max_parallel
    )

    if args.play is None:
        run_optimizer(args, obj)
    else:
        obj.store_video = True
        obj.n_test_episodes = 1
        obj.data_folder = os.path.dirname(data_folder)
        obj.play_check(args.play, render_mode="rgb_array_list", name="test")

    # best_test = obj.play(best, data_folder, "best", plot)
    # print("Test with best x (median max):", best_test)
    # mean_test = obj.play(mean, data_folder, "mean", plot)
    # print("Test with mean x (median max):", mean_test)

    time.sleep(1)
