import os
import pandas as pd
import numpy as np
from tqdm import tqdm

from objective import Objective
from setting import ENVIRONMENTS


def load_data(env_folder):
    stats_list = []
    for strat in os.listdir(env_folder):
        path = os.path.join(env_folder, strat)
        if not os.path.isdir(path):
            continue
        for i, (run) in enumerate(os.listdir(path)):
            path = os.path.join(env_folder, strat, run)
            if not os.path.isdir(path):
                continue
            stats = pd.read_csv(os.path.join(path, "stats.csv"), skipinitialspace=True)
            stats["run"] = i
            stats["folder"] = run
            stats["strat"] = strat
            sig, lamb = strat.split("sigma-")[1].split("-lambda-")
            stats["sigma0"] = float(sig)
            stats["lambda"] = int(lamb)
            stats_list.append(stats)

    return pd.concat(stats_list, ignore_index=True)


def get_objective(env_name, store_videos=False, data_folder=""):
    env_setting = ENVIRONMENTS[env_name]
    return Objective(
        env_setting,
        n_episodes=1,
        n_timesteps=None,
        n_eval_timesteps=None,
        n_layers=1,
        normalized=True,
        bias=False,
        eval_total_timesteps=False,
        n_test_episodes=5,
        store_video=store_videos,
        data_folder=data_folder,
        seed_train_envs=None,
    )


def get_loc(env_name, record):
    root = os.path.join(
        "./data",
        env_name,
        record.strat,
        record.folder,
        "policies",
        f"t-{record.generation}",
    )
    best = f"{root}-best"
    current = f"{root}-mean"
    assert os.path.isfile(f"{best}.npy") and os.path.isfile(f"{current}.npy")
    return best, current


def get_auc(data, n_top, time_measure="n_train_episodes"):
    df_auc = pd.DataFrame(columns=["method", "sigma0", "lambda", "auc"])
    for _, group in data.groupby("method"):
        time = np.sort(np.unique(group[time_measure])).astype(int)
        setting_test = []
        setting_parm = []
        for (
            _,
            setting,
        ) in group.groupby("strat"):
            y = setting.groupby(time_measure).expected_test.median()
            setting_test.append(np.interp(time, y.index.values, y.values))
            setting_parm.append(
                (
                    setting.method.iloc[0],
                    setting.sigma0.iloc[0],
                    setting["lambda"].iloc[0],
                )
            )

        setting_test = np.array(setting_test)
        setting_test += -setting_test.min()
        setting_auc = np.array([np.trapz(si, time) for si in setting_test])
        df_auc = pd.concat(
            [
                df_auc,
                pd.DataFrame(np.c_[setting_parm, setting_auc], columns=df_auc.columns),
            ]
        )

    df_auc = df_auc.astype({"sigma0": float, "lambda": int, "auc": float})
    top_performers = df_auc.groupby("method")["auc"].max()
    top_performers = df_auc[df_auc.auc.isin(top_performers)].drop_duplicates(
        subset="method", keep="first"
    )
    assert len(top_performers) == n_top, top_performers
    return df_auc, top_performers.reset_index(drop=True)


ENVS_W_CORRECTION = [
    "Hopper-v4",
    "Walker2d-v4",
    "Ant-v4",
]

ENVS_WO_CORRECTION = [
    "Swimmer-v4",
    "HalfCheetah-v4",
    # "Humanoid-v4",
    "CartPole-v1",
    "Acrobot-v1",
    "Pendulum-v1",
    "LunarLander-v2",
    "BipedalWalker-v3",
    "SpaceInvaders-v5",
    "Atlantis-v5",
]

ALL_GYM_ENVS = [
    # ("CartPole-v1", 0.1, 4),
    # ("Acrobot-v1", 0.05, 4),
    # ("Pendulum-v1", .1, 32),
    # ("LunarLander-v2", .1, 32),
    # ("BipedalWalker-v3", .1, 48),
    # ("Swimmer-v4", .1, 4),
    # ("HalfCheetah-v4", 0.05, 17),
    # ("Hopper-v4", 0.05, 32),
    # ("Walker2d-v4", 0.05, 51),
    ("Humanoid-v4", 0.01, 128),
    ("Ant-v4", 0.05, 108),
]

if __name__ == "__main__":
    copy_from_data = False
    use_expected = True
    for env_name, sig, lam in ALL_GYM_ENVS:
        # (
        #
        #     # "Atlantis-v5",
        #     # "BeamRider-v5",
        #     # "Boxing-v5",
        #     # "Pong-v5",
        #     # "CrazyClimber-v5",
        #     # "Enduro-v5",
        #     # "Qbert-v5",
        #     # "Seaquest-v5",
        #     # "SpaceInvaders-v5"
        #     # "Hopper-v4",
        #     # "Walker2d-v4",
        #     # "Ant-v4",
        # ):
        obj = get_objective(env_name)
        data = load_data(f"data/{env_name}")
        data = data[data.generation % 5 == 0].reset_index(drop=True)
        data["train"] = data[["best", "current"]].max(axis=1)
        data["method"] = data.strat.str.split("-norm-").str[0]
        data["test"] = -np.inf
        data["expected_test"] = data[["best_median", "current_median"]].max(axis=1)

        # print(env_name)
        # df_auc, top_performers = get_auc(data, 3 if not env_name == 'Humanoid-v4' else 2, "n_train_timesteps")
        # df_auc.auc /= 1e5
        # print(df_auc)
        # print(top_performers)

        # cols = ["method","sigma0", "lambda"]
        # index = pd.MultiIndex.from_frame(data[cols])
        # data = data[index.isin(top_performers[cols].values.tolist())]

        data = data[(data.sigma0 == sig) & (data["lambda"] == lam)]

        use_expected = env_name not in ("Hopper-v4", "Ant-v4", "Walker2d-v4")

        for idx, row in tqdm(data.iterrows(), total=len(data), disable=False):
            best_loc, current_loc = get_loc(env_name, row)
            if row.best_median > row.current_median:
                loc = best_loc
                expected = row.best_median
            else:
                loc = current_loc
                expected = row.current_median

            if row.method.startswith("sep") or use_expected:
                data.at[idx, "test"] = expected
            else:
                data.at[idx, "test"] = np.median(obj.play_check(loc, n_reps=5))

        data = data[
            [
                "method",
                "sigma0",
                "lambda",
                "run",
                "generation",
                "n_train_episodes",
                "n_train_timesteps",
                "test",
                "train",
                "expected_test",
            ]
        ]

        if copy_from_data:
            data2 = pd.read_pickle(f"data/{env_name}/data.pkl")
            data["test"] = data["expected_test"]
            data2 = data2[~data2.method.str.startswith("sep")]
            data = data[data.method.str.startswith("sep")]
            data = pd.concat([data, data2]).reset_index(drop=True)
        data.to_pickle(f"data/{env_name}/data_hyp.pkl")
