#!/usr/bin/env python3
"""
Convert MLP experiment results to pickle format aligned with collect_data.py.

Matches the data format used by paper_plots.ipynb:
  - Filters to every 5th generation (generation % 5 == 0)
  - train = max(best, current)
  - expected_test = max(best_median, current_median)
  - test = expected_test

Output format: DataFrame with columns:
  ['env', 'method', 'sigma0', 'lambda', 'run', 'generation',
   'n_train_episodes', 'n_train_timesteps', 'test', 'train',
   'expected_test', 'total_seconds', 'total_formatted', 'reward_shaping']
"""

import json
import os
import pickle
from pathlib import Path

import numpy as np
import pandas as pd


def parse_timing_file(timing_path: Path) -> tuple[float, str]:
    """Parse a timing.json file and extract total_seconds and total_formatted."""
    if not timing_path.exists():
        return None, None
    
    with open(timing_path, 'r') as f:
        data = json.load(f)
    
    total_seconds = data.get('total_seconds')
    total_formatted = data.get('total_formatted')
    
    # If total_formatted is not present, compute it from total_seconds
    if total_seconds is not None and total_formatted is None:
        hours, remainder = divmod(int(total_seconds), 3600)
        minutes, seconds = divmod(remainder, 60)
        total_formatted = f"{hours:02d}:{minutes:02d}:{seconds:02d}"
    
    return total_seconds, total_formatted


def parse_settings_file(settings_path: Path) -> dict:
    """Parse a settings.json file and extract sigma0 and lambda."""
    if not settings_path.exists():
        return {}
    with open(settings_path, 'r') as f:
        return json.load(f)


def parse_stats_file(stats_path: Path) -> pd.DataFrame:
    """Parse a stats.csv file, aligned with collect_data.py.

    - Filters to every 5th generation (generation % 5 == 0)
    - train = max(best, current)
    - expected_test = max(best_median, current_median)
    - test = expected_test
    """
    df = pd.read_csv(stats_path)

    # Keep only every 5th generation, matching collect_data.py
    df = df[df['generation'] % 5 == 0].reset_index(drop=True)

    # Training score: best of best individual and population mean
    df['train'] = df[['best', 'current']].max(axis=1)

    # Expected test score: median test returns of best/mean policy
    df['expected_test'] = df[['best_median', 'current_median']].max(axis=1)

    # Test = expected_test (same as collect_data.py)
    df['test'] = df['expected_test']

    result = df[['generation', 'n_train_episodes', 'n_train_timesteps',
                  'test', 'train', 'expected_test']].copy()
    return result


def get_method_name(folder_name: str) -> str:
    """Extract method name from folder name like 'mlp-lm-ma-es-norm-sigma-default'."""
    if 'lm-ma-es' in folder_name:
        return 'mlp-lm-ma-es'
    elif 'sep-cma-es' in folder_name:
        return 'mlp-sep-cma-es'
    else:
        # Generic fallback
        return folder_name.replace('mlp-', '').replace('-norm-sigma-default', '')


def is_mujoco_env(env_name: str) -> bool:
    """Check if environment is a MuJoCo environment (uses reward shaping)."""
    mujoco_envs = {
        'Ant-v4',
        'HalfCheetah-v4',
        'Hopper-v4',
        'Humanoid-v4',
        'Swimmer-v4',
        'Walker2d-v4',
    }
    return env_name in mujoco_envs


def load_mlp_experiments(data_dir: Path) -> pd.DataFrame:
    """Load all MLP experiment results from data directory."""
    all_data = []
    
    # Iterate over environment folders
    for env_dir in sorted(data_dir.iterdir()):
        if not env_dir.is_dir():
            continue
        
        env_name = env_dir.name
        print(f"Processing {env_name}...")
        
        # Iterate over method folders (e.g., mlp-lm-ma-es-norm-sigma-default)
        for method_dir in sorted(env_dir.iterdir()):
            if not method_dir.is_dir():
                continue
            
            method_name = get_method_name(method_dir.name)
            
            # Iterate over run folders (timestamps)
            run_num = 0
            for run_dir in sorted(method_dir.iterdir()):
                if not run_dir.is_dir():
                    continue
                
                run_num += 1
                stats_file = run_dir / 'stats.csv'
                timing_file = run_dir / 'timing.json'
                settings_file = run_dir / 'settings.json'
                
                if not stats_file.exists():
                    print(f"  Warning: No stats.csv in {run_dir}")
                    continue
                
                try:
                    df = parse_stats_file(stats_file)
                    df['env'] = env_name
                    df['method'] = method_name
                    df['run'] = run_num
                    
                    # Extract sigma0 and lambda from settings.json
                    settings = parse_settings_file(settings_file)
                    df['sigma0'] = settings.get('sigma0', np.nan)
                    df['lambda'] = settings.get('lambda_', settings.get('lamb', np.nan))
                    
                    # Parse timing information
                    total_seconds, total_formatted = parse_timing_file(timing_file)
                    df['total_seconds'] = total_seconds
                    df['total_formatted'] = total_formatted
                    
                    # Set reward shaping flag (True for MuJoCo envs)
                    df['reward_shaping'] = is_mujoco_env(env_name)
                    
                    all_data.append(df)
                except Exception as e:
                    print(f"  Error parsing {stats_file}: {e}")
    
    if not all_data:
        raise ValueError("No data found!")
    
    # Combine all data
    combined = pd.concat(all_data, ignore_index=True)
    
    # Reorder columns to match collect_data.py output format
    combined = combined[['env', 'method', 'sigma0', 'lambda', 'run', 'generation',
                         'n_train_episodes', 'n_train_timesteps', 'test', 'train',
                         'expected_test', 'total_seconds', 'total_formatted', 'reward_shaping']]
    
    return combined


def main():
    # Paths
    base_dir = Path(__file__).parent
    mlp_data_dirs = [
        base_dir / 'es' / 'data' / 'mlp_experiments',
        base_dir / 'es' / 'data' / 'mlp_experiments_missing',
    ]
    output_file = base_dir / 'mlp_experiments_data.pkl'
    
    # Load data from all directories
    frames = []
    for mlp_data_dir in mlp_data_dirs:
        if not mlp_data_dir.exists():
            print(f"Skipping (not found): {mlp_data_dir}")
            continue
        print(f"Loading MLP experiments from: {mlp_data_dir}")
        frames.append(load_mlp_experiments(mlp_data_dir))
    
    df = pd.concat(frames, ignore_index=True)
    
    print(f"\nLoaded {len(df)} rows")
    print(f"Shape: {df.shape}")
    print(f"Columns: {list(df.columns)}")
    print(f"\nEnvironments: {sorted(df['env'].unique())}")
    print(f"\nMethods: {sorted(df['method'].unique())}")
    print(f"\nRows per method/env/run (first 10):")
    print(df.groupby(['env', 'method'])['run'].value_counts().groupby(['env', 'method']).first().head(10))
    
    # Save to pickle
    df.to_pickle(output_file)
    print(f"\nSaved to: {output_file}")
    
    # Verify with Acrobot example
    acro = df[(df['env'] == 'Acrobot-v1') & (df['method'] == 'mlp-lm-ma-es')]
    if len(acro) > 0:
        print(f"\nVerification — Acrobot mlp-lm-ma-es:")
        print(f"  Total rows: {len(acro)}, runs: {acro['run'].nunique()}")
        print(f"  Rows per run: {dict(acro.groupby('run').size())}")
        print(f"  Sample (run 1):")
        print(acro[acro['run'] == 1][['env', 'method', 'run', 'generation',
              'n_train_timesteps', 'test', 'train', 'expected_test']].to_string(index=False))


if __name__ == '__main__':
    main()
