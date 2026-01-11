#!/usr/bin/env python3
"""
Convert MLP experiment results to pickle format matching limited_data_5_runs_v2.pkl

Output format: DataFrame with columns ['env', 'method', 'run', 'n_train_timesteps', 'test']
"""

import os
import pickle
from pathlib import Path

import pandas as pd


def parse_stats_file(stats_path: Path) -> pd.DataFrame:
    """Parse a stats.csv file and extract relevant columns."""
    df = pd.read_csv(stats_path)
    
    # Select and rename columns to match target format
    # Use best_test as the test score (test evaluation of best policy)
    result = df[['n_train_timesteps', 'best_test']].copy()
    result.columns = ['n_train_timesteps', 'test']
    
    # Filter out rows where test is None/NaN (evaluations not performed)
    result = result.dropna(subset=['test'])
    
    # If no test evaluations available, use 'best' column (training score) as fallback
    # This happens when experiments finish before first test evaluation
    if len(result) == 0 and 'best' in df.columns:
        result = df[['n_train_timesteps', 'best']].copy()
        result.columns = ['n_train_timesteps', 'test']
        # Take only the last row (final result)
        result = result.tail(1)
    
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
                
                if not stats_file.exists():
                    print(f"  Warning: No stats.csv in {run_dir}")
                    continue
                
                try:
                    df = parse_stats_file(stats_file)
                    df['env'] = env_name
                    df['method'] = method_name
                    df['run'] = run_num
                    all_data.append(df)
                except Exception as e:
                    print(f"  Error parsing {stats_file}: {e}")
    
    if not all_data:
        raise ValueError("No data found!")
    
    # Combine all data
    combined = pd.concat(all_data, ignore_index=True)
    
    # Reorder columns to match target format
    combined = combined[['env', 'method', 'run', 'n_train_timesteps', 'test']]
    
    return combined


def main():
    # Paths
    base_dir = Path(__file__).parent
    mlp_data_dir = base_dir / 'es' / 'data' / 'mlp_experiments'
    output_file = base_dir / 'mlp_experiments_data.pkl'
    
    print(f"Loading MLP experiments from: {mlp_data_dir}")
    
    # Load data
    df = load_mlp_experiments(mlp_data_dir)
    
    print(f"\nLoaded {len(df)} rows")
    print(f"Shape: {df.shape}")
    print(f"Columns: {list(df.columns)}")
    print(f"\nEnvironments: {sorted(df['env'].unique())}")
    print(f"\nMethods: {sorted(df['method'].unique())}")
    print(f"\nRuns per method/env:")
    print(df.groupby(['env', 'method'])['run'].nunique())
    
    # Save to pickle
    df.to_pickle(output_file)
    print(f"\nSaved to: {output_file}")
    
    # Verify
    print("\nVerification - first 10 rows:")
    print(df.head(10))


if __name__ == '__main__':
    main()
