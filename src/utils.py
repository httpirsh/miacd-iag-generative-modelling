"""Small reproducibility and reporting helpers."""

import numpy as np
import pandas as pd
import torch


def set_seed(seed=42):
    """Set random seed for reproducibility"""
    torch.manual_seed(seed)
    np.random.seed(seed)
    torch.cuda.manual_seed_all(seed)


def create_results_dataframe(results_dict):
    """Create pandas DataFrame from results across seeds"""
    data = []
    for seed, metrics in results_dict.items():
        row = {"Seed": seed}
        row.update(metrics)
        data.append(row)
    return pd.DataFrame(data)
