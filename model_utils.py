import numpy as np

def log_transform_func(x):
    "`Log1p transform used inside the XGBoost preprocessing pipeline."""
    return np.log1p(x)
