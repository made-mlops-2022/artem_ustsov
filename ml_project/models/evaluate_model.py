"""Copyright 2022 by Artem Ustsov"""

from sklearn.metrics import f1_score
from typing import Dict
import numpy as np
import pandas as pd


def evaluate_model(
    predicts: np.ndarray, target: pd.Series, use_log_trick: bool = False
) -> Dict[str, float]:
    if use_log_trick:
        target = np.exp(target)
    return {
        "f1_score": f1_score(target, predicts),
    }
