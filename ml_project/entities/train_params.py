"""Copyright 2022 by Artem Ustsov"""

from dataclasses import dataclass, field
from sklearn.model_selection import StratifiedKFold
from typing import Any


@dataclass()
class TrainingParams:
    model_type: str = field(default="LogisticRegressionCV")
    penalty: str = field(default="l2")
    max_iter: int = field(default=10000)
    tol: float = field(default=0.01)
