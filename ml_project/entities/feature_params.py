"""Copyright 2022 by Artem Ustsov"""

from dataclasses import dataclass
from typing import Any, Optional, List
import pandas as pd


@dataclass()
class FeatureParams:
    categorical_features: List[str]
    numerical_features: List[str]
    # features_to_drop: List[str]
    target_col: Optional[str]
