"""Copyright 2022 by Artem Ustsov"""

import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline


def predict_model(
    model: Pipeline,
    features: pd.DataFrame,
) -> np.ndarray:
    predicts = model.predict(features)
    return predicts
