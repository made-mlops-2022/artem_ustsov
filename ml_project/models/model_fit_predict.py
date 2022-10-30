"""Copyright 2022 by Artem Ustsov"""

import pickle
from typing import Dict

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegressionCV
from sklearn.metrics import f1_score
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.model_selection import StratifiedKFold

from ml_project.entities.train_params import TrainingParams

SklearnClassifierModel = LogisticRegressionCV


def train_model(
    features: pd.DataFrame, target: pd.Series, train_params: TrainingParams
) -> SklearnClassifierModel:
    cv = StratifiedKFold(10)
    if train_params.model_type == "LogisticRegressionCV":
        model = LogisticRegressionCV(
            penalty=train_params.penalty,
            cv=cv,
            max_iter=train_params.max_iter,
            tol=train_params.tol,
        )
    else:
        raise NotImplementedError()
    model.fit(features, target)
    return model


def predict_model(
    model: Pipeline,
    features: pd.DataFrame,
) -> np.ndarray:
    predicts = model.predict(features)
    return predicts


def evaluate_model(
    predicts: np.ndarray, target: pd.Series, use_log_trick: bool = False
) -> Dict[str, float]:
    if use_log_trick:
        target = np.exp(target)
    return {
        "f1_score": f1_score(target, predicts),
    }


def create_inference_pipeline(
    model: SklearnClassifierModel, transformer: FeatureUnion
) -> Pipeline:
    return Pipeline([("feature_part", transformer), ("model_part", model)])


def serialize_model(model: object, output: str) -> str:
    with open(output, "wb") as f:
        pickle.dump(model, f)
    return output
