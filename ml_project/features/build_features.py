"""Copyright 2022 by Artem Ustsov"""

import logging
import pickle
import boto3
from io import StringIO


from typing import Any, List, NoReturn
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from sklearn.impute import SimpleImputer
from sklearn.pipeline import FeatureUnion, Pipeline

from ml_project.entities.feature_params import FeatureParams


class FeatureSelector(BaseEstimator, TransformerMixin): # FIXME
    """Custom Transformer that extracts columns passed
    as argument to its constructor
    """

    def __init__(self, feature_names: List[str]) -> NoReturn:
        self._feature_names = feature_names

    def transform(self, x_data: pd.DataFrame) -> pd.DataFrame:
        """

        :param x_data:
        :return:
        """

        return x_data[self._feature_names]


class DataframeTransformer(BaseEstimator, TransformerMixin):
    """ """

    def transform(self, x_data: pd.DataFrame) -> pd.DataFrame:
        """

        :param x_data:
        :return:
        """

        _x_data = x_data.copy()
        _x_data.columns = [
            "age",
            "sex",
            "chest_pain_type",
            "resting_blood_pressure",
            "cholesterol",
            "fasting_blood_sugar",
            "rest_ecg",
            "max_heart_rate_achieved",
            "exercise_induced_angina",
            "st_depression",
            "st_slope",
            "num_major_vessels",
            "thalassemia",
            "condition",
        ]
        return _x_data


class CategoricalTransformer(BaseEstimator, TransformerMixin):  # FIXME
    """ """

    @staticmethod
    def process_sex(obj: Any) -> str:
        if obj == 0:
            return "female"
        if obj == 1:
            return "male"

    @staticmethod
    def process_chest_pain_type(obj: Any) -> str:
        if obj == 0:
            return "typical_angina"
        if obj == 1:
            return "atypical_angina"
        if obj == 2:
            return "non_anginal_pain"
        if obj == 3:
            return "asymptomatic"

    @staticmethod
    def process_rest_ecg(obj: Any) -> str:
        if obj == 0:
            return "normal"
        if obj == 1:
            return "ST-T_wave_abnormality"
        if obj == 2:
            return "left_ventricular_hypertrophy"

    @staticmethod
    def process_fasting_blood_sugar(obj: Any) -> str:
        if obj == 0:
            return "less_than_120mg/ml"
        if obj == 1:
            return "greater_than_120mg/ml"

    @staticmethod
    def process_exercise_induced_angina(obj: Any) -> str:
        if obj == 0:
            return "no"
        if obj == 1:
            return "yes"

    @staticmethod
    def process_st_slope(obj: Any) -> str:
        if obj == 0:
            return "upsloping"
        if obj == 1:
            return "flat"
        if obj == 2:
            return "downsloping"

    @staticmethod
    def process_thalassemia(obj: Any) -> str:
        if obj == 0:
            return "fixed_defect"
        if obj == 1:
            return "normal"
        if obj == 2:
            return "reversable_defect"

    def transform(self, x_data: pd.DataFrame) -> np.array:
        """

        :param x_data:
        :return:
        """
        # _x_data = x_data.copy()

        for cat_feature in x_data.columns:
            exec(
                f"x_data.loc[:, '{cat_feature}'] = x_data['{cat_feature}'].apply(self.process_{cat_feature})"
            )
        return x_data


class NumericalTransformer(BaseEstimator, TransformerMixin):  # FIXME
    """ """

    def __init__(self, new_feature: str = None) -> NoReturn:
        self.new_feature = new_feature

    def transform(self, x_data: pd.DataFrame) -> np.array:
        """

        :param x_data:
        :return:
        """
        _x_data = x_data.copy()

        if self.new_feature:
            x_data.loc[:, self.new_feature] = (
                x_data["max_heart_rate_achieved"] / x_data["resting_blood_pressure"]
            )

        # Converting any infinity values in the dataset to Nan
        _x_data = _x_data.replace([np.inf, -np.inf], np.nan)
        return _x_data


def build_raw_data_pipeline() -> Pipeline:
    # new_feature = "rest_max_blood_pres_ratio"
    raw_headers_pipeline = Pipeline(steps=[
                                            ("column_renames", DataframeTransformer())
                                          ]
    )
    # FIXME
    # numerical_raw_pipeline = Pipeline(steps=[
    #                                            ('num_selector', FeatureSelector(params.numerical_features)),
    #                                            ('num_transformer', NumericalTransformer()),
    #                                            # ('add_new_feature', params.numerical_features.append(new_feature))
    #                                         ]
    # )
    #
    # categorical_raw_pipeline = Pipeline(steps=[
    #                                              ('cat_selector', FeatureSelector(params.categorical_features)),
    #                                              ('cat_transformer', CategoricalTransformer()),
    #                                           ]
    # )
    #
    # full_raw = FeatureUnion(transformer_list=[('raw_headers_pipeline', raw_headers_pipeline),
    #                                           ('categorical_raw_pipeline', categorical_raw_pipeline),
    #                                           ('numerical_raw_pipeline', numerical_raw_pipeline)])
    #
    # full_raw_pipeline = Pipeline(steps=[
    #                                      ('full_raw_pipeline', full_raw)
    #                                    ]
    # )
    return raw_headers_pipeline


def process_raw_data(raw_data_df: pd.DataFrame) -> pd.DataFrame:
    logger = logging.getLogger(__name__)
    logger.info('Rename dataframe columns')
    raw_data_pipeline = build_raw_data_pipeline()
    return pd.DataFrame(raw_data_pipeline.transform(raw_data_df))


def build_categorical_pipeline() -> Pipeline:
    categorical_pipeline = Pipeline(
        steps=[
            ("impute", SimpleImputer(missing_values=np.nan, strategy="most_frequent")),
            ("one_hot_encoder", OneHotEncoder(sparse=False)),
        ]
    )
    return categorical_pipeline


def process_categorical_features(categorical_df: pd.DataFrame) -> pd.DataFrame:
    logger = logging.getLogger(__name__)
    logger.info('Impute missing with most frequent. Make one hot encoding')
    categorical_pipeline = build_categorical_pipeline()
    return pd.DataFrame(categorical_pipeline.fit_transform(categorical_df).toarray())


def build_numerical_pipeline() -> Pipeline:
    numerical_pipeline = Pipeline(
        steps=[
            ("impute", SimpleImputer(missing_values=np.nan, strategy="mean")),
            ("std_scaler", StandardScaler()),
        ]
    )
    return numerical_pipeline


def process_numerical_features(numerical_df: pd.DataFrame) -> pd.DataFrame:
    logger = logging.getLogger(__name__)
    logger.info('Imputing missing values by median. Make standard scaling')
    num_pipeline = build_numerical_pipeline()
    return pd.DataFrame(num_pipeline.fit_transform(numerical_df))


def make_features(transformer: ColumnTransformer, df: pd.DataFrame) -> pd.DataFrame:
    logger = logging.getLogger(__name__)
    logger.info('Making features')
    return transformer.transform(df)


def build_transformer(params: FeatureParams) -> ColumnTransformer:
    transformer = ColumnTransformer(
        [
            (
                "categorical_pipeline",
                build_categorical_pipeline(),
                params.categorical_features,
            ),
            (
                "numerical_pipeline",
                build_numerical_pipeline(),
                params.numerical_features,
            ),
        ]
    )
    return transformer


def extract_target(df: pd.DataFrame, params: FeatureParams) -> pd.Series:
    target = df[params.target_col]
    return target
