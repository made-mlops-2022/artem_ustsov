"""Copyright 2022 by Artem Ustsov"""

import logging
import pickle
import boto3
from io import StringIO

import dvc.api
import hydra
from typing import Any, List, NoReturn, Tuple
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import FeatureUnion, Pipeline

# from ml_project.conf.config import Config
# from ml_project.data import DATA_PATH, MODEL_PATH

CAT_FEATURES = ['sex', 'chest_pain_type', 'fasting_blood_sugar', 'rest_ecg',
                'exercise_induced_angina', 'st_slope', 'thalassemia']

NUMERICAL_FEATURES = ['age', 'resting_blood_pressure', 'cholesterol',
                      'max_heart_rate_achieved', 'st_depression', 'num_major_vessels']


# Custom Transformer that extracts columns passed as argument to its constructor
class FeatureSelector(BaseEstimator, TransformerMixin):
    """

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
    """

    """

    def transform(self, x_data: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
        """

        :param x_data:
        :return:
        """

        _X = x_data.copy()
        _X.columns = [
            'age', 'sex', 'chest_pain_type', 'resting_blood_pressure', 'cholesterol',
            'fasting_blood_sugar', 'rest_ecg', 'max_heart_rate_achieved',
            'exercise_induced_angina', 'st_depression', 'st_slope', 'num_major_vessels',
            'thalassemia', 'condition',
        ]
        _y = _X['condition']
        _X = _X.drop('condition', axis=1)
        return _X, _y


class CategoricalTransformer(BaseEstimator, TransformerMixin):
    """

    """

    @staticmethod
    def process_sex(obj: Any) -> str:
        if obj == 0:
            return 'female'
        if obj == 1:
            return 'male'

    @staticmethod
    def process_chest_pain_type(obj: Any) -> str:
        if obj == 0:
            return 'typical_angina'
        if obj == 1:
            return 'atypical_angina'
        if obj == 2:
            return 'non_anginal_pain'
        if obj == 3:
            return 'asymptomatic'

    @staticmethod
    def process_rest_ecg(obj: Any) -> str:
        if obj == 0:
            return 'normal'
        if obj == 1:
            return 'ST-T_wave_abnormality'
        if obj == 2:
            return 'left_ventricular_hypertrophy'

    @staticmethod
    def process_fasting_blood_sugar(obj: Any) -> str:
        if obj == 0:
            return 'less_than_120mg/ml'
        if obj == 1:
            return 'greater_than_120mg/ml'

    @staticmethod
    def process_exercise_induced_angina(obj: Any) -> str:
        if obj == 0:
            return 'no'
        if obj == 1:
            return 'yes'

    @staticmethod
    def process_st_slope(obj: Any) -> str:
        if obj == 0:
            return 'upsloping'
        if obj == 1:
            return 'flat'
        if obj == 2:
            return 'downsloping'

    @staticmethod
    def process_thalassemia(obj: Any) -> str:
        if obj == 0:
            return 'fixed_defect'
        if obj == 1:
            return 'normal'
        if obj == 2:
            return 'reversable_defect'

    def transform(self, x_data: pd.DataFrame) -> np.array:
        """

        :param x_data:
        :return:
        """

        for cat_feature in x_data.columns:
            exec(f"X.loc[:, '{cat_feature}'] = X['{cat_feature}'].apply(self.process_{cat_feature})")
        return x_data.values


class NumericalTransformer(BaseEstimator, TransformerMixin):
    """

    """

    def __init__(self, rest_max_blood_pres_ratio: bool = True) -> NoReturn:
        self._rest_max_blood_pres_ratio = rest_max_blood_pres_ratio

    def transform(self, x_data: pd.DataFrame) -> np.array:
        """

        :param x_data:
        :return:
        """

        if self._rest_max_blood_pres_ratio:
            x_data.loc[:, 'rest_max_blood_pres_ratio'] = x_data['max_heart_rate_achieved'] / \
                                                         x_data['resting_blood_pressure']

        # Converting any infinity values in the dataset to Nan
        x_data = x_data.replace([np.inf, -np.inf], np.nan)
        return x_data.values

# TO DO CONFIG
@hydra.main(version_base=None, config_path='../conf', config_name="config")
def main(cfg: Config):
    try:
        dvc_params = dvc.api.params_show()
        [setattr(cfg, k, v) for k, v in dvc_params.items()]
    except FileNotFoundError:
        pass

    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    logger = logging.getLogger('process data')

    logger.info('Begin preprocessing...')




    # TO DO CONFIG
    session = boto3.session.Session()
    s3_client = session.client(
        service_name='s3',
        region_name='ru-msk',
        endpoint_url='https://hb.bizmrg.com',
        aws_access_key_id='6CKG3ZF3Mxs91VfNrw3c9Z',
        aws_secret_access_key='47vCFUUq3su1EhCeLzpXDDL2iBvtV6DudxJDcNsh9kKp'
    )

    bucket_name = 'ml_project'
    object_key = 'dataset/heart_cleveland_upload.csv'

    # TO DO CONFIG
    csv_obj = s3_client.get_object(Bucket=bucket_name, Key=object_key)
    csv_string = csv_obj['Body'].read().decode('utf-8')
    df = pd.read_csv(StringIO(csv_string))





    # TO DO CONFIG
    raw_data = pd.read_csv(DATA_PATH.joinpath('raw/heart_cleveland_upload.csv'))

    df_transformer = DataframeTransformer()
    train_data, target = df_transformer.transform(raw_data)

    if cfg.preprocessing.categorical_features == 'all':
        categorical_features = CAT_FEATURES_ONE_HOT + CAT_FEATURES_LABEL
    else:
        categorical_features = CAT_FEATURES_ONE_HOT

    categorical_pipeline = Pipeline(steps=[('cat_selector', FeatureSelector(CAT_FEATURES)),
                                           ('cat_transformer', CategoricalTransformer()),
                                           ('one_hot_encoder', OneHotEncoder(sparse=False))])

    numerical_pipeline = Pipeline(steps=[('num_selector', FeatureSelector(NUMERICAL_FEATURES)),
                                         ('num_transformer', NumericalTransformer()),
                                         ('std_scaler', StandardScaler())])

    full_pipeline = FeatureUnion(transformer_list=[('categorical_pipeline', categorical_pipeline),
                                                   ('numerical_pipeline', numerical_pipeline)])


    logger.info('Save encoder...')

    with open(MODEL_PATH.joinpath('encoder_baseline.pkl'), 'wb') as file:
        pickle.dump(encoder, file)

    full_pipeline_m = Pipeline(steps=[('full_pipeline', full_pipeline)])
    full_pipeline_m.fit(train_data)
    train_transformed_data = full_pipeline_m.transform(train_data)
    train_transformed_data.to_csv(DATA_PATH.joinpath('processed/heart_cleveland_upload.csv'), index=False)

    logger.info('Preprocess finished')


if __name__ == '__main__':
    main()
