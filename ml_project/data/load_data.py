"""Copyright 2022 by Artem Ustsov"""

from typing import Tuple, NoReturn

import pandas as pd
import boto3
from sklearn.model_selection import train_test_split

from ml_project.entities.split_params import SplittingParams
import logging


def download_data_from_s3(s3_bucket: str, s3_path: str, output: str) -> NoReturn:
    logger = logging.getLogger(__name__)
    logger.info(f"Download from {s3_path}")
    session = boto3.session.Session()
    s3_client = session.client(
        service_name='s3',
        region_name='ru-msk',
        endpoint_url='https://hb.bizmrg.com',
        aws_access_key_id='6CKG3ZF3Mxs91VfNrw3c9Z',
        aws_secret_access_key='47vCFUUq3su1EhCeLzpXDDL2iBvtV6DudxJDcNsh9kKp'
    )
    with open(output, 'wb') as f:
        s3_client.download_fileobj(s3_bucket, s3_path, f)


def read_data(path: str) -> pd.DataFrame:
    print(path)
    data = pd.read_csv(path)
    return data


def split_train_val_data(
    data: pd.DataFrame, params: SplittingParams
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    :rtype: object
    """
    train_data, val_data = train_test_split(
        data, test_size=params.val_size, random_state=params.random_state
    )
    return train_data, val_data
