"""Copyright 2022 by Artem Ustsov"""

import os
from datetime import timedelta

from airflow.models import Variable

LOCAL_DATA_DIR = Variable.get("local_data_dir")

default_args = {
    "owner": "artem_ustsov",
    "retries": 1,
    "retry_delay": timedelta(minutes=5),
}


def wait_file(file_name: str) -> bool:
    return os.path.exists(file_name)
