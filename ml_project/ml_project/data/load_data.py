"""Copyright 2022 by Artem Ustsov"""

import logging

from dotenv import find_dotenv, load_dotenv

from ml_project.data import DATA_PATH


def main():
    logger = logging.getLogger('download dataset')
    logger.info('Try kaggle auth...')

    try:
        from kaggle.api.kaggle_api_extended import KaggleApi
        api = KaggleApi()
        api.authenticate()
    except OSError:
        logger.error('KAGGLE_KEY and KAGGLE_USERNAME not found in environment. Exit.')
        return

    logger.info('Downloading dataset...')
    api.dataset_download_files('cherngs/heart-disease-cleveland-uci',
                               path=DATA_PATH.joinpath('raw'),
                               unzip=True)
    logger.info('Dataset downloaded')


if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    load_dotenv(find_dotenv())
    main()
