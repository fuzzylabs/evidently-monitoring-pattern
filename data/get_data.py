import logging
import zipfile
import numpy as np
import pandas as pd

from kaggle.api.kaggle_api_extended import KaggleApi


def setup_logger() -> None:
    logging.basicConfig(
        level=logging.INFO, 
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[
            logging.StreamHandler()
        ]
    )


def authenticate_api() -> KaggleApi():
    '''
    Authenticate the Kaggle API with your set environment username and token.
    
    Retrun the authenticated API object.
    '''
    api = KaggleApi()
    logging.info("Authenticating API keys")
    api.authenticate()
    logging.info("Keys authenticated")
    return api


def download_dataset(api: KaggleApi()) -> None:
    '''
    Download the dataset using the authenticated API.
    '''
    logging.info("Downloading house pricing data from Kaggle")
    api.dataset_download_files('harlfoxem/housesalesprediction', path="data")


def prepare_dataset() -> None:
    '''
    Unzip the downloaded dataset and set the date column as index.

    Save the dataset as csv.
    '''
    logging.info("Extracting dataset zip file")
    with zipfile.ZipFile('data/housesalesprediction.zip', 'r') as zip_ref:
        zip_ref.extractall('data')

    logging.info("Processing data")

    house_data = pd.read_csv("data/kc_house_data.csv")

    house_data['date'] = pd.to_datetime(house_data['date'])

    house_data.set_index('date', inplace=True)
    house_data.to_csv('data/processed_house_data.csv')

    house_data.head()
    logging.info("processed_house_data.csv generated")


if __name__ == "__main__":
    setup_logger()
    api = authenticate_api()
    download_dataset(api)
    prepare_dataset()
