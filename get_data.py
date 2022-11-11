"""Get data from Kaggle."""
import zipfile

import pandas as pd
from kaggle.api.kaggle_api_extended import KaggleApi

from config.config import logger


def authenticate_api() -> KaggleApi():
    """Authenticate the Kaggle API with your set environment username and token.

    Returns:
        KaggleApi: the authenticated API object.
    """
    api = KaggleApi()
    logger.info("Authenticating API keys")
    api.authenticate()
    logger.info("Keys authenticated")
    return api


def download_dataset(api: KaggleApi(), output_path: str) -> None:
    """Download the dataset using the authenticated API.

    Args:
        api (KaggleApi): Kaggle API object
        output_path (str): Path to directory to save dataset
    """
    logger.info("Downloading house pricing data from Kaggle")
    api.dataset_download_files(
        "harlfoxem/housesalesprediction", path=output_path
    )


def prepare_dataset(output_path: str) -> None:
    """Unzip the downloaded dataset and set the date column as index.

    Save the dataset as csv.

    Args:
        output_path (str): Path to directory to save dataset
    """
    logger.info("Extracting dataset zip file")
    with zipfile.ZipFile(
        f"{output_path}/housesalesprediction.zip", "r"
    ) as zip_ref:
        zip_ref.extractall(output_path)

    logger.info("Processing data")

    house_data = pd.read_csv(f"{output_path}/kc_house_data.csv")

    house_data["date"] = pd.to_datetime(house_data["date"])

    house_data.set_index("date", inplace=True)
    house_data.to_csv(f"{output_path}/processed_house_data.csv")

    house_data.head()
    logger.info("processed_house_data.csv generated")


if __name__ == "__main__":
    data_output_path = "datasets"
    api = authenticate_api()
    download_dataset(api, data_output_path)
    prepare_dataset(data_output_path)
