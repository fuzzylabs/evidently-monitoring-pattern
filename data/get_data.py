"""Download data from Kaggle."""
import logging
import zipfile

import gdown
import pandas as pd


def setup_logger() -> None:
    """Set logger basic config."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[logging.StreamHandler()],
    )


def download_dataset(url: str, output: str) -> None:
    """Download the dataset using the authenticated API.

    Args:
        url (str): the link to the dataset on google drive
        output (str): the name of the file and the output path
    """
    logging.info("Downloading house pricing data from google drive")
    gdown.download(url, output, quiet=False)


def prepare_dataset() -> None:
    """Unzip datasetm set the date column as index and save the dataset as csv."""
    logging.info("Extracting dataset zip file")
    with zipfile.ZipFile("data/housesalesprediction.zip", "r") as zip_ref:
        zip_ref.extractall("data")

    logging.info("Processing data")

    house_data = pd.read_csv("data/kc_house_data.csv")

    house_data["date"] = pd.to_datetime(house_data["date"])

    house_data.set_index("date", inplace=True)
    house_data.to_csv("data/processed_house_data.csv")

    house_data.head()
    logging.info("processed_house_data.csv generated")


if __name__ == "__main__":
    setup_logger()

    dataset_url = (
        "https://drive.google.com/uc?id=1YTeSOebJhD2skONp9lJoO6YK6FhxcOd2"
    )
    output = "data/housesalesprediction.zip"
    download_dataset(dataset_url, output)
    prepare_dataset()
