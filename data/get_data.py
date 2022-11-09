"""Download data from Google drive and preprocess dataset."""
import logging
import os
import zipfile

import gdown
import pandas as pd


def download_dataset(url: str, output: str) -> None:
    """Download the dataset using the gdown library.

    Args:
        url (str): the link to the dataset on google drive
        output (str): the name of the file and the output path
    """
    logging.info("Downloading house pricing data from google drive")
    # Download the dataset using gdown library
    gdown.download(url, output, quiet=False)


def preprocess_dataset(output_path: str) -> None:
    """Unzip dataset and set the date column as index and saves the dataset as csv.

    Args:
        output_path (str): the path to the output file
    """
    # Extract dataset
    logging.info("Extracting dataset zip file")
    with zipfile.ZipFile(output_path, "r") as zip_ref:
        zip_ref.extractall("data")

    logging.info(f"Downloaded dataset at path: {output_path}")
    logging.info("Processing data...")

    root_dir = os.path.dirname(output_path)
    filename = "kc_house_data.csv"
    # Read data
    house_data = pd.read_csv(os.path.join(root_dir, filename))
    # Convert to datetime using pandas
    house_data["date"] = pd.to_datetime(house_data["date"])
    # Set date column as index
    house_data.set_index("date", inplace=True)
    # Save as new dataset
    new_filename = "processed_house_data.csv"
    save_path = os.path.join(root_dir, new_filename)
    house_data.to_csv(save_path)
    logging.info(f"Saved processed dataset at path: {save_path}")
