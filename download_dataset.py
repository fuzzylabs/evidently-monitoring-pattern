"""Download and preprocess downloaded dataset."""
from config.config import logger
from data.get_data import download_dataset, preprocess_dataset

""" Jon:
The code below should be split into several functions and there should be
a 'main' function which calls them. It's not the best practise to run this
within the code execution if statement.

You could also make the dataset_url and ouput either parameters that are
passed to the file, or better, are stored in some form of configuration file.

It's worth considering whether the 'preproces_dataset' function should be called in here,
it's mixing functionality and you already have a 'prepare_dataset' file which
deals with these sorts of things - this file should only contain code to 
download the data.
"""
if __name__ == "__main__":
    # Path to the dataset on google drive
    dataset_url = (
        "https://drive.google.com/uc?id=1YTeSOebJhD2skONp9lJoO6YK6FhxcOd2"
    )
    # Path to save the downloaded dataset
    output = "data/housesalesprediction.zip"
    # Download dataset from google drive
    download_dataset(dataset_url, output)
    # Preprocess dataset to set "date" column as index
    preprocess_dataset(output)
