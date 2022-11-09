"""Download and preprocess downloaded dataset."""
from config.config import logger
from data.get_data import download_dataset, preprocess_dataset

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
