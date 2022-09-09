import numpy as np
import pandas as pd
import math
import os


def split_raw_dataset(raw_data_path: str) -> None:
    '''
    Parameters:
    raw_data_path (str): path of the downloaded dataset from Kaggle.

    Load the dataset using pandas.

    Split raw dataset with a 50/50 ratio. Save the split data as reference and production csv in a folder named "house_price_random_forest" within the datasets folder.
    '''
    raw_dataset = pd.read_csv(raw_data_path)
    n_rows = len(raw_dataset) # Total number of rows in the raw dataset.
    reference_data = raw_dataset[:math.floor(n_rows/2)] # Take the first 50% of the raw data.
    production_data = raw_dataset[math.floor(n_rows/2):] # Take the remaining 50% of the raw data.
    production_data = raw_dataset.drop(["price"], axis = 1) # Drop the price column.
    print(reference_data)
    print(production_data)

    # Set path to store the reference and production csvs.
    datasets_path = "datasets/house_price_random_forest"

    # Check if the path already exists
    if os.path.exists(datasets_path):
        reference_data.to_csv(os.path.join(datasets_path, "reference.csv"), index=False)
        production_data.to_csv(os.path.join(datasets_path, "production.csv"), index=False)
    else:
        # If the datasets folder does not exist, create new folder and store csvs.
        os.makedirs("datasets/house_price_random_forest")
        reference_data.to_csv(os.path.join(datasets_path, "reference.csv"), index=False)
        production_data.to_csv(os.path.join(datasets_path, "production.csv"), index=False)


if __name__ == "__main__":
    raw_data_path = "data/processed_house_data.csv"
    split_raw_dataset(raw_data_path)