"""Download and generate reference and production dataset with 2 scenarios (drift/no-drift) for data monitoring."""
import logging
import os
import zipfile

import pandas as pd

import gdown

from .prob_distribution import ProbDistribution


def laod_data(dataset_path: str, features: list, no_rows: int) -> pd.DataFrame:
    """Loads the dataset from the `dataset_path`,  select the `features` and number of rows upto `no_rows`.

    Args:
        dataset_path (str): the dataset path
        features (list): a list of features to use from the orignal dataset
        no_rows (int): number of rows to use

    Returns:
        pd.DataFrame: a pandas dataframe containing the dataset
    """
    df = pd.read_csv(dataset_path)
    logging.info(
        f"Using {len(features)} features from total {len(df.columns)} features of the dataset"
    )
    # select specific features
    df = df[features]
    logging.info(
        f"Using first {no_rows} rows from total {len(df)} rows of the dataset"
    )
    # select specific rows
    df = df[0:no_rows]
    return df


def compute_dist(feature: pd.Series) -> dict:
    """This function compute the distribution of a feature from the dataset.

    Args:
        feature (pd.Series): the feature to compute distribution for

    Returns:
        dict: the distribution
    """
    # Converting a 1D pandas data structure to a list
    feature_list = feature.to_list()
    # Count the number of values in the list
    feature_count = len(feature_list)

    # A dictonary to store the probability distribution
    dist = {}

    # Compute the occurance of each distinct value in the list
    for val in feature_list:
        if val in dist:
            dist[val] += 1
        else:
            dist[val] = 1

    # Compute the probabilty distribution for each distinct value store in dictonary
    dist = {key: val / feature_count for key, val in dist.items()}

    return dist


def generate_reference_data(
    dataset_path: str, features: list, no_rows: int, save_dir: str
) -> pd.DataFrame:
    """Generate reference data used by Evidently as reference.

    Args:
        dataset_path (str): the dataset path
        features (list): a list of features to use from the orignal dataset
        no_rows (int): number of rows to use
        save_dir (str): path to save generated dataset

    Returns:
        pd.DataFrame: a pandas dataframe containing the dataset
    """
    # load data with specific features and upto specified number of rows
    reference_df = laod_data(
        dataset_path=dataset_path, features=features, no_rows=no_rows
    )
    # save reference dataset
    save_path = os.path.join(save_dir, "reference.csv")
    reference_df.to_csv(save_path, index=False)
    logging.info(f"Saved reference data at path: {save_path}")
    return reference_df


def generate_production_data(reference_df: pd.DataFrame) -> pd.DataFrame:
    """Generate production data which from which drift and no-drift dataset will be derived.

    This function uses a copy of reference dataset and drops price column to create production dataset.

    Args:
        reference_df (pd.DataFrame): the referencce dataset

    Returns:
        pd.DataFrame: the production dataset
    """
    # use a copy of reference dataset
    production_df = reference_df.copy()
    # drop the price column
    production_df = production_df.drop(["price"], axis=1)
    return production_df


def create_data_simulator(reference_df: pd.DataFrame) -> tuple:
    """Create a data simulator for bedroom and condition column using reference dataset.

    This function uses `ProbDistribution` class to generated skew dataset for drift scenario and
    non-skewed dataset for no-drift scenario

    Args:
        reference_df (pd.DataFrame): the referencce dataset

    Returns:
        tuple : A pair of data generators for bedroom and condition column
    """
    # Compute the probability distribution of the bedrooms feature
    bedrooms_dist = compute_dist(reference_df["bedrooms"])
    # Stores the orginal probability distribution and the skewd distribution
    bedrooms_generator = ProbDistribution(bedrooms_dist)

    # Similar to the bedrooms feature above
    condition_dist = compute_dist(reference_df["condition"])
    condition_generator = ProbDistribution(condition_dist)
    return bedrooms_generator, condition_generator


def generate_production_no_drift_data(
    production_df: pd.DataFrame,
    bedrooms_generator: ProbDistribution,
    condition_generator: ProbDistribution,
    save_dir: str,
) -> pd.DataFrame:
    """Generate drift dataset from production dataset.

    This function skews the distribution for some entries in dataset for columns ["bedroom", "condition"].
    The skew is done by ....

    Args:
        production_df (pd.DataFrame): production dataset
        bedrooms_generator (ProbDistribution): Instance of ProbDistribution class to generate data for bedroom column
        condition_generator (ProbDistribution): Instance of ProbDistribution class to generate data for condition column
        save_dir (str): path to save generated dataset

    Returns:
        pd.DataFrame: production dataset
    """
    # fmt: off
    # use same distribution for both columns: ["bedroom", "condition"] as production dataset to generate new values
    for index, row in production_df.iterrows():
        production_df.at[index, "bedrooms"] = bedrooms_generator.generate_val(shuffle_dist=False)
        production_df.at[index, "condition"] = condition_generator.generate_val(shuffle_dist=False)
    # fmt: on
    save_path = os.path.join(save_dir, "production_no_drift.csv")
    production_df.to_csv(save_path, index=False)
    logging.info(f"Saved production data with no drift at path: {save_path}")
    return production_df


def generate_production_with_drift_data(
    production_df: pd.DataFrame,
    bedrooms_generator: ProbDistribution,
    condition_generator: ProbDistribution,
    save_dir: str,
) -> pd.DataFrame:
    """Generate drift dataset from production dataset.

    This function skews the distribution for some entries in dataset for columns ["bedroom", "condition"].
    The skew is done by ....

    Args:
        production_df (pd.DataFrame): production dataset
        bedrooms_generator (ProbDistribution): Instance of ProbDistribution class to generate data for bedroom column
        condition_generator (ProbDistribution): Instance of ProbDistribution class to generate data for condition column
        save_dir (str): path to save generated dataset

    Returns:
        pd.DataFrame: production dataset
    """
    # use skewed distribution for both columns: ["bedroom", "condition"] as production dataset
    # to generate new values at certain indexes [index >= 10 and index < 15]
    counter = 0
    # fmt: off
    for index, row in production_df.iterrows():
        if counter < 10:
            # shuffle_dist=False mean the original distribution is used
            production_df.at[index, "bedrooms"] = bedrooms_generator.generate_val(shuffle_dist=False)
            production_df.at[index, "condition"] = condition_generator.generate_val(shuffle_dist=False)
            counter += 1
        elif counter >= 10 and counter < 15:
            # shuffle_dist=True mean the skewed distribution is used
            production_df.at[index, "bedrooms"] = bedrooms_generator.generate_val(shuffle_dist=True)
            production_df.at[index, "condition"] = condition_generator.generate_val(shuffle_dist=True)
            counter += 1
        elif counter == 15:
            counter = 0
    # fmt: on
    save_path = os.path.join(save_dir, "production_with_drift.csv")
    production_df.to_csv(save_path, index=False)
    logging.info(f"Saved production data with drift at path: {save_path}")
    return production_df

def download_dataset(url: str, output: str) -> None:
    """Download the dataset using the gdown library.

    Args:
        url (str): the link to the dataset on google drive
        output (str): the name of the file and the output path
    """
    logging.info("Downloading house pricing data from google drive")
    # Download the dataset using gdown library
    gdown.download(url, output, quiet=False)


def preprocess_dataset(dataset_path: str) -> None:
    """Unzip dataset and set the date column as index and saves the dataset as csv.

    Args:
        dataset_path (str): the path to the downloaded zip dataset file
    """
    # Extract dataset
    logging.info("Extracting dataset zip file")
    with zipfile.ZipFile(dataset_path, "r") as zip_ref:
        zip_ref.extractall("data")

    logging.info(f"Downloaded dataset at path: {dataset_path}")
    logging.info("Processing data...")

    root_dir = os.path.dirname(dataset_path)
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