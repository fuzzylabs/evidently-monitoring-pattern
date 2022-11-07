"""Generate reference and production dataset for data monitoring."""
import logging
import os

import numpy as np
import pandas as pd

from setup_logger import setup_logger


def laod_data(dataset_path: str, features: list, no_rows: int) -> pd.DataFrame:
    """Loads the dataset in the specified path and select the features specified.

    Arguments:
        dataset_path (str): the dataset path
        features (list): a list of features to use from the orignal dataset
        no_rows (int): number of rows to use

    Returns:
        pd.DataFrame: a pandas dataframe containing the dataset
    """
    df = pd.read_csv(dataset_path)
    df = df[features]
    df = df[0:no_rows]

    logging.info("Kaggle dataset loaded")
    return df


class ProbDistribution:
    """Store and generate values based on the skewed distribution of a feature."""

    def __init__(self, dist: dict) -> None:
        """Initialise class variables.

        Arguments:
            dist (dict): distribution of a feature e.g. {Apple: 0.3, Banana: 0.5, Pear: 0.2}
        """
        self.no_items = list(dist.keys())
        self.items_dist = list(dist.values())
        self.shuffled_dist = self.skew_dist(self.items_dist)

    def skew_dist(self, items_dist: list) -> dict:
        """Skew the distrubtion.

        Arguments:
            items_dist (list): the distribution probability

        Returns:
            dict: a skew distribution e.g. {Apple: 0.0, Banana: 0.0, Pear: 1.0}
        """
        min_dist_idx = items_dist.index(min(items_dist))
        shuffled_dist = [0 for x in range(len(items_dist))]
        shuffled_dist[min_dist_idx] = 1.0

        return shuffled_dist

    def generate_val(self, shuffle_dist: bool = False) -> float:
        """Generate a value base on the probability distribution.

        Arguments:
            shuffle_dist (bool): whether to use skewed distribution or not

        Returns:
            float: a value based on the probability distribution used
        """
        if not shuffle_dist:
            val = np.random.choice(self.no_items, p=self.items_dist)
        else:
            val = np.random.choice(self.no_items, p=self.shuffled_dist)

        return val


def compute_dist(feature: pd.Series) -> dict:
    """This function compute the distribution of a feature from the dataset.

    Arguments:
        feature (pd.Series): the feature to compute distribution for

    Returns:
        dict: the distribution
    """
    feature_list = feature.to_list()
    feature_count = len(feature_list)

    dist = {}

    for val in feature_list:
        if val in dist:
            dist[val] += 1
        else:
            dist[val] = 1

    dist = {key: val / feature_count for key, val in dist.items()}

    return dist


if __name__ == "__main__":
    setup_logger()

    dataset_path = "data/processed_house_data.csv"
    save_path = "datasets/house_price_random_forest"

    features = [
        "date",
        "bedrooms",
        "bathrooms",
        "sqft_living",
        "sqft_lot",
        "floors",
        "waterfront",
        "view",
        "condition",
        "grade",
        "yr_built",
        "price",
    ]

    reference_df = laod_data(
        dataset_path=dataset_path, features=features, no_rows=1000
    )
    production_df = reference_df.copy()
    production_df = production_df.drop(["price"], axis=1)

    min_beds = reference_df.min(axis=0)["bedrooms"]
    max_beds = reference_df.max(axis=0)["bedrooms"]

    min_condition = reference_df.min(axis=0)["condition"]
    max_condition = reference_df.max(axis=0)["condition"]

    bedrooms_dist = compute_dist(reference_df["bedrooms"])
    bedrooms_generator = ProbDistribution(bedrooms_dist)

    condition_dist = compute_dist(reference_df["condition"])
    condition_generator = ProbDistribution(condition_dist)

    counter = 0

    for index, row in production_df.iterrows():
        if counter < 10:
            production_df.at[
                index, "bedrooms"
            ] = bedrooms_generator.generate_val(shuffle_dist=False)
            production_df.at[
                index, "condition"
            ] = condition_generator.generate_val(shuffle_dist=False)
            counter += 1
        elif counter >= 10 and counter < 15:
            production_df.at[
                index, "bedrooms"
            ] = bedrooms_generator.generate_val(shuffle_dist=True)
            production_df.at[
                index, "condition"
            ] = condition_generator.generate_val(shuffle_dist=True)
            counter += 1
        elif counter == 15:
            counter = 0

    if not os.path.exists(save_path):
        os.makedirs(save_path)

    production_df.to_csv(
        os.path.join(save_path, "production_with_drift.csv"), index=False
    )

    for index, row in production_df.iterrows():
        production_df.at[index, "bedrooms"] = bedrooms_generator.generate_val(
            shuffle_dist=False
        )
        production_df.at[index, "condition"] = condition_generator.generate_val(
            shuffle_dist=False
        )

    production_df.to_csv(
        os.path.join(save_path, "production_no_drift.csv"), index=False
    )

    reference_df.to_csv(os.path.join(save_path, "reference.csv"), index=False)
    logging.info("Reference and 2 scenarios production data are generated")
