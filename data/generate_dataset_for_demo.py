import pandas as pd
import numpy as np
import os
import logging
import random


def setup_logger() -> None:
    logging.basicConfig(
        level=logging.INFO, 
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[
            logging.StreamHandler()
        ]
    )


def laod_data(dataset_path: str, features: list, no_rows: int) -> pd.DataFrame:
    '''
    This function loads the dataset in the specified path and uses only the features specified in the list, return the dataset as a pandas dataframe.
    '''
    df = pd.read_csv(dataset_path)
    df = df[features]
    df = df[0:no_rows]

    logging.info("Kaggle dataset loaded")
    return df


class ProbDistribution:
    '''
    This class generate a value using the distribution of the reference data.
    '''
    def __init__(self, dist: dict):
        self.dist = dist
        self.no_bedrooms = list(dist.keys())
        self.bedrooms_dist = list(dist.values())

    def generate_val(self) -> float:
        val = np.random.choice(self.no_bedrooms, p=self.bedrooms_dist)
        return val


def compute_dist(feature: pd.Series) -> dict:
    '''
    This function compute the distribution of a feature from the dataset.
    '''
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

    features = ['date', 'bedrooms', 'bathrooms', 'sqft_living', 'sqft_lot', 'floors', 
                'waterfront', 'view', 'condition', 'grade', 'yr_built', 'price']

    reference_df = laod_data(dataset_path=dataset_path, features=features, no_rows=1000)
    production_df = reference_df.copy()
    production_df = production_df.drop(['price'], axis=1)

    min_beds = reference_df.min(axis=0)["bedrooms"]
    max_beds = reference_df.max(axis=0)["bedrooms"]

    dist = compute_dist(reference_df['bedrooms'])
    feature_generator = ProbDistribution(dist)

    counter = 0

    for index, row in production_df.iterrows():
        if counter < 10:
            production_df.at[index, "bedrooms"] = feature_generator.generate_val()
            counter += 1
        elif counter >= 10 and counter < 15:
            production_df.at[index, "bedrooms"] = random.randint(max_beds, max_beds * 10)
            counter += 1
        elif counter == 15:
            counter = 0

    if not os.path.exists(save_path):
        os.makedirs(save_path)
    
    production_df.to_csv(os.path.join(save_path, "production_with_drift.csv"), index=False)

    for index, row in production_df.iterrows():
        production_df.at[index, "bedrooms"] = feature_generator.generate_val()
            
    production_df.to_csv(os.path.join(save_path, "production_no_drift.csv"), index=False) 

    reference_df.to_csv(os.path.join(save_path, "reference.csv"), index=False)
    logging.info("Reference and 2 scenarios production data are generated")