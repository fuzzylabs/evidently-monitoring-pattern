"""Prepare dataset to run demo."""
import argparse
import os

from config.config import logger
from pipeline.train import (
    evaluate,
    model_setup,
    prepare_data,
    save_model,
    train,
)
from utils.prepare_data import (
    create_data_simulator,
    download_dataset,
    generate_production_data,
    generate_production_no_drift_data,
    generate_production_with_drift_data,
    generate_reference_data,
    preprocess_dataset,
)


def get_features(is_train) -> list:
    """Return list of features.

    Args:
        is_train (bool): is train use different features if in training mode

    Returns:
        list: List of features
    """
    features = [
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
    ]
    if is_train:
        return features
    else:
        # add date and price column not used in training
        features.extend(["date", "price"])
        return features


def create_dir(path_dir: str):
    """Create a directory if it does not exists.

    Args:
        path_dir (str): Path to directory to create
    """
    if not os.path.exists(path_dir):
        logger.info(f"Creating directory at path: {path_dir}")
        os.makedirs(path_dir)


def download_preprocess_data(
    url: str, output_path: str, output_name: str
) -> str:
    """Download the dataset using the gdown library and preprocess by the date column to the index.

    Args:
        url (str): Path to ggoogle drive shareable link containing dataset
        output_path (str): Path to directory to save dataset
        output_name (str): Name of the dataset

    Returns:
        str: Path to preprocessed data
    """
    create_dir(output_path)
    output_path = os.path.join(output_path, output_name)
    download_dataset(url, output_path)
    return preprocess_dataset(output_path)


def prepare(dataset_path: str, save_dir: str, features: str) -> None:
    """Create reference and 2 production datasets required for running demo

    Args:
        dataset_path (str): path to dataset csv
        save_dir (str): path to save new dataset
        features (str): features to use
    """
    # create a direectory to save new datasets
    create_dir(save_dir)
    logger.info("Generating reference data")
    # select first 1000 rows and select features to get reference dataset from original dataset
    reference_df = generate_reference_data(
        dataset_path=dataset_path,
        features=features,
        no_rows=1000,
        save_dir=save_dir,
    )

    logger.info("Generating production data")
    # get production dataset
    production_df = generate_production_data(reference_df=reference_df)
    # generator data simulator for columns : ["bedroom", "condition"]
    bedrooms_generator, condition_generator = create_data_simulator(
        reference_df=reference_df
    )

    logger.info("Generating production dataset with drift")
    # generate production dataset with drift
    production_df = generate_production_with_drift_data(
        production_df=production_df,
        bedrooms_generator=bedrooms_generator,
        condition_generator=condition_generator,
        save_dir=save_dir,
    )

    logger.info("Generating production dataset with no drift")
    # generate production dataset without drift
    production_df = generate_production_no_drift_data(
        production_df=production_df,
        bedrooms_generator=bedrooms_generator,
        condition_generator=condition_generator,
        save_dir=save_dir,
    )

    logger.info(
        "1 reference and 2 scenarios (drift and no-drift) production datasets are generated"
    )


def training(
    reference_dataset_path: str,
    model_save_path: str,
    features: list,
    test_size: float,
):
    """Train a random forest regression model using the reference dataset.

    Args:
        reference_dataset_path (str): Path to reference data csv
        model_save_path (str): Path to save the trained model
        features (list): List of features to use for training
        test_size (float): split ratio to split dataset into train and test datasets
    """
    save_dir = os.path.dirname(model_save_path)
    create_dir(save_dir)
    # target column
    target = "price"
    x_train, x_test, y_train, y_test = prepare_data(
        data_path=reference_dataset_path,
        features=features,
        target=target,
        test_size=test_size,
    )
    # Create a random forest regressor using sklearn
    model = model_setup()
    # Fit the model
    train(model, x_train, y_train)
    # Evaluate the performance
    evaluate(model, x_test, y_test)
    # Saving the model
    save_model(model, model_save_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Script for running drift/no-drift demo"
    )
    parser.add_argument(
        "-d",
        "--download",
        default=False,
        action="store_true",
        help="Download dataset from google drive",
    )
    parser.add_argument(
        "-p",
        "--prepare",
        default=False,
        action="store_true",
        help="Prepare dataset for model training and monitoring",
    )
    parser.add_argument(
        "-t",
        "--train",
        default=False,
        action="store_true",
        help="Train a regression model using the reference dataset",
    )
    args = parser.parse_args()

    # path to shareable google drive link containing kaggle dataset
    g_drive_url = (
        "https://drive.google.com/uc?id=1YTeSOebJhD2skONp9lJoO6YK6FhxcOd2"
    )
    # path to download dataset
    data_path = "datasets"
    file_name = "housesalesprediction.zip"
    save_dir = f"{data_path}/house_price_random_forest"
    # path to reference data
    reference_dataset_path = f"{save_dir}/reference.csv"
    # path and name of ML model to save
    model_save_path = "models/model.pkl"
    # train test split
    test_size = 0.2

    # use portion of all features
    features = get_features(is_train=False)
    train_features = get_features(is_train=True)

    # download and preprocess dataset
    if args.download:
        logger.info("Downloading dataset from google drive")
        dataset_path = download_preprocess_data(
            g_drive_url, data_path, file_name
        )
    # prepare reference and 2 production datasets required for running demo
    if args.prepare:
        logger.info("Preparing dataset for training and data drift monitoring")
        prepare(dataset_path, save_dir, features)
    # train a random forest regression model for inference server to make requests
    if args.train:
        logger.info("Training the regression model")
        training(
            reference_dataset_path,
            model_save_path,
            train_features,
            test_size,
        )
