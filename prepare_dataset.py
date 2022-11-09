import os
from data.generate_dataset_for_demo import (
    create_data_simulator,
    generate_reference_data,
    generate_production_data,
    generate_production_no_drift_data,
    generate_production_with_drift_data,
)
from config.config import logger


if __name__ == "__main__":
    logger.info("Generating reference data")
    # same path where preprocessed dataset is stored by preprocess_dataset function in download_dataset.py
    dataset_path = "data/processed_house_data.csv"
    # create a new directory to store reference and production datasets with and without drift
    save_dir = "datasets/house_price_random_forest"

    if not os.path.exists(save_dir):
        logger.info(f"Creating directory at path: {save_dir}")
        os.makedirs(save_dir)
    # use portion of all features
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
    # select first 1000 rows and select features to get reference dataset from original dataset
    reference_df = generate_reference_data(
        dataset_path=dataset_path, features=features, no_rows=1000, save_dir=save_dir
    )
    # get production dataset
    production_df = generate_production_data(reference_df=reference_df)
    # generator data simulator for columns : ["bedroom", "condition"]
    bedrooms_generator, condition_generator = create_data_simulator(
        reference_df=reference_df
    )
    # generate production dataset with drift
    production_df = generate_production_with_drift_data(
        production_df=production_df,
        bedrooms_generator=bedrooms_generator,
        condition_generator=condition_generator,
        save_dir=save_dir,
    )
    # generate production dataset without drift
    production_df = generate_production_no_drift_data(
        production_df=production_df,
        bedrooms_generator=bedrooms_generator,
        condition_generator=condition_generator,
        save_dir=save_dir,
    )

    logger.info(
        "Reference and 2 scenarios (drift and no-drift) production data are generated"
    )
