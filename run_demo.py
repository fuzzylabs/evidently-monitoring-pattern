"""Run drift/no-drift demo."""
import os
import time
import argparse

from setup_logger import logging
from docker_utils import check_docker_installation, run_docker_compose


def check_dataset(dataset_path: str):
    """Check if dataset has been downloaded and prepared.

    Args:
        dataset_path (str) : Path to the toy dataset

    """
    if not os.path.exists(dataset_path):
        logging.error(f"Dataset do not exist in path: {dataset_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Script for running drift/no-drift demo"
    )
    parser.add_argument(
        "-d",
        "--drift",
        default=False,
        action="store_true",
    )
    parser.add_argument(
        "-no-d",
        "--no-drift",
        default=True,
        action="store_true",
    )
    args = parser.parse_args()

    dataset_path = "datasets/house_price_random_forest"
    # check if docker compose is installed
    check_docker_installation()
    # check if dataset is present at `dataset_path` path
    check_dataset(dataset_path)
    # run docker compose
    run_docker_compose()
    time.sleep(5)
    # if drift scenario
    if args.drift:
        os.system("python scenarios/drift.py")
    # if no drift scenario
    elif args.no_drift:
        os.system("python scenarios/no_drift.py")
