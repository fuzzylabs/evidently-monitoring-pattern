"""Run drift/no-drift demo."""
import argparse
import os
import time

from config.config import logger
from utils.docker_utils import (
    check_docker_installation,
    run_docker_compose,
    stop_docker_compose,
)


def check_dataset(dataset_path: str):
    """Check if dataset has been downloaded and prepared.

    Args:
        dataset_path (str) : Path to the toy dataset

    Raises:
        FileNotFoundError : If dataset is not found at path `dataset_path`
    """
    if not os.path.exists(dataset_path):
        logger.error(f"Dataset do not exist in path: {dataset_path}")
        raise FileNotFoundError(f"Dataset do not exist in path: {dataset_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Script for running drift/no-drift demo"
    )
    parser.add_argument(
        "-d",
        "--drift",
        default=False,
        action="store_true",
        help="Send drift data for inference to evidently server",
    )
    parser.add_argument(
        "-no-d",
        "--no-drift",
        default=False,
        action="store_true",
        help="Send no drift data for inference to evidently server",
    )
    parser.add_argument(
        "-s",
        "--stop",
        default=False,
        action="store_true",
        help="Stop docker compose and remove container images",
    )
    args = parser.parse_args()
    # Path to directory containing reference and dirft/no-drift datasets
    dataset_path = "datasets/house_price_random_forest"
    # Check if docker compose is installed
    check_docker_installation()
    # Check if dataset is present at `dataset_path` path
    check_dataset(dataset_path)
    # Run docker compose
    run_docker_compose()
    # Wait for command to start
    time.sleep(5)
    try:
        # If drift scenario
        if args.drift:
            logger.info("Sending drifted data")
            os.system("python scenarios/drift.py")
        # If no drift scenario
        elif args.no_drift:
            logger.info("Sending non drifted data")
            os.system("python scenarios/no_drift.py")
        logger.info("Visit http://localhost:3000/ for Grafana dashboard")
    except KeyboardInterrupt:
        logger.info("Interrupt detected.")
    # Stop docker compose
    if args.stop:
        logger.info(
            "Stopping all services running in docker compose and removing container images"
        )
        stop_docker_compose()
