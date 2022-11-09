"""Simulate a scenario with data drift."""
import argparse
import json
import logging
import time

import pandas as pd
import requests


def request_drift_prediction(sleep_timeout: int, model_server_url: str) -> None:
    """Send an instance from the production_with_drift.csv to the inference server after each sleep time out.

    Args:
        sleep_timeout (int): seconds to wait before requesting the next row for inference
        model_server_url (str): the address of the inference server
    """
    dataset_path = (
        "datasets/house_price_random_forest/production_with_drift.csv"
    )

    dataset = pd.read_csv(dataset_path)

    for _, row in dataset.iterrows():
        features = row.to_json(orient="index")
        features = json.loads(features)
        logging.info(f"Sending a request")

        try:
            requests.post(model_server_url, json=features)
            logging.info(
                f"Waiting for {sleep_timeout} seconds till next request."
            )
            time.sleep(sleep_timeout)

        except requests.exceptions.ConnectionError:
            logging.error(f"Cannot reach the inference server.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Script for data sending to Evidently metrics integration demo service"
    )

    parser.add_argument(
        "-t",
        "--timeout",
        type=float,
        default=2,
        help="Sleep timeout between data send tries in seconds.",
    )

    parser.add_argument(
        "-H",
        "--host",
        type=str,
        default="127.0.0.1",
        help="Server host address",
    )

    parser.add_argument(
        "-p", "--port", type=str, default="5050", help="Port of host"
    )

    args = parser.parse_args()

    model_server_url = "http://127.0.0.1:5050/predict"
    request_drift_prediction(args.timeout, model_server_url)
