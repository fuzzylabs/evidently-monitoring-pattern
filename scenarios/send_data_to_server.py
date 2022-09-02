import os
import logging
import argparse
import json
import time
from typing import Dict

import numpy as np
import pandas as pd
import requests


def setup_logger():
    logging.basicConfig(
        level=logging.INFO, 
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[
            logging.StreamHandler()
        ]
    )

# The encoder converts NumPy types in source data to JSON-compatible types
class NumpyEncoder(json.JSONEncoder):
    '''
    Inherit the class json.JSONEncoder and then implement the default method.

    If object type contains types that are not JSON serializable, return object as JSON compatible types.
    '''
    def default(self, obj):
        if isinstance(obj, np.void):
            return None

        if isinstance(obj, (np.generic, np.bool_)):
            return obj.item()

        if isinstance(obj, np.ndarray):
            return obj.tolist()

        return obj


def send_data_row(dataset_name: str, data: Dict, host: str, port: str) -> None:
    '''
    Parameters:
    dataset_name (str): Name of the datasets within the datasets folder.
    data (Dict): Data of the dataset with name specified with the parameter "dataset_name".
    host (str): The host address of where the data should send to.
    port (str): The port of the host.

    Make a post request to the host with production data 
    '''
    logging.info(f"Sending a data item for {dataset_name}")
    try:
        response = requests.post(
            f"http://{host}:{port}/iterate/{dataset_name}",
            data=json.dumps([data], cls=NumpyEncoder),
            headers={"content-type": "application/json"},
        )

        if response.status_code == 200:
            print(f"Success.")

        else:
            print(
                f"Got an error code {response.status_code} for the data chunk. "
                f"Reason: {response.reason}, error text: {response.text}"
            )

    except requests.exceptions.ConnectionError as error:
        logging.error(f"Cannot reach a metrics application, error: {error}, data: {data}")


def main(sleep_timeout: int, host: str, port: str) -> None:
    '''
    Parameters:
    sleep_timeout (int): The period of time to wait before sending another row of data.
    host (str): The host address of where the data should send to.
    port (str): The port of the host.

    Load all the datasets stores in the datasets folder and iteravitely send each rows of data from each dataset using the send_data_row function.
    '''
    datasets_path = "datasets"

    if not os.path.exists(datasets_path):
        exit("Cannot find datasets")

    logging.info(f"Get production data from {datasets_path} and send the data to monitoring service each {args.timeout} seconds")

    datasets = {}
    max_index = 0

    for dataset_name in os.listdir(datasets_path):
        production_data_path = os.path.join(datasets_path, dataset_name, "production.csv")
        new_data = pd.read_csv(production_data_path)
        datasets[dataset_name] = new_data
        max_index = max(max_index, new_data.shape[0])

    for idx in range(0, max_index):
        for dataset_name, dataset in datasets.items():
            dataset_size = dataset.shape[0]
            data = dataset.iloc[idx % dataset_size].to_dict()
            send_data_row(dataset_name, data, host, port)

        logging.info(f"Wait {sleep_timeout} seconds till the next try.")
        time.sleep(sleep_timeout)


if __name__ == "__main__":
    setup_logger()

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
        help="Server host address"
    )
    parser.add_argument(
        "-p",
        "--port",
        type=str,
        default="5000",
        help="Port of host"
    )

    args = parser.parse_args()
    main(args.timeout, args.host, args.port)