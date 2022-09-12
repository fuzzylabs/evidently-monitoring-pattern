import hashlib

import os
import logging
from dataclasses import dataclass # Automatically adding generated special methods such as __init__() and __repr__().
from datetime import datetime, timedelta

from typing import Dict # NOTE Type hinting is deprecate after python 3.9, used for init variable without specifying its values
from typing import List
from typing import Optional

import flask
import prometheus_client
from flask import Flask, request
import pandas as pd
import yaml
from werkzeug.middleware.dispatcher import DispatcherMiddleware

from evidently.pipeline.column_mapping import ColumnMapping
from evidently.model_monitoring import DataDriftMonitor
from evidently.model_monitoring import RegressionPerformanceMonitor
from evidently.model_monitoring import ModelMonitoring # Specify monitors to use and return specific metrics of monitors

from prometheus_client import Gauge

from evidently.runner.loader import DataLoader # Basically pd.read_csv()
from evidently.runner.loader import DataOptions # Set a column for date, header and separtor etc...


app = Flask(__name__)


def setup_logger() -> None:
    logging.basicConfig(
        level=logging.INFO, 
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[
            logging.StreamHandler()
        ]
    )


# Add prometheus wsgi middleware to route /metrics requests
app.wsgi_app = DispatcherMiddleware(app.wsgi_app, {"/metrics": prometheus_client.make_wsgi_app()})


@dataclass
class MonitoringServiceOptions:
    datasets_path: str
    min_reference_size: int
    use_reference: bool
    moving_reference: bool
    window_size: int
    calculation_period_sec: int


@dataclass
class LoadedDataset: # Stores the dataset config once loaded
    name: str
    references: pd.DataFrame
    monitors: List[str]
    column_mapping: ColumnMapping


EVIDENTLY_MONITORS_MAPPING = {
    "data_drift": DataDriftMonitor
}


class MonitoringService:
    '''
    This class defines the the monitoring service object which is use to listen to data sent by the inference server.
    '''
    #datasets: List[str] # A list of dataset to monitor, dont think its needed
    metric: Dict[str, prometheus_client.Gauge]
    #last_run: Optional[datetime.datetime] # ?? Not needed?
    reference: Dict[str, pd.DataFrame] #
    current: Dict[str, Optional[pd.DataFrame]] #
    monitoring: Dict[str, ModelMonitoring] #
    #calculation_period_sec: float #
    window_size: int #

    def __init__(self, datasets: Dict[str, LoadedDataset], window_size: int, calculation_period_sec: float):
        self.reference = {}
        self.current = {}
        self.monitoring = {}
        self.column_mapping = {}
        self.window_size = window_size
        self.calculation_period_sec = calculation_period_sec

        for dataset_info in datasets.values():
            bedroom_only = dataset_info.references[['bedrooms']]
            # self.reference[dataset_info.name] = dataset_info.references
            self.reference[dataset_info.name] = bedroom_only
            self.monitoring[dataset_info.name] = ModelMonitoring(
                monitors = [EVIDENTLY_MONITORS_MAPPING[monitor]() for monitor in dataset_info.monitors]
            )
            self.column_mapping[dataset_info.name] = dataset_info.column_mapping

        # print(self.reference['house_price_random_forest']['bedrooms'])
        
        self.metrics = {}
        self.next_run_time = {}
        self.hash = hashlib.sha256(pd.util.hash_pandas_object(self.reference["house_price_random_forest"]).values).hexdigest()
        self.hash_metric = prometheus_client.Gauge("Evidently:reference_dataset_hash", "", labelnames=["hash"])

    def iterate(self, dataset_name: str, new_rows: pd.DataFrame):
        # new_rows = new_rows.drop(['price'], axis = 1) # Drop price column as we only care about features drift for now.
        new_rows = new_rows[['bedrooms']]
        print(new_rows)
        window_size = self.window_size

        if dataset_name in self.current: # Check if have recevied data before
            current_data = pd.concat([self.current[dataset_name], new_rows], ignore_index = True)
        else: # If first time receive data
            current_data = new_rows

        current_size = current_data.shape[0]

        if current_size > self.window_size: # If there are more rows in current data then specified window size
            current_data.drop(index = list(range(0, current_size - self.window_size)), inplace=True)
            current_data.reset_index(drop = True, inplace = True)
        
        self.current[dataset_name] = current_data

        #print(self.current[dataset_name])
        
        if current_size < window_size:
            logging.info(f"Currenlty has less data than set window size: {current_size} of {window_size}, waiting for more data")
            return
        
        next_run_time = self.next_run_time.get(dataset_name)

        if next_run_time is not None and next_run_time > datetime.now():
            logging.info(f"Next data request for dataset {dataset_name} in {next_run_time}")
            return
        
        self.next_run_time[dataset_name] = datetime.now() + timedelta(
            seconds = self.calculation_period_sec               
        )

        self.monitoring[dataset_name].execute(
            self.reference[dataset_name], current_data, self.column_mapping[dataset_name]
        ) # The exectue method inherits from the Pipeline class

        self.hash_metric.labels(hash=self.hash).set(1)

        for metric, value, labels in self.monitoring[dataset_name].metrics():
            metric_key = f"Evidently:{metric.name}"
            #logging.info(metric_key)
            found = self.metrics.get(metric_key)

            if not labels:
                labels = {}
            
            labels["dataset_name"] = dataset_name
            
            if isinstance(value, str): # Check if the value variable is a string
                continue

            if found is None:
                found = Gauge(metric_key, "", list(sorted(labels.keys())))
                self.metrics[metric_key] = found
            
            try:
                found.labels(**labels).set(value)

            except ValueError as error:
                # ignore errors sending other metrics
                logging.error("Value error for metric %s, error: ", metric_key, error)
            
            # print(metric.name)
            # print(value)
            # print(labels)

            # if metric.name == "data_drift:dataset_drift" and value == True:
            #     print("Data drift detected")

            ## STOPPED HERE LAST TIME, CONTINUE FROM HERE NEXT TIME


SERVICE: Optional[MonitoringService] = None


@app.before_first_request
def configure_service() -> None:
    global SERVICE
    config_file_path = "monitoring_server/config.yaml" # This get path of the config.yaml file

    # Check if a config file exists?
    if not os.path.exists(config_file_path): # Will return false if not exists
        logging.error(f"Config file does not exists in path: {config_file_path}")
        exit("Failed to config metrics service")
    
    # If config file found
    with open(config_file_path, "rb") as config_file:
        configs = yaml.safe_load(config_file) # A dict
        #print(configs['service'])
    
    monitoring_service_options =  MonitoringServiceOptions(**configs["service"]) # Init with config file, ** = dict unpack

    # Load and set up reference dataset
    datasets_path = monitoring_service_options.datasets_path
    data_loader = DataLoader()

    datasets = {}

    for dataset_name in os.listdir(datasets_path):
        logging.info(f"Loading reference data from '{dataset_name}'")
        reference_data_path = os.path.join(datasets_path, dataset_name, "reference.csv")
        
        if dataset_name in configs["datasets"]:
            dataset_configs = configs["datasets"][dataset_name]
            reference_data = data_loader.load(
                reference_data_path,
                DataOptions(
                    date_column = dataset_configs["column_mapping"].get("datetime", None), # If no date_column specified, used none
                    separator = dataset_configs["data_format"]["separator"],
                    header = dataset_configs["data_format"]["header"]
                ),            
            )

            datasets[dataset_name] = LoadedDataset(
                name = dataset_name,
                references = reference_data,
                monitors = dataset_configs["monitors"],
                column_mapping = ColumnMapping(**dataset_configs["column_mapping"])
            )

            logging.info(f"Reference data of {dataset_name} dataset is loaded, containing {len(reference_data)} rows.")

        else:
            logging.error(f"{dataset_name} is not configured within the config.yaml file")

    SERVICE = MonitoringService(datasets = datasets, window_size = monitoring_service_options.window_size, calculation_period_sec = monitoring_service_options.calculation_period_sec)


@app.route('/')
def home() -> None:
    return "Hello world from the metric server."


@app.route("/iterate/<dataset>", methods=["POST"])
def iterate(dataset: str) -> None:
    '''
    Get the data from the inference server and call the iterate method from a MonitoringService object.
    '''
    item = request.json

    global SERVICE
    if SERVICE is None:
        return "Internal Server Error: service not found", 500

    SERVICE.iterate(dataset_name=dataset, new_rows=pd.DataFrame.from_dict(item))
    return "ok"


if __name__ == "__main__":
    setup_logger()
    app.run(host = "0.0.0.0", port = "8085", debug = True)