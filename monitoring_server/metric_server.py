"""Evidently's monitoring service."""
# fmt: off
import hashlib
import logging
import os
from dataclasses import (  # Automatically adding generated special methods such as __init__() and __repr__().
    dataclass,
)
from datetime import datetime, timedelta

import pandas as pd
import prometheus_client
import yaml
from evidently.model_monitoring import (  # Specify monitors to use and return specific metrics of monitors
    DataDriftMonitor,
    ModelMonitoring,
)
from evidently.options.data_drift import (  # Set data drift options e.g share drift etc..
    DataDriftOptions,
)
from evidently.pipeline.column_mapping import ColumnMapping
from evidently.runner.loader import DataLoader  # Basically pd.read_csv()
from evidently.runner.loader import (  # Set a column for date, header and separtor etc...
    DataOptions,
)
from flask import Flask, request
from prometheus_client import Gauge
from werkzeug.middleware.dispatcher import DispatcherMiddleware

app = Flask(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler()],
)

# Add prometheus wsgi middleware to route /metrics requests
app.wsgi_app = DispatcherMiddleware(
    app.wsgi_app, {"/metrics": prometheus_client.make_wsgi_app()}
)

@dataclass
class MonitoringServiceOptions:
    """Monitoring service option parameters."""

    datasets_path: str
    use_reference: bool
    moving_reference: bool
    window_size: int
    calculation_period_sec: int


@dataclass
class LoadedDataset:
    """Parameters for storing the dataset config once loaded."""

    name: str
    references: pd.DataFrame
    monitors: list[str]
    column_mapping: ColumnMapping


EVIDENTLY_MONITORS_MAPPING = {"data_drift": DataDriftMonitor}


class MonitoringService:
    """This class defines the the monitoring service object which is use to listen to data sent by the inference server."""

    metric: dict[str, prometheus_client.Gauge]
    reference: dict[str, pd.DataFrame]
    current: dict[str, pd.DataFrame | None]
    monitoring: dict[str, ModelMonitoring]
    window_size: int  #

    def __init__(
        self,
        datasets: dict[str, LoadedDataset],
        window_size: int,
        calculation_period_sec: float,
    ) -> None:
        """Initalise the class variables.

        Args:
            datasets (dict): datasets to be monitored
            window_size (int): number of rows to be used for calculation
            calculation_period_sec (float): frequency for calculation
        """
        self.reference = {}
        self.current = {}
        self.monitoring = {}
        self.column_mapping = {}
        self.window_size = window_size
        self.calculation_period_sec = calculation_period_sec
        self.options = DataDriftOptions(drift_share=1)

        for dataset_info in datasets.values():
            features = dataset_info.references[["bedrooms", "condition"]]
            self.reference[dataset_info.name] = features
            self.monitoring[dataset_info.name] = ModelMonitoring(
                monitors=[EVIDENTLY_MONITORS_MAPPING[monitor]() for monitor in dataset_info.monitors],
                options=[self.options],
            )
            self.column_mapping[dataset_info.name] = dataset_info.column_mapping

        self.metrics = {}
        self.next_run_time = {}
        self.hash = hashlib.sha256(pd.util.hash_pandas_object(self.reference["house_price_random_forest"]).values).hexdigest()
        self.hash_metric = prometheus_client.Gauge("Evidently:reference_dataset_hash", "", labelnames=["hash"])
        self.n_feature = 2
        self.n_feature_metric = prometheus_client.Gauge("Evidently:n_features", "", labelnames=["dataset_name"])

    def iterate(self, dataset_name: str, new_rows: pd.DataFrame) -> None:
        """Get a new row of data for monitoring.

        Args:
            dataset_name (str): name of the dataset
            new_rows (pd.DataFrame): a row of data used for inference from the inference server
        """
        # new_rows = new_rows.drop(['price'], axis = 1) # Drop price column if we only care about features drift for now.
        # We only want the bedroom and the condition feature
        new_rows = new_rows[["bedrooms", "condition"]]
        logging.info(new_rows)
        window_size = self.window_size

        if dataset_name in self.current:  # Check if have recevied data before
            current_data = pd.concat([self.current[dataset_name], new_rows], ignore_index=True)
        else:  # If first time receive data
            current_data = new_rows

        current_size = current_data.shape[0]

        if (
            current_size > self.window_size
        ):  # If there are more rows in current data then specified window size
            current_data.drop(index=list(range(0, current_size - self.window_size)), inplace=True,)
            current_data.reset_index(drop=True, inplace=True)

        self.current[dataset_name] = current_data

        if current_size < window_size:
            logging.info(f"Currenlty has less data than set window size: {current_size} of {window_size}, waiting for more data")
            return

        next_run_time = self.next_run_time.get(dataset_name)

        if next_run_time is not None and next_run_time > datetime.now():
            logging.info(f"Next data request for dataset {dataset_name} in {next_run_time}")
            return

        self.next_run_time[dataset_name] = datetime.now() + timedelta(seconds=self.calculation_period_sec)
        # The exectue method inherits from the Pipeline class
        self.monitoring[dataset_name].execute(
            self.reference[dataset_name],
            current_data,
            self.column_mapping[dataset_name],
        )
        self.hash_metric.labels(hash=self.hash).set(1)
        self.n_feature_metric.labels(**{"dataset_name": dataset_name}).set(self.n_feature)

        for metric, value, labels in self.monitoring[dataset_name].metrics():
            metric_key = f"Evidently:{metric.name}"
            found = self.metrics.get(metric_key)

            if not labels:
                labels = {}

            labels["dataset_name"] = dataset_name

            if isinstance(value, str):  # Check if the value variable is a string
                continue

            if found is None:
                found = Gauge(metric_key, "", list(sorted(labels.keys())))
                self.metrics[metric_key] = found

            try:
                found.labels(**labels).set(value)

            except ValueError as error:
                # ignore errors sending other metrics
                logging.error("Value error for metric %s, error: ", metric_key, error)

            if metric.name == "data_drift:dataset_drift" and value is True:
                logging.info("Data drift detected")


SERVICE: MonitoringService | None = None


@app.before_first_request
def configure_service() -> None:
    """Configure evidently's monitoring service on server start."""
    global SERVICE
    config_file_path = "config.yaml"  # This get path of the config.yaml file

    # Check if a config file exists?
    if not os.path.exists(config_file_path):  # Will return false if not exists
        logging.error(f"Config file does not exists in path: {config_file_path}")
        exit("Failed to config metrics service")

    # If config file found
    with open(config_file_path, "rb") as config_file:
        configs = yaml.safe_load(config_file)

    # Init with config file, ** = dict unpack
    monitoring_service_options = MonitoringServiceOptions(**configs["service"])
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
                    # If no date_column specified, used none
                    date_column=dataset_configs["column_mapping"].get("datetime", None),
                    separator=dataset_configs["data_format"]["separator"],
                    header=dataset_configs["data_format"]["header"],
                ),
            )

            datasets[dataset_name] = LoadedDataset(
                name=dataset_name,
                references=reference_data,
                monitors=dataset_configs["monitors"],
                column_mapping=ColumnMapping(**dataset_configs["column_mapping"]),
            )

            logging.info(f"Reference data of {dataset_name} dataset is loaded, containing {len(reference_data)} rows.")

        else:
            logging.error(f"{dataset_name} is not configured within the config.yaml file")

    SERVICE = MonitoringService(
        datasets=datasets,
        window_size=monitoring_service_options.window_size,
        calculation_period_sec=monitoring_service_options.calculation_period_sec,
    )


@app.route("/")
def home() -> None:
    """The message for default route.

    Returns:
        str: the message to return
    """
    return "Hello world from the metric server."


@app.route("/iterate/<dataset>", methods=["POST"])
def iterate(dataset: str) -> str:
    """Get the data from the inference server and call the iterate method from a MonitoringService object.

    Args:
        dataset (str): the dataset to monitor

    Returns:
        str: message to indicate whether the server is running or not.
    """
    item = request.json

    global SERVICE
    if SERVICE is None:
        return "Internal Server Error: service not found", 500

    SERVICE.iterate(dataset_name=dataset, new_rows=pd.DataFrame.from_dict(item))
    return "ok"


if __name__ == "__main__":
    app.run(host="0.0.0.0", port="8085", debug=True)
# fmt: on
