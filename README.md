# Introduction

This repo is a complete demo of real-time data monitoring using Evidently. Using a Random Forest Regressor to predict house prices and simulate data drift by sending drifted feature(s) to the model. Evidently calculates the metrics for data drift, send them to Prometheus and demonstrate these on a pre-built Grafana dashboard.

# Contents

- [Outline](#outline)
- [Running locally](#running-locally)
- [How does the demo work?](#how-does-the-demo-work)

# Outline

<!-- TODO: add detail to this description -->

Within the repo, you will find:

* [`data`](data): contains two scripts. Running the `get_data.py` will automatically download a Kaggle house sale prices dataset for model training and data monitoring (drift monitoring); the dataset is saved to this directory. The `generate_dataset_for_demo.py` script will split the house sale prices dataset into a production and a reference dataset which will be saved to a new directory named `datasets`.

    NOTE: The Kaggle dataset has been uploaded to Google Drive for easy access.
* [`pipeline`](pipeline): a model training script which will use the reference data to create and train a Random Forest Regressor model.
* [`inference_server`](model_server): a model server that exposes our house price model through a REST API.
* [`monitoring_server`](monitoring_server): an Evidently model monitoring service which collects inputs and predictions from the model and computes metrics such as data drift.
* [`scenarios`](scenarios): Two scripts to simulate different scenarios. A scenario where there is no drift in the inputs and a scenario which the input data contains drifted data.
* [`dashboards`](dashboards): a data drift monitoring dashboard which uses Prometheus and Grafana to visualise Evidently's monitoring metrics in real-time.
* A [`run_demo_no_drift.py`](run_demo_no_drift.py) script to run the demo **with no data drift** using docker compose.
* A [`run_demo_drift.py`](run_demo_drift.py) script to run the demo **with data drift** using docker compose.
* A docker-compose file to run the whole thing.

# Running locally

## Pre-requisites

You'll need Python 3, and Docker and Docker Compose V2.

## Getting started

1. **Download and install [Docker](https://www.docker.com/) if you don't have it.**

2. **Clone this repo:**
```bash
git clone git@github.com:fuzzylabs/evidently-monitoring-demo.git
```

3. **Go to the demo directory:**
```bash
cd evidently-monitoring-demo
```

4. **Create a new Python virtual environment and activate it.** For Linux/MacOS users:

```bash
python3 -m venv demoenv
source demoenv/bin/activate
pip install -r requirements.txt
```

## Jupyter Notebook or Terminal

From this point, you have the option to continue the demo by following the instructions below or you continue this demo with [`demo.ipynb`](demo.ipynb) (included in this repo) using Jupyter Notebook. The notebook will provide an breif explaination as we go through each steps. Alternatively, you can check out [**How does the demo work?**](#how-does-the-demo-work) to see how each individual component works with each other and how are the datasets generated.

## Download and prepare data for the model

1. **Run the `download_dataset.py` script**: <a name="step2"></a>

```bash
python download_dataset.py
```

This will download and save the data from Google drive.

3. **Split the dataset into production and reference**:

```bash
python prepare_dataset.py
```

This will split the dataset into 1 reference and 2 production datasets, 1 with drifted data and 1 without.

## Training the Random Forest Regressor

1. **Run the `train_model.py` script to train the model**:

```bash
python train_model.py
```
- Once the model is trained, it will be saved as `model.pkl` inside the `models` folder.

## Running the demo

**At the moment, there are two scenarios we can simulate by running the demo:**

- **Scenario 1:** No data drift

```bash
python run_demo.py --no-drift
```

OR

- **Scenario 2:** Data drift

```bash
python run_demo.py --drift
```

Once docker compose is running, the demo will start sending data to the inference server for price prediction which will then be monitored by the Evidently metric server.

The metric server will receive the price prediction along with the feature(s) (model inputs) used for the prediction. The features are used to monitor data drift by Evidently using the data drift monitor.

The metrics produced by Evidently will be logged to Prometheus's database which will be available at port 9090. To access Prometheus web interface, go to your browser and open: http://localhost:9090/

To visualise these metrics, Grafana is connected to Prometheus's database to collect data for the dashboard. Grafana will be available at port 3030. To access Grafana web interface, go to your browser and open: http://localhost:3000/ . If this is your **first time using Grafana**, you will be asked to enter a username and password, by default, both the username and password is "admin".

To stop the demo, press ctrl+c and shut down docker compose by running the following command:

```bash
python run_demo.py --stop
```

# How does the demo work?

![Flow](images/Monitoring_Flow_Chart.png)

The demo is comprised of 5 core components:

- Scenario scripts: within the [`scenarios`](scenarios) folder, it contains two scripts namely [`drift.py`](scenarios/drift.py) and [`no_drift.py`](scenarios/no_drift.py). Both scripts send production data to the model server for price prediction. The difference between the two is that one would send data from the `production_no_drift.csv` and the other would send data from the `production_with_drift.csv` which contains drifted data. The [How are the data generated?](#how-are-the-data-generated) section will explain how are the two production csv generated.

- The inference server: this is a model server that will return a price prediction when a request is sent to the server. The request would consists of the features of a house such as the number of bedrooms, etc... After a prediction is made by the model, the server would send the predictions along with the features to the metric server.

- The Evidently metric server: this is the Evidently metrics server which will monitor both the inputs and outputs of the inference server to calculate the metrics for data drift.

- Prometheus: once the Evidently monitors have produced some metrics, they will be logged into Prometheus's database as time series data.

- Grafana: this is what we can use to visualise the metrics produced by Evidently in real time. A pre-built dashboard for visualising data drift is include in the [`dashboards`](dashboards) directory.

## How are the data generated?

To monitor data drift or outliers, etc., Evidently requires at least two datasets to perform comparison. The house price data downloaded from Google Drive is split into a reference and a production dataset. The reference dataset is used as the baseline data and for training the model. The second dataset is the current production data which will be used to compared against the reference dataset to identify data drift. Production datasets do not include the price column as the price will be predicted by the regression model.

The original dataset downloaded contains 20 features. To make this demo simple and easy to understand, we are only going to select 2 features from the original dataset.

The [`prepare_dataset.py`](data/generate_dataset_for_demo.py) scripts will create two scenarios of production data, 1 with data drift, 1 without and 1 reference dataset. These will be stored under the `datasets` folder.

For the no data drift production dataset, the number of bedrooms and the condition features for each row of data is generated using the same distribution as the reference dataset to ensure that no data drift will be detected.

For the data drift dataset, both the number of bedrooms and the condition features are generated using a skewed distribution of the the reference's dataset. E.g. At Looking at the reference dataset, if 7 bedrooms has the lowest probability distribution, then the `generate_dataset_for_demo.py` script will generate a value of 7 to simulate data drift.

Once the datasets are generated, the Random Forest Regressor is trained using the reference dataset.

## Histogram visualisation

Distribution comparison between the reference datasets and the **non-drifted** production dataset:

![NoDriftHistogram](images/No_Drift_Histogram.png)

Distribution comparison between the reference datasets and the **drifted** production dataset:

![DriftHistogram](images/Drift_Histogram.png)

## The dashboards

![Drift](images/No_Drift.png)

- When the no data drift scenario is running, the Grafana's dashboard should show no data drift is detected. However, as there are some randomness in dataset generation, it is possible to see data drift every once a while.

![Drift](images/Data_Drift.png)

- When the data drift scenario is running, Grafana's dashboard should show data drift at a relatively constant time frame, e.g. no data drift for 10 seconds -> data drift detected for 5 seconds -> no data drift for 10 seconds etc...
