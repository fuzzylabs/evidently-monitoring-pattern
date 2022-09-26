# Introduction

This repo is a complete demo of real-time model monitoring using Evidently. Using a simple ML model which predicts house prices, we simulate model drift and outliers, and demonstrate these on a dashboard.

# Outline

<!-- TODO: add detail to this description -->

Within the repo, you will find:

* `data`: contains a script to download a training dataset; the data is also saved to this directory. A script to generate the production and reference dataset for evidently to monitor data drift and outliers. The reference and production dataset will be saved to a new directory named `datasets`.
* `pipeline`: a model training script which will use the data to create and train a simple model.
* `inference_server`: a model server that exposes our house price model through a REST API.
* `monitoring_server`: an Evidently model monitoring service which collects inputs and predictions from the model and computes metrics such as drift.
* `scenarios`: some scripts that simulate different scenarios: e.g. model drift vs normal (no drift) input.
* A monitoring dashboard which uses Prometheus and Grafana to visualise in real-time the monitoring metrics.
* A Docker Compose file to run the whole thing.

# Running locally

## Pre-requisites

You'll need Python 3, and Docker plus Docker-compose.

## Getting started

1. **Create a new Python virtual environment and activate it**. For Linux/MacOS:

```bash
python3 -m venv env
source env/bin/activate 
pip install -r requirements.txt
```

## Download data and prepare for the model

1. **Get and set up Kaggle API token**

- Go to [Kaggle](https://www.kaggle.com) to log in or create an account.
- Get into your account settings page.
- Under the API section, click on `create a new API token`.
- This will prompt you to download the `.json` file into your system. Open the file, and copy the username and key.
- Export your Kaggle username and token to the environment

```bash
export set KAGGLE_USERNAME=<your-kaggle-username>
export set KAGGLE_KEY=<your-kaggle-api-key>
```

2. **Run the `get_data.py` script**:

```bash
python data/get_data.py
```

3. **Split the dataset into production and reference**:

```bash
python data/generate_dataset_for_demo.py
```

- To monitor data drift or outliers, etc.., two datasets are required to perform comparison. The house price data downloaded from Kaggle is split into a reference and a production dataset. The reference dataset is used as the baseline data and for training the model. The second dataset is the current production data which will be used to compared against the reference dataset to identify data drift and detect outliers. The production dataset does not include the price column as the price will be predicted by the regression model. The scripts will create two scenarios of production data, one with data drift and one without.

## Train the house prices model

1. **Run the `train.py` script to train the model**:

```bash
python pipeline/train.py
```
- Once the model is trained, it will be saved as `model.pkl` inside the `models` folder.

## Running the demo

**At the moment, there are two scenarios we can simulate by running the demo:**

- **Scenario 1: No data drift**

```bash
python run_demo_no_drift.py
```

OR

- **Scenario 2: Data drift**

```bash
python run_demo_drift.py
```

Once docker compose is running, the demo will start sending data to the inference server for price prediction which will then be monitor by the Evidently metric server.

The metric server will receive the price prediction along with the features (model inputs) used for the prediction. The features are used to monitor data drift by Evidently using the data drift monitor.

The metrics produced by Evidently will be logged to Promethus's database which will be available at port 9090. To access Prometheus web interface, go to your browser and open: http://localhost:9090/

To visualise these metrics, Grafana is connected to Promethus's database to collect data for the dashboard. Grafana will be available at port 3030. To access Grafana web interface, go to your browser and open: http://localhost:3000/

To stop the demo, press ctrl+c and shut down docker compose by running the following command:

```bash
docker compose down
```

## Components of the demo

- Training the regression model: The model is trained using the reference dataset which is split from the original dataset downloaded from Kaggle.

- The inference server: This is a model server that will return a price prediction when a request is sent to the server. The request would consists of the features of a house such as the number of bedrooms, etc... After a prediction is made by the model, the server would send the predictions along with the features to the metric server.

- The metric server: This is the Evidently metrics server which will monitor the predictions output by the inferenece server to detect data drift and outliers.

- Grafana & Prometheus: Once these metrics has been produced by Evidently, they will be logged to the Promethus's database and can be visualised using Grafana.

- When the no data drift scneario is running, the Grafana's dashboard should show no data drift is detected. However, as there are some randomness in dataset generation, it is normal to see data drift every once a while.

- When the data drift scneratio is running, Grafana's dashboard should show data drift at a constant time frame, e.g. no data drfit for 10 seconds -> data drift detected for 5 seconds -> no data drift for 10 seconds etc...