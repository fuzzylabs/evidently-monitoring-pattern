# Introduction

This repo is a complete demo of real-time model monitoring using Evidently. Using a simple ML model which predicts house prices, we simulate model drift and outliers, and demonstrate these on a dashboard.

# Outline

<!-- TODO: add detail to this description -->

Within the repo, you will find:

* `data`: contains a script to download a training dataset; the data is also saved to this directory.
* `pipeline`: a model training script which will use the data to create a simple model.
* `inference_server`: a model server that exposes our house price model through a REST API.
* `monitoring_server`: An Evidently model monitoring service which collects inputs and predictions from the model and computes metrics such as drift.
* `scenarios`: some scripts that simulate different scenarios: e.g. model drift vs normal input.
* A monitoring dashboard which uses Prometheus and Grafana to visualise in real-time the monitoring metrics.
* A Docker Compose file to run the whole thing.

# Running locally

## Pre-requisites

You'll need Python 3, and Docker plus Docker-compose.

## Getting started

1. **Create a new Python virtual environment and activate it**. For Linux/MacOS:

```bash
python -m venv env
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

- To evaluate data drift or model's perfromance, etc.., two datasets are required to perform comparison. The house price data downloaded from Kaggle is split into a reference and a production dataset. The reference dataset is used as the baseline data and for training the model. The second dataset is the current production data which will be used to compared against the reference dataset to identify data drift or evaluate the regression performance. The production dataset does not include the price column as the price will be predicted by the regression model. The scripts will create two scenarios of production data, one with data drift and one without.

## Train the house prices model

1. **Run the `train.py` script to train the model**:

```bash
python pipeline/train.py
```
- Once the model is trained, it will be saved as `model.pkl` inside the `models` folder.

## Run the model server

1. **Start the server**:

```bash
python model_server/inference_server.py
```

2. **Test it using Curl**

```bash
curl -XPOST http://127.0.0.1:5050/predict -w '\n' -H 'Content-type: application/json' -d '{"bedrooms": 1, "bathrooms": 1, "sqft_living": 50, "sqft_lot": 50, "floors": 1, "waterfront": 0, "view": 0, "condition": 0, "grade": 0, "yr_built": 1960}'
```

The server will return with a price, e.g. `100000`.


## Run the metric server

1. **Start the server**:

```bash
python monitoring_server/metric_server.py
```

2. **Test it by sending data to it**

```bash
python scenarios/send_data_to_server.py
```
- By default, the host is set to "127.0.0.1" and port 5000, this can be changed by passing two arguments using the -H and -p flags:
```bash
python scenarios/send_data_to_server.py -H "127.0.0.1" -p "5000"
```
- For now, the server will return a message saying "Data drift detected" when data drift is detected.
- The data that are sending to the metric server are the predictions output by the regression model together with the features used for the predictions. This is done by the inference server and not the scenarios' script.

## Components of the demo

- Training the regression model: The model is trained using the reference dataset which is split from the original dataset downloaded from Kaggle.
- The inference server: This is a model server that will return a price prediction when a request is sent to the server. The request would consists of the features of a house such as the number of bedrooms, etc... After a prediction is made by the model, the server would send the predictions along with the features to the metric server.
- The metric server: This is the Evidently metrics server which will monitor the predictions output by the inferenece server to detect data drift and regression performance.
