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

## Download and prepare data for the model

1. **Create a new Python virtual environment and activate it**. For Linux/MacOS:

```bash
python -m venv env
source venv/bin/activate 
pip install -r requirements.txt
```
2. **Get and set up Kaggle API token**

- Go to [Kaggle](https://www.kaggle.com) to log in or create an account.
- Get into your account settings page.
- Under the API section, click on `create a new API token`.
- This will prompt you to download the `.json` file into your system. Open the file, and copy the username and key.
- Export your Kaggle username and token to the environment

```bash
export set KAGGLE_USERNAME=<your-kaggle-username>
export set KAGGLE_KEY=<your-kaggle-api-key>
```

3. **Run the `get_data.py` script**:

```bash
python data/get_data.py
```

## Train the house prices model

1. **Run the `train.py` script to train the model**:

```bash
python pipeline/train.py
```
- Once the model is trained, it will be saved as `model.pkl` inside the `models` folder.

## Run the model server

1. **Start the server**:

```bash
python inference_server/server.py
```

2. **Test it using Curl**

```bash
curl -XPOST http://127.0.0.1:5000/predict -H 'Content-type: application/json' -d '{"bedrooms": 1, "bathrooms": 1, "sqft_living": 50, "sqft_lot": 50, "floors": 1, "waterfront": 0, "view": 0, "condition": 0, "grade": 0, "yr_built": 1960}'
```

The server will return with a price, e.g. `100000`.


## Run the metric server
1. **Split the dataset into production and reference**:

```bash
python data/split_data.py
```

2. **Start the server**:

```bash
python monitoring_server/metric_server.py
```

3. **Test it by sending data to it**

```bash
python monitoring_server/send_data_to_server.py
```

- For now, the server will return a message saying "Data drift detected" when data drift is detected.
- The data that are sending to the metric server using the send_data_to_server.py script are getting data from the datasets file. The dataset file contains a reference.csv and a production.csv which is used for testing and building the server. These two csv files are created using the split_data.py scipt within the data folder.