# Introduction

This repo is a complete demo of real-time model monitoring using Evidently. Within the repo, you will find:

* A script to download and process the training data
* A model training pipeline to train a simple regression model that predicts house prices.
* A model server that exposes our house price model through a REST API.
* Some scripts that simulate different scenarios: e.g. model drift vs normal input.
* An Evidently model monitoring service which collects inputs and predictions from the model and computes metrics such as drift.
* A monitoring dashboard which uses Prometheus and Grafana to visualise in real-time the monitoring metrics.
* Docker Compose to run the whole thing.

## How to download and prepare data for the model?

1. **Create a new Python virtual environment and activate it**. For Linux/MacOS:
```bash
pip install virtualenv
virtualenv venv
source venv/bin/activate 
```
2. **Get the `evidently-monitoring-demo` code example**:
```bash
git clone git@github.com:fuzzylabs/evidently-monitoring-demo.git
```
3. **Install dependencies**:
- Go to the demo directory:
```bash
cd evidently-monitoring-demo
```
- install dependencies for the data script
```bash
pip install -r requirements.txt
```
4. **Get and Set Up Kaggle API token**
- Go to https://www.kaggle.com/ to create an account.
- Get into your account settings page.
- Under the API section, click on Create a new API token.
- This will prompt you to download the .json file into your system. Open the file, and copy the username and key.
- Export your Kaggle username and token to the environment
```bash
export KAGGLE_USERNAME=<your-kaggle-username>
export KAGGLE_KEY=<your-kaggle-api-key>
```
5. **Run the `get_data.py` script**:
```bash
python data/get_data.py
```

## How to train a regression model to predict house prices?
1. **Run the `train.py` script to train a Random Forest Regressor**:
```bash
python pipeline/train.py
```
- Once the model is trained, it will be automatically saved as model.pkl inside the pipeline folder.