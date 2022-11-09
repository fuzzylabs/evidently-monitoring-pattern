# Introduction

This repo is a complete demo of real-time data monitoring using Evidently. Using a Random Forest Regressor to predict house prices and simulate data drift by sending drifted feature(s) to the model. Evidently calculates the metrics for data drift, send them to Prometheus and visualize the results on a pre-built Grafana dashboard.

# Contents

- [Outline](#outline)
- [Running locally](#running-locally)
- [How does the demo work?](#how-does-the-demo-work)
    - [Scenarios](#scenarios)
        - [How are the data generated?](#how-are-the-data-generated)
        - [Scenario scripts](#scenario-scripts)
    - [Inference server](#inference-server)
    - [Evidently server](#evidently-server)
    - [Prometheus](#prometheus)
    - [Grafana](#grafana)

# Outline

<!-- TODO: add detail to this description -->

Within the repo, you will find:

- [`data`](data): contains two scripts. Running the `get_data.py` will automatically download a Kaggle house sale prices dataset for model training and data monitoring (drift monitoring); the dataset is saved to this directory. The `preppare_dataset.py` script will split the house sale prices dataset into a production and a reference dataset which will be saved to a new directory named `datasets`.

    NOTE: The Kaggle dataset has been uploaded to Google Drive for easy access.
- [`pipeline`](pipeline): a model training script which will use the reference data to create and train a Random Forest Regressor model.
- [`inference_server`](model_server): a model server that exposes our house price model through a REST API.
- [`monitoring_server`](monitoring_server): an Evidently model monitoring service which collects inputs and predictions from the model and computes metrics such as data drift.
- [`scenarios`](scenarios): Two scripts to simulate different scenarios. A scenario where there is no drift in the inputs and a scenario which the input data contains drifted data.
- [`dashboards`](dashboards): a data drift monitoring dashboard which uses Prometheus and Grafana to visualise Evidently's monitoring metrics in real-time.
- A [`run_demo.py`](run_demo_.py) script to run the demo using docker compose.
- A docker-compose file to run the whole thing.

# Running demo

## Pre-requisites

You'll need following pre-requisites to run the demo:

- [Python 3](https://www.python.org/downloads/)
- [Docker](https://docs.docker.com/engine/install/)
- [Docker Compose V2](https://docs.docker.com/compose/install/).

## Getting started

1. Clone this repo:

    ```bash
    git clone git@github.com:fuzzylabs/evidently-monitoring-demo.git
    ```

2. Go to the demo directory:

    ```bash
    cd evidently-monitoring-demo
    ```

3. Create a new Python virtual environment and activate it. For Linux/MacOS users:

    ```bash
    python3 -m venv demoenv
    source demoenv/bin/activate
    pip install -r requirements.txt
    ```

## Jupyter Notebook or Terminal

From this point, you have the option to continue the demo by following the instructions below or you continue this demo with [`demo.ipynb`](demo.ipynb) (included in this repo) using Jupyter Notebook. The notebook will provide an breif explaination as we go through each steps. Alternatively, you can check out [**How does the demo work?**](#how-does-the-demo-work) to see how each individual component works with each other and how are the datasets generated.

## Download and prepare data for the model

1. Run the `download_dataset.py` script:

    ```bash
    python download_dataset.py
    ```

    This script will download and preprocess the data from Google drive.

2. Split the dataset into production and reference:

    ```bash
    python prepare_dataset.py
    ```

    This script will split the dataset into 1 reference and 2 production datasets (with drift data and  without drift data).

## Training the Random Forest Regressor

- Run the `train_model.py` script to train the model:

    ```bash
    python train_model.py
    ```

    Once the model is trained, it will be saved as `model.pkl` inside the `models` folder.

## Running the demo

At the moment, there are two scenarios we can simulate by running the demo:

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

The metrics produced by Evidently will be logged to Prometheus's database which will be available at port 9090. To access Prometheus web interface, go to your browser and open: <http://localhost:9090/>

To visualise these metrics, Grafana is connected to Prometheus's database to collect data for the dashboard. Grafana will be available at port 3030. To access Grafana web interface, go to your browser and open: <http://localhost:3000/> . If this is your **first time using Grafana**, you will be asked to enter a username and password, by default, both the username and password is "admin".

To stop the demo, press ctrl+c and shut down docker compose by running the following command:

```bash
python run_demo.py --stop
```

# How does the demo work?

![Flow](docs/assets/images/Monitoring_Flow_Chart.png)

The demo is comprised of 5 core components:

- Scenarios
- Inference server
- Evidently metric server
- Prometheus
- Grafana

## Scenarios

### How are the data generated?

This demo makes use of the Kaggle dataset [House Sales in King County, USA](https://www.kaggle.com/datasets/harlfoxem/housesalesprediction). The dataset includes homes from between May 2014 to May 2015. The dataset contains 21,613 rows and 21 columns that represent the features of the homes sold. The Kaggle dataset has been uploaded to Google Drive for easy acces using [gdown](https://pypi.org/project/gdown/), so we don't need to generate and add the Kaggle API key as environment variables for using the Kaggle API.

- [download_dataset.py](download_dataset.py): This script uses 2 functions from [data/get_data.py](data/get_data.py), the "download_dataset" and "preprocess_dataset function".

    It first download the Kaggle dataset as a zip file "housesalesprediction.zip". All the files downloaded or generated by this script are saved under the data folder. The downloaded zip file will be extracted as "kc_house_data.csv". Finally, to help Evidently to process the input data, the "date" column of "kc_house_data.csv" will be converted to a pandas datetime object and a new dataset will be saved as "processed_house_data.csv".

- [prepare_dataset.py](prepare_dataset.py): This script uses 5 functions from [data/generate_dataset_for_demo.py](data/generate_dataset_for_demo.py), the "create_data_simulator", "generate_production_data", "generate_production_no_drift_data", "generate_production_with_drift_data" and "generate_reference_data" function.

    To monitor data drift or outliers, Evidently requires at least two datasets to perform comparison, a production dataset and a reference dataset. All the files downloaded or generated by this script are saved under the datasets folder. Using "processed_house_data.csv", this script will create two scenarios of production data, 1 with data drift, 1 without and 1 reference dataset. The original dataset downloaded contains 21 features.

    The reference dataset is used as the baseline data by Evidently and for training the model. As we do not have a actual house selling website, the production dataset will simulate live data which will be used to compared against the reference dataset to identify data drift. Production datasets do not include the price column as the price will be predicted by a regression model.

    To simplify this demo and make it easier to understand what is happening, we are only going to select 2 features from the original dataset.

    The distribution of each feature is computed using the the "compute_dist" function which returns a dictionary of probabilty distribution. For example, if 50% of the bedroom has a value of 2, 30% has a value of 3 and 20% has a value of 4. Then, the "compute_dist" function would return a dictionary of {2: 0.5, 3: 0.3, 4: 0.2}. For the no data drift production dataset, the number of bedrooms and the condition feature for each row of data is generated using the same distribution as the reference dataset to ensure that no data drift will be detected. 
    
    These feature distribution will be used to create a value generator object using the ProbDistribution class within the [prob_distribution.py](prob_distribution.py). When a generator object is initialised, it will compute a skewed distribution based on the orginal probability distribution. For example, using the example bedroom distribution {2: 0.5, 3: 0.3, 4: 0.2}. The generator will skew the distribtuion by changing the probability of each value. To ensure drift will be detected for this demo, the value with the lowest probability will have a proability of 1 after skewed. The skewed distribution will be stored as a dictionary {2: 0.0, 3: 0.0, 4: 1.0} as an attribute of the generator object.

    ### Histogram visualisation

    #### Distribution comparison between the reference datasets and the **non-drifted** production dataset:

    ![NoDriftHistogram](docs/assets/images/No_Drift_Histogram.png)

    The histogram above shows that for the bedroom feature, 7 bedrooms appears the least meaning that it has the lowest probabilty of being generated in the no data drift production dataset.

    #### Distribution comparison between the reference datasets and the **drifted** production dataset:

    ![DriftHistogram](docs/assets/images/Drift_Histogram.png)

    The histogram above shows that for the bedroom feature, 7 bedrooms appears much more frequently compare to the no data drift production dataset.

    Once the datasets are generated, the Random Forest Regressor is trained using the reference dataset with [train.py](train.py).

### Scenario scripts
- Within the [`scenarios`](scenarios) folder, it contains two scripts namely [`drift.py`](scenarios/drift.py) and [`no_drift.py`](scenarios/no_drift.py). Both scripts send production data to the model server for price prediction. The difference between the two is that one would send data from the `production_no_drift.csv` and the other would send data from the `production_with_drift.csv` which contains drifted data.

## Inference server
- This is a model server that will return a price prediction when a request is sent to the server. The request would consists of the features of a house such as the number of bedrooms, etc... After a prediction is made by the model, the server would send the predictions along with the features to the metric server.

## Evidently server
- The Evidently metric server: this is the Evidently metrics server which will monitor both the inputs and outputs of the inference server to calculate the metrics for data drift.
To simplify this demo and make it easier to understand what is happening, we are only going to select 2 features from the original dataset

## Prometheus
- Prometheus: once the Evidently monitors have produced some metrics, they will be logged into Prometheus's database as time series data.

## Grafana
- Grafana: this is what we can use to visualise the metrics produced by Evidently in real time. A pre-built dashboard for visualising data drift is include in the [`dashboards`](dashboards) directory.

### The dashboards

![Drift](docs/assets/images/No_Drift.png)

- When the no data drift scenario is running, the Grafana's dashboard should show no data drift is detected. However, as there are some randomness in dataset generation, it is possible to see data drift every once a while.

![Drift](docs/assets/images/Data_Drift.png)

- When the data drift scenario is running, Grafana's dashboard should show data drift at a relatively constant time frame, e.g. no data drift for 10 seconds -> data drift detected for 5 seconds -> no data drift for 10 seconds etc...
