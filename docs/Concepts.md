# How does the demo work?

![Flow](assets/images/Monitoring_Flow_Chart.png)

The demo is comprised of 5 core components:

- [Scenarios](#scenarios)
  - [How are data generated?](#how-are-the-data-generated)
  - [Histogram Visualization](#histogram-visualisation)
- [Inference server](#inference-server)
- [Evidently server](#evidently-server)
- [Prometheus](#prometheus)
- [Grafana](#grafana)
  - [How to read dashboards](#how-to-read-dashboards)
    - [No drift Scenario](#no-drift-scenario)
    - [Drift Scenario](#drift-scenario)

We will go through each component in great detail.

## Scenarios

This demo makes use of the Kaggle dataset [House Sales in King County, USA](https://www.kaggle.com/datasets/harlfoxem/housesalesprediction). The dataset includes homes from between May 2014 to May 2015. The dataset contains 21,613 rows and 21 columns that represent the features of the homes sold. The Kaggle dataset has been uploaded to Google Drive for easy access.

### How are the data generated?

This section will help you understand the motivation behind creating 1 reference and 2 production datasets.

- Download and Preprocess data Step

    ```bash
    python prepare_demo.py --download
    ```

    This step downlaods the Kaggle dataset as a zip file `housesalesprediction.zip` inside `datasets` folder. The downloaded zip file will be extracted as `kc_house_data.csv`. Finally, to help Evidently to process the input data, the `date` column of `kc_house_data.csv` will be converted to a pandas datetime object and a new dataset will be saved as `processed_house_data.csv` inside the same folder.

- Prepare datasets step

    ```bash
    python prepare_demo.py --prepare
    ```

    To monitor data drift or outliers, Evidently requires at least two datasets to perform comparison, a production dataset and a reference dataset. Using `processed_house_data.csv` from previous step, this script will create two scenarios of production data, 1 with data drift, 1 without data drift. The reference dataset is first 1000 rows of original dataset. The original dataset downloaded contains 21 features. To simplify this demo and make it easier to understand what is happening, we are only going to select 2 features from the original dataset i.e. ["bedroom", "condition"] columns are selected.

    The reference dataset is used as the baseline data by Evidently. Since we do not have a actual house selling website, the production dataset will simulate live data which will be used to compared against the reference dataset to identify data drift. Production datasets do not include the price column as the price will be predicted by a regression model.

    The distribution of each feature is computed using the the [compute_dist](../src/prepare_data.py#L37) function which returns a dictionary of probabilty distribution. For example, if 50% of the bedroom has a value of 2, 30% has a value of 3 and 20% has a value of 4. Then, the `compute_dist` function would return a dictionary of {2: 0.5, 3: 0.3, 4: 0.2}. For the no data drift production dataset, the number of bedrooms and the condition feature for each row of data is generated using the same distribution as the reference dataset to ensure that no data drift will be detected.

    These feature distribution will be used to create a value generator object using the `ProbDistribution` class within the [prob_distribution.py](../src/prob_distribution.py). When a generator object is initialised, it will compute a skewed distribution based on the orginal probability distribution. For example, using the example bedroom distribution {2: 0.5, 3: 0.3, 4: 0.2}. The generator will skew the distribtuion by changing the probability of each value. To ensure drift will be detected for this demo, the value with the lowest probability will have a proability of 1 after skewed. The skewed distribution will be stored as a dictionary {2: 0.0, 3: 0.0, 4: 1.0} as an attribute of the generator object.

- Train Random Forest Regression Model.

    ```bash
    python prepare_demo.py --train
    ```

    Once the datasets are generated, the Random Forest Regressor is trained using the reference dataset with [train.py](../pipeline/train.py).

    Within the [`scenarios`](../scenarios) folder, there are two scripts namely [`drift.py`](../scenarios/drift.py) and [`no_drift.py`](../scenarios/no_drift.py). Both scripts send production data to the model server for price prediction. The difference between the two is that one would send data from the `production_no_drift.csv` for no-drift scenario and the other would send data from the `production_with_drift.csv` which contains drifted data for drift scenario.

### Histogram visualisation

In this section, we visualize the historgram for 2 features in our toy dataset. We compare the histogram for drift and no-drift scenarios against reference dataset to visualize what the distribution looks like.

**Distribution comparison between the reference datasets and the non-drifted production dataset**

![NoDriftHistogram](assets/images/No_Drift_Histogram.png)

The histogram above shows that for the bedroom feature, 7 bedrooms appears the least meaning that it has the lowest probabilty of being generated in the no data drift production dataset.

**Distribution comparison between the reference datasets and the drifted production dataset**

![DriftHistogram](assets/images/Drift_Histogram.png)

The histogram above shows that for the bedroom feature, 7 bedrooms appears much more frequently compare to the no data drift production dataset.

## Inference server

[Inference server](../server/model_server/inference_server.py) is basically a flask application that serves the model for inference. Whenever we send a POST request from either of the scenarios from [scnearios](../scenarios/) folder, it reaches this inference server which sends an output, the predicted price of the house.

The [drift scenario](../scenarios/drift.py) sends the data from `production_with_drift.csv` created in previous step. We read each row from the dataset and send a POST request to this inference server.

The same happens in case of [no drift scenario](../scenarios/no_drift.py) as well with only difference being the dataset. In this case, `production_no_drift.csv` also created in previous step is used.

The model endpoint for prediction `http://127.0.0.1:5050/predict`. Once the POST request is sent, the inference server will also send both the predicted price and features of the input to Evidently metric server. In next section, we will see how the Evidently metric server works and what it does with these metrics.

## Evidently server

This section is one of the complex parts of the project. [Evidently metric server](../server/monitoring_server/metric_server.py) is also a flask application that sends various metrics to Prometheus endpoint.

It receives the predicted price and features of input from inference server from previous step. Evidently calculates data drift between reference data and current data which is sent by inference server. The metrics such as "n_features" (number of features), "dataset_drift" and all [different metrics](https://docs.evidentlyai.com/reference/all-metrics#data-drift) calculated using Evidently library are sent to Prometheus endpoint.

This service is exposed at endpoint : <http://localhost:8085/>

## Prometheus

[Prometheus](https://prometheus.io/) is open source monitoring tool that scrapes metrics from HTTP endpoints. This server stores the scraped metrics in a time series DB. The Evidently metric server in previous step sends all different evidently related metrics to this server.

Prometheus endpoint is exposed at <http://localhost:9090/>.

Metrics by themselves won't make sense if we don't create a simple visualization tool to visualize the metrics. This is where Grafana plays a vital role which we discuss in the next section.

## Grafana

[Grafana](https://grafana.com/) is open source project that helps creating beatiful dashboards for visualizing different metrics. We can use it to visualise the metrics produced by Evidently in real time. A pre-built dashboard for visualising data drift is include in the [`dashboards`](../dashboards) directory.

Once the application is started, you can see the results on Grafana dashboard at <http://localhost:3000/>. The default login credentials are username: admin and password: admin.

To see the monitoring dashboard in the Grafana interface, click "General" and navigate to the chosen dashboard (e.g. "Evidently Drift Monitoring"). Under the dashboard, you will see a graph that shows the drift detection results.

### How to read dashboards?

#### No Drift Scenario

![No Drift](assets/images/No_Drift.png)

When the no data drift scenario is running, the Grafana's dashboard should show no data drift is detected. This can be monitored by looking at the "# of drifted features" block in the Grafana dashboard. The no drift scenario shows that currently there are 2 features and no drift is detected for either of features.

#### Drift Scenario

![Drift](assets/images/Data_Drift.png)

The drift scenarios show that both the features are drifted.
