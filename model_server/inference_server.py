import json
import logging
from flask import Flask, request
import numpy as np
import pickle
import requests


app = Flask(__name__)


def setup_logger() -> None:
    logging.basicConfig(
        level=logging.INFO, 
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[
            logging.StreamHandler()
        ],
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


@app.before_first_request
def load_model():
    '''
    Load the trained model from the specified path.

    Return the trained model.
    '''
    model_path = "models/model.pkl"
    logging.info(f"Loading model in path: {model_path}")
    with open(model_path, 'rb') as f:
        MODEL = pickle.load(f)
    logging.info(f"Model loaded")
    return MODEL


@app.route('/')
def home() -> str:
    return "Hello world"


@app.route('/predict', methods=['POST'])
def predict() -> str:
    '''
    Process the JSON payload and select features that can be used by the model to make predictions.

    Return the price prediction as a string.
    '''
    features = ['bedrooms', 'bathrooms', 'sqft_living', 'sqft_lot', 'floors', 
            'waterfront', 'view', 'condition', 'grade', 'yr_built']

    logging.info(f"Received inference request {request.json}")
    request_data = dict((f, request.json[f]) for f in features if f in request.json)
    features = list(request_data.values())
    features = np.array(features)
    features = features.reshape(1, -1)

    pred = MODEL.predict(features)

    logging.info(f"The predicted prices is: {pred[0]}")
    return str(pred[0])


@app.after_request
def send_pred_to_metric_server(response):
    '''
    This function sends the predictions made by the model together with the features are used to make the predictions to the metric server.
    '''
    request_features = request.get_json()
    pred_price = response.get_data(as_text = True)
    pred_price = {"price": float(pred_price)}
    features_n_pred = request_features | pred_price

    metric_server_url = "http://127.0.0.1:5000/iterate/house_price_random_forest"
    logging.info(f"Sending predictions to metric server.")
    try:
        r = requests.post(
            metric_server_url, 
            data = json.dumps([features_n_pred], cls = NumpyEncoder), 
            headers = {"content-type": "application/json"})

        if response.status_code == 200:
            print(f"Success.")

        else:
            print(
                f"Got an error code {response.status_code} for the data chunk. "
                f"Reason: {response.reason}, error text: {response.text}"
            )

    except requests.exceptions.ConnectionError as e:
        logging.error(f"Cannot reach the metric server")

    return response


if __name__ == "__main__":
    setup_logger()
    MODEL = load_model()
    app.run(port = "5050", debug = True)