"""Server for inference."""
import json
import logging
import pickle

import numpy as np
import requests
from flask import Flask, request
from sklearn.ensemble import RandomForestRegressor

app = Flask(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler()],
)


# The encoder converts NumPy types in source data to JSON-compatible types
class NumpyEncoder(json.JSONEncoder):
    """If object type contains types that are not JSON serializable, return object as JSON compatible types.

    Args:
        json.JSONEncoder: inherit the class json.JSONEncoder and then implement the default method
    """

    def default(self, obj: object) -> object:
        """Implement the default method for the JSONEncoder class.

        Args:
            obj (object): the object to encode

        Returns:
            object: the encoded object
        """
        if isinstance(obj, np.void):
            return None

        if isinstance(obj, (np.generic, np.bool_)):
            return obj.item()

        if isinstance(obj, np.ndarray):
            return obj.tolist()

        return obj


@app.before_first_request
def load_model() -> RandomForestRegressor:
    """Load the trained model from the specified path.

    Returns:
        RandomForestRegressor: the trained model
    """
    model_path = "models/model.pkl"
    logging.info(f"Loading model in path: {model_path}")

    with open(model_path, "rb") as f:
        MODEL = pickle.load(f)
    logging.info("Model loaded")

    return MODEL


@app.route("/")
def home() -> str:
    """The message for default route.

    Returns:
        str: the message to return
    """
    return "Hello world from the inference server."


@app.route("/predict", methods=["POST"])
def predict() -> str:
    """Process the JSON payload and select features that can be used by the model to make predictions.

    Returns:
        str: the price prediction as a string
    """
    features = [
        "bedrooms",
        "bathrooms",
        "sqft_living",
        "sqft_lot",
        "floors",
        "waterfront",
        "view",
        "condition",
        "grade",
        "yr_built",
    ]

    logging.info(f"Received inference request {request.json}")
    request_data = dict(
        (f, request.json[f]) for f in features if f in request.json
    )
    features = list(request_data.values())
    features = np.array(features)
    features = features.reshape(1, -1)

    pred = MODEL.predict(features)
    pred_price = float(pred[0])

    logging.info(f"The predicted prices is: {pred_price}")
    send_pred_to_metric_server(pred_price)
    return str(pred[0])


def send_pred_to_metric_server(pred_price: float) -> None:
    """This function sends the predictions made by the model together with the features are used to make the predictions to the metric server.

    Args:
        pred_price (float): the predicted price
    """
    request_features = request.get_json()
    pred_price = {"price": float(pred_price)}
    features_n_pred = request_features | pred_price

    metric_server_url = (
        "http://evidently_service:8085/iterate/house_price_random_forest"
    )

    logging.info("Sending predictions to metric server.")
    try:
        response = requests.post(
            metric_server_url,
            data=json.dumps([features_n_pred], cls=NumpyEncoder),
            headers={"content-type": "application/json"},
        )

        if response.status_code == 200:
            logging.info("Success.")

        else:
            logging.Error(
                f"Got an error code {response.status_code} for the data chunk. "
                f"Reason: {response.reason}, error text: {response.text}"
            )

    except requests.exceptions.ConnectionError:
        logging.error("Cannot reach the metric server")


if __name__ == "__main__":
    MODEL = load_model()
    app.run(host="0.0.0.0", port="5050", debug=True)
