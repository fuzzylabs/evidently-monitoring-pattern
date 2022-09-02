import json
import logging
from flask import Flask, request
import numpy as np
import pickle
import os

app = Flask(__name__)

def setup_logger():
    logging.basicConfig(
        level=logging.INFO, 
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[
            logging.StreamHandler()
        ]
    )


@app.before_first_request
def load_model():
    global MODEL
    model_path = "models/model.pkl"
    logging.info(f"Loading model in path: {model_path}")
    with open(model_path, 'rb') as f:
        MODEL = pickle.load(f)
    logging.info(f"Model loaded")


@app.route('/')
def home():
    return "Hello world"


@app.route('/predict', methods=['POST'])
def predict():
    global MODEL

    logging.info(f"Received inference request {request.json}")
    request_data = request.json
    features = list(request_data.values())
    features = np.array(features)
    features = features.reshape(1, -1)

    pred = MODEL.predict(features)

    return str(pred[0])


if __name__ == "__main__":
    setup_logger()
    app.run(debug=True)
