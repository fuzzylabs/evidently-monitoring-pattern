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

def load_model():
    model_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'models/model.pkl')
    print(model_path)
    with open(model_path, 'rb') as f:
        model = pickle.load(f)
    return model

@app.route('/')
def home():
    return "Hello world"

@app.route('/predict', methods=['POST'])
def predict():
    logging.info(f"Received inference request {request.json}")
    request_data = request.json
    features = list(request_data.values())
    features = np.array(features)
    features = features.reshape(1, -1)

    model = load_model()
    pred = model.predict(features)

    return str(pred[0])

if __name__ == "__main__":
    setup_logger()
    app.run(debug=True)
