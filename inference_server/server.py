import json
import logging
from flask import Flask, request

def setup_logger():
    logging.basicConfig(
        level=logging.INFO, 
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[
            logging.StreamHandler()
        ]
    )

app = Flask(__name__)

@app.route('/')
def home():
    return "Hello world"

@app.route('/predict', methods=['POST'])
def predict():
    logging.info(f"Received inference request {request.json}")
    return "100000"

if __name__ == "__main__":
    setup_logger()
    app.run(debug=True)
