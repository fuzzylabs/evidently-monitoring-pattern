import pandas as pd
import numpy as np
import logging
import pickle

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

def setup_logger():
    logging.basicConfig(
        level=logging.INFO, 
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[
            logging.StreamHandler()
        ]
    )

def prepare_data():
    logging.info("Preparing data for train and test")
    df = pd.read_csv('data/processed_house_data.csv', index_col='date')

    target = 'price'
    prediction = 'predicted_price'
    datetime = 'date'

    features = ['bedrooms', 'bathrooms', 'sqft_living', 'sqft_lot', 'floors',
        'waterfront', 'view', 'condition', 'grade', 'yr_built']

    X = df[features].values
    y = df[target].values

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=28)

    return X_train, X_test, y_train, y_test

def model_setup():
    logging.info("Creating Random Forest Regressor model")
    model = RandomForestRegressor(random_state=28, verbose=1)
    return model

def train(model):
    logging.info("Training model")
    model.fit(X_train, y_train)
    logging.info("Training Completed")

def evaluate(model):
    logging.info("Evaluating model on test set")
    predictions = model.predict(X_test)
    mse = mean_squared_error(y_test, predictions)
    mae = mean_absolute_error(y_test, predictions)
    rmse = mean_squared_error(y_test, predictions, squared=False)
    r2 = r2_score(y_test, predictions)

    logging.info(f"Mean Squared Error: {mse}")
    logging.info(f"Mean Absolute Error: {mae}")
    logging.info(f"Root Mean Squared Error: {rmse}")
    logging.info(f"R-Squared: {r2}")

def save_model(model):
    with open('pipeline/model.pkl','wb') as f:
        pickle.dump(model, f)

if __name__ == "__main__":
    setup_logger()
    X_train, X_test, y_train, y_test = prepare_data()
    model = model_setup()
    train(model)
    evaluate(model)
    save_model(model)
