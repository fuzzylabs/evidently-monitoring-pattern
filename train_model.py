"""Train a random forest regressor using the reference dataset."""
from config.config import logger
from pipeline.train import prepare_data, model_setup, train, evaluate, save_model

if __name__ == "__main__":
    # Path to the reference dataset
    data_path = "datasets/house_price_random_forest/reference.csv"
    # Path for saving the trained model
    save_path = "models/model.pkl" 

    # Selecting portion of all features to be used for training
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
    # Training target
    target = "price"
    # Test set size for train test split
    test_size = 0.2

    x_train, x_test, y_train, y_test = prepare_data(
        data_path, features, target, test_size
    )

    # Create a random forest regressor using sklearn
    model = model_setup()
    # Fit the model
    train(model, x_train, y_train)
    # Evaluate the performance
    evaluate(model, x_test, y_test)
    # Saving the model
    save_model(model, save_path)