import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

data_path = "data/processed_house_data.csv"

data_df = pd.read_csv(data_path)

bedrooms = data_df['bedrooms']

print(data_df['bedrooms'].value_counts().describe())