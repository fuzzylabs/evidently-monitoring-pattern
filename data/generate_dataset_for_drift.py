from math import prod
import pandas as pd
import numpy as np
import random
import os
import math


dataset_path = "data/processed_house_data.csv"
features = ['date', 'bedrooms', 'bathrooms', 'sqft_living', 'sqft_lot', 'floors', 
            'waterfront', 'view', 'condition', 'grade', 'yr_built']


df = pd.read_csv(dataset_path)
df = df[features]
df = df[:20]

production_df = []

for x in range(100):
    production_data = {}
    for feature in features:
        if feature != 'date':
            min = df.min(axis=0)[feature]
            max = df.max(axis=0)[feature]
            production_data[feature] = random.randint(int(min), int(max))
        else:
            production_data[feature] = (df.sample()['date'].values)[0]
    production_df.append(production_data)

production_df = pd.DataFrame(production_df)
print(production_df)

# out_of_range_features = {}

# for feature in features:
#     if feature != 'date':
#         max_val = df.max(axis=0)[feature]
#         production_df[feature] = random.randint(int(math.ceil(max_val)), int(math.ceil(max_val * 2)))
#     else:
#         print(df.sample()['date'].values)
#         production_df[feature] = df.sample()['date'].values

# production_df = pd.DataFrame(data = production_df, index = [0])
# newdf = pd.DataFrame(np.repeat(production_df.values, 1000, axis=0))
# newdf.columns = df.columns
# production_df = newdf


datasets_path = "datasets/house_price_random_forest"

df.to_csv(os.path.join(datasets_path, "reference.csv"), index = False)
production_df.to_csv(os.path.join(datasets_path, "production.csv"), index = False)