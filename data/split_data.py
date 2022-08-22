import numpy as np
import pandas as pd
import os

dataset_dir = r'processed_house_data.csv'

def data(dataset_dir):
    raw = pd.read_csv(dataset_dir)
    ref = raw[:10000]
    pro = raw[10000:20001]
    print(ref)

    ref.to_csv(os.path.join('datasets\house_price_random_forest', "reference.csv"), index=False)
    pro.to_csv(os.path.join('datasets\house_price_random_forest', "production.csv"), index=False)

data(dataset_dir)