import numpy as np
import pandas as pd
import os

dataset_dir = r'data\processed_house_data.csv'

def data(dataset_dir):
    raw = pd.read_csv(dataset_dir)
    ref = raw[:10000]
    pro = raw[10000:20001]
    print(ref)

    datasets_path = "datasets\house_price_random_forest"

    if os.path.exists(datasets_path):
        ref.to_csv(os.path.join(datasets_path, "reference.csv"), index=False)
        pro.to_csv(os.path.join(datasets_path, "production.csv"), index=False)
    else:
        os.makedirs("datasets\house_price_random_forest")
        ref.to_csv(os.path.join(datasets_path, "reference.csv"), index=False)
        pro.to_csv(os.path.join(datasets_path, "production.csv"), index=False)
        
data(dataset_dir)