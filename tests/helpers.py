import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler


def load_credit():
    data = pd.read_csv('/new_creditcard.csv')
    data = data.values

    X, y = data[:, 1:-1], data[:, -1]
    return X, y


def scale_data(data):
    scaler = StandardScaler()
    scaler.fit(data)
    scaler.transform(data)
    return data


def classify_array(svc, X):
    y_tag = []
    for x_el in X:
        x_el = svc.seal_ops.encrypt(x_el)
        print(f'encrypted {x_el}')
        y_tag += [svc.predict(x_el)]
    return np.array(y_tag)
