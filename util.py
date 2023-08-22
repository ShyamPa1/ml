import json
import pickle
import numpy as np
from sklearn.preprocessing import OneHotEncoder

__locations = None
__data_columns = None
__model = None

def get_estimated_price(location, sqft, bhk, bath):
    try:
        loc_index = __data_columns.index(location.lower())
    except:
        loc_index = -1
    x = np.zeros(len(__data_columns))
    x[0] = sqft
    x[1] = bath
    x[2] = bhk
    if loc_index >= 0:
        x[loc_index] = 1

    # Transform x into a 2D array
    x_2d = np.array(x).reshape(1, -1)

    return round(__model.predict(x_2d)[0], 2)

def get_location_names():
    return __locations

def load_saved_artifacts():
    print("loading saved artifacts...start")
    global __data_columns
    global __locations
    global __model
    with open("./artifacts/columns.json", 'r') as f:
        __data_columns = json.load(f)['data_columns']
        __locations = __data_columns[4:]
    with open("./artifacts/bengaluru_prices.pickle", 'rb') as f:
        __model = pickle.load(f)
    print("loading saved artifacts done")

if __name__ == "__main__":
    load_saved_artifacts()
    print(get_estimated_price('1st block jayanagar', 1030, 3, 3))
    print(get_location_names())
