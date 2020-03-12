import pandas as pd
from location_order import LOCATIONS
from sklearn.model_selection import train_test_split


def read_data(filepath):
    """
    Takes a file path and returns a Pandas dataframe
    :param filepath:
    :return:
    """
    data = pd.read_csv(filepath)
    data.dropna(axis=0)

    return data


def preprocess_data(data, features, y_label):
    """
    Takes a Pandas dataframe and a list of features as an array to process and split
    :param data:
    :param features:
    :return:
    """

    # Manually label encode dwelling types
    types = ['apartment', 'house']

    for index, type in enumerate(types):
        data.replace(type, index, inplace=True)

    # Manually label encode locations in order of value (ordered in location_order.py)
    for index, location in enumerate(LOCATIONS):
        data.replace(location, index, inplace=True)

    x = data[features]
    y = data[y_label]

    x_train, x_test, y_train, y_test = train_test_split(x, y, random_state=1)

    return x_train, x_test, y_train, y_test