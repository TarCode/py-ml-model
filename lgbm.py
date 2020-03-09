import pandas as pd
from location_order import LOCATIONS
from sklearn import metrics
from sklearn.model_selection import train_test_split
import lightgbm as lgb


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


def train_model(x_train, y_train):
    dtrain = lgb.Dataset(x_train, label=y_train)

    param = {'num_leaves': 64, 'objective': 'regression', 'metric': 'auc'}
    num_round = 1000
    bst = lgb.train(param, dtrain, num_round)

    return bst


def evaluate_model(model, x_test, y_test):
    ypred = model.predict(x_test)
    score = metrics.roc_auc_score(y_test, ypred, multi_class="ovr")
    return score

