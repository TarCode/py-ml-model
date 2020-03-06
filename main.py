import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.model_selection import cross_val_score
from sklearn.metrics import mean_absolute_error
from location_order import LOCATIONS


def preprocess_data(data_path):
    data = pd.read_csv(data_path)
    data = data.dropna(axis=0)

    types = ['apartment', 'house']

    for index, type in enumerate(types):
        data.replace(type, index, inplace=True)

    for index, location in enumerate(LOCATIONS):
        data.replace(location, index, inplace=True)

    features = ['bedrooms', 'type', 'location', 'area', 'bathrooms', 'parking']

    X = data[features]
    y = data.price

    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=1)

    return X_train, X_test, y_train, y_test


def get_scores_for_n_estimators(n_estimators, X_train, y_train):
    my_pipeline = Pipeline(steps=[
        ('preprocessor', SimpleImputer()),
        ('model', RandomForestRegressor(n_estimators=n_estimators, random_state=0))
    ])

    scores = -1 * cross_val_score(my_pipeline, X_train, y_train, cv=3, scoring='neg_mean_absolute_error')
    return scores.mean()


def get_scores_for_max_leaf_nodes(max_leaf_nodes, X_train, y_train):
    my_pipeline = Pipeline(steps=[
        ('preprocessor', SimpleImputer()),
        ('model', RandomForestRegressor(max_leaf_nodes=max_leaf_nodes, random_state=0))
    ])

    scores = -1 * cross_val_score(my_pipeline, X_train, y_train, cv=3, scoring='neg_mean_absolute_error')
    return scores.mean()


def get_scores_with_xg_boost(n_estimators, X_train, y_train, X_test, y_test):
    model = XGBRegressor(n_estimators=n_estimators, learning_rate=0.1, n_jobs=2)
    model.fit(X_train, y_train)

    predictions = model.predict(X_test)
    mae = mean_absolute_error(y_test, predictions)
    return "Mean Absolute Error %f" % mae


X_train, X_test, y_train, y_test = preprocess_data('data/cape-town-property-listings.csv')
#
# max_leaf_nodes_results = {}
# for i in [34, 55, 89, 144, 233, 377, 610, 947]:
#     max_leaf_nodes_results[i] = get_scores_for_max_leaf_nodes(i, X_train, y_train)
#
# n_estimators_results = {}
#
# for i in range(400, 950, 50):
#     n_estimators_results[i] = get_scores_for_n_estimators(i, X_train, y_train)

# print('MAX LEAF NODES RESULTS')
# print(max_leaf_nodes_results)
#
# print('N ESTIMATORS RESULTS')
# print(n_estimators_results)

xgboost_results = get_scores_with_xg_boost(5000, X_train, y_train, X_test, y_test)
print('XG BOOST RESULTS')
print(xgboost_results)