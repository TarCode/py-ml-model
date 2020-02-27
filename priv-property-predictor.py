import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split

"""
    TODO: Enrich the dataset by getting bathrooms, parking spaces and pets/no pets
"""


def get_mae(max_leaf_nodes, train_X, val_X, train_y, val_y):
    model = RandomForestRegressor(max_leaf_nodes=max_leaf_nodes, random_state=1)
    model.fit(train_X, train_y)
    preds_val = model.predict(val_X)
    mae = mean_absolute_error(val_y, preds_val)
    print("Validation data: ")
    print(val_X)
    print("Predicted values: ")
    print(preds_val)
    print("Mean Absolute Error: ", mae)
    return mae


data_path = 'data/private-property-listings-cpt-cbd.csv'
data = pd.read_csv(data_path)
data = data.dropna(axis=0)

features = ['bedrooms', 'price']

X = data[features]
y = data.price

train_X, val_X, train_y, val_y = train_test_split(X, y, random_state=1)

my_mae = get_mae(50, train_X, val_X, train_y, val_y)

# for max_leaf_nodes in [20, 50, 100, 500, 1000]:
#     my_mae = get_mae(max_leaf_nodes, train_X, val_X, train_y, val_y)
#     print('Max leaf nodes: %d \t\t Mean absolute error: %d' % (max_leaf_nodes, my_mae))
