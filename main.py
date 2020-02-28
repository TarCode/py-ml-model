import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split
from location_order import LOCATIONS

"""
    TODO: Enrich the dataset by getting bathrooms, parking spaces and pets/no pets
"""


def get_mae(max_leaf_nodes, train_X, val_X, train_y, val_y):
    model = RandomForestRegressor(max_leaf_nodes=max_leaf_nodes, random_state=1, n_estimators=100)
    model.fit(train_X, train_y)
    preds_val = model.predict(val_X)
    mae = mean_absolute_error(val_y, preds_val)
    # print("Validation data: ")
    # print(val_X)
    # print("Predicted values: ")
    # print(preds_val)
    # print("Mean Absolute Error: ", mae)
    return mae


data_path = 'data/cape-town-property-listings.csv'
data = pd.read_csv(data_path)
data = data.dropna(axis=0)

#
# locations = [
#     'Seawinds',
#     'Maitland',
#     'Lakeside',
#     'Plumstead',
#     'Tokai',
#     'Pinelands',
#     'Rondebosch',
#     'Claremont',
#     'Claremont Upper',
#     'Oranjezicht',
#     'Sea Point',
#     'Clifton',
#     'Camps Bay'
# ]

print(data['type'].unique())

types = ['apartment', 'house']

for index, type in enumerate(types):
    data.replace(type, index, inplace=True)

for index, location in enumerate(LOCATIONS):
    data.replace(location, index, inplace=True)

features = ['bedrooms', 'location', 'area', 'bathrooms', 'parking', 'price']

X = data[features]
y = data.price

train_X, val_X, train_y, val_y = train_test_split(X, y, random_state=1)

# my_mae = get_mae(50, train_X, val_X, train_y, val_y)

for max_leaf_nodes in [2, 3, 5, 8, 13, 21, 34, 55, 89, 144, 233, 377, 610, 947]:
    my_mae = get_mae(max_leaf_nodes, train_X, val_X, train_y, val_y)
    print('Max leaf nodes: %d \t\t Mean absolute error: %d' % (max_leaf_nodes, my_mae))
