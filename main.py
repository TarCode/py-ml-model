import pandas as pd
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split

"""
    TODO: Process the CSV to clean the data and remove all strings 
    (Add bedrooms field and remove 'R' and space from pricing)
"""


def get_mae(max_leaf_nodes, train_X, val_X, train_y, val_y):
    model = DecisionTreeRegressor(max_leaf_nodes=max_leaf_nodes, random_state=0)
    model.fit(train_X, train_y)
    preds_val = model.predict(val_X)
    print('Prediction value', preds_val)
    mae = mean_absolute_error(val_y, preds_val)
    return mae


data_path = './property-listings.csv'
data = pd.read_csv(data_path, dtype={"Close": float})
data = data.dropna(axis=0)

print('PRICE COLUMNS')
print(data.columns)

features = ['title', 'location', 'area', 'price']

X = data[features]
y = data.price

train_X, val_X, train_y, val_y = train_test_split(X, y, random_state=1)


print('DATA DESCRIBE')
print(X.describe())

model = DecisionTreeRegressor(random_state=1)

model.fit(X, y)

print('MAKE PREDICTIONS FOR THE FOLLOWING 5 ROWS')
print(data.head())

predicted_prices = model.predict(X)
print('THE PREDICTIONS ARE')
print(predicted_prices)

print('MEAN ABSOLUTE ERROR')
print(mean_absolute_error(y, predicted_prices))

for max_leaf_nodes in [2, 3, 5, 8, 10, 2000]:
    my_mae = get_mae(max_leaf_nodes, train_X, val_X, train_y, val_y)
    print('Max leaf nodes: %d \t\t Mean absolute error: %d' % (max_leaf_nodes, my_mae))
