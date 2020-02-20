import pandas as pd
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split


def get_mae(max_leaf_nodes, train_X, val_X, train_y, val_y):
    model = DecisionTreeRegressor(max_leaf_nodes=max_leaf_nodes, random_state=0)
    model.fit(train_X, train_y)
    preds_val = model.predict(val_X)
    print('Prediction value', preds_val)
    mae = mean_absolute_error(val_y, preds_val)
    return mae


crypto_price_path = './crypto_prices.csv'
crypto_data = pd.read_csv(crypto_price_path, dtype={"Close": float})
crypto_data = crypto_data.dropna(axis=0)

print('CRYPTO PRICE COLUMNS')
print(crypto_data.columns)

features = ['Open', 'Close', 'Volume', 'High', 'Low']

X = crypto_data[features]
y = crypto_data.Close

train_X, val_X, train_y, val_y = train_test_split(X, y, random_state=1)

print('CRYPTO DATA COLUMNS')
print(X.columns)

print('CRYPTO DATA DESCRIBE')
print(X.describe())

crypto_model = DecisionTreeRegressor(random_state=1)

crypto_model.fit(X, y)

print('MAKE PREDICTIONS FOR THE FOLLOWING 5 CRYPTOCURRENCIES')
print(crypto_data.head())

predicted_crypto_closing_prices = crypto_model.predict(X)
print('THE PREDICTIONS ARE')
print(predicted_crypto_closing_prices)

print('MEAN ABSOLUTE ERROR')
print(mean_absolute_error(y, predicted_crypto_closing_prices))

for max_leaf_nodes in [2, 3, 5, 8, 10, 2000]:
    my_mae = get_mae(max_leaf_nodes, train_X, val_X, train_y, val_y)
    print('Max leaf nodes: %d \t\t Mean absolute error: %d' % (max_leaf_nodes, my_mae))
