import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_absolute_error

crypto_price_path = './crypto_prices.csv'
crypto_data = pd.read_csv(crypto_price_path, dtype={"Close": float})
crypto_data = crypto_data.dropna(axis=0)

print('CRYPTO PRICE COLUMNS')
print(crypto_data.columns)

features = ['Open', 'Close', 'Volume', 'High', 'Low']

X = crypto_data[features]
y = crypto_data.Close

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
