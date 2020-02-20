import pandas as pd

crypto_price_path = './crypto_prices.csv'
crypto_data = pd.read_csv(crypto_price_path)
crypto_data = crypto_data.dropna(axis=0)

print('CRYPTO PRICE COLUMNS')
print(crypto_data.columns)

features = ['Symbol', 'Open', 'Close', 'Volume', 'High', 'Low', 'DateTime']

X = crypto_data[features]

print('CRYPTO DATA COLUMNS')
print(X.columns)

print('CRYPTO DATA HEAD')
print(X.head())

print('CRYPTO DATA DESCRIBE')
print(X.describe())
