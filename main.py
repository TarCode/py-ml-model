from lgbm import read_data, preprocess_data, train_model, evaluate_model

data = read_data('./data/cape-town-property-listings.csv')

features = ['bedrooms', 'type', 'location', 'area', 'bathrooms', 'parking']
y_label = 'price'

x_train, x_test, y_train, y_test = preprocess_data(data, features, y_label)

bst = train_model(x_train, y_train)

score = evaluate_model(bst, x_test, y_test)

print("SCORE", score)
