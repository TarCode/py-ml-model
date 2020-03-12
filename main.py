from preprocess import read_data, preprocess_data
import lgbm
import cboost

data = read_data('./data/cape-town-property-listings.csv')

features = ['bedrooms', 'type', 'location', 'area', 'bathrooms', 'parking']
y_label = 'price'

x_train, x_test, y_train, y_test = preprocess_data(data, features, y_label)

# LightGBM model and score
print("LIGHT GBM")
bst = lgbm.train_model(x_train, y_train)
score = lgbm.evaluate_model(bst, x_test, y_test)
print("SCORE", score)

print("**************")
print("CATBOOST")
model = cboost.train_model(x_train, y_train)
score = cboost.evaluate_model(model, x_test, y_test)
print("SCORE", score)