from sklearn import metrics
from catboost import CatBoostRegressor, Pool


def train_model(x_train, y_train):
    train_pool = Pool(x_train, y_train)
    model = CatBoostRegressor(iterations=100, depth=16, learning_rate=0.3, loss_function='RMSE')
    model.fit(train_pool)
    return model


def evaluate_model(model, x_test, y_test):
    ypred = model.predict(x_test)

    score = metrics.mean_absolute_error(y_test, ypred)

    return score

