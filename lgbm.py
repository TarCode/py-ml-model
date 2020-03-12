from sklearn import metrics
import lightgbm as lgb


def train_model(x_train, y_train):
    dtrain = lgb.Dataset(x_train, label=y_train)

    param = {
        'boosting_type': 'gbdt',
        'objective': 'regression',
        'num_leaves': 64,
        'learning_rate': 0.3,
        'feature_fraction': 0.6,
    }
    num_round = 1000
    bst = lgb.train(param, dtrain, num_round)

    return bst


def evaluate_model(model, x_test, y_test):
    ypred = model.predict(x_test)

    score = metrics.mean_absolute_error(y_test, ypred)

    return score

