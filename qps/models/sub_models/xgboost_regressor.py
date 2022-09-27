import numpy as np
import pandas as pd

import xgboost
import optuna
from optuna import Trial
from optuna.samplers import TPESampler
from sklearn.model_selection import cross_val_score
import matplotlib.pyplot as plt

from qps.models.qps_model import SubModel


class XGBoostRegressorSubModel(SubModel):
    def __init__(self):
        super().__init__(name='XGBoost_Regressor',
                         version=xgboost.__version__,
                         estimator=xgboost.XGBRegressor())


    def objective(self, trial:Trial, X: pd.DataFrame, y: pd.Series, cv, tunning_scoring):
        params = {
            'lambda': trial.suggest_loguniform('lambda', 1e-3, 10.0),
            'learning_rate': trial.suggest_loguniform('learning_rate', 0.01, 0.5),
            'n_estimators': trial.suggest_int('n_estimators', 500, 700, 100),
            'subsample': trial.suggest_loguniform('subsample', 0.8, 1),
            'eta': trial.suggest_loguniform('eta', 0.05, 0.3),
            'min_child_weight': trial.suggest_int('min_child_weight', 1, 3)

            # 'alpha': trial.suggest_loguniform('alpha', 1e-3, 10.0),
            # 'colsample_bytree': trial.suggest_categorical('colsample_bytree',
            #                                               [0.8, 0.9, 1.0]),
            # 'learning_rate': trial.suggest_loguniform('learning_rate', 0.01, 0.5),
            # 'n_estimators': trial.suggest_categorical('n_estimators', [100, 500, 700]),
            # 'max_depth': trial.suggest_int('max_depth', 4, 10),
            # 'random_state': 42,
            # 'min_child_weight': trial.suggest_int('min_child_weight', 1, 300)
                    }
        model = self.estimator.set_params(**params)
        result = cross_val_score(model, X, y, cv=cv, scoring=tunning_scoring)
        print('result : ', result)
        return np.mean(result)


    def set_important_features(self, data: pd.DataFrame):
        imp_feats = pd.Series(self.model.feature_importances_,
                              index=data.columns)
        imp_feats = imp_feats[imp_feats > imp_feats.mean()]
        self.feature_importance = imp_feats



if __name__=='__main__':
    X_selected = pd.read_csv('./data/train_x_selected.csv')
    y = pd.read_csv('./data/train.csv')['OUTPUT_VALUE']

    model = XGBoostRegressorSubModel()
    yhat = model.fit_predict(X_selected, y)

    plt.figure()
    plt.plot(yhat, label='yhat', alpha=0.7)
    plt.plot(y, label='y', alpha=0.7)
    plt.legend()
    plt.show()