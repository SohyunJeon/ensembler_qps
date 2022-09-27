import sklearn
import pandas as pd
import numpy as np

from sklearn.ensemble import RandomForestRegressor
import matplotlib.pyplot as plt
from sklearn.model_selection import cross_val_score

from optuna import Trial
from qps.models.qps_model import SubModel


class RandomForestSubModel(SubModel):
    def __init__(self):
        super().__init__(name='RandomForestRegressor',
                         version=sklearn.__version__,
                         estimator=RandomForestRegressor(n_jobs=-1))


    def objective(self, trial:Trial, X: pd.DataFrame, y: pd.Series, cv, tunning_scoring):
        params = {
            'n_estimators': trial.suggest_int('n_estimators', 100, 1000, 100),
        }
        model = self.estimator.set_params(**params)
        result = cross_val_score(model, X, y, cv=cv,
                                 scoring=tunning_scoring,
                                 n_jobs=-1)
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

    model = RandomForestSubModel()
    yhat = model.fit_predict(X_selected, y)

    plt.figure()
    plt.plot(yhat, label='yhat', alpha=0.7)
    plt.plot(y, label='y', alpha=0.7)
    plt.legend()
    plt.show()

