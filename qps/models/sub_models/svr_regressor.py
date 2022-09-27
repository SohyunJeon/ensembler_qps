import sklearn
import numpy as np
import pandas as pd
from sklearn.svm import SVR
import matplotlib.pyplot as plt

from sklearn.model_selection import KFold
from optuna import Trial
from qps.models.qps_model import SubModel
from sklearn.model_selection import cross_val_score


class SVRSubModel(SubModel):
    def __init__(self):
        super().__init__(name='SVMRegressor',
                         version=sklearn.__version__,
                         estimator=SVR(),
                         )

    def objective(self, trial: Trial, X: pd.DataFrame, y: pd.Series, cv,
                  tunning_scoring):
        params = {
            'kernel': 'linear',
            'gamma': trial.suggest_categorical('gamma', ['scale', 'auto']),
            'epsilon': trial.suggest_float('epsilon', 0.1, 0.3),
            'C': trial.suggest_float('C', 0.8, 1.2)
        }
        model = self.estimator.set_params(**params)
        result = cross_val_score(model, X, y, cv=cv,
                                 scoring=tunning_scoring,
                                 n_jobs=-1)
        print('result : ', result)
        return np.mean(result)


    def set_important_features(self, data: pd.DataFrame):
        imp_feats = pd.Series(abs(self.model.coef_.ravel()),
                              index=data.columns)
        imp_feats = imp_feats[imp_feats > imp_feats.mean()]
        self.feature_importance = imp_feats



if __name__=='__main__':
    X_selected = pd.read_csv('./data/train_x_selected.csv')
    y = pd.read_csv('./data/train.csv')['OUTPUT_VALUE']

    model = SVRSubModel()
    yhat = model.fit_predict(X_selected, y)

    plt.figure()
    plt.plot(yhat, label='yhat', alpha=0.7)
    plt.plot(y, label='y', alpha=0.7)
    plt.legend()
    plt.show()