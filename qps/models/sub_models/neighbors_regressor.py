import sklearn
import numpy as np
import pandas as pd
from sklearn.neighbors import KNeighborsRegressor
import matplotlib.pyplot as plt

from optuna import Trial
from qps.models.qps_model import SubModel
from sklearn.model_selection import cross_val_score



class KNRSubModel(SubModel):
    def __init__(self):
        super().__init__(name='KNeighborsRegressor',
                         version=sklearn.__version__,
                         estimator=KNeighborsRegressor(),
                         )

    def objective(self, trial: Trial, X: pd.DataFrame, y: pd.Series, cv,
                  tunning_scoring):
        params = {

            'n_neighbors': trial.suggest_int('n_neighbors', 3, 10, 1),
            'weights': trial.suggest_categorical('weights', ['uniform', 'distance']),
            'algorithm': trial.suggest_categorical('algorithm', ['auto',
                                                                 'ball_tree',
                                                                 'kd_tree',
                                                                 'brute']),
        }
        model = self.estimator.set_params(**params)
        result = cross_val_score(model, X, y, cv=cv,
                                 scoring=tunning_scoring,
                                 n_jobs=-1)
        print('result : ', result)
        return np.mean(result)

    def set_important_features(self, data: pd.DataFrame):
        self.feature_importance = None


if __name__=='__main__':
    X_selected = pd.read_csv('./data/train_x_selected.csv')
    y = pd.read_csv('./data/train.csv')['OUTPUT_VALUE']

    model = KNRSubModel()
    yhat = model.fit_predict(X_selected, y)

    plt.figure()
    plt.plot(yhat, label='yhat', alpha=0.7)
    plt.plot(y, label='y', alpha=0.7)
    plt.legend()
    plt.show()