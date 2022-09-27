import pandas as pd
import numpy as np
import timeit
from catboost import CatBoostRegressor
import catboost
from optuna import Trial
from qps.models.qps_model import SubModel
from sklearn.model_selection import cross_val_score
import matplotlib.pyplot as plt


class CatboostSubModel(SubModel):
    def __init__(self):
        super().__init__(name='CatboostRegressor',
                         version=catboost.__version__,
                         estimator=CatBoostRegressor(),
                         )


    def objective(self, trial: Trial, X: pd.DataFrame, y: pd.Series, cv,
                  tunning_scoring):
        params = {
            'verbose': False,
            'iterations':trial.suggest_int('iterations', 100, 500, 100),
            'learning_rate':trial.suggest_float('learning_rate', 0.05, 0.1),
            'depth': trial.suggest_int('depth', 2, 8, 2)
        }
        model = self.estimator.set_params(**params)
        result = cross_val_score(model, X, y, cv=cv,
                                 scoring=tunning_scoring,
                                 n_jobs=-1)
        print('result : ', result)
        return np.mean(result)


    def set_important_features(self, data: pd.DataFrame):
        imp_feats = pd.Series(self.model.get_feature_importance(),
                              index=data.columns)
        imp_feats = imp_feats[imp_feats > imp_feats.mean()]
        self.feature_importance = imp_feats



if __name__ == '__main__':
    X_selected = pd.read_csv('./data/train_x_selected.csv')
    y = pd.read_csv('./data/train.csv')['OUTPUT_VALUE']

    model = CatboostSubModel()
    start = timeit.default_timer()
    yhat = model.fit_predict(X_selected, y)
    print('Elapse :', timeit.default_timer() - start)


    plt.figure()
    plt.plot(yhat, label='yhat', alpha=0.7)
    plt.plot(y, label='y', alpha=0.7)
    plt.legend()
    plt.show()