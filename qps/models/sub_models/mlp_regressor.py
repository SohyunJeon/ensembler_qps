import sklearn
import numpy as np
import pandas as pd
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold,train_test_split
import shap
import ray


from optuna import Trial
from qps.models.qps_model import SubModel
from sklearn.model_selection import cross_val_score
from qps.models import evaluation



class MLPRegressorSubModel(SubModel):
    def __init__(self):
        super().__init__(name='MLPRegressor',
                         version=sklearn.__version__,
                         estimator=MLPRegressor(),
                         )

    def objective(self, trial: Trial, X: pd.DataFrame, y: pd.Series, cv,
                  tunning_scoring):
        params = {
            'learning_rate': trial.suggest_categorical('learning_rate', ['constant',
                                                                        'invscaling',
                                                                        'adaptive']),
            'early_stopping': True,
            'max_iter': trial.suggest_int('max_iter', 600, 2000, 200)
        }
        model = self.estimator.set_params(**params)
        result = cross_val_score(model, X, y, cv=cv,
                                 scoring=tunning_scoring,
                                 n_jobs=-1)
        print('result : ', result)
        return np.mean(result)


    def fit_predict(self, X: pd.DataFrame, y: pd.Series, n_tunning:int=5,
                    tunning_scoring:str='neg_root_mean_squared_error')-> np.array:
        X_values = X.values
        self.select_params(X_values, y, n_tunning, tunning_scoring)
        self.model = self.estimator.set_params(**self.best_params)

        tr_X, te_X, tr_y, te_y = train_test_split(X, y, shuffle='False')
        self.model.fit(tr_X.values, tr_y)
        yhat = self.model.predict(te_X.values)

        # cv = KFold(n_splits=self.n_kfold)
        # yhat = np.array([])
        # # Leave One-and-out의 yhat
        # for tr_i, te_i in cv.split(X):
        #     tr_X, tr_y = X_values[tr_i, :], y[tr_i]
        #     te_X, te_y = X_values[te_i, :], y[te_i]
        #     self.model.fit(tr_X, tr_y)
        #     yhat_part = self.model.predict(te_X)
        #     yhat = np.concatenate([yhat, yhat_part])
        # y_eval = y[len(y) - len(yhat):]

        self.scores = {
                    'rmse':evaluation.cal_rmse(te_y, yhat),
                     'r2': evaluation.cal_r2(te_y, yhat),
                       }
        # 최종 모델 fit
        self.model.fit(X_values, y)
        final_yhat = self.model.predict(X_values)
        self.set_important_features(X)
        print('Scores : ', self.scores)
        return final_yhat


    def set_important_features(self, data: pd.DataFrame):
        explainer = shap.AdditiveExplainer(self.model.predict, data)
        imp_feats = pd.Series(abs(explainer(data)[0].values),
                              index=data.columns)
        imp_feats = imp_feats[imp_feats > imp_feats.mean()]
        self.feature_importance = imp_feats



if __name__=='__main__':
    X_selected = pd.read_csv('./data/train_x_selected.csv')
    y = pd.read_csv('./data/train.csv')['OUTPUT_VALUE']

    scaler = StandardScaler()
    scaler.fit(X_selected)
    X_train = scaler.transform(X_selected)

    model = MLPRegressorSubModel()
    yhat = model.fit_predict(X_selected, y)

    plt.figure()
    plt.plot(yhat, label='yhat', alpha=0.7)
    plt.plot(y, label='y', alpha=0.7)
    plt.legend()
    plt.show()
