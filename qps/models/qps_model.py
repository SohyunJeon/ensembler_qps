import numpy as np
import pandas as pd
from sklearn.model_selection import KFold, train_test_split
from sklearn.preprocessing import MinMaxScaler
import optuna
from optuna.samplers import TPESampler
import shap
import ray

from qps.models import evaluation


class SubModel:
    def __init__(self, name: str, version: str, estimator, n_kfold: int=3, use_scale_data: bool=False):
        self.name = name
        self.version = version
        self.estimator = estimator
        self.n_kfold = n_kfold
        self.use_scale_data = use_scale_data
        self.model = None
        self.best_params = None
        self.feature_importance = None
        self.scores = None


    def __str__(self):
        return f'{self.name}: {self.version}'


    def select_params(self, X: pd.DataFrame, y: pd.Series, n_tunning:int, tunning_scoring:str)-> list:
        # cv = LeaveOneOut()
        cv = KFold(n_splits=self.n_kfold)
        study = optuna.create_study(direction='maximize', sampler=TPESampler())
        study.optimize(lambda trial: self.objective(trial, X, y, cv, tunning_scoring),
                       n_trials=n_tunning)
        print(f'Best trial: score - {study.best_trial.values} / params - {study.best_trial.params} ')
        self.best_params = study.best_trial.params


    def objective(self):
        pass


    def set_important_features(self, data: pd.DataFrame):
        pass


    def fit_predict(self, X: pd.DataFrame, y: pd.Series, n_tunning:int=5,
                    tunning_scoring:str='neg_root_mean_squared_error')-> np.array:
        self.select_params(X, y, n_tunning, tunning_scoring)
        self.model = self.estimator.set_params(**self.best_params)

        ## sub-model 점수 계산을 위한 부분, Kfold에서 속도를 위해 단순 split

        tr_X, te_X, tr_y, te_y = train_test_split(X, y, shuffle='False')
        self.model.fit(tr_X, tr_y)
        yhat = self.model.predict(te_X)

        # cv = KFold(n_splits=self.n_kfold)
        # yhat = np.array([])
        # for tr_i, te_i in cv.split(X):
        #     tr_X, tr_y = X.iloc[tr_i, :], y[tr_i]
        #     te_X, te_y = X.iloc[te_i, :], y[te_i]
        #     self.model.fit(tr_X, tr_y)
        #     yhat_part = self.model.predict(te_X)
        #     yhat = np.concatenate([yhat, yhat_part])
        # y_eval = y[len(y) - len(yhat):]

        self.scores = {
                    'rmse':evaluation.cal_rmse(te_y, yhat),
                     'r2': evaluation.cal_r2(te_y, yhat),
                       }
        # 최종 모델 fit
        self.model.fit(X, y)
        final_yhat = self.model.predict(X)
        self.set_important_features(X)
        print('Scores : ', self.scores)
        return final_yhat



class MetaModel:
    def __init__(self, name: str, version: str, estimator, n_kfold: int=4, use_scale_data: bool=False):
        self.name = name
        self.version = version
        self.estimator = estimator
        self.use_scale_data = use_scale_data
        self.n_kfold = n_kfold
        self.model = None
        self.best_params = None
        self.scores = None

    def __str__(self):
        return f'{self.name}: {self.version}'

    def fit_predict(self, X: pd.DataFrame, y: pd.Series):
        self.model = self.estimator

        tr_X, te_X, tr_y, te_y = train_test_split(X, y, shuffle='False')
        self.model.fit(tr_X, tr_y)
        yhat = self.model.predict(te_X)

        # cv = KFold(n_splits=self.n_kfold)
        # yhat = np.array([])
        # for tr_i, te_i in cv.split(X):
        #     tr_X, tr_y = X.iloc[tr_i, :], y[tr_i]
        #     te_X, te_y = X.iloc[te_i, :], y[te_i]
        #     self.model.fit(tr_X, tr_y)
        #     yhat_part = self.model.predict(te_X)
        #     yhat = np.concatenate([yhat, yhat_part])
        # y_eval = y[len(y) - len(yhat):]

        self.scores = {
            'rmse': evaluation.cal_rmse(te_y, yhat),
            'r2': evaluation.cal_r2(te_y, yhat),
        }
        # 최종 모델 fit
        self.model.fit(X, y)
        final_yhat = self.model.predict(X)
        print('Scores : ', self.scores)
        return final_yhat




class StackedModel:
    def __init__(self, meta_model, sub_models: dict, variables: list):
        self.meta_model = meta_model
        self.sub_models = sub_models
        self.variables = variables
        self.set_feature_importance()
        self.explainer = None

    def __str__(self):
        return f'Meta-model : {self.meta_model.name} \n' \
               f'Sub-model count : {len(self.sub_models)}'


    def set_feature_importance(self):
        intersect_feats = pd.DataFrame()
        for sub_model in self.sub_models.values():
            if type(sub_model.feature_importance)==type(None):
                continue
            else:
                temp = sub_model.feature_importance.reset_index()
            if len(intersect_feats) == 0:
                intersect_feats = temp.copy()
            else:
                intersect_feats = pd.merge(intersect_feats, temp, on='index',
                                           how='outer')
        intersect_feats = intersect_feats.set_index('index')
        # 기준이 다른 중요도를 0~1로 조정
        scale_model = MinMaxScaler()
        scale_model.fit(intersect_feats)
        scaled_feats = pd.DataFrame(scale_model.transform(intersect_feats),
                                    index=intersect_feats.index). \
            fillna(0)

        scaled_feats['mean'] = scaled_feats.mean(axis=1)
        scaled_feats = scaled_feats.sort_values('mean', ascending=False)
        feats_dict = dict(scaled_feats['mean'])
        self.feature_importance = {k: float(v) for k, v in feats_dict.items()}


    def predict_with_model_for_contribution(self, X:pd.DataFrame):
        new_X = pd.DataFrame()
        for sub_name, sub_model in self.sub_models.items():
            sub_yhat = sub_model.model.predict(X)
            new_X[sub_name] = sub_yhat.ravel()
        final_yhat = self.meta_model.model.predict(new_X)
        return final_yhat


    def set_explainer(self, X: pd.DataFrame):
        self.explainer = shap.explainers.Additive(
            self.predict_with_model_for_contribution, X)


