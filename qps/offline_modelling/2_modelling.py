import pandas as pd
import numpy as np
import timeit
import matplotlib.pyplot as plt
from types import SimpleNamespace
import dill

from qps.models.sub_models.catboost_regressor import CatboostSubModel
from qps.models.sub_models.mlp_regressor import MLPRegressorSubModel
from qps.models.sub_models.neighbors_regressor import KNRSubModel
from qps.models.sub_models.pls_regressor import PLSSubModel
from qps.models.sub_models.rf_regressor import RandomForestSubModel
from qps.models.sub_models.svr_regressor import SVRSubModel
from qps.models.sub_models.xgboost_regressor import XGBoostRegressorSubModel
from qps.models.meta_models.linear_regressor import LRMetaModel
from qps.models.qps_model import StackedModel

from qps.models import evaluation
from modelling_config import model_config
from qps.models.feature_selection import FeatureSelection
from qps.models import preprocessing as prep_func
from qps.models import model_save
from qpslib_model_manager import model_client


#%% Setting ====================================================================
DATA = SimpleNamespace(**model_config['DATA'])
PREP = SimpleNamespace(**model_config['PREPROCESSING'])
SERVICE = SimpleNamespace(**model_config['SERVICE'])




#%% Load Train Data ============================================================
train = pd.read_csv('./data/train_summary.csv')



#%% Preprocess x ===============================================================
df = prep_func.drop_na(train)
df = prep_func.remove_same_values(df, PREP.same_value_ratio)
df = prep_func.remove_high_corr(df, PREP.high_corr)

X, y = df.drop([DATA.id, DATA.time, DATA.y], axis=1), df[DATA.y]


#%% select feature =============================================================
start = timeit.default_timer()

feats_model = FeatureSelection()
selected_feats = feats_model.select(X, y)

X_selected = X.loc[:, selected_feats]

stats = [{'name': name, 'mean': mean, 'std': std} for name, mean, std in
         zip(X_selected.columns, X_selected.mean().values, X_selected.std().values)]
model_meta_info = {'train_stats': stats}



#%% Sub-modelling ==============================================================
X_train = X_selected.copy()
y_train = y.copy()



## 1
cat_sub = CatboostSubModel()
cat_yhat = cat_sub.fit_predict(X_train, y_train)

## 2
mlp_sub = MLPRegressorSubModel()
mlp_yhat = mlp_sub.fit_predict(X_train, y_train)

## 3
knr_sub = KNRSubModel()
knr_yhat = knr_sub.fit_predict(X_train, y_train)

## 4
pls_sub = PLSSubModel()
pls_yhat = pls_sub.fit_predict(X_train, y_train)

## 5
rf_sub = RandomForestSubModel()
rf_yhat = rf_sub.fit_predict(X_train, y_train)

## 6
svr_sub = SVRSubModel()
svr_yhat = svr_sub.fit_predict(X_train, y_train)

## 7
xgb_sub = XGBoostRegressorSubModel()
xgb_yhat = xgb_sub.fit_predict(X_train, y_train)

print('Sub Modeling Elapse: ', timeit.default_timer() - start)


#%% Create new X ===============================================================
sub_yhats = [cat_yhat, mlp_yhat, knr_yhat, pls_yhat.ravel(), rf_yhat,
             svr_yhat, xgb_yhat]
new_X = pd.DataFrame({str(i): yhat for i, yhat in enumerate(sub_yhats)})


#%% Meta-modeling ==============================================================
reg_meta = LRMetaModel()
final_yhat = reg_meta.fit_predict(new_X, y_train)


plt.figure()
plt.plot(y_train, label='y', linewidth=0.7, alpha=0.7)
plt.plot(final_yhat, label='yhat', linewidth=0.7, alpha=0.7)
plt.legend()
plt.show()


#%% Stacked-model define =======================================================
sub_models = {cat_sub.name: cat_sub,
              mlp_sub.name: mlp_sub,
              knr_sub.name: knr_sub,
              pls_sub.name: pls_sub,
              rf_sub.name: rf_sub,
              svr_sub.name: svr_sub,
              xgb_sub.name: xgb_sub}
stacked_model = StackedModel(meta_model=reg_meta,
                             sub_models=sub_models,
                             variables=selected_feats)
stacked_model.set_explainer(X_train)

with open('./temp_data/stacked_model.pkl', 'wb') as f:
    dill.dump(stacked_model, f)

#%% Define Model ===============================================================
client = model_client.QPSModelClient(SERVICE.host, SERVICE.company,
                                     SERVICE.target, SERVICE.result_type)

sub_models_name = 'Init Sub-Models'
stacked_model_name = 'Init Stacked-Model'
model_score = {'rmse': float(np.mean([x.scores['rmse'] for x in sub_models.values()])),
               'r2': float(np.mean([x.scores['r2'] for x in sub_models.values()]))}


#%% Save Model ===============================================================

sub_save_resp = model_save.save_sub_models(client=client,
                                           sub_models=sub_models,
                                           name=sub_models_name)

stacked_save_resp = model_save.save_stacked_model(client=client,
                                                  stacked_model=stacked_model,
                                                  ref_id=sub_save_resp.id,
                                                  score=model_score,
                                                  # score = stacked_model.meta_model.scores,
                                                  name=stacked_model_name,
                                                  meta=model_meta_info,
                                                  feats=stacked_model.feature_importance)
client.set_best_model(stacked_save_resp.id)



