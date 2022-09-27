import timeit
from types import SimpleNamespace
import pandas as pd
import numpy as np
from itertools import chain
import dill as dill
import json
import grpc
import traceback
from types import SimpleNamespace

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
from qps.models import model_save
from qps.models import preprocessing as prep_func
from qps.models.feature_selection import FeatureSelection
from qps.models.summary import SummaryDB
from modelling_config import model_config
from qps.models.qps_model import StackedModel
from qpslib_model_manager import model_client
from qpslib_retrain_manager import retrain_client
from common.error import make_error_msg
from common.handler import Handler

import config



class QPSModelUpdate(Handler):
    def __init__(self):
        self.DATA = SimpleNamespace(**model_config['DATA'])
        self.PREP = SimpleNamespace(**model_config['PREPROCESSING'])
        self.SERVICE = SimpleNamespace(**model_config['SERVICE'])
        self.SETTING = SimpleNamespace(**model_config['UPDATE'])
        self.stacked_model = None

    # @concurrent.process(timeout=30) # not work with
    def run(self, data: SimpleNamespace):
        ## Initialize
        self.client = model_client.QPSModelClient(self.SERVICE.host,
                                             data.company,
                                             data.target,
                                             data.result_type)
        self.rt_client = retrain_client.QPSRetrainClient(self.SERVICE.host,
                                                    data.company,
                                                    data.target,
                                                    data.master_id,
                                                    data.result_type)
        output = {}
        self.comments = []

        ## 1. Load best model
        try:
            model_info = self.client.get_best_model()
            print(f'model_id: {model_info.id}')
            output['model_id'] = model_info.id
            self.stacked_model = dill.loads(self.client.download_model(model_info.id))
        except grpc.RpcError as e:
            output['error'] = {'message': e.details()}
            print(f'output: {output}')
            return output


        ## 2. Load Retrain data & Evaluation data
        try:
            # summ_df = SummaryDB().load_data(self.SETTING.retain_data_count)
            summ_df = SummaryDB().load_data(500)
            retrain_raw_df = summ_df.loc[summ_df[self.DATA.y] != None, :].\
                reset_index(drop=True)                                          # y==None 제거
            eval_df = retrain_raw_df.loc[len(retrain_raw_df)-(self.SETTING.eval_data_count):           # row
                                     , :]              # column
        except Exception as e:
            output['error'] = make_error_msg(str(e), f'Load Data : {traceback.format_exc()}')
            return output


        ## 3. Meta-model update & save
        try:
            updated_model = self.update_model(retrain_raw_df)
            self.stacked_model.meta_model = updated_model
            stacked_save_resp = model_save.\
                save_stacked_model(client=self.client,
                                   stacked_model=self.stacked_model,
                                   ref_id=model_info.ref,
                                   score=model_info.score,
                                   name='Meta-model updated',
                                   meta=model_info.meta,
                                   feats=model_info.feature_importance)
            output['new_model'] = stacked_save_resp.id
        except Exception as e:
            output['error'] = make_error_msg(str(e), f'Meta-model update : {traceback.format_exc()}')
            return output


        ## 4. Load models and evaluate
        try:
            eval_models = self.load_managing_models()
            models_eval_list = self.evaluate_models(eval_models, eval_df)

            r2_list = [m['r2'] for m in models_eval_list]
            rmse_list = [m['rmse'] for m in models_eval_list]
            best_model_id = models_eval_list[np.argmin(rmse_list)]['stacked_model']

            output['evaluation'] = {
                'used_data': eval_df[self.DATA.id].tolist(),
                'models': models_eval_list,
                'best_model': best_model_id
            }
        except Exception as e:
            output['error'] = make_error_msg(str(e), f'Model evaluation : {traceback.format_exc()}')
            return output


        ## 5. Decide retrain model or not
        if np.all(np.array(r2_list) < self.SETTING.r2_limit):
            self.comments.append(f'Max R2 : {max(r2_list)}')
            if self.rt_client.is_retraining():
                self.comments.append(f'Previous retraining has not finished yet.')
                do_retrain = False
            else:
                do_retrain = True
        else:
            do_retrain = False

        if do_retrain:
            try:
                self.rt_client.start_retrain()
                new_stacked_model, model_meta_info = self.retrain_model(retrain_raw_df)
                new_save_resp = self.save_models(new_stacked_model, model_meta_info)
                output['retrain'] = {
                    'sub_model': new_save_resp.ref,
                    'stacked_model': new_save_resp.id
                }
                self.rt_client.finish_retrain()
            except Exception as e:
                output['error'] = make_error_msg(str(e),
                                                 f'Retrain : {traceback.format_exc()}')
                self.rt_client.finish_retrain()
                return output
        else:
            self.client.set_best_model(best_model_id)

        output['comment'] = '||'.join(self.comments)
        return output




    def load_managing_models(self) -> list:
        """
        관리하고 있는 모델 불러오기
        현재 기준 : 최근 생성된 sub_model_count 개수만큼 sub model과
        해당 sub model을 사용한 stacked model중 rmse가 작은 순으로 stacked_model_count개수 만큼 로드
        즉, sub_model_count * stacked_model_count
        :return: stacked model info 리스트
        """
        model_list = []
        sub_query = {'type': 'SUB_MODEL'}
        sub_sort = {'created_at': -1}
        sub_list = self.client.list_model_info(sub_query, sub_sort,
                                               self.SETTING.sub_model_count)
        sub_id_list = [model.id for model in sub_list]
        for sub_id in sub_id_list:
            stacked_query = {'ref': sub_id}
            stacked_sort = {'score.rmse': 1}
            stacked_list = self.client.list_model_info(stacked_query,
                                                       stacked_sort,
                                                       self.SETTING.stacked_model_count)
            model_list += stacked_list
        return model_list


    def evaluate_models(self, model_info_list: list, eval_data:pd.DataFrame)\
            -> list:
        """
        평가 데이터를 이용해 각 평가 모델들을 평가하고 모델의 점수를 업데이트한다.

        :param model_info_list: stacked-model info 리스트
        :param eval_data: 전체 평가 데이터
        :return: 각 stacked-model의 결과 리스트
        """
        models_eval_list = []

        for model_info in model_info_list:
            # 개별 stacked model 평가 :
            # 모델 다운로드 -> 평가 데이터 지정 -> 예측 -> 점수 계산 -> 각 모델 점수 업데이트
            model = dill.loads(self.client.download_model(model_info.id))
            eval_X = eval_data.loc[:, model.variables]
            eval_y = eval_data[self.DATA.y]
            yhat = self.predict_with_model(eval_X)
            rmse = evaluation.cal_rmse(eval_y, yhat)
            r2 = evaluation.cal_r2(eval_y, yhat)
            model_save.save_model_score(self.client, model_info.id,
                                        {'rmse': rmse, 'r2': r2})
            model_eval = {'sub_model': model_info.ref,
                          'stacked_model': model_info.id,
                          'rmse': rmse,
                          'r2': r2}
            models_eval_list.append(model_eval)
        return models_eval_list



    def update_model(self, data: pd.DataFrame) -> LRMetaModel:
        X = data.loc[:, self.stacked_model.variables]
        y = data[self.DATA.y]
        new_X = self.predict_with_submodel(X)
        reg_meta = LRMetaModel()
        final_yhat = reg_meta.fit_predict(new_X, y)
        return reg_meta


    def predict_with_model(self, X: pd.DataFrame) -> float:
        new_X = self.predict_with_submodel(X)
        final_yhat = self.stacked_model.meta_model.model.predict(new_X)
        return final_yhat


    def predict_with_submodel(self, X: pd.DataFrame) -> pd.DataFrame:
        sub_yhats = []
        for sub_model in self.stacked_model.sub_models.values():
            sub_yhat = sub_model.model.predict(X)
            sub_yhats.append(sub_yhat)
        new_X = pd.DataFrame(sub_yhats).T
        return new_X


    def retrain_model(self, data: pd.DataFrame) -> StackedModel:
        """
        모델 재학습
        :param X:
        :param y:
        :return:
        """
        # Preprocessing
        df = prep_func.drop_na(data)
        df = prep_func.remove_same_values(df, self.PREP.same_value_ratio)
        df = prep_func.remove_high_corr(df, self.PREP.high_corr)

        X = df.drop([self.DATA.id, self.DATA.time, self.DATA.y], axis=1)
        y = df[self.DATA.y]

        # Select variables
        feats_model = FeatureSelection()
        selected_feats = feats_model.select(X, y)

        X_selected = X.loc[:, selected_feats]
        stats = [{'name': name, 'mean': mean, 'std': std} for name, mean, std in
                 zip(X_selected.columns, X_selected.mean().values,
                     X_selected.std().values)]
        model_meta_info = {'train_stats': stats}

        # Sub-modeling
        cat_sub = CatboostSubModel() # 1
        cat_yhat = cat_sub.fit_predict(X_selected, y)
        mlp_sub = MLPRegressorSubModel() # 2
        mlp_yhat = mlp_sub.fit_predict(X_selected, y)
        knr_sub = KNRSubModel() # 3
        knr_yhat = knr_sub.fit_predict(X_selected, y)
        pls_sub = PLSSubModel() # 4
        pls_yhat = pls_sub.fit_predict(X_selected, y)
        rf_sub = RandomForestSubModel() # 5
        rf_yhat = rf_sub.fit_predict(X_selected, y)
        svr_sub = SVRSubModel() # 6
        svr_yhat = svr_sub.fit_predict(X_selected, y)
        xgb_sub = XGBoostRegressorSubModel() # 7
        xgb_yhat = xgb_sub.fit_predict(X_selected, y)

        # Create new X
        sub_yhats = [cat_yhat, mlp_yhat, knr_yhat, pls_yhat.ravel(), rf_yhat,
                     svr_yhat, xgb_yhat]
        new_X = pd.DataFrame({str(i): yhat for i, yhat in enumerate(sub_yhats)})

        # Meta-modeling
        reg_meta = LRMetaModel()
        final_yhat = reg_meta.fit_predict(new_X, y)

        # Stacked-model define
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
        stacked_model.set_explainer(X_selected)

        return stacked_model, model_meta_info


    def save_models(self, model: StackedModel, model_meta_info: None):
        sub_models_name = 'Retrained Sub-Models'
        stacked_model_name = 'Retrained Stacked-Model'
        model_score = {'rmse': float(
            np.mean([x.scores['rmse'] for x in model.sub_models.values()])),
            'r2': float(np.mean(
                [x.scores['r2'] for x in model.sub_models.values()]))}

        sub_save_resp = model_save.save_sub_models(client=self.client,
                                                   sub_models=model.sub_models,
                                                   name=sub_models_name)

        stacked_save_resp = model_save.save_stacked_model(client=self.client,
                                                          stacked_model=model,
                                                          ref_id=sub_save_resp.id,
                                                          score=model_score,
                                                          name=stacked_model_name,
                                                          meta=model_meta_info,
                                                          feats=model.feature_importance)
        self.client.set_best_model(stacked_save_resp.id)
        return stacked_save_resp




if __name__ == '__main__':
    esbr_input = '''{
    "company":"BRIQUE",
    "target":"TTA",
    "service_type":"QPS",
    "input_case":"MODEL_UPDATE",
    "result_type":"REGRESSION",
    "master_id":"12JB439A_03",
    "residual":null
   }'''

    start = timeit.default_timer()
    config.load_config('../../config.yml')

    data = SimpleNamespace(**json.loads(esbr_input))

    runner = QPSModelUpdate()
    output = runner.run(data)
    print('Elase : ', timeit.default_timer() - start)
    print(output)

