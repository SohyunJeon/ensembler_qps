import pandas as pd

from sklearn.ensemble import RandomForestRegressor
from sklearn.cross_decomposition import PLSRegression
from sklearn.feature_selection import SelectFromModel
from catboost import CatBoostRegressor

from modelling_config import model_config
from types import SimpleNamespace



class FeatureSelection:
    def __init__(self):
        pass

    def select(self, X:pd.DataFrame, y: pd.Series) -> list:
        rfr_selector = SelectFromModel(estimator=RandomForestRegressor(n_jobs=-1)).\
            fit(X, y)
        rfr_selected = X.columns[rfr_selector.get_support()]

        pls_selector = SelectFromModel(estimator=PLSRegression()).fit(X, y)
        pls_selected = X.columns[pls_selector.get_support()]

        cat_selector = SelectFromModel(estimator=CatBoostRegressor(verbose=False)).fit(X, y)
        cat_selected = X.columns[cat_selector.get_support()]

        selected_feats = list(rfr_selected & pls_selected & cat_selected)
        return selected_feats




if __name__ == '__main__':
    DATA = SimpleNamespace(**model_config['DATA'])
    PREP = SimpleNamespace(**model_config['PREPROCESSING'])

    train = pd.read_csv('./data/train_summary.csv')
    X, y = train.drop([DATA.id, DATA.y], axis=1), train[DATA.y]

    model = FeatureSelection()
    selected_feats = model.select(X, y)