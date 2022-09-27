import sklearn
import pandas as pd
import numpy as np

from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
from sklearn.model_selection import cross_val_score

from optuna import Trial
from qps.models.qps_model import MetaModel


class LRMetaModel(MetaModel):
    def __init__(self):
        super().__init__(name='LinearRegression',
                         version=sklearn.__version__,
                         estimator=LinearRegression())





if __name__=='__main__':
    new_X = pd.read_csv('./temp_data/new_X.csv')
    y_train = pd.read_csv('./temp_data/y_train.csv')['OUTPUT_VALUE']

    model = LRMetaModel()
    yhat = model.fit_predict(new_X, y_train)

    plt.figure()
    plt.plot(yhat, label='yhat', alpha=0.7)
    plt.plot(y_train, label='y', alpha=0.7)
    plt.legend()
    plt.show()

