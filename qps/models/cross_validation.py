import numpy as np
from sklearn.model_selection import TimeSeriesSplit



class BlockingTimeSeriesSplit:
    def __init__(self, n_splits: float):
        self.n_splits = n_splits

    def get_n_splits(self, groups):
        return self.n_splits

    def split(self, X, y=None, groups=None):
        n_samples = len(X)
        k_fold_size = n_samples // self.n_splits
        indices = np.arange(n_samples)
        margine = 0
        for i in range(self.n_splits):
            start = i * k_fold_size
            stop = start + k_fold_size
            mid = int(0.8 * (stop - start)) + start
            yield indices[start: mid], indices[mid + margine: stop]


class NestedTimeSeriesCV:
    def __init__(self, estimator, inner_cv:TimeSeriesSplit, outer_cv:BlockingTimeSeriesSplit ):
        self.estimator = estimator
        self.inner_cv = inner_cv
        self.outer_cv = outer_cv

