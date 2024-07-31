# from cuml.preprocessing import StandardScaler
from sklearn.preprocessing import StandardScaler, OneHotEncoder
import numpy as np


class StandardScalerd(object):
    # another possibility is to compute the mean and standard deviation only using partial_fit
    def __init__(self, keys, copy=True, with_mean=True, with_std=True, continuous_variables=None):
        self.keys = keys
        self.copy = copy
        self.with_mean = with_mean
        self.with_std = with_std
        self.continuous_variables = continuous_variables
        self.continuous_vars_indexes = None
        self.transformer = {k: StandardScaler(copy=copy, with_mean=with_mean, with_std=with_std) for k in keys}

    def __call__(self, data_dict):
        # here comes in a dictionary with numpy arrays or tensors
        for k in self.keys:
            if k in data_dict.keys():
                data_dict[k][self.continuous_vars_indexes] = self.transformer[k].transform(
                    data_dict[k][[self.continuous_vars_indexes]])
        return data_dict

    def fit(self, data_pd):
        # here comes in a pandas dataframe with tabular data
        if self.continuous_variables is None:
            self.continuous_variables = data_pd.columns
        self.continuous_vars_indexes = data_pd.columns.get_indexer(self.continuous_variables)
        for k in self.keys:
            assert len(list(data_pd.shape)) == 2
            self.transformer[k].fit(data_pd.loc[:, self.continuous_variables].values)
