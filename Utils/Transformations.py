# from cuml.preprocessing import StandardScaler
from sklearn.preprocessing import StandardScaler, OneHotEncoder
import numpy as np
import monai
import torchvision


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


def transform_pipeline_old(config):
    img_keys = [k for k in config['MODALITY'].keys() if config['MODALITY'][k]]
    records_keys = ['records'] if config['RECORDS']['records'] else []

    if len(records_keys) > 0 or len(img_keys) > 0:
        train_transform = []
        val_transform = []

        if len(records_keys) > 0:
            if 'continuous_cols' not in config['DATA'].keys():
                non_continuous = [config['DATA']['target'], config['DATA']['censor_label'],
                                  config['DATA']['subject_label']]
                config['DATA']['continuous_cols'] = [col for col in config['DATA']['clinical_cols']
                                                     if col not in non_continuous]
            train_transform += [
                StandardScalerd(keys=records_keys, continuous_variables=config['DATA']['continuous_cols']),]
            val_transform += [
                StandardScalerd(keys=records_keys, continuous_variables=config['DATA']['continuous_cols']),]

        if len(img_keys) > 0:
            condition = (('RTSTRUCT' not in config['MODALITY'].keys()) or (not config['MODALITY']['RTSTRUCT']) and
                         (config['MODALITY']['CT']) and ('CT' in config['MODALITY'].keys()))
            train_transform = [
                monai.transforms.EnsureChannelFirstd(keys=img_keys + ['RTSTRUCT'] if condition else img_keys),
                monai.transforms.CropForegroundd(keys=img_keys, source_key='RTSTRUCT', select_fn=threshold_at_one),
                monai.transforms.Resized(keys=img_keys, spatial_size=config['DATA']['dim']),
                monai.transforms.RandAffined(keys=img_keys),
                monai.transforms.RandHistogramShiftd(keys=img_keys),
                monai.transforms.RandAdjustContrastd(keys=img_keys),
                monai.transforms.RandGaussianNoised(keys=img_keys),
                monai.transforms.ScaleIntensityd(keys=list(set(img_keys).difference(set(['RTDOSE'])))),
            ]

            val_transform = [
                monai.transforms.EnsureChannelFirstd(keys=img_keys + ['RTSTRUCT'] if condition else img_keys),
                monai.transforms.CropForegroundd(keys=img_keys, source_key='RTSTRUCT', select_fn=threshold_at_one),
                monai.transforms.Resized(keys=img_keys, spatial_size=config['DATA']['dim']),
                monai.transforms.ScaleIntensityd(list(set(img_keys).difference(set(['RTDOSE'])))),
            ]

            if not config['DATA']['crop_foreground']:
                del train_transform[-7]  # remove crop foreground
                del val_transform[-3]  # remove crop foreground

        train_transform = torchvision.transforms.Compose(train_transform)
        val_transform = torchvision.transforms.Compose(val_transform)
    else:
        train_transform = None
        val_transform = None

    return train_transform, val_transform
