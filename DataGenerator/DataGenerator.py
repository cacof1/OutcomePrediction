from pytorch_lightning import LightningDataModule, LightningModule, Trainer, seed_everything
from torch.utils.data import DataLoader
import numpy as np
import torch
from sklearn.model_selection import train_test_split, StratifiedShuffleSplit
import pandas as pd
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler, LabelEncoder, OrdinalEncoder
from sklearn.compose import ColumnTransformer
from pathlib import Path
from collections import Counter
from monai.visualize.utils import matshow3d
from monai.transforms import LoadImage, EnsureChannelFirstd, ResampleToMatchd, ResizeWithPadOrCropd

class DataGenerator(torch.utils.data.Dataset):
    def __init__(self, SubjectList, config=None, keys=['CT'], transform=None, inference=False,
                 clinical_cols=None, session=None, **kwargs):
        super().__init__()
        self.config = config
        self.SubjectList = SubjectList
        self.keys = keys
        self.transform = transform
        self.inference = inference
        self.clinical_cols = clinical_cols

    def __len__(self):
        return int(self.SubjectList.shape[0])

    def __getitem__(self, i):

        data = {}
        meta = {}
        data['slabel']  = self.SubjectList.loc[i, 'subject_label']
        ## Load CT
        if 'CT' in self.keys:
            CTPath = self.SubjectList.loc[i, 'CT_Path']
            CT_Path = Path(CTPath, 'CT.nii.gz')
            data['CT'] = LoadImage(image_only=True)(CT_Path)

        ## Load RTDOSE
        if 'RTDOSE' in self.keys:
            RTDOSEPath = self.SubjectList.loc[i, 'RTDOSE_Path']
            RTDOSEPath = Path(RTDOSEPath, 'Dose.nii.gz')
            data['RTDOSE'] = LoadImage(image_only=True)(RTDOSEPath)
            data['RTDOSE'] = data['RTDOSE'] / 67  ## Probably need to make it a variable

        ## Load PET
        if 'PET' in self.keys:
            PETPath = self.SubjectList.loc[i, 'PET_Path']
            if self.config['DATA']['Nifty']:
                PETPath = Path(PETPath, 'pet.nii.gz')
            data['PET'] = LoadImage(image_only=True)(PETPath)

        ## Load Mask
        if 'RTSTRUCT' in self.keys:
            RSPath = self.SubjectList.loc[i, 'RTSTRUCT_Path']
            data['RTSTRUCT'] = LoadImage(image_only=True)(Path(RSPath, self.config['DATA']['Structs']))

        if self.transform: data = self.transform(data)

        if self.config['DATA']['Multichannel']:
            old_keys = list(self.keys)
            data['Image'] = np.concatenate([data[key] for key in old_keys], axis=0)
            for key in old_keys: data.pop(key)
        else:
            if 'RTSTRUCT' in data.keys() and 'RTDOSE' in data.keys():
                data['RTDOSE'] = np.concatenate([data[key] for key in ['RTDOSE', 'RTSTRUCT']], axis=0)
                data.pop('RTSTRUCT')
            elif 'RTSTRUCT' in data.keys():
                data.pop('RTSTRUCT')  ## No need for mask in single-channel multi-branch

        ## Add clinical record at the end
        if 'Records' in self.config.keys():
            data['Records'] = self.SubjectList.loc[i, self.clinical_cols].values.astype('float')

        if self.inference:
            return data

        else:  ##Training
            label = np.float32(self.SubjectList.loc[i, self.config['DATA']['target']])
            if 'threshold' in self.config['DATA'].keys():  ## Classification
                label = np.where(label > self.config['DATA']['threshold'], 1, 0)
                label = torch.as_tensor(label, dtype=torch.float32)
            if 'censor_label' in self.config['DATA'].keys():
                censor_status = np.float32(
                    self.SubjectList.loc[i, self.config['DATA']['censor_label']]).astype('bool')
                return data, censor_status, label
            else:
                return data, label
### DataLoader
class DataModule(LightningDataModule):
    def __init__(self, SubjectList, config=None, train_transform=None, val_transform=None, train_size=0.7, rd=None,
                 rd_tv=None, num_workers=1, **kwargs):
        super().__init__()
        self.batch_size = config['MODEL']['batch_size']
        self.num_workers = 32

        train_list, val_test_list = train_test_split(SubjectList, train_size=train_size,random_state=rd_tv) ## 0.7/0.3

        val_list, test_list = train_test_split(val_test_list, train_size=0.5,random_state=rd_tv)## 0.15/0.15

        self.train_list = train_list.reset_index(drop=True)
        self.val_list   = val_list.reset_index(drop=True)
        self.test_list  = test_list.reset_index(drop=True)

        self.train_data = DataGenerator(self.train_list, config=config, transform=train_transform, **kwargs)
        self.val_data   = DataGenerator(self.val_list, config=config, transform=val_transform, **kwargs)
        self.test_data  = DataGenerator(self.test_list, config=config, transform=val_transform, **kwargs)

    def train_dataloader(self): return DataLoader(self.train_data, batch_size=self.batch_size,
                                                  num_workers=self.num_workers, pin_memory=True, shuffle=False)

    def val_dataloader(self):   return DataLoader(self.val_data, batch_size=self.batch_size,
                                                  num_workers=self.num_workers, pin_memory=True, shuffle=False)

    def test_dataloader(self):  return DataLoader(self.test_data, batch_size=self.batch_size,
                                                  num_workers=self.num_workers, pin_memory=True)



