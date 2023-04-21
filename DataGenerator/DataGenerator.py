from pytorch_lightning import LightningDataModule, LightningModule, Trainer, seed_everything
from torch.utils.data import DataLoader
import numpy as np
import torch
from monai.data import MetaTensor
from sklearn.model_selection import train_test_split, StratifiedShuffleSplit
import xnat
import matplotlib.pyplot as plt
from monai.transforms import (
    LoadImage, EnsureChannelFirstd, ResampleToMatchd, ResizeWithPadOrCropd
)
from Utils.DicomTools import *
from Utils.XNATXML import XMLCreator
from io import StringIO
import requests
import pandas as pd
import xmltodict
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler, LabelEncoder, OrdinalEncoder
from rt_utils import RTStructBuilder
from sklearn.compose import ColumnTransformer
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
import concurrent


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
        subject_id = self.SubjectList.loc[i, 'subjectid']
        slabel = self.SubjectList.loc[i, 'subject_label']
        data['slabel'] = slabel
        ## Load CT
        if 'CT' in self.keys:
            CTPath = self.SubjectList.loc[i, 'CT_Path']
            CTPath = Path(CTPath, 'CT.nii.gz')
            data['CT'], meta['CT'] = LoadImage()(CTPath)

        ## Load Dose
        if 'Dose' in self.keys:
            DosePath = self.SubjectList.loc[i, 'Dose_Path']
            DosePath = Path(DosePath, 'Dose.nii.gz')
            data['Dose'], meta['Dose'] = LoadImage()(DosePath)
            data['Dose'] = data['Dose'] / 67  ## Probably need to make it a variable

        ## Load PET
        if 'PET' in self.keys:
            PETPath = self.SubjectList.loc[i, 'PET_Path']
            if self.config['DATA']['Nifty']:
                PETPath = Path(PETPath, 'pet.nii.gz')
            data['PET'], meta['PET'] = LoadImage()(PETPath)

        ## Load Mask
        if 'Structs' in self.keys:
            RSPath = self.SubjectList.loc[i, 'Structs_Path']
            data['Structs'], meta['Structs'] = LoadImage()(Path(RSPath, self.config['DATA']['Structs']))

        if self.transform: data = self.transform(data)

        if self.config['DATA']['Multichannel']:
            old_keys = list(self.keys)
            data['Image'] = np.concatenate([data[key] for key in old_keys], axis=0)
            for key in old_keys: data.pop(key)
        else:
            if 'Structs' in data.keys():
                data.pop('Structs')  ## No need for mask in single-channel multi-branch

        ## Add clinical record at the end
        if 'Records' in self.config.keys(): data['Records'] = torch.tensor(self.SubjectList.loc[i, self.clinical_cols],
                                                                           dtype=torch.float32)
        if self.inference:
            return data
        else:  ##Training
            label = torch.tensor(
                np.float(self.SubjectList.loc[i, "xnat_subjectdata_field_map_" + self.config['DATA']['target']]))
            censor_status = not (np.int8(
                self.SubjectList.loc[i, 'xnat_subjectdata_field_map_' + self.config['DATA']['censor_label']]).astype(
                'bool'))
            if 'threshold' in self.config['DATA'].keys():  ## Classification
                label = torch.where(label > self.config['DATA']['threshold'], 1, 0)
                label = torch.as_tensor(label, dtype=torch.float32)
            #label = (censored_label, label)
            return data, censor_status, label


### DataLoader
class DataModule(LightningDataModule):
    def __init__(self, SubjectList, config=None, train_transform=None, val_transform=None, train_size=0.85,
                 num_workers=10, **kwargs):
        super().__init__()
        self.batch_size = config['MODEL']['batch_size']
        self.num_workers = num_workers
        data_trans = class_stratify(SubjectList, config)
        ## Split Test with fixed seed
        train_val_list, test_list = train_test_split(SubjectList, train_size=train_size, random_state=75,
                                                     stratify=data_trans)

        data_trans = class_stratify(train_val_list, config)
        ## Split train-val with random seed
        train_list, val_list = train_test_split(train_val_list, train_size=train_size,
                                                random_state=np.random.randint(10000),
                                                stratify=data_trans)

        train_list = train_list.reset_index(drop=True)
        val_list = val_list.reset_index(drop=True)
        test_list = test_list.reset_index(drop=True)

        self.train_data = DataGenerator(train_list, config=config, transform=train_transform, **kwargs)
        self.val_data = DataGenerator(val_list, config=config, transform=val_transform, **kwargs)
        self.test_data = DataGenerator(test_list, config=config, transform=val_transform, **kwargs)

    def train_dataloader(self): return DataLoader(self.train_data, batch_size=self.batch_size,
                                                  num_workers=self.num_workers, pin_memory=True, drop_last=True,
                                                  shuffle=True)

    def val_dataloader(self):   return DataLoader(self.val_data, batch_size=self.batch_size,
                                                  num_workers=self.num_workers, pin_memory=True, drop_last=True,
                                                  shuffle=False)

    def test_dataloader(self):  return DataLoader(self.test_data, batch_size=self.batch_size,
                                                  num_workers=self.num_workers, pin_memory=True, drop_last=True)


def QuerySubjectList(config, session):
    root_element = "xnat:subjectData"
    XML = XMLCreator(root_element)  # , search_field, search_where)
    print("Querying from Server")
    ## Target
    XML.Add_search_field(
        {"element_name": "xnat:subjectData", "field_ID": "XNAT_SUBJECTDATA_FIELD_MAP=" + str(config['DATA']['target']),
         "sequence": "1", "type": "int"})
    if 'censor_label' in config['DATA'].keys():
        XML.Add_search_field(
            {"element_name": "xnat:subjectData",
             "field_ID": "XNAT_SUBJECTDATA_FIELD_MAP=" + str(config['DATA']['censor_label']),
             "sequence": "1", "type": "int"})
    ## Label
    XML.Add_search_field(
        {"element_name": "xnat:subjectData", "field_ID": "SUBJECT_LABEL", "sequence": "1", "type": "string"})

    if 'Records' in config.keys():
        feats_list = [item for sublist in config['Records'].values() for item in sublist]
        for value in feats_list:
            dict_temp = {"element_name": "xnat:subjectData", "field_ID": "XNAT_SUBJECTDATA_FIELD_MAP=" + str(value),
                         "sequence": "1", "type": "int"}
            XML.Add_search_field(dict_temp)

    ## Where Condition
    templist = []
    for value in config['SERVER']['Projects']:
        templist.append({"schema_field": "xnat:subjectData.PROJECT", "comparison_type": "=", "value": str(value)})
    if (len(templist)): XML.Add_search_where(templist, "OR")

    templist = []
    for key, value in config['CRITERIA'].items():
        templist.append({"schema_field": "xnat:subjectData.XNAT_SUBJECTDATA_FIELD_MAP=" + key, "comparison_type": "=",
                         "value": str(value)})
    if (len(templist)): XML.Add_search_where(templist, "AND")  ## if any items in here

    templist = []
    for key, value in config['MODALITY'].items():
        templist.append(
            {"schema_field": "xnat:ctSessionData.SCAN_COUNT_TYPE=" + key, "comparison_type": ">=", "value": str(value)})
    if (len(templist)): XML.Add_search_where(templist, "AND")  ## if any items in here

    templist = []
    if (config['FILTER']):
        for value in config['FILTER']['patient_id']:
            dict_temp = {"schema_field": "xnat:subjectData.SUBJECT_LABEL", "comparison_type": "!=", "value": str(value)}

            templist.append(dict_temp)
    if (len(templist)): XML.Add_search_where(templist, "AND")

    xmlstr = XML.ConstructTree()
    response = session.post(config['SERVER']['Address'] + '/data/search/', data=xmlstr, format='csv')
    SubjectList = pd.read_csv(StringIO(response.text), dtype=str)
    # print('Query: ', SubjectList)
    return SubjectList


def class_stratify(SubjectList, config):
    ptarget = SubjectList['xnat_subjectdata_field_map_' + config['DATA']['target']]
    kbins = KBinsDiscretizer(n_bins=15, encode='ordinal', strategy='uniform')
    ptarget = np.array(ptarget).reshape((len(ptarget), 1))
    data_trans = kbins.fit_transform(ptarget)
    return data_trans


def SynchronizeData(config, SubjectList):
    session = xnat.connect(config['SERVER']['Address'], user=config['SERVER']['User'],
                           password=config['SERVER']['Password'])
    for subjectlabel, subjectid in zip(SubjectList['subject_label'], SubjectList['subjectid']):
        if (not Path(config['DATA']['DataFolder'], subjectlabel).is_dir()):
            xnatsubject = session.create_object('/data/subjects/' + subjectid)
            print("Synchronizing ", subjectid, subjectlabel)
            xnatsubject.download_dir(config['DATA']['DataFolder'])  ## Download data


def get_subject_info(config, session, subjectid):
    r = session.get(config['SERVER']['Address'] + '/data/subjects/' + subjectid, format='xml')
    data = xmltodict.parse(r.text, force_list=True)
    return data


def QuerySubjectInfo(config, SubjectList):
    for i in range(len(SubjectList)):
        subject_label = SubjectList.loc[i, 'subject_label']
        for key in config['MODALITY'].keys():
            if key == 'Structs':
                SubjectList.loc[i, key + '_Path'] = Path(config['DATA']['DataFolder'], subject_label, 'struct_TS')
            else:
                SubjectList.loc[i, key + '_Path'] = Path(config['DATA']['DataFolder'], subject_label)


def LoadClinicalData(config, PatientList):
    category_cols = []
    numerical_cols = []
    for col in config['Records']['category_feat']:
        category_cols.append('xnat_subjectdata_field_map_' + col)

    for row in config['Records']['numerical_feat']:
        numerical_cols.append('xnat_subjectdata_field_map_' + row)

    target = config['DATA']['target']

    ct = ColumnTransformer(
        [("CatTrans", OneHotEncoder(), category_cols),
         ("NumTrans", MinMaxScaler(), numerical_cols), ])

    X = PatientList.loc[:, category_cols + numerical_cols]
    yc = X[category_cols].astype('float32')
    X[category_cols] = yc.fillna(yc.mean().astype('int'))
    yn = X[numerical_cols].astype('float32')
    X[numerical_cols] = yn.fillna(yn.mean())  # X.loc[:, numerical_cols] = yn.fillna(yn.mean())
    X_trans = ct.fit_transform(X)
    if not isinstance(X_trans, (np.ndarray, np.generic)): X_trans = X_trans.toarray()

    df_trans = pd.DataFrame(X_trans, index=X.index, columns=ct.get_feature_names_out())
    clinical_col = list(df_trans.columns)
    df_trans['xnat_subjectdata_field_map_' + target] = PatientList.loc[:, 'xnat_subjectdata_field_map_' + target]
    df_trans['subject_label'] = PatientList.loc[:, 'subject_label']
    df_trans['subjectid'] = PatientList.loc[:, 'subjectid']
    if 'censor_label' in config['DATA'].keys():
        df_trans['xnat_subjectdata_field_map_' + config['DATA']['censor_label']] = PatientList.loc[:,
                                                                                   'xnat_subjectdata_field_map_' +
                                                                                   config['DATA']['censor_label']]
    return df_trans, clinical_col
