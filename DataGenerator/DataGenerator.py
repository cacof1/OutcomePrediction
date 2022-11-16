from pytorch_lightning import LightningDataModule, LightningModule, Trainer, seed_everything
from torch.utils.data import DataLoader
import numpy as np
import torch
from sklearn.model_selection import train_test_split, StratifiedShuffleSplit
import xnat
import matplotlib.pyplot as plt
from monai.transforms import (
    LoadImage,EnsureChannelFirstd, ResampleToMatchd
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
    def __init__(self, SubjectList, SubjectInfo, config=None, keys=['CT'],  transform=None, inference=False, clinical_cols=None, session = None,**kwargs):
        super().__init__()
        self.config        = config
        self.session       = session
        self.SubjectList   = SubjectList
        self.SubjectInfo   = SubjectInfo
        self.keys          = keys
        self.transform     = transform
        self.inference     = inference
        self.clinical_cols = clinical_cols
        
    def GeneratePath(self, subjectid, Modality):
        subject       = self.SubjectInfo[subjectid]['xnat:Subject'][0]
        subject_label = subject['@label']
        experiments   = subject['xnat:experiments'][0]['xnat:experiment']

        ## Won't work with many experiments yet
        for experiment in experiments:
            experiment_label = experiment['@label']
            scans            = experiment['xnat:scans'][0]['xnat:scan']
            for scan in scans:
                if (scan['@type'] in Modality):
                    scan_label = scan['@ID'] + '-' + scan['@type']
                    resources_label = scan['xnat:file'][0]['@label']
                    path = os.path.join(self.config['DATA']['DataFolder'], subject_label, experiment_label, 'scans', scan_label, 'resources', resources_label, 'files')
                    return path

    def __len__(self):
        return int(self.SubjectList.shape[0])

    def __getitem__(self, i):

        data = {}
        meta = {}
        subject_id = self.SubjectList.loc[i, 'subjectid']
        slabel = self.SubjectList.loc[i, 'subject_label']
        ## Load CT
        if 'CT' in self.keys:
            CTPath                 = self.GeneratePath(subject_id, 'CT')
            data['CT'], meta['CT'] = LoadImage()(CTPath)
        ## Load Mask            
        if 'Mask' in self.config['DATA'].keys():
            RSPath = glob.glob(self.GeneratePath(subject_id, 'Structs') + '/*dcm')
            mask_imgs = np.zeros_like(data['CT'])
            mask = get_RS_masks(slabel,CTPath, mask_imgs, RSPath[0], self.config['DATA']['Mask'])
            mask = np.rot90(mask)
            mask = np.flip(mask, axis=0)
            data['Mask'] = mask

        else: data["Mask"] = np.ones_like(data['CT']) ## No ROI target defined

        ## Load Dose        
        if 'Dose' in self.keys:
            DosePath                   = self.GeneratePath(subject_id, 'Dose')
            data['Dose'], meta['Dose'] = LoadImage()(DosePath+"/1-1.dcm")
            data['Dose']               = data['Dose'] * np.double(meta['Dose']['3004|000e'])

        ## Load PET            
        if 'PET' in self.keys:
            PETPath                    = self.GeneratePath(subject_id, 'PET')
            data['PET'], meta['PET']   = LoadImage()(PETPath+"/1-1.dcm")

        ## Apply transforms on all
        if self.transform : data = self.transform(data)
        # Decide between multi-branch single-channel/multi-channel single-branch
        if self.config['DATA']['Multichannel']:
            old_keys = list(data.keys())
            data['Image'] = np.concatenate([data[key] for key in data.keys()],axis=0)
            for key in old_keys: data.pop(key)
        else: data.pop('Mask') ## No need for mask in single-channel multi-branch
        
        ## Add clinical record at the end
        if 'Records' in self.keys: data['Records'] = torch.tensor(self.SubjectList.loc[i, self.clinical_cols], dtype=torch.float32)

        if self.inference: return data
        else:
            label = torch.tensor(self.SubjectList.loc[i, "xnat_subjectdata_field_map_" + self.config['DATA']['target']])
            if self.config['DATA']['threshold'] is not None:  label = torch.where(label > self.config['DATA']['threshold'], 1, 0)
            return data, label

### DataLoader
class DataModule(LightningDataModule):
    def __init__(self, SubjectList, SubjectInfo, config=None, train_transform=None, val_transform=None, train_size=0.7,
                 val_size=0.2, test_size=0.1, num_workers=64, **kwargs):
        super().__init__()
        self.batch_size = config['MODEL']['batch_size']
        self.num_workers = num_workers
        #data_trans = class_stratify(SubjectList, config)
        ## Split Test with fixed seed
        train_val_list, test_list = train_test_split(SubjectList,test_size=0.15,random_state=42)
        #                                            stratify=data_trans)

        #data_trans = class_stratify(train_val_list, config) 
        ## Split train-val with random seed
        train_list, val_list = train_test_split(train_val_list,test_size=0.15, random_state=np.random.randint(10000))
        #stratify=data_trans)
        
        train_list = train_list.reset_index(drop=True)
        val_list   = val_list.reset_index(drop=True)
        test_list  = test_list.reset_index(drop=True)

        self.train_data = DataGenerator(train_list, SubjectInfo, config=config, transform=train_transform, **kwargs)
        self.val_data   = DataGenerator(val_list,   SubjectInfo, config=config, transform=val_transform, **kwargs)
        self.test_data  = DataGenerator(test_list,  SubjectInfo, config=config, transform=val_transform, **kwargs)
        
    def train_dataloader(self): return DataLoader(self.train_data, batch_size=self.batch_size, num_workers=self.num_workers, pin_memory=True,drop_last=True, shuffle=True)
    def val_dataloader(self):   return DataLoader(self.val_data, batch_size=self.batch_size, num_workers=self.num_workers, pin_memory=True,drop_last=True, shuffle=False)
    def test_dataloader(self):  return DataLoader(self.test_data, batch_size=self.batch_size, num_workers=self.num_workers, pin_memory=True, drop_last=True)

def QuerySubjectList(config, session):
    root_element = "xnat:subjectData"
    XML = XMLCreator(root_element)  # , search_field, search_where)
    print("Querying from Server")
    ## Target
    XML.Add_search_field(
        {"element_name": "xnat:subjectData", "field_ID": "XNAT_SUBJECTDATA_FIELD_MAP=" + str(config['DATA']['target']),
         "sequence": "1", "type": "int"})
    ## Label
    XML.Add_search_field(
        {"element_name": "xnat:subjectData", "field_ID": "SUBJECT_LABEL", "sequence": "1", "type": "string"})

    if 'Records' in config['DATA']['module']:
        for value in config['DATA']['clinical_columns']:
            dict_temp = {"element_name": "xnat:subjectData", "field_ID": "XNAT_SUBJECTDATA_FIELD_MAP=" + str(value), "sequence": "1", "type": "int"}
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
    if(config['FILTER']):
        for value in config['FILTER']['patient_id']:
            dict_temp = {"schema_field": "xnat:subjectData.SUBJECT_LABEL", "comparison_type": "!=",  "value": str(value)}
                     
            templist.append(dict_temp)
    if (len(templist)): XML.Add_search_where(templist, "AND")

    xmlstr = XML.ConstructTree()
    response = session.post(config['SERVER']['Address'] + '/data/search/', data=xmlstr, format='csv')
    SubjectList = pd.read_csv(StringIO(response.text))
    #print('Query: ', SubjectList)
    return SubjectList

def SynchronizeData(config, SubjectList):
    session = xnat.connect(config['SERVER']['Address'], user=config['SERVER']['User'],password=config['SERVER']['Password'])
    for subjectlabel, subjectid in zip(SubjectList['subject_label'], SubjectList['subjectid']):
        if (not Path(config['DATA']['DataFolder'], subjectlabel).is_dir()):
            xnatsubject = session.create_object('/data/subjects/' + subjectid)
            print("Synchronizing ", subjectid, subjectlabel)
            xnatsubject.download_dir(config['DATA']['DataFolder'])  ## Download data

def get_subject_info(config, session, subjectid):
    r = session.get(config['SERVER']['Address'] + '/data/subjects/' + subjectid, format='xml')
    data = xmltodict.parse(r.text, force_list=True)
    return data

def QuerySubjectInfo(config, SubjectList,session):
    SubjectInfo = {}
    with ThreadPoolExecutor(max_workers=10) as executor:
        future_to_url = {executor.submit(get_subject_info, config, session, subjectid) for subjectid in
                         SubjectList['subjectid']}
        executor.shutdown(wait=True)
    for future in concurrent.futures.as_completed(future_to_url):
        subjectdata = future.result()
        subjectid = subjectdata["xnat:Subject"][0]["@ID"]
        SubjectInfo[subjectid] = subjectdata
    return SubjectInfo


def LoadClinicalData(config, PatientList):
    category_cols = []
    numerical_cols = []
    for col in config['CLINICAL']['category_feat']:
        category_cols.append('xnat_subjectdata_field_map_' + col)

    for row in config['CLINICAL']['numerical_feat']:
        numerical_cols.append('xnat_subjectdata_field_map_' + row)

    target = config['DATA']['target']

    ct = ColumnTransformer(
        [("CatTrans", OneHotEncoder(), category_cols),
         ("NumTrans", MinMaxScaler(), numerical_cols), ])

    X = PatientList.loc[:, category_cols + numerical_cols]
    yc = X.loc[:, category_cols].astype('float32')
    X.loc[:, category_cols] = yc.fillna(yc.mean().astype('int'))
    yn = X.loc[:, numerical_cols].astype('float32')
    X.loc[:, numerical_cols] = yn.fillna(yn.mean())
    X_trans = ct.fit_transform(X)
    if not isinstance(X_trans, (np.ndarray, np.generic)): X_trans = X_trans.toarray()
        
    df_trans = pd.DataFrame(X_trans, index=X.index, columns=ct.get_feature_names_out())
    clinical_col = list(df_trans.columns)
    df_trans['xnat_subjectdata_field_map_' + target] = PatientList.loc[:, 'xnat_subjectdata_field_map_' + target]
    df_trans['subject_label'] = PatientList.loc[:, 'subject_label']
    df_trans['subjectid'] = PatientList.loc[:, 'subjectid']
    return df_trans, clinical_col
