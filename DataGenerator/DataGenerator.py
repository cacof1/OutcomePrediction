from pytorch_lightning import LightningDataModule, LightningModule, Trainer, seed_everything
from torch.utils.data import DataLoader
import numpy as np
import torch
from sklearn.model_selection import train_test_split, StratifiedShuffleSplit
import xnat
import SimpleITK as sitk
from Utils.DicomTools import *
from Utils.XNATXML import XMLCreator
from io import StringIO
import requests
import pandas as pd
import xmltodict
from sklearn.preprocessing import OneHotEncoder,MinMaxScaler,LabelEncoder,OrdinalEncoder
from rt_utils import RTStructBuilder
from sklearn.compose import ColumnTransformer
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
import concurrent

class DataGenerator(torch.utils.data.Dataset):
    def __init__(self, SubjectList, SubjectInfo, config=None,keys=['CT'], targetROI=None, transform=None, inference=False, clinical_cols=None, **kwargs):
        super().__init__()
        self.config = config
        self.SubjectList = SubjectList
        self.SubjectInfo = SubjectInfo
        self.targetROI = config['DATA']['mask_name']
        self.keys = keys
        self.transform = transform
        self.inference = inference
        self.clinical_cols = clinical_cols

    def GeneratePath(self, subjectid, Modality):
        subject = self.SubjectInfo[subjectid]['xnat:Subject'][0]
        subject_label = subject['@label']
        experiments   = subject['xnat:experiments'][0]['xnat:experiment']


    def __len__(self):
        return int(self.PatientList.shape[0])

    def GeneratePath(self, subjectid, Modality):
        subject = self.SubjectInfo[subjectid]['xnat:Subject'][0]
        subject_label = subject['@label']
        experiments   = subject['xnat:experiments'][0]['xnat:experiment']

        ## Won't work with many experiments
        for experiment in experiments:
            experiment_label = experiment['@label']
            scans = experiment['xnat:scans'][0]['xnat:scan']
            for scan in scans:
                if(scan['@type'] in Modality):
                    scan_label = scan['@ID']+'-'+scan['@type']
                    resources_label = scan['xnat:file'][0]['@label']
                    return os.path.join(self.config['DATA']['DataFolder'],subject_label, experiment_label, 'scans',scan_label,'resources',resources_label,'files')

    def __len__(self):
        return int(self.SubjectList.shape[0])

    def __getitem__(self, i):
        data       = {}
        subject_id = self.SubjectList.loc[i,'subjectid']
        CTPath     = self.GeneratePath(subject_id, 'CT')
        CTSession  = ReadDicom(CTPath)
        CTSession.SetOrigin(CTSession.GetOrigin())
        CTArray    = sitk.GetArrayFromImage(CTSession)
        ## First define the ROI based on target
        if self.targetROI:
            RSPath    = glob.glob(self.GeneratePath(subject_id,'Structs') + '/*dcm')
            RS        = RTStructBuilder.create_from(dicom_series_path=CTPath, rt_struct_path=RSPath[0])
            roi_names = RS.get_roi_names()
            if self.targetROI in roi_names:
                mask_img = RS.get_roi_mask_by_name(self.targetROI)
                mask_img = mask_img.transpose(2, 0, 1)
                mask_img = np.flip(mask_img, 0) ## Same frame of reference as CT
            else:
                message = "No ROI of name " + self.targetROI +" found in RTStruct"
                raise ValueError(message)
                
        else: ## No ROI target defined
            mask_img   = np.ones_like(CTArray)
        ## Load image within each channel for the target ROI
        for channel in self.keys:
            if channel == 'CT':
                data['CT']      = get_masked_img_voxel(CTArray, mask_img)
                data['CT']      = np.expand_dims(data['CT'], 0)
                if self.transform: data['CT'] = self.transform(data['CT'])
                data['CT']      = torch.as_tensor(data['CT'], dtype=torch.float32)

            if channel == 'RTDose':
                DosePath        = self.GeneratePath(subject_id, 'Dose')
                DoseSession     = ReadDicom(DosePath)[..., 0]
                DoseSession     = ResamplingITK(DoseSession, CTSession) ## 1 mm3 or ish
                DoseArray       = sitk.GetArrayFromImage(DoseSession)
                DoseArray       = DoseArray * np.double(DoseSession.GetMetaData('3004|000e'))
                data['RTDose']  = get_masked_img_voxel(DoseArray, mask_img)
                data['RTDose']  = np.expand_dims(data['RTDose'], 0)
                if self.transform: data['RTDose'] = self.transform(data['RTDose'])
                data['RTDose']  = torch.as_tensor(data['RTDose'], dtype=torch.float32)

            if channel == 'PET':
                PETPath         = self.GeneratePath(subject_id, 'PET')
                PETSession      = ReadDicom(PETPath)
                PETSession      = ResamplingITK(PETSession, CTSession)
                PETArray        = sitk.GetArrayFromImage(PETSession)
                data['PET']     = get_masked_img_voxel(PETArray, mask_img)
                data['PET']     = np.expand_dims(data['PET'], 0)
                if self.transform: data['PET'] = self.transform(data['PET'])
                data['PET']     = torch.as_tensor(data['PET'], dtype=torch.float32)

            if channel == 'Records':
                records         = self.SubjectList.loc[:,self.clinical_cols].to_numpy()
                data['Records'] = torch.as_tensor(records, dtype=torch.float32)
        if self.inference: return data

        else:
            label = self.SubjectList.loc[i,"xnat_subjectdata_field_map_"+self.config['DATA']['target']]
            if self.config['MODEL']['Classification_threshold'] is not None:  label = np.array(label > self.config['MODEL']['Classification_threshold'])
            label = torch.as_tensor(label, dtype=torch.int64)
            return data, label
       
### DataLoader
class DataModule(LightningDataModule):
    def __init__(self, SubjectList, SubjectInfo, config=None,  train_transform=None, val_transform=None,  train_size=0.7, val_size=0.2, test_size=0.1, num_workers=0, **kwargs):
        super().__init__()
        self.batch_size  = config['MODEL']['batch_size']
        self.num_workers = num_workers
        ## Split Test with fixed seed
        train_val_list, test_list = train_test_split(SubjectList,
                                                     test_size=0.15,
                                                     random_state=42,
                                                     stratify=SubjectList['xnat_subjectdata_field_map_' + config['DATA']['target']] >= config['MODEL']['Classification_threshold'])
        ## Split train-val with random seed
        train_list, val_list      = train_test_split(train_val_list,
                                                     test_size=0.15,
                                                     random_state=np.random.randint(1, 10000),
                                                     stratify=train_val_list['xnat_subjectdata_field_map_' + config['DATA']['target']] >= config['MODEL']['Classification_threshold'])

        train_list = train_list.reset_index(drop=True)
        val_list   = val_list.reset_index(drop=True)
        test_list  = test_list.reset_index(drop=True)

        self.train_data = DataGenerator(train_list, SubjectInfo, config = config, transform=train_transform, **kwargs)
        self.val_data   = DataGenerator(val_list,   SubjectInfo, config = config, transform=val_transform, **kwargs)
        self.test_data  = DataGenerator(test_list,  SubjectInfo, config = config, transform=val_transform, **kwargs)

    def train_dataloader(self):
        return DataLoader(self.train_data, batch_size=self.batch_size, num_workers=self.num_workers, pin_memory=True, drop_last=True, shuffle=True)

    def val_dataloader(self):
        return DataLoader(self.val_data, batch_size=self.batch_size, num_workers=self.num_workers, pin_memory=True, drop_last=True, shuffle=False)
    
    def test_dataloader(self):
        return DataLoader(self.test_data, batch_size=self.batch_size, num_workers=self.num_workers, pin_memory=True, drop_last = True)
                                                  
def QuerySubjectList(config, **kwargs):
    root_element = "xnat:subjectData"
    XML          = XMLCreator(root_element)#, search_field, search_where)    
    print("Querying from Server")
    ## Target
    XML.Add_search_field({"element_name":"xnat:subjectData","field_ID":"XNAT_SUBJECTDATA_FIELD_MAP="+str(config['DATA']['target']),"sequence":"1", "type":"int"})
    ## Label
    XML.Add_search_field({"element_name":"xnat:subjectData","field_ID":"SUBJECT_LABEL","sequence":"1", "type":"string"})

    if 'Records' in config['DATA']['module']:
        for value in config['DATA']['clinical_columns']:
            dict_temp = {"element_name": "xnat:subjectData", "field_ID": "XNAT_SUBJECTDATA_FIELD_MAP=" + str(value),
                         "sequence": "1", "type": "int"}
            XML.Add_search_field(dict_temp)

    ## Where Condition
    templist = []
    for value in config['SERVER']['Projects']:
        templist.append({"schema_field":"xnat:subjectData.PROJECT","comparison_type":"=","value":str(value)})
    if(len(templist)): XML.Add_search_where(templist, "OR")

    templist = []
    for key,value in config['CRITERIA'].items():
        templist.append({"schema_field":"xnat:subjectData.XNAT_SUBJECTDATA_FIELD_MAP="+key, "comparison_type":"=","value":str(value)})
    if(len(templist)): XML.Add_search_where(templist, "AND") ## if any items in here
    
    templist = []
    for key,value in config['MODALITY'].items():
        templist.append({"schema_field":"xnat:ctSessionData.SCAN_COUNT_TYPE="+key,"comparison_type":">=","value":str(value)})
    if(len(templist)): XML.Add_search_where(templist, "AND") ## if any items in here

    templist = []
    for value in config['FILTER']['patient_id']:
        dict_temp = {"schema_field": "xnat:subjectData.SUBJECT_LABEL", "comparison_type": "!=",
                     "value": str(value)}
        templist.append(dict_temp)
    if (len(templist)): XML.Add_search_where(templist, "AND")



    xmlstr       = XML.ConstructTree()
    params       = {'format': 'csv'}
    response     = requests.post(config['SERVER']['Address']+'/data/search/', params=params, data=xmlstr, auth=(config['SERVER']['User'], config['SERVER']['Password']))
    SubjectList  = pd.read_csv(StringIO(response.text))
    print('Query: ', SubjectList)
    return SubjectList

def SynchronizeData(config, SubjectList):
    session = xnat.connect(config['SERVER']['Address'], user=config['SERVER']['User'], password=config['SERVER']['Password'])
    for subjectlabel, subjectid in zip(SubjectList['subject_label'],SubjectList['subjectid']):
        if (not Path(config['DATA']['DataFolder'], subjectlabel).is_dir()):
            xnatsubject = session.create_object('/data/subjects/'+subjectid)
            print("Synchronizing ", subjectid, subjectlabel)
            xnatsubject.download_dir(config['DATA']['DataFolder']) ## Download data

def get_subject_info(config, session,  subjectid):
    params = {'format':'xml'}
    r = session.get(config['SERVER']['Address']+'/data/subjects/'+subjectid, params=params)
    data = xmltodict.parse(r.text,force_list=True)
    return data

def QuerySubjectInfo(config, SubjectList):
    params          = {'format': 'xml'}
    SubjectInfo = {}
    with requests.Session() as session:
        session.auth = (config['SERVER']['User'], config['SERVER']['Password'])
        r = session.get(config['SERVER']['Address']+'/data/JSESSION')        
        with ThreadPoolExecutor(max_workers=20) as executor:
            future_to_url = {executor.submit(get_subject_info, config, session, subjectid) for subjectid in SubjectList['subjectid']}
            executor.shutdown(wait=True)
        for future in concurrent.futures.as_completed(future_to_url):
            subjectdata = future.result()
            subjectid   = subjectdata["xnat:Subject"][0]["@ID"]
            SubjectInfo[subjectid] = subjectdata
    return SubjectInfo


def LoadClinicalData(config, PatientList):
    category_cols = []
    numerical_cols = []
    for col in config['CLINICAL']['category_feat']:
        category_cols.append('xnat_subjectdata_field_map_' + col)

    for row in config['CLINICAL']['numerical_feat']:
        numerical_cols.append('xnat_subjectdata_field_map_' + row)

    target = config['DATA']['target'][0]

    ct = ColumnTransformer(
        [("CatTrans", OneHotEncoder(), category_cols),
         ("NumTrans", MinMaxScaler(), numerical_cols), ])

    X = PatientList.loc[:, category_cols + numerical_cols]
    X.loc[:, category_cols] = X.loc[:, category_cols].astype('str')
    X.loc[:, numerical_cols] = X.loc[:, numerical_cols].astype('float32')
    X_trans = ct.fit_transform(X)
    df_trans = pd.DataFrame(X_trans, index=X.index, columns=ct.get_feature_names_out())
    clinical_col = list(df_trans.columns)
    df_trans['xnat_subjectdata_field_map_' + target] = PatientList.loc[:, 'xnat_subjectdata_field_map_' + target]
    df_trans['subject_label'] = PatientList.loc[:, 'subject_label']

    return df_trans, clinical_col


# def LoadClinicalData(config, SubjectList, ClinicalDataset):
#     category_cols = config['CLINICAL']['category_feat']
#     numerical_cols = config['CLINICAL']['numerical_feat']
#     target = config['DATA']['Target']
#
#     subject_list = SubjectList['subject_label'].tolist()
#     ClinicalDataset = ClinicalDataset[ClinicalDataset.SubjectID.isin(subject_list)].reset_index(drop=True)
#
#     ct = ColumnTransformer(
#         [("CatTrans", OneHotEncoder(), category_cols),
#          ("NumTrans", MinMaxScaler(), numerical_cols), ])
#
#     X = ClinicalDataset.loc[:, category_cols + numerical_cols]
#     X.loc[:, category_cols] = X.loc[:, category_cols].astype('str')
#     X.loc[:, numerical_cols] = X.loc[:, numerical_cols].astype('float32')
#     X_trans = ct.fit_transform(X)
#     df_trans = pd.DataFrame(X_trans, index=X.index, columns=ct.get_feature_names_out())
#     df_trans[target] = ClinicalDataset.loc[:, target]
#     df_trans['subject_label'] = ClinicalDataset.loc[:, 'SubjectID']
#     return df_trans


