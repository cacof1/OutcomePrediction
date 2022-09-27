import glob
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

from rt_utils import RTStructBuilder
import xmltodict
from sklearn.preprocessing import OneHotEncoder,MinMaxScaler,LabelEncoder,OrdinalEncoder
from sklearn.compose import ColumnTransformer
from pathlib import Path
import xml.etree.ElementTree as ET
class DataGenerator(torch.utils.data.Dataset):
    def __init__(self, PatientList, config=None,
                 selected_channel=['CT','RTDose','Records'], ROIRange=[60,60,10],
                 transform=None, inference=False, clinical_cols=None, **kwargs):
        super().__init__()
        self.config = config
        self.PatientList = PatientList
        self.target = config['DATA']['target']
        self.threshold = config['MODEL']['Classification_threshold']
        self.targetROI = config['DATA']['mask_name']
        self.ROIRange = ROIRange
        self.selected_channel = selected_channel
        self.dicom_folder = config['DATA']['DataFolder']
        self.transform = transform
        self.inference = inference
        self.clinical_cols = clinical_cols

    def __len__(self):
        return int(self.PatientList.shape[0])

    def GeneratePath(self, patientid, Modality):
        params = {'format': 'xml'}
        response = requests.get(self.config['SERVER']['Address'] + '/data/subjects/' + patientid, params=params,
                                auth=(self.config['SERVER']['User'], self.config['SERVER']['Password']))
        resp = xmltodict.parse(response.text, force_list=True)
        subject = resp['xnat:Subject'][0]
        subject_label = subject['@label']
        project = subject['@project']
        experiments = subject['xnat:experiments'][0]['xnat:experiment']
        ## Won't work with many experiments
        for experiment in experiments:
            experiment_label = experiment['@label']
            scans = experiment['xnat:scans'][0]['xnat:scan']
            for scan in scans:
                if (scan['@type'] == Modality):
                    scan_label = scan['@ID'] + '-' + scan['@type']
                    resources_label = scan['xnat:file'][0]['@label']
                    break
        ModalityPath = Path(self.config['DATA']['DataFolder'], project, subject_label, experiment_label, 'scans', scan_label,
                            'resources', resources_label, 'files')
        return ModalityPath

    def __getitem__(self, i):

        data = {}
        patient_id = self.PatientList.loc[i,'subjectid']
        CTPath    = self.GeneratePath(patient_id, 'CT')
        CTSession = ReadDicom(str(CTPath))
        CTSession.SetOrigin(CTSession.GetOrigin())
        CTArray   = sitk.GetArrayFromImage(CTSession)

        ## First define the ROI
        if self.targetROI is not None:
            RTSSPath = self.GeneratePath(patient_id,'Structs')
            RTSSPath = glob.glob(str(RTSSPath) + '/*dcm')
            rtstruct = RTStructBuilder.create_from(
                dicom_series_path=CTPath,
                rt_struct_path=RTSSPath[0]
            )
            roi_names = rtstruct.get_roi_names()
            if self.targetROI in roi_names:
                mask_img = rtstruct.get_roi_mask_by_name(self.targetROI)
                mask_img = mask_img.transpose(2, 0, 1)
                mask_img = np.flip(mask_img, 0)
                properties = regionprops(mask_img.astype(np.int8), mask_img)
                cropbox = properties[0].bbox
            else:
                mask_img = np.ones_like(CTArray)
                cropbox = [0, 0, 0, CTArray.shape[0], CTArray.shape[1], CTArray.shape[2]]

        #     contours  = RTSStoContour(str(RTSSPath), targetROI=self.targetROI) ##Returns coordinate of the target ROI
        #     mask_voxel, bbox_voxel, ROI_voxel, img_indices = get_ROI_voxel(contours, CTPath, roi_range=self.ROIRange)
        # else:
        #     mask_voxel  = np.ones((self.ROIRange[0], self.ROIRange[1]))
        #     bbox_voxel  = np.ones((self.ROIRange[0], self.ROIRange[1]))
        #     ROI_voxel   = np.ones((self.ROIRange[0], self.ROIRange[1]))
        #     Z_indices   = range(self.ROIRange[-1]) ## Slices of the ROI in the CT

        ## Load image within each channel for the target ROI
        for channel in self.selected_channel:
            if channel == 'CT':
                data['CT'] = np.expand_dims(MaskCrop(CTArray, cropbox), 0)
                if self.transform is not None: data['CT'] = self.transform['CT'](data['CT'])
                data['CT'] = torch.as_tensor(data['CT'], dtype=torch.float32)

            if channel == 'RTDose':
                DosePath    = self.GeneratePath(patient_id, 'Dose')
                DoseSession = ReadDicom(DosePath)[..., 0]
                DoseSession = ResamplingITK(DoseSession, CTSession)
                DoseArray = sitk.GetArrayFromImage(DoseSession)
                DoseArray = DoseArray * np.double(DoseSession.GetMetaData('3004|000e'))
                data['RTDose'] = np.expand_dims(MaskCrop(DoseArray, cropbox), 0)
                if self.transform is not None: data['RTDose'] = self.transform['Dose'](data['RTDose'])
                data['RTDose'] = torch.as_tensor(data['RTDose'], dtype=torch.float32)

            if channel == 'PET':
                PETPath     = self.GeneratePath(patient_id, 'PET')
                PETSession  = ReadDicom(PETPath)
                PETSession  = ResamplingITK(PETSession, CTSession)
                PETArray    = sitk.GetArrayFromImage(PETSession)
                data['PET'] = np.expand_dims(MaskCrop(PETArray, cropbox), 0)
                if self.transform is not None: data['PET'] = self.transform['PET'](data['PET'])
                data['PET'] = torch.as_tensor(data['PET'], dtype=torch.float32)

            if channel == 'Records':
                records = self.PatientList.loc[i, self.clinical_cols].to_numpy()
                data['Records'] = torch.as_tensor(list(records), dtype=torch.float32)

        if self.inference:
            return data

        else:
            label = self.PatientList.loc[i, 'xnat_subjectdata_field_map_' + self.target]
            if self.threshold is not None:  label = np.array(label > self.threshold)
            label = torch.as_tensor(label, dtype=torch.float32)
            return data, label

def MaskCrop(img, bbox):
    return img[bbox[0]:bbox[3], bbox[1]:bbox[4], bbox[2]:bbox[5]]



### DataLoader
class DataModule(LightningDataModule):
    def __init__(self, PatientList, config=None, train_transform=None, val_transform=None,
                 batch_size=8, train_size=0.7, val_size=0.2, test_size=0.1, num_workers=64,
                 threshold = 24,
                 target='survival_month',
                 **kwargs):
        super().__init__()
        self.batch_size  = config['MODEL']['batch_size']
        self.num_workers = num_workers
        ## Split Test with fixed seed
        train_val_list, test_list = train_test_split(PatientList,
                                                     test_size=0.15,
                                                     random_state=42,
                                                     stratify=PatientList['xnat_subjectdata_field_map_' + config['DATA']['target']] >= config['MODEL']['Classification_threshold'])
        ## Split train-val with random seed
        train_list, val_list      = train_test_split(train_val_list,
                                                     test_size=0.15,
                                                     random_state=np.random.randint(1, 10000),
                                                     stratify=train_val_list['xnat_subjectdata_field_map_' + config['DATA']['target']] >= config['MODEL']['Classification_threshold'])

        train_list = train_list.reset_index(drop=True)
        val_list   = val_list.reset_index(drop=True)
        test_list  = test_list.reset_index(drop=True)

        self.train_data = DataGenerator(train_list, config = config, transform=train_transform, **kwargs)
        self.val_data   = DataGenerator(val_list,   config = config, transform=val_transform, **kwargs)
        self.test_data  = DataGenerator(test_list,  config = config, transform=val_transform, **kwargs)

    def train_dataloader(self):
        return DataLoader(self.train_data, batch_size=self.batch_size, num_workers=self.num_workers, pin_memory=True, drop_last=True, shuffle=True)

    def val_dataloader(self):
        return DataLoader(self.val_data, batch_size=self.batch_size, num_workers=self.num_workers, pin_memory=True, drop_last=True, shuffle=False)

    def test_dataloader(self):
        return DataLoader(self.test_data, batch_size=self.batch_size, num_workers=self.num_workers, pin_memory=True, drop_last = True)


def QueryFromServer(config, **kwargs):
    print("Querying from Server")
    search_field = []
    search_where = []
    # add search field for clinical features
    if 'Records' in config['DATA']['module']:
        for value in config['DATA']['clinical_columns']:
            dict_temp = {"element_name": "xnat:subjectData", "field_ID": "XNAT_SUBJECTDATA_FIELD_MAP=" + str(value),
                         "sequence": "1", "type": "int"}
            search_field.append(dict_temp)
    ## Target
    dict_temp = {"element_name":"xnat:subjectData","field_ID":"XNAT_SUBJECTDATA_FIELD_MAP="+str(config['DATA']['target']),"sequence":"1", "type":"int"}
    search_field.append(dict_temp)

    ##Project
    search_field.append({"element_name":"xnat:subjectData","field_ID":"PROJECT","sequence":"1", "type":"string"})
    ##Label
    search_field.append(
        {"element_name": "xnat:subjectData", "field_ID": "SUBJECT_LABEL", "sequence": "1", "type": "string"})

    ## Where Condition
    for value in config['SERVER']['Projects']:
        dict_temp = {"schema_field": "xnat:subjectData.PROJECT", "comparison_type": "=", "value": str(value)}
        search_where.append(dict_temp)

    # dict_temp = {"schema_field": "xnat:subjectData.PROJECT", "comparison_type": "!=", "value": "tciaradiogenomics"}
    # search_where.append(dict_temp)

    for key, value in config['CRITERIA'].items():
        dict_temp = {"schema_field": "xnat:subjectData.XNAT_SUBJECTDATA_FIELD_MAP=" + key, "comparison_type": "=",
                     "value": str(value)}
        search_where.append(dict_temp)

    for value in config['FILTER']['patient_id']:
        dict_temp = {"schema_field": "xnat:subjectData.SUBJECT_LABEL", "comparison_type": "!=",
                     "value": str(value)}
        search_where.append(dict_temp)

    # here only include ctSessionData, petSessionData might be used
    for key, value in config['MODALITY'].items():
        dict_temp = {"schema_field": "xnat:ctSessionData.SCAN_COUNT_TYPE=" + key, "comparison_type": ">=",
                     "value": str(value)}
        search_where.append(dict_temp)

    root_element = "xnat:subjectData"
    XML          = XMLCreator(root_element, search_field, search_where)
    xmlstr       = XML.ConstructTree()
    params       = {'format': 'csv'}
    response     = requests.post(config['SERVER']['Address']+'/data/search/', params=params, data=xmlstr, auth=(config['SERVER']['User'], config['SERVER']['Password']))
    PatientList  = pd.read_csv(StringIO(response.text))
    print(PatientList)
    print("Queried from Server")
    return PatientList

def SynchronizeData(config, PatientList):
    session = xnat.connect(config['SERVER']['Address'], user='admin', password='mortavar1977')
    for patientlabel, patientid, project in zip(PatientList['subject_label'], PatientList['subjectid'], PatientList['project']):
        if (not Path(config['DATA']['DataFolder'], project, patientlabel).is_dir()):
            xnatsubject = session.create_object('/data/subjects/'+patientid)
            print("Synchronizing ", patientid, patientlabel)
            xnatsubject.download_dir(Path(config['DATA']['DataFolder'], project)) ## Download data

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

