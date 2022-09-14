import glob
from pytorch_lightning import LightningDataModule, LightningModule, Trainer, seed_everything
from torch.utils.data import DataLoader
import numpy as np
import torch
from sklearn.model_selection import train_test_split, StratifiedShuffleSplit
import xnat
import SimpleITK as sitk
from pathlib import Path
from Utils.DicomTools import *
from Utils.XNATXML import XMLCreator
from io import StringIO
import requests
import pandas as pd

from sklearn.preprocessing import OneHotEncoder, MinMaxScaler, LabelEncoder, OrdinalEncoder, StandardScaler
from sklearn.compose import ColumnTransformer, make_column_transformer
import xml.etree.ElementTree as ET


class DataGenerator(torch.utils.data.Dataset):
    def __init__(self, PatientList,
                 target="pCR", selected_channel=['CT', 'RTDose', 'Records'], targetROI='PTV', ROIRange=[60, 60, 10],
                 dicom_folder=None, transform=None, inference=False, clinical_cols=None, threshold=None, **kwargs):

        super().__init__()
        self.PatientList = PatientList
        self.target = target
        self.targetROI = targetROI
        self.ROIRange = ROIRange
        self.selected_channel = selected_channel
        self.dicom_folder = dicom_folder
        self.transform = transform
        self.inference = inference
        self.clinical_cols = clinical_cols
        self.threshold = threshold

    def __len__(self):
        return int(self.PatientList.shape[0])

    def __getitem__(self, i):
        data = {}
        patient_id = self.PatientList.loc[i, 'subject_label']
        DicomPath = os.path.join(self.dicom_folder, patient_id, self.PatientList.loc[i, 'subject_label'], 'scans/')
        # DicomPath = os.path.join(self.dicom_folder, patient_id, patient_id, 'scans/')
        CTPath = glob.glob(DicomPath + '*CT')
        CTPath = os.path.join(CTPath[0], 'resources', 'DICOM', 'files/')
        CTSession = ReadDicom(CTPath)
        # CTSession.SetOrigin(CTSession.GetOrigin())
        CTArray = sitk.GetArrayFromImage(CTSession)
        if self.targetROI is not None:
            RTSSPath = glob.glob(DicomPath + '*Structs')
            RTSSPath = os.path.join(RTSSPath[0], 'resources', 'secondary', 'files/')
            contours = RTSStoContour(RTSSPath, targetROI=self.targetROI)
            mask_voxel, bbox_voxel, ROI_voxel, image_voxel, img_indices = get_ROI_voxel(contours, CTPath,
                                                                                        roi_range=self.ROIRange)
        else:
            mask_voxel = np.ones((self.ROIRange[0], self.ROIRange[1]))
            bbox_voxel = np.ones((self.ROIRange[0], self.ROIRange[1]))
            ROI_voxel = np.ones((self.ROIRange[0], self.ROIRange[1]))
            img_indices = range(self.ROIRange[-1])

        for channel in self.selected_channel:
            if channel == 'CT':
                data['CT'] = get_masked_img_voxel(CTArray[img_indices], mask_voxel, bbox_voxel, ROI_voxel)
                data['CT'] = np.expand_dims(data['CT'], 0)
                if self.transform is not None: data['CT'] = self.transform['CT'](data['CT'])
                data['CT'] = torch.as_tensor(data['CT'], dtype=torch.float32)

            if channel == 'RTDose':
                DosePath = glob.glob(DicomPath + '*-Dose')
                DosePath = os.path.join(DosePath[0], 'resources', 'DICOM', 'files/')
                DoseSession = ReadDicom(DosePath)[..., 0]
                DoseSession = ResamplingITK(DoseSession, CTSession)
                DoseArray = sitk.GetArrayFromImage(DoseSession)
                # DoseArray = DoseArray * np.double(DoseSession.GetMetaData('3004|000e'))
                data['RTDose'] = get_masked_img_voxel(DoseArray[img_indices], mask_voxel, bbox_voxel, ROI_voxel,
                                                      visImage=True)
                data['RTDose'] = np.expand_dims(data['RTDose'], 0)
                if self.transform is not None: data['RTDose'] = self.transform['Dose'](data['RTDose'])
                data['RTDose'] = torch.as_tensor(data['RTDose'], dtype=torch.float32)

            if channel == 'PET':
                PETPath = glob.glob(DicomPath + '*PET')
                PETPath = os.path.join(PETPath[0], 'resources', 'DICOM', 'files/')
                PETSession = ReadDicom(PETPath)
                PETSession = ResamplingITK(PETSession, CTSession)
                PETArray = sitk.GetArrayFromImage(PETSession)

                data['PET'] = get_masked_img_voxel(PETArray[img_indices], mask_voxel, bbox_voxel, ROI_voxel)
                data['PET'] = np.expand_dims(data['PET'], 0)
                if self.transform is not None: data['PET'] = self.transform(data['PET'])
                data['PET'] = torch.as_tensor(data['PET'], dtype=torch.float32)

            if channel == 'Records':
                records = self.PatientList.loc[i, self.clinical_cols].to_numpy()
                data['Records'] = torch.as_tensor(list(records), dtype=torch.float32)

        if self.inference:
            return data
        else:
            label = self.PatientList.loc[i, 'xnat_subjectdata_field_map_' + self.target[0]]
            if self.threshold is not None:  label = np.array(label > self.threshold)
            label = torch.as_tensor(label, dtype=torch.float32)
            return data, label


### DataLoader
class DataModule(LightningDataModule):
    def __init__(self, PatientList, train_transform=None, val_transform=None,
                 batch_size=8, train_size=0.7, val_size=0.2, test_size=0.1, num_workers=0,
                 threshold = 24,
                 target='survival_month',
                 **kwargs):
        super().__init__()

        self.batch_size = batch_size
        self.num_workers = num_workers

        # Convert regression value to histogram class
        train_val_list, test_list = train_test_split(PatientList, test_size=0.15, random_state=42,
                                                     stratify=PatientList['xnat_subjectdata_field_map_' + target[
                                                         0]] >= threshold)

        train_list, val_list = train_test_split(train_val_list, test_size=0.15, random_state=np.random.randint(1, 10000),
                                                stratify=train_val_list['xnat_subjectdata_field_map_' + target[
                                                    0]] >= threshold)
        
        train_list.reset_index(drop=True)
        val_list.reset_index(drop=True)
        test_list.reset_index(drop=True)

        self.train_data = DataGenerator(train_list, transform=train_transform, target=target, threshold=threshold, **kwargs)
        self.val_data = DataGenerator(val_list, transform=val_transform, target=target, threshold=threshold, **kwargs)
        self.test_data = DataGenerator(test_list, transform=val_transform, target=target, threshold=threshold, **kwargs)

    def train_dataloader(self):
        return DataLoader(self.train_data, batch_size=self.batch_size, num_workers=self.num_workers, pin_memory=True,
                          drop_last=True, shuffle=True, collate_fn=None)

    def val_dataloader(self):
        return DataLoader(self.val_data, batch_size=self.batch_size, num_workers=self.num_workers, drop_last=True,
                          pin_memory=True, collate_fn=None)

    def test_dataloader(self):
        return DataLoader(self.test_data, batch_size=self.batch_size, num_workers=self.num_workers, drop_last=True,
                          pin_memory=True, collate_fn=None)


def QueryFromServer(config, **kwargs):
    print("Querying from Server")
    search_field = []
    search_where = []

    if 'Records' in config['DATA']['module']:
        for value in config['DATA']['clinical_columns']:
            dict_temp = {"element_name": "xnat:subjectData", "field_ID": "XNAT_SUBJECTDATA_FIELD_MAP=" + str(value),
                         "sequence": "1", "type": "int"}
            search_field.append(dict_temp)

    ## ITEMS TO QUERY
    for value in config['DATA']['target']:
        dict_temp = {"element_name": "xnat:subjectData", "field_ID": "XNAT_SUBJECTDATA_FIELD_MAP=" + str(value),
                     "sequence": "1", "type": "int"}
        search_field.append(dict_temp)

    ##Project
    search_field.append({"element_name": "xnat:subjectData", "field_ID": "PROJECT", "sequence": "1", "type": "string"})
    ##Label
    search_field.append(
        {"element_name": "xnat:subjectData", "field_ID": "SUBJECT_LABEL", "sequence": "1", "type": "string"})

    ## WHERE CONDITION
    for value in config['SERVER']['Projects']:
        dict_temp = {"schema_field": "xnat:subjectData.PROJECT", "comparison_type": "=", "value": str(value)}
        search_where.append(dict_temp)
    for key, value in config['CRITERIA'].items():
        dict_temp = {"schema_field": "xnat:subjectData.XNAT_SUBJECTDATA_FIELD_MAP=" + key, "comparison_type": "=",
                     "value": str(value)}
        search_where.append(dict_temp)

    for value in config['FILTER']['patient_id']:
        dict_temp = {"schema_field": "xnat:subjectData.SUBJECT_LABEL", "comparison_type": "!=",
                     "value": str(value)}
        search_where.append(dict_temp)

    for key, value in config['MODALITY'].items():
        dict_temp = {"schema_field": "xnat:ctSessionData.SCAN_COUNT_TYPE=" + key, "comparison_type": ">=",
                     "value": str(value)}
        search_where.append(dict_temp)

    root_element = "xnat:subjectData"
    XML = XMLCreator(root_element, search_field, search_where)
    xmlstr = XML.ConstructTree()
    params = {'format': 'csv'}
    response = requests.post('http://128.16.11.124:8080/xnat/data/search/', params=params, data=xmlstr,
                             auth=(config['SERVER']['User'], config['SERVER']['Password']))
    PatientList = pd.read_csv(StringIO(response.text))
    print(PatientList)
    print("Queried from Server")
    return PatientList


def SynchronizeData(config, subject_list):
    params = {'format': 'csv'}
    response = requests.get('http://128.16.11.124:8080/xnat/data/subjects/XNAT01_S00800', params=params,
                            auth=(config['SERVER']['User'], config['SERVER']['Password']))
    # print(response.text)
    import xmltodict
    subject_dict = xmltodict.parse(response.text)
    print(subject_dict['xnat:Subject']["xnat:experiments"]["xnat:experiment"])
    # subject_query = ET.fromstring(response.text)

    # for experiments in subject_query:
    #    print(experiments)
    #    for experiment in experiments:
    #        print(experiment.tag)
    #        for scan in experiment:
    #            print(scan.tag)
    ## Verify if data exists in data folder
    for subject in subject_list['subject_label']:
        # print(subject.label, subject.fulluri, dir(subject), subject.uri)
        # scans = subject.experiments[subject.label].scans['CT'].fulldata
        if (not Path(config['DATA']['DataFolder'], subject).is_dir()):
            print("Synchronizing ", subject.id, subject.label)
            subject.download_dir(config['DATA']['DataFolder'])

    ## Download data


def custom_collate(original_batch):
    filtered_data = {}
    filtered_target = []

    # Init the dict
    for key in original_batch[0][0].keys():
        filtered_data[key] = None

    i = 0
    for patient in original_batch:
        none_found = False
        if "Anatomy" in patient[0].keys():
            if patient[0]["Anatomy"] is None:
                none_found = True
        if "Dose" in patient[0].keys():
            if patient[0]["Dose"] is None:
                none_found = True

        if not none_found:
            if i == 0:
                for key in patient[0].keys():
                    t_shape = (1, patient[0][key].shape[0], patient[0][key].shape[1], patient[0][key].shape[2],
                               patient[0][key].shape[3])
                    filtered_data[key] = torch.reshape(patient[0][key], t_shape)
            else:
                for key in patient[0].keys():
                    t_shape = (1, patient[0][key].shape[0], patient[0][key].shape[1], patient[0][key].shape[2],
                               patient[0][key].shape[3])
                    filtered_data[key] = torch.vstack((filtered_data[key], torch.reshape(patient[0][key], t_shape)))

            filtered_target.append(patient[1])
            i += 1

    return filtered_data, torch.FloatTensor(filtered_target)


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
