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

class DataGenerator(torch.utils.data.Dataset):
    def __init__(self, PatientList,
                 target="pCR", selected_channel=['CT','RTDose','Records'], targetROI='PTV', ROIRange=[60,60,10],
                 dicom_folder=None, transform=None, inference=False, clinical_cols=None,**kwargs):

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

    def __len__(self):
        return int(self.PatientList.shape[0])

    def __getitem__(self, i):
        data = {}
        patient_id = self.PatientList.loc[i,'subject_label']
        DicomPath = os.path.join(self.dicom_folder, patient_id, patient_id, 'scans/')
        if self.targetROI is not None:
            RTSSPath = glob.glob(DicomPath + '*Structs')
            RTSSPath = os.path.join(RTSSPath[0], 'resources/secondary/files/')
            contours = RTSStoContour(RTSSPath, targetROI=self.targetROI)
            CTPath = glob.glob(DicomPath + '*CT')
            CTPath = os.path.join(CTPath[0], 'resources/DICOM/files/')
            CTSession = ReadDicom(CTPath)
            CTSession.SetOrigin(CTSession.GetOrigin())
            CTArray = sitk.GetArrayFromImage(CTSession)
            mask_voxel, bbox_voxel, ROI_voxel, img_indices = get_ROI_voxel(contours, CTPath, roi_range=self.ROIRange)

        for channel in self.selected_channel:
            if channel == 'CT':
                data['CT'] = get_masked_img_voxel(CTArray[img_indices], mask_voxel, bbox_voxel, ROI_voxel)
                data['CT'] = np.expand_dims(data['CT'], 0)
                if self.transform is not None: data['CT'] = self.transform(data['CT'])
                data['CT'] = torch.as_tensor(data['CT'], dtype=torch.float32)

            if channel == 'RTDose':
                DosePath = glob.glob(DicomPath + '*Dose')
                DosePath = os.path.join(DosePath[0], 'resources/DICOM/files/')
                DoseSession = ReadDicom(DosePath)[..., 0]
                DoseSession = ResamplingITK(DoseSession, CTSession)
                DoseArray = sitk.GetArrayFromImage(DoseSession)
                data['RTDose'] = get_masked_img_voxel(DoseArray[img_indices], mask_voxel, bbox_voxel, ROI_voxel)
                data['RTDose'] = np.expand_dims(data['RTDose'], 0)
                if self.transform is not None: data['RTDose'] = self.transform(data['RTDose'])
                data['RTDose'] = torch.as_tensor(data['RTDose'], dtype=torch.float32)

            if channel == 'PET':
                PETPath = glob.glob(DicomPath + '*PET')
                PETPath = os.path.join(PETPath[0], 'resources/DICOM/files/')
                PETSession = ReadDicom(PETPath)
                PETSession = ResamplingITK(PETSession, CTSession)
                PETArray = sitk.GetArrayFromImage(PETSession)
                data['PET'] = get_masked_img_voxel(PETArray[img_indices], mask_voxel, bbox_voxel, ROI_voxel)
                data['PET'] = np.expand_dims(data['PET'], 0)
                if self.transform is not None: data['PET'] = self.transform(data['PET'])
                data['PET'] = torch.as_tensor(data['PET'], dtype=torch.float32)

            if channel == 'Records':
                records = self.PatientList.loc[:,self.clinical_cols].toarray()
                records = torch.as_tensor(records, dtype=torch.float32)
                data['Records'] = records

        if self.inference:
            return data
        else:
            llabel = self.PatientList.loc[i,self.target]
            label = torch.as_tensor(label, dtype=torch.int64)
            return data, label

### DataLoader
class DataModule(LightningDataModule):
    def __init__(self, PatientList, train_transform=None, val_transform=None, batch_size=8, train_size=0.7, val_size=0.2, test_size=0.1,
                 target="pCR",selected_channel=['CT','RTDose','Records'], targetROI='PTV',ROIRange=[60,60,20],
                 dicom_folder=None, num_workers=0, clinical_cols=None,**kwargs):
        super().__init__()

        self.batch_size = batch_size
        self.num_workers = num_workers

        train_list, val_list = train_test_split(PatientList, test_size=(val_size+test_size), train_size=train_size)
        val_list, test_list = train_test_split(val_list, test_size=(test_size/val_size), train_size=(1-test_size/val_size))
        
        train_list.reset_index(inplace=True,drop=True)
        val_list.reset_index(inplace=True, drop=True)
        test_list.reset_index(inplace=True, drop=True)
        
        self.train_data = DataGenerator(train_list, target=target, selected_channel=selected_channel,
                                        targetROI=targetROI,ROIRange=ROIRange,dicom_folder=dicom_folder,
                                        transform=train_transform, clinical_cols=clinical_cols, **kwargs)
        self.val_data = DataGenerator(val_list, target=target, selected_channel=selected_channel,
                                        targetROI=targetROI,ROIRange=ROIRange,dicom_folder=dicom_folder,
                                        transform=val_transform, clinical_cols=clinical_cols, **kwargs)
        self.test_data = DataGenerator(test_list, target=target, selected_channel=selected_channel,
                                      targetROI=targetROI, ROIRange=ROIRange, dicom_folder=dicom_folder,
                                      transform=val_transform, clinical_cols=clinical_cols, **kwargs)

    def train_dataloader(self):
        return DataLoader(self.train_data, batch_size=self.batch_size, num_workers=self.num_workers, pin_memory=True, shuffle=True)

    def val_dataloader(self):
        return DataLoader(self.val_data, batch_size=self.batch_size, num_workers=self.num_workers, pin_memory=True)
    
    def test_dataloader(self):
        return DataLoader(self.test_data, batch_size=self.batch_size, num_workers=self.num_workers, pin_memory=True)

def QueryFromServer(config, **kwargs):
    print("Querying from Server")

    search_field = [{"element_name": "xnat:subjectData", "field_ID": "SUBJECT_LABEL", "sequence": "1", "type": "1"}]
    search_where = [
        {"schema_field": "xnat:subjectData.XNAT_SUBJECTDATA_FIELD_MAP=survival_status", "comparison_type": "=",
         "value": "1"},
        {"schema_field": "xnat:ctSessionData.SCAN_COUNT_TYPE=Dose", "comparison_type": ">=", "value": "1"}, ]

    root_element = "xnat:subjectData"
    test = XMLCreator(root_element, search_field, search_where)
    params = {'format': 'csv'}

    files = {'file': open('example.xml', 'rb')}
    response = requests.post('http://128.16.11.124:8080/xnat/data/search/', params=params, files=files,
                             auth=(config['SERVER']['User'], config['SERVER']['Password']))
    PatientList = pd.read_csv(StringIO(response.text))
    print(PatientList)
    print("Queried from Server")

    return PatientList

def SynchronizeData(config, subject_list):
    ## Data Storage Format --> Idem as XNAT

    ## Verify if data exists in data folder
    for subject in subject_list:
        # print(subject.label, subject.fulluri, dir(subject), subject.uri)
        # scans = subject.experiments[subject.label].scans['CT'].fulldata
        if (not Path(config['DATA']['DataFolder'], subject.label).is_dir()):
            print("Synchronizing ", subject.id, subject.label)
            subject.download_dir(config['DATA']['DataFolder'])

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
    category_cols = config['CLINICAL']['category_feat']
    numerical_cols = config['CLINICAL']['numerical_feat']

    category_feats = []
    numerical_feats = []
    for i, patient in enumerate(PatientList):
        clinical_features = patient.fields
        numerical_feat = [clinical_features[x] for x in numerical_cols]
        category_feat = [clinical_features[x] for x in category_cols]
        category_feats.append(category_feat)
        numerical_feats.append(numerical_feat)

    return category_feats, numerical_feats
