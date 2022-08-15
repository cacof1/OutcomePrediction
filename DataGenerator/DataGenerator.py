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

class DataGenerator(torch.utils.data.Dataset):
    def __init__(self, PatientList,
                 target="pCR", selected_channel=['CT','RTDose','Records'], targetROI='PTV', ROIRange=[60,60,20],
                 dicom_folder=None, transform=None, inference=False, **kwargs):

        super().__init__()
        self.PatientList = PatientList
        self.target = target
        self.targetROI = targetROI
        self.ROIRange = ROIRange
        self.selected_channel = selected_channel
        self.dicom_folder = dicom_folder
        self.transform = transform
        self.inference = inference

        self.n_norm = kwargs['numerical_norm']
        self.c_norm = kwargs['category_norm']

    def __len__(self):
        return int(len(self.PatientList))

    def __getitem__(self, i):
        data = {}
        patient_id = self.PatientList[id].label
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
                category_feat, numerical_feat = LoadClinicalData(self.config, [self.PatientList[id]])
                n_category_feat = self.c_norm.transform(category_feat).toarray()
                n_numerical_feat = self.n_norm.transform(numerical_feat)
                records = np.concatenate((n_numerical_feat, n_category_feat), axis=1)
                records = np.squeeze(records)
                data['Records'] = records

        if self.inference:
            return data
        else:
            label = self.PatientList[id].fields[self.target]
            label = torch.as_tensor(label, dtype=torch.int64)
            return data, label

### DataLoader
class DataModule(LightningDataModule):
    def __init__(self, patient_df, train_transform=None, val_transform=None, batch_size=8, train_size=0.7, val_size=0.3,
                 target="pCR",selected_channel=['CT','RTDose','Records'], targetROI='PTV',ROIRange=[60,60,20],
                 dicom_folder=None, num_workers=0, **kwargs):
        super().__init__()

        self.batch_size = batch_size
        self.num_workers = num_workers

        patient_df_train, patient_df_val = train_test_split(patient_df, test_size=val_size, train_size=train_size)
        patient_df_train.reset_index(drop=True, inplace=True)
        patient_df_val.reset_index(drop=True, inplace=True)

        self.train_data = DataGenerator(patient_df_train, target=target, selected_channel=selected_channel,
                                        targetROI=targetROI,ROIRange=ROIRange,dicom_folder=dicom_folder,
                                        transform=train_transform, **kwargs)
        self.val_data = DataGenerator(patient_df_val, target=target, selected_channel=selected_channel,
                                        targetROI=targetROI,ROIRange=ROIRange,dicom_folder=dicom_folder,
                                        transform=val_transform, **kwargs)

    def train_dataloader(self):
        return DataLoader(self.train_data, batch_size=self.batch_size, num_workers=self.num_workers, pin_memory=True, shuffle=True)

    def val_dataloader(self):
        return DataLoader(self.val_data, batch_size=self.batch_size, num_workers=self.num_workers, pin_memory=True)

def QueryFromServer(config, **kwargs):
    print("Querying from Server")
    ## Get List of Patients
    session = xnat.connect('http://128.16.11.124:8080/xnat/', user=config["SERVER"]["User"],
                           password=config["SERVER"]["Password"])
    project = session.projects[config["SERVER"]["Project"]]
    ## Verify fit with clinical criteria
    subject_list = []
    clinical_keys = list(config['CRITERIA'].keys())
    target = config['DATA']['target']

    for nb, subject in enumerate(project.subjects.values()):
        # print("Criteria", subject, nb)
        # if(nb>10): break
        subject_keys = subject.fields.keys()  # .key_map
        # print(set(subject_dict))
        # subject_keys = list(subject_dict.keys())
        if set(clinical_keys).issubset(subject_keys):
            if (all(subject.fields[k] in str(v) for k, v in config['CRITERIA'].items())):  subject_list.append(subject)

    # Verify availability of images
    rm_subject_list = []
    for k, v in config['MODALITY'].items():
        for nb, subject in enumerate(subject_list):
            # if(nb>10): break
            # print("Modality", subject, nb)
            # keys = np.concatenate([list(experiment.scans.key_map.keys()) for experiment in subject.experiments.values()]
            #                       , axis=0)

            # remove the target is nan
            if subject.fields[target] == 'nan':
                rm_subject_list.append(subject)
                break

            # verity the images
            if len(config['ImageSession'].items()) > 1:
                for experiment in subject.experiments.values():
                    if config['ImageSession'].get(k) in experiment.label:
                        keys = experiment.scans.key_map.keys()
            else:
                keys = np.concatenate([list(experiment.scans.key_map.keys()) for experiment in
                                       subject.experiments.values()], axis=0)

            if v not in keys:
                # if v not in scan_dict.keys() and 'Fx1Dose' not in scan_dict.keys():
                rm_subject_list.append(subject)
    # Verify the clinical features
    if 'Clinical' in config['DATA']['module']:
        clinical_feat = np.concatenate([feat for feat in config['CLINICAL'].values()])
        for v in clinical_feat:
            for nb, subject in enumerate(subject_list):
                if v not in subject.fields.keys():
                    if subject not in rm_subject_list:
                        rm_subject_list.append(subject)

    rm_subject_list = list(set(rm_subject_list))
    for subject in rm_subject_list:
        subject_list.remove(subject)

    # for k, v in config['MODALITY'].items():
    #     for nb, subject in enumerate(subject_list):
    #         # if(nb>10): break
    #         # print("Modality", subject, nb)
    #         for experiment in subject.experiments.values():
    #             scan_dict = experiment.scans.key_map
    #             if (v not in scan_dict.keys() and 'Fx1Dose' not in scan_dict.keys()):
    #                 subject_list.remove(subject)
    #                 break
    print("Queried from Server")
    return subject_list

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
