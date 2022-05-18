import matplotlib.pyplot as plt
from rt_utils import RTStructBuilder
from pytorch_lightning import LightningDataModule, LightningModule, Trainer, seed_everything
from torch.utils.data import DataLoader
from torchvision import transforms
import pandas as pd
import os
from pathlib import Path
from torch.utils.data import Dataset
import torchvision.models as models
import numpy as np
import fnmatch
import torch
import sys, glob
from sklearn.model_selection import train_test_split, StratifiedShuffleSplit
import SimpleITK as sitk
import scipy.ndimage as ndi
from skimage.measure import regionprops
from Utils.GenerateSmoothLabel import get_smoothed_label_distribution
from sklearn.preprocessing import StandardScaler
import xnat
import SimpleITK as sitk
from monai.data import image_reader
from monai.transforms import LoadImage, LoadImaged
from numpy import array
from scipy.interpolate import RegularGridInterpolator as rgi
from scipy.ndimage import map_coordinates

class DataGenerator(torch.utils.data.Dataset):
    def __init__(self, PatientList, config, keys, transform=None, **kwargs):
        super().__init__()
        self.transform = transform
        self.keys = keys
        self.inference = kwargs['inference']
        self.PatientList = PatientList
        # self.n_norm = kwargs['numerical_norm']
        # self.c_norm = kwargs['category_norm']
        self.config = config

    def __len__(self):
        return int(len(self.PatientList))

    def __getitem__(self, id):

        datadict = {}
        roiSize = [10, 40, 40]
        patient_id = self.PatientList[id].label
        ScanPath = self.config['DATA']['DataFolder'] + patient_id + '\\' + patient_id + '\\' + 'scans\\'
        subfolder_list = os.listdir(ScanPath)
        reader = image_reader.ITKReader()
        label = self.PatientList[id].fields[self.config['DATA']['target']]
        CT_match_folder = [match for match in subfolder_list if "CT" in match]
        full_CT_path = ScanPath + CT_match_folder[0] + '\\resources\\DICOM\\files\\'
        dicom_files = os.listdir(full_CT_path)
        correct_Origin = reader.read(full_CT_path + dicom_files[0])
        itkObj = reader.read(full_CT_path)
        itkObj.SetOrigin(correct_Origin.GetOrigin())
        # Get the mask of PTV
        if "Dose" in self.keys or "Anatomy" in self.keys:
            if self.config['DATA']['Use_mask']:
                RT_match_folder = [match for match in subfolder_list if "Structs" in match]
                full_RT_path = ScanPath + RT_match_folder[0] + '\\resources\\secondary\\files\\1-1.dcm'
                # read the rtstruct
                rtstruct = RTStructBuilder.create_from(
                    dicom_series_path=full_CT_path,
                    rt_struct_path=full_RT_path
                )
                mask_img = rtstruct.get_roi_mask_by_name(self.config['DATA']['mask_name'])
                # need process to convert 2D points to 3D masks
                properties = regionprops(mask_img.astype(np.int8), mask_img)
                cropbox = properties[0].bbox

        if "Dose" in self.keys:
            Dose_match_folders = [match for match in subfolder_list if "Dose" in match]
            full_Dose_path = [ScanPath + Dose_match_folder + '\\resources\\DICOM\\files\\' for Dose_match_folder in
                              Dose_match_folders]
            if len(full_Dose_path) > 1:
                for dose_path in full_Dose_path:
                    itkObjD = reader.read(dose_path + '1-1.dcm')
                    dose, dose_info = reader.get_data(itkObjD)
                    for value in dose_info.values():
                        if 'totalhetero' in value:
                            break
            else:
                itkObjD = reader.read(full_Dose_path[0] + '1-1.dcm')
                dose, dose_info = reader.get_data(itkObjD)

            ResampledDose = DoseMatchCT(itkObjD, dose, itkObj)

            # maxDoseCoords = findMaxDoseCoord(dose) # Find coordinates of max dose
            # checkCrop(maxDoseCoords, roiSize, dose.shape, self.mastersheet["DosePath"].iloc[id]) # Check if the crop works (min-max image shape costraint)
            # datadict["Dose"]  = np.expand_dims(CropImg(dose, maxDoseCoords, roiSize),0)
            if self.config['DATA']['Use_mask']:
                datadict["Dose"] = np.expand_dims(MaskCrop(ResampledDose, cropbox), 0)
            else:
                datadict["Dose"] = np.expand_dims(ResampledDose, 0)

            if self.transform:
                transformed_data = self.transform(datadict["Dose"])
                if transformed_data is None:
                    datadict["Dose"] = None
                else:
                    datadict["Dose"] = torch.from_numpy(transformed_data)

        if "Anatomy" in self.keys:
            anatomy, _ = reader.get_data(itkObj)
            # anatomy = LoadImg(self.mastersheet["CTPath"].iloc[id])
            # datadict["Anatomy"] = np.expand_dims(CropImg(anatomy, maxDoseCoords, roiSize), 0)
            if self.config['DATA']['Use_mask']:
                datadict["Anatomy"] = np.expand_dims(MaskCrop(anatomy, cropbox), 0)
            else:
                datadict["Anatomy"] = np.expand_dims(anatomy, 0)

            # datadict["Anatomy"] = np.expand_dims(anatomy, 0)

            if self.transform:
                transformed_data = self.transform(datadict["Anatomy"])
                if transformed_data is None:
                    datadict["Anatomy"] = None
                else:
                    datadict["Anatomy"] = torch.from_numpy(transformed_data)

            # print(datadict["Anatomy"].size, type(datadict["Anatomy"]))
        if "Clinical" in self.keys:
            numerical_data, category_data = LoadClinicalData(self.mastersheet)
            # data = clinical_data.iloc[id].to_numpy()
            num_data = self.n_norm.transform([numerical_data.iloc[id]])
            cat_data = self.c_norm.transform([category_data.iloc[id]]).toarray()
            data = np.concatenate((num_data, cat_data), axis=1)
            datadict["Clinical"] = data.flatten()

        if (self.inference):
            return datadict
        else:
            return datadict, label.astype(np.float32)


### DataLoader
class DataModule(LightningDataModule):
    def __init__(self, PatientList, config, keys, train_transform=None, val_transform=None, batch_size=64, **kwargs):
        super().__init__()
        self.batch_size = batch_size

        # Convert regression value to histogram class
        train, val_test = train_test_split(PatientList, train_size=0.7)
        test, val = train_test_split(val_test, test_size=0.66)

        self.train_data = DataGenerator(train, config, keys, transform=train_transform, **kwargs)
        self.val_data = DataGenerator(val, config, keys, transform=val_transform, **kwargs)
        self.test_data = DataGenerator(test, config, keys, transform=val_transform, **kwargs)

    def train_dataloader(self): return DataLoader(self.train_data, batch_size=self.batch_size, shuffle=True,
                                                  num_workers=0, collate_fn=custom_collate)

    def val_dataloader(self):   return DataLoader(self.val_data, batch_size=self.batch_size, num_workers=0,
                                                  collate_fn=custom_collate)

    def test_dataloader(self):  return DataLoader(self.test_data, batch_size=self.batch_size, collate_fn=custom_collate)


def QueryFromServer(config, **kwargs):
    print("Querying from Server")
    ## Get List of Patients
    session = xnat.connect('http://128.16.11.124:8080/xnat/', user=config["SERVER"]["User"],
                           password=config["SERVER"]["Password"])
    project = session.projects[config["SERVER"]["Project"]]
    ## Verify fit with clinical criteria
    subject_list = []
    clinical_keys = list(config['CRITERIA'].keys())
    for nb, subject in enumerate(project.subjects.values()):
        # print("Criteria", subject, nb)
        # if(nb>10): break
        subject_keys = subject.fields.keys()  # .key_map
        # print(set(subject_dict))
        # subject_keys = list(subject_dict.keys())
        if set(clinical_keys).issubset(subject_keys):
            if (all(subject.fields[k] == str(v) for k, v in config['CRITERIA'].items())):  subject_list.append(subject)

    ## Verify availability of images
    for k, v in config['MODALITY'].items():
        for nb, subject in enumerate(subject_list):
            # if(nb>10): break
            # print("Modality", subject, nb)
            for experiment in subject.experiments.values():
                scan_dict = experiment.scans.key_map
                if (v not in scan_dict.keys()):
                    subject_list.remove(subject)
                    break
    print("Queried from Server")
    return subject_list


def DoseMatchCT(itkObjD, DoseVolume, itkObj):
    originD = itkObjD.GetOrigin()
    spaceD = itkObjD.GetSpacing()
    origin = itkObj.GetOrigin()
    space = itkObj.GetSpacing()
    dx = np.arange(0, itkObjD.shape[2])*spaceD[2] + originD[0]
    dy = np.arange(0, itkObjD.shape[1])*spaceD[1] + originD[1]
    dz = -np.arange(0, itkObjD.shape[0])*spaceD[0] + originD[2]
    dz.sort()

    cz = -np.arange(0, itkObj.shape[0]) * space[2] + origin[2]
    cy = np.arange(0, itkObj.shape[1]) * space[1] + origin[1]
    cx = np.arange(0, itkObj.shape[2]) * space[0] + origin[0]
    cz.sort()

    cxv, cyv, czv = np.meshgrid(cx, cy, cz, indexing='ij')

    Vi = interp3(dx, dy, dz, DoseVolume, cxv, cyv, czv)
    Vf = np.flip(Vi, 2)
    return Vf

def SynchronizeData(config, subject_list):
    ## Data Storage Format --> Idem as XNAT

    ## Verify if data exists in data folder
    for subject in subject_list:
        # print(subject.label, subject.fulluri, dir(subject), subject.uri)
        scans = subject.experiments[subject.label].scans['CT'].fulldata
        if (not Path(config['DATA']['DataFolder'], subject.label).is_dir()):
            print("Synchronizing ", subject.id, subject.label)
            subject.download_dir(config['DATA']['DataFolder'])

    ## Download data


def LoadImg(path):
    img = sitk.ReadImage(path)
    return sitk.GetArrayFromImage(img).astype(np.float32)


def MaskCrop(img, bbox):
    return img[bbox[0]:bbox[3], bbox[1]:bbox[4], bbox[2]:bbox[5]]


def CropImg(img, center, delta):  ## Crop image
    return img[center[0] - delta[0]:center[0] + delta[0], center[1] - delta[1]:center[1] + delta[1],
           center[2] - delta[2]:center[2] + delta[2]]


def findMaxDoseCoord(img):
    result = np.where(img == np.amax(img))
    listOfCordinates = list(zip(result[0], result[1], result[2]))

    return listOfCordinates[int(len(listOfCordinates) / 2)]


def checkCrop(center, delta, imgShape, fn):
    if (center[0] - delta[0] < 0 or
            center[0] + delta[0] > imgShape[0] or
            center[1] - delta[1] < 0 or
            center[1] + delta[1] > imgShape[1] or
            center[2] - delta[2] < 0 or
            center[2] + delta[2] > imgShape[2]):
        print("ERROR! Invalid crop for file %s" % (fn))
        exit()


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


def LoadClinicalData(MasterSheet):
    clinical_columns = ['arm', 'age', 'gender', 'race', 'ethnicity', 'zubrod',
                        'histology', 'nonsquam_squam', 'ajcc_stage_grp', 'rt_technique',
                        # 'egfr_hscore_200', 'received_conc_cetuximab','rt_compliance_physician',
                        'smoke_hx', 'rx_terminated_ae', 'rt_dose',
                        'volume_ptv', 'dmax_ptv', 'v100_ptv',
                        'v95_ptv', 'v5_lung', 'v20_lung', 'dmean_lung', 'v5_heart',
                        'v30_heart', 'v20_esophagus', 'v60_esophagus', 'Dmin_PTV_CTV_MARGIN',
                        'Dmax_PTV_CTV_MARGIN', 'Dmean_PTV_CTV_MARGIN',
                        'rt_compliance_ptv90', 'received_conc_chemo',
                        ]

    numerical_cols = ['age', 'volume_ptv', 'dmax_ptv', 'v100_ptv',
                      'v95_ptv', 'v5_lung', 'v20_lung', 'dmean_lung', 'v5_heart',
                      'v30_heart', 'v20_esophagus', 'v60_esophagus', 'Dmin_PTV_CTV_MARGIN',
                      'Dmax_PTV_CTV_MARGIN', 'Dmean_PTV_CTV_MARGIN']

    category_cols = list(set(clinical_columns).difference(set(numerical_cols)))

    numerical_cols = list(set(numerical_cols).intersection(set(MasterSheet.keys())))
    category_cols = list(set(category_cols).intersection(set(MasterSheet.keys())))

    numerical_data = MasterSheet[numerical_cols]  # pd.DataFrame()
    category_data = MasterSheet[category_cols]

    return numerical_data, category_data


def interp3(x, y, z, v, xi, yi, zi, **kwargs):
    """Sample a 3D array "v" with pixel corner locations at "x","y","z" at the
    points in "xi", "yi", "zi" using linear interpolation. Additional kwargs
    are passed on to ``scipy.ndimage.map_coordinates``."""
    def index_coords(corner_locs, interp_locs):
        index = np.arange(len(corner_locs))
        if np.all(np.diff(corner_locs) < 0):
            corner_locs, index = corner_locs[::-1], index[::-1]
        return np.interp(interp_locs, corner_locs, index)

    orig_shape = np.asarray(xi).shape
    xi, yi, zi = np.atleast_1d(xi, yi, zi)
    for arr in [xi, yi, zi]:
        arr.shape = -1

    output = np.empty(xi.shape, dtype=float)
    coords = [index_coords(*item) for item in zip([x, y, z], [xi, yi, zi])]

    map_coordinates(v, coords, order=1, output=output, **kwargs)

    return output.reshape(orig_shape)