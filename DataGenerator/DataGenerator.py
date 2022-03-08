import matplotlib.pyplot as plt
from pytorch_lightning import LightningDataModule, LightningModule, Trainer,seed_everything
from torch.utils.data import DataLoader
from torchvision import transforms
import pandas as pd
import os
from pathlib import Path
from torch.utils.data import Dataset
import torchvision.models as models
import numpy as np
import torch
#import openslide
import sys, glob
import torch.nn.functional as F
from sklearn.model_selection import train_test_split
import SimpleITK as sitk
import scipy.ndimage as ndi

class DataGenerator(torch.utils.data.Dataset):
    def __init__(self, mastersheet, label, keys, inference=False, transform=None, target_transform = None):
        super().__init__()
        self.keys             = keys
        self.transform        = transform
        self.target_transform = target_transform
        self.label            = label
        self.inference        = inference
        self.mastersheet      = mastersheet
    def __len__(self):
        return int(self.mastersheet.shape[0])

    def __getitem__(self, id):

        datadict = {}
        roiSize = [10, 40, 40]

        label    = self.mastersheet[self.label].iloc[id]

        if "Dose" in self.keys:
            dose     = LoadImg(self.mastersheet["DosePath"].iloc[id])
            maxDoseCoords = findMaxDoseCoord(dose) # Find coordinates of max dose
            checkCrop(maxDoseCoords, roiSize, dose.shape, self.mastersheet["DosePath"].iloc[id]) # Check if the crop works (min-max image shape costraint)
            datadict["Dose"]  = np.expand_dims(CropImg(dose, maxDoseCoords, roiSize),0)
            if self.transform:
                transformed_data = self.transform(datadict["Dose"])
                if transformed_data is None:
                    datadict["Dose"]  = None
                else:
                    datadict["Dose"]  = torch.from_numpy(transformed_data)

        if "Anatomy" in self.keys:
            anatomy  = LoadImg(self.mastersheet["CTPath"].iloc[id])
            datadict["Anatomy"] = np.expand_dims(CropImg(anatomy, maxDoseCoords, roiSize), 0)
            if self.transform:
                transformed_data = self.transform(datadict["Anatomy"])
                if transformed_data is None:
                    datadict["Anatomy"]  = None
                else:
                    datadict["Anatomy"]  = torch.from_numpy(transformed_data)

        #print(datadict["Anatomy"].size, type(datadict["Anatomy"]))
        if "Clinical" in self.keys:
            datadict["Clinical"] = LoadClinical(self.mastersheet.iloc[id])

        if(self.inference): return datadict
        else: return datadict, label.astype(np.float32)

### DataLoader
class DataModule(LightningDataModule):
    def __init__(self, mastersheet, label, keys, train_transform = None, val_transform = None, batch_size = 8, **kwargs):
        super().__init__()
        self.batch_size      = batch_size
        train, val_test       = train_test_split(mastersheet, train_size=0.7)
        val, test             = train_test_split(val_test,test_size =0.66)

        self.train_data      = DataGenerator(train, label, keys, transform = train_transform, **kwargs)
        self.val_data        = DataGenerator(val,   label, keys, transform = val_transform, **kwargs)
        self.test_data       = DataGenerator(test,  label, keys, transform = val_transform, **kwargs)

    def train_dataloader(self): return DataLoader(self.train_data, batch_size=self.batch_size, shuffle=True, num_workers=0, collate_fn=custom_collate)
    def val_dataloader(self):   return DataLoader(self.val_data,   batch_size=self.batch_size,num_workers=0, collate_fn=custom_collate)
    def test_dataloader(self):  return DataLoader(self.test_data,  batch_size=self.batch_size, collate_fn=custom_collate)

def PatientQuery(config, **kwargs):
    mastersheet = pd.read_csv(config['DATA']['Mastersheet'],index_col='patid')
    for key,item in config['CRITERIA'].items(): mastersheet = mastersheet[mastersheet[key]==item]
    return mastersheet

def LoadImg(path):
    img = sitk.ReadImage(path)
    return sitk.GetArrayFromImage(img).astype(np.float32)

def CropImg(img, center, delta): ## Crop image 
    return img[center[0]-delta[0]:center[0]+delta[0], center[1]-delta[1]:center[1]+delta[1], center[2]-delta[2]:center[2]+delta[2]]

def findMaxDoseCoord(img):
    result = np.where(img == np.amax(img))
    listOfCordinates = list(zip(result[0], result[1], result[2]))

    return listOfCordinates[int(len(listOfCordinates)/2)]

def checkCrop(center, delta, imgShape, fn):
    if (center[0]-delta[0] < 0 or
        center[0]+delta[0] > imgShape[0] or
        center[1]-delta[1] < 0 or
        center[1]+delta[1] > imgShape[1] or
        center[2]-delta[2] < 0 or
        center[2]+delta[2] > imgShape[2]):

        print("ERROR! Invalid crop for file %s" % (fn))
        exit()

def LoadClinical(df): ## Not finished
    all_cols  = ['arm','age','gender','race','ethnicity','zubrod',
                 'histology','nonsquam_squam','ajcc_stage_grp','rt_technique',
                 'egfr_hscore_200','smoke_hx','rx_terminated_ae','rt_dose',
                 'volume_ptv','dmax_ptv','v100_ptv',
                 'v95_ptv','v5_lung','v20_lung','dmean_lung','v5_heart',
                 'v30_heart','v20_esophagus','v60_esophagus','Dmin_PTV_CTV_MARGIN',
                 'Dmax_PTV_CTV_MARGIN','Dmean_PTV_CTV_MARGIN','rt_compliance_physician',
                 'rt_compliance_ptv90','received_conc_chemo','received_conc_cetuximab',
                 'received_cons_chemo','received_cons_cetuximab',
    ]

    numerical_cols = ['age','volume_ptv','dmax_ptv','v100_ptv',
                      'v95_ptv','v5_lung','v20_lung','dmean_lung','v5_heart',
                      'v30_heart','v20_esophagus','v60_esophagus','Dmin_PTV_CTV_MARGIN',
                      'Dmax_PTV_CTV_MARGIN','Dmean_PTV_CTV_MARGIN']

    category_cols = list(set(all_cols).difference(set(numerical_cols)))
    for categorical in category_cols:
        df.loc[:, df.columns.str.startswith(categorical)]
        temp_col = pd.get_dummies(outcome[categorical], prefix=categorical)

    for numerical in numerical_cols:
        X_test = X_test.join([outcome[numerical]])
    return y_init

def custom_collate(original_batch):

    filtered_data = []
    filtered_target = []

    for patient in original_batch:
        none_found = False
        if "Anatomy" in patient[0].keys():
            if patient[0]["Anatomy"] is None:
                none_found = True
        if "Dose" in patient[0].keys():
            if patient[0]["Dose"] is None:
                none_found = True

        if not none_found:
            filtered_data.append(patient[0])
            filtered_target.append(patient[1])

    return filtered_data, filtered_target

