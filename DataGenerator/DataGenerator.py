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
from DataGenerator.DataProcessing import LoadClincalData
from sklearn.preprocessing import StandardScaler
class DataGenerator(torch.utils.data.Dataset):
    def __init__(self, mastersheet, label, keys, inference=False, clinical_norm = None, transform=None, target_transform = None):
        super().__init__()
        self.keys             = list(keys)
        self.transform        = transform
        self.target_transform = target_transform
        self.label            = label
        self.inference        = inference
        self.mastersheet      = mastersheet
        self.clinical_norm = clinical_norm

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
                datadict["Dose"]  = self.transform(datadict["Dose"])

                
        if "Anatomy" in self.keys:
            anatomy  = LoadImg(self.mastersheet["CTPath"].iloc[id])
            datadict["Anatomy"] = np.expand_dims(CropImg(anatomy, maxDoseCoords, roiSize), 0)
            if self.transform:            
                datadict["Anatomy"]  = torch.from_numpy(self.transform(datadict["Anatomy"]))

            print(datadict["Anatomy"].size, type(datadict["Anatomy"]))
        if "Clinical" in self.keys:
            clinical_data = LoadClincalData(self.mastersheet)
            #data = clinical_data.iloc[id].to_numpy()
            data = self.clinical_norm.transform([clinical_data.iloc[id].to_numpy()])
            datadict["Clinical"] = data.flatten()

        if(self.inference): return datadict
        else: return datadict, label.astype(np.float32)

### DataLoader
class DataModule(LightningDataModule):
    def __init__(self, mastersheet, label, keys, train_transform = None, val_transform = None, batch_size = 64, Norm = None, **kwargs):
        super().__init__()
        self.batch_size      = batch_size
        self.Norm = Norm
        train, val_test       = train_test_split(mastersheet, train_size=0.7)
        val, test             = train_test_split(val_test,test_size =0.66)

        self.train_data  = DataGenerator(train, label, keys, clinical_norm = self.Norm, transform = train_transform, **kwargs)
        self.val_data        = DataGenerator(val,   label, keys, clinical_norm = self.Norm, transform = val_transform, **kwargs)
        self.test_data       = DataGenerator(test,  label, keys, clinical_norm = self.Norm, transform = val_transform, **kwargs)
        print('test')

    def train_dataloader(self): return DataLoader(self.train_data, batch_size=self.batch_size,num_workers=0)
    def val_dataloader(self):   return DataLoader(self.val_data,   batch_size=self.batch_size,num_workers=0)
    def test_dataloader(self):  return DataLoader(self.test_data,  batch_size=self.batch_size)

def PatientQuery(mastersheet, **kwargs):
    for key,item in kwargs.items(): mastersheet = mastersheet[mastersheet[key]==item]
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
        df.loc[df.index.str.startswith(categorical)]
        temp_col = pd.get_dummies(df[categorical], prefix=categorical)

    for numerical in numerical_cols:
        X_test = X_test.join([outcome[numerical]])
    return y_init
