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

class DataGenerator(torch.utils.data.Dataset):
    def __init__(self, mastersheet, label, inference=False, transform=None, target_transform = None):
        super().__init__()
        self.transform        = transform
        self.target_transform = target_transform
        self.label            = label
        self.inference        = inference
        self.mastersheet      = mastersheet
    def __len__(self):
        return int(self.mastersheet.shape[0]) 

    def __getitem__(self, id):
        
        # Load image
        label    = self.mastersheet[self.label].iloc[id]
        anatomy  = np.expand_dims(LoadImg(self.mastersheet["CTPath"].iloc[id], [100,100,100], [40,40,10]),0)
        dose     = np.expand_dims(LoadImg(self.mastersheet["DosePath"].iloc[id],[100,100,100],[40,40,10]),0)
        #clinical = LoadClinical(self.mastersheet.iloc[id])
        
        ## Transform - Data Augmentation
        if self.transform:
            anatomy  = self.transform(anatomy)
            dose     = self.transform(dose)
            
        if(self.inference): return anatomy, dose#, clinical
        else: return anatomy, dose, label#, clinical ,label
        
### DataLoader
class DataModule(LightningDataModule):
    def __init__(self, mastersheet, label, train_transform = None, val_transform = None, batch_size = 8, **kwargs):
        super().__init__()
        self.batch_size      = batch_size
        ids_split            = np.round(np.array([0.7, 0.8, 1.0])*len(mastersheet)).astype(np.int32)
        self.train_data      = DataGenerator(mastersheet[:ids_split[0]],              label, transform = train_transform, **kwargs)
        self.val_data        = DataGenerator(mastersheet[ids_split[0]:ids_split[1]],  label, transform = val_transform, **kwargs)
        self.test_data       = DataGenerator(mastersheet[ids_split[1]:ids_split[-1]], label, transform = val_transform, **kwargs)

    def train_dataloader(self): return DataLoader(self.train_data, batch_size=self.batch_size,num_workers=10)
    def val_dataloader(self):   return DataLoader(self.val_data,   batch_size=self.batch_size,num_workers=10)
    def test_dataloader(self):  return DataLoader(self.test_data,  batch_size=self.batch_size)

def PatientQuery(mastersheet, **kwargs):  
    for key,item in kwargs.items(): mastersheet = mastersheet[mastersheet[key]==item]
    return mastersheet

def LoadImg(path, cm, delta): ## Select a region of size 2*delta^3 around the center of mass of the tumour
    img = sitk.ReadImage(path)
    img = sitk.GetArrayFromImage(img).astype(np.float32)
    img = img[cm[2]-delta[2]:cm[2]+delta[2], cm[1]-delta[1]:cm[1]+delta[1], cm[0]-delta[0]:cm[0]+delta[0]]
    return img


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
