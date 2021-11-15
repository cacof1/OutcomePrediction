import matplotlib.pyplot as plt
from pytorch_lightning import LightningDataModule, LightningModule, Trainer,seed_everything
import segmentation_models_pytorch as smp
from torch.utils.data import DataLoader
from torchvision import transforms
import pandas as pd
import os
from pathlib import Path
from sklearn.utils import shuffle
##Normalization
from Normalization.Macenko import MacenkoNormalization, TorchMacenkoNormalizer

from torch.utils.data import Dataset
from pytorch_lightning.loggers import TensorBoardLogger
import torchvision.models as models
import numpy as np
import torch
import openslide
import sys, glob
import torch.nn.functional as F
from sklearn.model_selection import train_test_split
from wsi_core.WholeSlideImage import WholeSlideImage
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping

class DataGenerator(torch.utils.data.Dataset):
    def __init__(self, data_file, label_file, inference=False, transform=None, target_transform = None):
        super().__init__()
        self.transform        = transform
        self.target_transform = target_transform
        self.data_file        = data_file
        self.label_file       = label_file
        self.inference        = inference
        
    def __len__(self):
        return int(self.data_file.shape[0]) 

    def __getitem__(self, id):
        
        # load image
        image = self.data_file[id,:]
        label = torch.tensor(self.label_file[id],dtype=torch.float)

        ## Transform - Data Augmentation
        if self.transform: image  = self.transform(image)

        if(self.inference): return image
        else: return image,label

### DataLoader
class DataModule(LightningDataModule):
    def __init__(self, data_file, label_file, train_transform = None, val_transform = None, batch_size = 8, **kwargs):
        super().__init__()
        self.batch_size      = batch_size

        data_file, label_file = shuffle(data_file, label_file, random_state=0)
        ids_split            = np.round(np.array([0.7, 0.8, 1.0])*len(data_file)).astype(np.int32)

        self.train_data      = DataGenerator(data_file[:ids_split[0]], label_file[:ids_split[0]],                           transform = train_transform, **kwargs)
        self.val_data        = DataGenerator(data_file[ids_split[0]:ids_split[1]], label_file[ids_split[0]:ids_split[1]],   transform = val_transform, **kwargs)
        self.test_data       = DataGenerator(data_file[ids_split[1]:ids_split[-1]], label_file[ids_split[1]:ids_split[-1]], transform = val_transform, **kwargs)

    def train_dataloader(self): return DataLoader(self.train_data, batch_size=self.batch_size,num_workers=10)
    def val_dataloader(self):   return DataLoader(self.val_data, batch_size=self.batch_size,num_workers=10)
    def test_dataloader(self):  return DataLoader(self.test_data, batch_size=self.batch_size)

