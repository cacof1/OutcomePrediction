import torch
import torchvision
from torch import nn
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from pytorch_lightning.strategies import DDPStrategy
from pytorch_lightning import LightningDataModule, LightningModule, Trainer, seed_everything
import sys, os
#import torchio as tio
import monai
torch.cuda.empty_cache()
## Module - Dataloaders
from DataGenerator.DataGenerator import *
from Models.Classifier import Classifier
from Models.Linear import Linear
from Models.MixModel import MixModel
from monai.transforms import EnsureChannelFirstd, ScaleIntensityd, ResampleToMatchd
## Main
from sklearn.preprocessing import StandardScaler, OneHotEncoder
import toml
from Utils.GenerateSmoothLabel import get_smoothed_label_distribution, get_module
from Utils.PredictionReports import PredictionReports
from pathlib import Path
#import torchio as tio
from torchmetrics import ConfusionMatrix
import torchmetrics
import nibabel as nib

config = toml.load(sys.argv[1])
## First Connect to XNAT
session = xnat.connect(config['SERVER']['Address'], user=config['SERVER']['User'],password=config['SERVER']['Password'])


SubjectList = QuerySubjectList(config, session)
SynchronizeData(config, SubjectList)
## GeneratePath
for key in config['MODALITY'].keys():
    SubjectList[key+'_Path'] = ""
QuerySubjectInfo(config, SubjectList, session)

path = '/home/dgs1/data/Nifty_Data/RTOG0617/'

for i in range(len(SubjectList)):
    print(i)
    sub = SubjectList.iloc[i]
    print(sub)
    patient_label = sub['subject_label']
    RSPath = sub['Structs_Path']
    for idx, roi in enumerate(config['DATA']['Structs']):
        try:
            data, meta = LoadImage()(Path(RSPath, roi + '.nii.gz'))
            print(data.shape)
            if idx == 0:
                masks_img = np.zeros_like(data)
        except:
            raise ValueError(patient_label + " has no ROI of name " + roi + " found in RTStruct")
        masks_img = BitSet(masks_img, idx * np.ones_like(masks_img), data)

    ni_img = nib.Nifti1Image(np.double(masks_img), affine=meta['affine'])
    spath = Path(path, patient_label, 'struct_TS')
    nib.save(ni_img, Path(spath, 'maskswc.nii.gz'))



def BitSet(n, p, b):
    p = p.astype(int)
    n = n.astype(int)
    b = b.astype(int)
    mask = 1 << p
    bm = b << p
    return (n & ~mask) | bm
