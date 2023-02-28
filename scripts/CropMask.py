import nibabel as nib
import sys, glob
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import ndimage
from pathlib import Path

path = '/home/dgs1/data/InnerEye/nii_root_folder'
data = pd.read_csv('/home/dgs1/data/InnerEye/CSV/dataset_lung_spinal.csv')
patient_id = data['subject'].unique()

crop_oar = set(data['channel'].unique()).difference(set(['lung_left','lung_right','ct']))

for i in range(len(patient_id)):
    pat = data[data.subject == i]

    mask_nii = pat[pat.channel == 'lung_left']['filePath']
    info = nib.load(Path(path, list(mask_nii)[0]))
    temp = info.get_fdata()

    mask_nii = pat[pat.channel == 'lung_right']['filePath']
    info = nib.load(Path(path, list(mask_nii)[0]))
    temp = temp + info.get_fdata()

    Zm = np.sum(temp, axis=(0,1))
    index = np.argwhere(Zm == 0).squeeze()
    for oar in crop_oar:
        mask_nii = pat[pat.channel == oar]['filePath']
        info = nib.load(Path(path, list(mask_nii)[0]))
        img = info.get_fdata()
        img[:, :, index] = np.nan
