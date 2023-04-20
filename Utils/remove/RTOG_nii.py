import nibabel as nib
import sys, glob
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import ndimage
from pathlib import Path
import os

path = '/home/dgs1/data/InnerEye/nii_root_folder/RTOG/'
data = pd.read_csv('/home/dgs1/data/InnerEye/nii_root_folder/RTOG/dataset.csv')
patient_id = data['subject'].unique()
cdata = pd.read_excel('/home/dgs1/Downloads/LUNG_EXCEL.xlsx')
roi_set = set(['LEFT','RIGHT'])
for i in range(142, len(patient_id), 1):
    pat = data[data.subject == i]
    mask_nii = pat[pat.channel == 'lung_ipsi']['filePath']
    id = list(mask_nii)[0].split('/')[0]
    ind = list(cdata[cdata['Patient ID'] == id]['LUNG_IPSI'])[0]
    value = list(cdata[cdata['Patient ID'] == id]['LUNG_CNTR'])
    ipsi_name = 'lung_' + list(roi_set.difference(set(value)))[0].lower() + '.nii.gz'

    old = Path(path, list(mask_nii)[0])
    new = Path(path, id, ipsi_name)
    if not os.path.exists(new):
        os.rename(old, new)

    mask_nii = pat[pat.channel == 'lung_cntr']['filePath']
    cntr_name =  'lung_' + value[0].lower() + '.nii.gz'

    old = Path(path, list(mask_nii)[0])
    new = Path(path, id, cntr_name)
    if not os.path.exists(new):
        os.rename(old, new)




