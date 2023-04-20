import sys
sys.path.insert(0, '/home/dgs1/Software/OutcomePrediction/')
import csv
import glob
import os
from pathlib import Path
import nibabel as nib
import numpy as np
import pandas
from scipy import ndimage
import toml
from DataGenerator.DataGenerator import QuerySubjectList, SynchronizeData, QuerySubjectInfo
from Utils.DicomTools import *
import xnat

session = xnat.connect('http://128.16.11.124:8080/xnat', user='admin', password='mortavar1977')
config = toml.load(sys.argv[1])
session = xnat.connect(config['SERVER']['Address'], user=config['SERVER']['User'],
                       password=config['SERVER']['Password'])
SubjectList = QuerySubjectList(config, session)
print(SubjectList)
SynchronizeData(config, SubjectList)

if 'Records' in config.keys():
    SubjectList, clinical_cols = LoadClinicalData(config, SubjectList)

else:
    clinical_cols = None

for key in config['MODALITY'].keys():
    SubjectList[key + '_Path'] = ""
QuerySubjectInfo(config, SubjectList, session)

path = '/home/dgs1/data/InnerEye/Seg'
header = ['idx', 'subject', 'survival_months', 'ptv', 'heart_atrium_left', 'heart_atrium_right', 'heart_myocardium',
          'heart_ventricle_left', 'heart_ventricle_right', 'pulmonary_artery', 'position', 'position_label']
idx = 0
mask_list = ['ptv.nii.gz', 'heart_atrium_left.nii.gz', 'heart_atrium_right.nii.gz', 'heart_myocardium.nii.gz',
             'heart_ventricle_left.nii.gz', 'heart_ventricle_right.nii.gz', 'pulmonary_artery.nii.gz']

clinical_cols = ['xnat_subjectdata_field_map_survival_months', 'xnat_subjectdata_field_map_grade3_toxicity']

with open('position_dmean.csv', 'w', encoding='UTF8', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(header)
    for patient_folder in glob.glob(path + "/0617-*"):
        id = patient_folder.split('/')[-1]
        organs = os.listdir(patient_folder)
        data = [idx, id]

        col_data = SubjectList.loc[
            SubjectList.subjectid == subjectid, 'xnat_subjectdata_field_map_' + config['DATA']['target']]
        data.append(col_data)

        if 'ptv.nii.gz' in organs:
            dose_info = nib.load(Path(patient_folder, 'dose.nii.gz'))
            dose_img = dose_info.get_fdata()

            for mask in mask_list:
                mask_info = nib.load(Path(patient_folder, mask))
                mask_img = mask_info.get_fdata()
                dmean = dose_img[mask_img > 0].mean()
                data.append(dmean)

            ptv_info = nib.load(Path(patient_folder, 'ptv.nii.gz'))
            ptv_img = ptv_info.get_fdata()
            gtv_img = ndimage.binary_erosion(ptv_img, structure=np.ones((10, 10, 5))).astype(ptv_img.dtype)
            g_slice = gtv_img.sum(axis=(0, 1))
            g_index = [i for i, v in enumerate(g_slice) if v > 0]

            PTV_COM = list(map(int, ndimage.center_of_mass(ptv_img)))
            # print('PTV_COM: ', PTV_COM)
            T7_info = nib.load(Path(patient_folder, 'vertebrae_T7.nii.gz'))
            T7_img = T7_info.get_fdata()
            t_slice = T7_img.sum(axis=(0, 1))
            t_index = [i for i, v in enumerate(t_slice) if v > 0]
            T7_COM = list(map(int, ndimage.center_of_mass(T7_img)))
            pos = PTV_COM[2] - T7_COM[2]
            data.append(pos)
            intersect = set(g_index).intersection(set(t_index))
            # print('T7_COM:', T7_COM)
            if len(intersect) == 0 & pos > 0:
                data.append(0)
            else:
                data.append(1)
            writer.writerow(data)
            idx = idx + 1
print(idx)
