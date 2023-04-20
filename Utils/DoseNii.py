import sys
import toml
import glob
import nibabel as nib
import numpy as np
from Utils.DicomTools import *
from pathlib import Path
from DataGenerator.DataGenerator import QuerySubjectList, SynchronizeData, QuerySubjectInfo
from Utils.DicomTools import *
import os
from rt_utils import RTStructBuilder
import xnat
from Utils.FixRTSS import *
session = xnat.connect('http://128.16.11.124:8080/xnat', user='admin', password='mortavar1977')
import csv
from monai.transforms import Spacing, LoadImage, EnsureChannelFirst, ResampleToMatch
from monai.data import MetaTensor
import itk
config = toml.load(sys.argv[1])

session = xnat.connect(config['SERVER']['Address'], user=config['SERVER']['User'],password=config['SERVER']['Password'])
SubjectList = QuerySubjectList(config, session)
print(SubjectList)
SynchronizeData(config, SubjectList)

for key in config['MODALITY'].keys():
    SubjectList[key+'_Path'] = ""
QuerySubjectInfo(config, SubjectList, session)

path = config['DATA']['NiiFolder']
sPatient = SubjectList

for i in range(301,len(sPatient), 1):
    print(i)
    subjectid = sPatient.loc[i, 'subjectid']
    subject_label = sPatient.loc[i,'subject_label']

    CTPath = SubjectList.loc[i, 'CT_Path']
    CTArray, meta = LoadImage()(CTPath)
    DosePath = SubjectList.loc[i,'Dose_Path']
    DoseArray, dmeta = LoadImage()(DosePath)
    DoseArray = DoseArray * np.double(dmeta['3004|000e'])

    ct = EnsureChannelFirst()(CTArray)
    ct = Spacing(pixdim=(1, 1, 3))(ct)
    DoseArray = EnsureChannelFirst()(DoseArray)
    dose = ResampleToMatch()(DoseArray, ct)

    names_generator = itk.GDCMSeriesFileNames.New()
    names_generator.SetUseSeriesDetails(True)
    names_generator.AddSeriesRestriction("0008|0021")  # Series Date
    names_generator.SetDirectory(list(CTPath)[0])
    series_uid = names_generator.GetSeriesUIDs()

    if len(series_uid) > 1:
        print(series_uid)
    else:
        ct_array = ct.array.squeeze()
        ni_ct = nib.Nifti1Image(np.double(ct_array), affine=ct.affine)
        dose_array = dose.array.squeeze()
        ni_dose = nib.Nifti1Image(np.double(dose_array), affine=dose.affine)
        spath = Path(path, subject_label)
        print(subject_label)

        if not os.path.isdir(spath):
           os.mkdir(spath)
        nib.save(ni_ct, Path(spath, 'ct.nii.gz'))
        nib.save(ni_dose, Path(spath, 'dose.nii.gz'))



