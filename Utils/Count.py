import sys
import toml
import glob
import nibabel as nib
import itertools
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
from monai.transforms import Spacing, LoadImage, EnsureChannelFirst
from monai.data import MetaTensor
import itk
from collections import Counter
config = toml.load(sys.argv[1])

session = xnat.connect(config['SERVER']['Address'], user=config['SERVER']['User'],password=config['SERVER']['Password'])
SubjectList = QuerySubjectList(config, session)
SubjectList.dropna(subset=['xnat_subjectdata_field_map_survival_months'], inplace=True)
print(SubjectList)
SynchronizeData(config, SubjectList)

for key in config['MODALITY'].keys():
    SubjectList[key+'_Path'] = ""
QuerySubjectInfo(config, SubjectList, session)
roi_series = []

path = config['DATA']['NiiFolder']
sPatient = SubjectList
r1 = [x.lower() for x in roi_series]

for i in range(0, len(SubjectList), 1):
    print(i)
    subjectid = sPatient.loc[i, 'subjectid']
    subject_label = sPatient.loc[i,'subject_label']
    CTPath = SubjectList[SubjectList.subjectid == subjectid]['CT_Path']
    CTArray, meta = LoadImage()(CTPath)
    ct = EnsureChannelFirst()(CTArray)
    ct = Spacing(pixdim=(1, 1, 3))(ct)

    names_generator = itk.GDCMSeriesFileNames.New()
    names_generator.SetUseSeriesDetails(True)
    names_generator.AddSeriesRestriction("0008|0021")  # Series Date
    names_generator.SetDirectory(list(CTPath)[0])
    series_uid = names_generator.GetSeriesUIDs()
    if len(series_uid) > 1:
        print(subject_label)
    else:
        ct_array = ct.array.squeeze()
        # First define the ROI based on target
        RSPath = SubjectList[SubjectList.subjectid == subjectid]['Structs_Path']
        # FixRTSS(list(RSPath)[0], list(CTPath)[0])
        try:
           RS = RTStructBuilder.create_from(dicom_series_path=list(CTPath)[0], rt_struct_path=list(RSPath)[0])
           roi_names = RS.get_roi_names()
           strList = [x.lower() for x in roi_names]
           roi_series.append(strList)
        except:
           print('RS error: pid', subject_label)

letter_counts = Counter(itertools.chain.from_iterable(roi_series))
print(letter_counts)
