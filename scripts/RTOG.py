import sys
#sys.path.insert(0, '/home/dgs1/Software/OutcomePrediction/')
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
from monai.transforms import Spacing, LoadImage, EnsureChannelFirst
from monai.data import MetaTensor
import itk
import pandas as pd
from pydicom import dcmread
config = toml.load(sys.argv[1])

session = xnat.connect(config['SERVER']['Address'], user=config['SERVER']['User'],password=config['SERVER']['Password'])
SubjectList = QuerySubjectList(config, session)
print(SubjectList)
SynchronizeData(config, SubjectList)
data = pd.read_excel('/home/dgs1/Downloads/LUNG_EXCEL.xlsx')

for key in config['MODALITY'].keys():
    SubjectList[key+'_Path'] = ""
QuerySubjectInfo(config, SubjectList, session)
sPatient = SubjectList

roi_set = set(['LEFT','RIGHT'])

for i in range(0, len(SubjectList), 1):
    subjectid = sPatient.loc[i, 'subjectid']
    subject_label = sPatient.loc[i,'subject_label']
    print(subject_label)
    RSPath = glob.glob(list(SubjectList[SubjectList.subjectid == subjectid]['Structs_Path'])[0] + '/*dcm')
    info = dcmread(RSPath[0])
    ind = list(data[data['Patient ID'] == subject_label]['LUNG_CNTR'])[0]
    value = list(data[data['Patient ID'] == subject_label]['LUNG_IPSI'])
    if not (ind == 'NO TUMOUR?'):
        for r in range(len(info.StructureSetROISequence)):
            if info.StructureSetROISequence[r].ROIName == 'LUNG_IPSI':
                info.StructureSetROISequence[r].ROIName = 'LUNG_' + value[0]
            if info.StructureSetROISequence[r].ROIName == 'LUNG_CNTR':
                info.StructureSetROISequence[r].ROIName = 'LUNG_' + list(roi_set.difference(set(value)))[0]
        info.save_as(RSPath[0])



