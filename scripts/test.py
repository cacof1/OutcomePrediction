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
config = toml.load('/home/dgs1/Software/OutcomePrediction/SettingsRTOG_Inner2.ini')

session = xnat.connect(config['SERVER']['Address'], user=config['SERVER']['User'],password=config['SERVER']['Password'])
SubjectList = QuerySubjectList(config, session)
print(SubjectList)
# SynchronizeData(config, SubjectList)

for key in config['MODALITY'].keys():
    SubjectList[key+'_Path'] = ""
QuerySubjectInfo(config, SubjectList, session)
sPatient = SubjectList

roi_set = set(['LEFT','RIGHT'])
plist = []

for i in range(0, len(SubjectList), 1):
    subjectid = sPatient.loc[i, 'subjectid']
    subject_label = sPatient.loc[i,'subject_label']
    # print(subject_label)
    RSPath = list(SubjectList[SubjectList.subjectid == subjectid]['Structs_Path'])[0]
    pl = RSPath.split('/')
    plc = '/'.join(pl[0:8])
    ll = glob.glob(plc+'/*DOSE')
    if len(ll) > 1:
        plist.append(subject_label)
        print(subject_label)
        print(ll)


print('test')

