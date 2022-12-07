import toml
import sys
import glob
import nibabel as nib
import numpy as np
from Utils.DicomTools import *
from pathlib import Path
from DataGenerator.DataGenerator import QuerySubjectList, SynchronizeData, QuerySubjectInfo
import os
from rt_utils import RTStructBuilder
import xnat
from monai.transforms import LoadImage, ResampleToMatchd, EnsureChannelFirstd
from monai.data.image_writer import ITKWriter

session = xnat.connect('http://128.16.11.124:8080/xnat', user='admin', password='mortavar1977')
config = toml.load(sys.argv[1])
from pydicom import dcmread

session = xnat.connect(config['SERVER']['Address'], user=config['SERVER']['User'],
                       password=config['SERVER']['Password'])
SubjectList = QuerySubjectList(config, session)
print(SubjectList)
SynchronizeData(config, SubjectList)
QuerySubjectInfo(config, SubjectList, session)
for i in range(0,len(SubjectList),1):
    print(i)
    CTPath = SubjectList['CT_Path'][i].split('/')
    scanPath = '/'.join(CTPath[0:8])
    Dosefile = glob.glob(scanPath + '/*-RTDOSE')
    data = {}
    meta = {}
    if len(Dosefile) > 1:
        for j in range(len(Dosefile)):
            file = glob.glob(Dosefile[j] + '/**/*.dcm', recursive=True)
            if j == 0:
                info = dcmread(file[0])
            data['Dose'+ str(j)], meta['Dose'+str(j)] = LoadImage()(file[0])

        data = EnsureChannelFirstd(data.keys())(data)
        data = ResampleToMatchd(list(set(data.keys()).difference(set(['Dose0']))), key_dst='Dose0')(data)

        total_array = np.zeros_like(data['Dose0'])
        for key in data.keys():
            total_array = total_array + data[key]*np.float64(meta[key]['3004|000e'])/np.float64(meta['Dose0']['3004|000e'])

        total_array = total_array.squeeze()
        total_array = np.transpose(total_array, (2, 1, 0))
        total_array = np.uint32(total_array)

        info.PixelData = total_array.tobytes()
        if not os.path.isdir(Path('/home/dgs1/data/dose/', CTPath[5])):
            os.mkdir(Path('/home/dgs1/data/dose/', CTPath[5]))
        dpath = Path('/home/dgs1/data/dose/', CTPath[5], '1-1.dcm')
        info.save_as(dpath)
