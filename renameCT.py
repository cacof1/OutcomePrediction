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
config = toml.load(sys.argv[1])

session = xnat.connect(config['SERVER']['Address'], user=config['SERVER']['User'],password=config['SERVER']['Password'])
SubjectList = QuerySubjectList(config, session)
print(SubjectList)
SynchronizeData(config, SubjectList)

for key in config['MODALITY'].keys():
    SubjectList[key+'_Path'] = ""
QuerySubjectInfo(config, SubjectList, session)
#
# roi_series = ['Heart', 'Oesophagus', 'Spinal Canal', 'Prox bronch tree', 'Proximal trachea', 'Cwall & ribs', 'L Lung',
#        'R Lung', 'Brachial Plexus']
# roi_name = ['HEART', 'ESOPHAGUS', 'SPINAL_CORD', 'PROX_BRONCH_TREE', 'PROXIMAL_TRACHEA', 'CWALL_RIBS', 'LUNG_LEFT',
#        'LUNG_RIGHT', 'BRACHIAL_PLEXUS']
# roi_series = ['Heart', 'Esophagus', 'Lung_L', 'Lung_R','SpinalCord']
# roi_name = ['HEART', 'ESOPHAGUS', 'LUNG_LEFT','LUNG_RIGHT', 'SPINAL_CORD']

#roi_series = ['HEART', 'PTV', 'LUNG_LEFT', 'LUNG_RIGHT', 'SPINAL_CORD', 'ESOPHAGUS']
#roi_name = ['HEART', 'PTV', 'LUNG_LEFT', 'LUNG_RIGHT', 'SPINAL_CORD', 'ESOPHAGUS']
roi_series = ['PTV']
roi_name = ['PTV']

path = config['DATA']['NiiFolder']
sPatient = SubjectList
r1 = [x.lower() for x in roi_series]

for i in range(1, len(SubjectList), 1):
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
        print(i)
    else:
        ct_array = ct.array.squeeze()
        # First define the ROI based on target
        RSPath = SubjectList[SubjectList.subjectid == subjectid]['Structs_Path']
        try:
            RS = RTStructBuilder.create_from(dicom_series_path=list(CTPath)[0], rt_struct_path=list(RSPath)[0])
            #%%
            roi_names = RS.get_roi_names()
            strList = [x.lower() for x in roi_names]
            ni_img = nib.Nifti1Image(np.double(ct_array), affine=ct.affine)
            spath = Path(path, subject_label)

            if not os.path.isdir(spath):
                os.mkdir(spath)
            nib.save(ni_img, Path(spath, 'ct.nii.gz'))

            for i in range(len(r1)):
                roi = r1[i]
                if roi in strList:
                    #print(roi)
                    index = strList.index(roi.lower())
                    mask_img = RS.get_roi_mask_by_name(roi_names[index])
                    mask_img = np.rot90(mask_img)
                    mask_img = np.flip(mask_img, 0)
                    mask = MetaTensor(mask_img.copy(), meta=meta)
                    mask = EnsureChannelFirst()(mask)
                    mask = Spacing(pixdim=(1, 1, 3))(mask)
                    mask_array = mask.array.squeeze()
                    ni_mask = nib.Nifti1Image(mask_array.astype('int'), affine=mask.affine, dtype='uint8')
                    nib.save(ni_mask, Path(spath, roi_name[i].lower() + '.nii.gz'))
                else:
                    print(subject_label)
        except:
            #print('No.{}:'.format(i))
            continue



