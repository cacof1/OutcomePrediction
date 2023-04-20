import sys
# sys.path.insert(0, '/home/dgs1/Software/OutcomePrediction/')
import toml
import glob
import nibabel as nib
import numpy as np
from Utils.DicomTools import *
from pathlib import Path
from DataGenerator.DataGenerator import QuerySubjectList, SynchronizeData
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
from scipy import ndimage

session = xnat.connect(config['SERVER']['Address'], user=config['SERVER']['User'],
                       password=config['SERVER']['Password'])
SubjectList = QuerySubjectList(config, session)
print(SubjectList)
SynchronizeData(config, SubjectList)

for key in config['MODALITY'].keys():
    SubjectList[key + '_Path'] = ""

QuerySubjectInfo(config, SubjectList, session)

roi_name = 'gtv'
path = config['DATA']['NiiFolder']
sPatient = SubjectList

se = ndimage.generate_binary_structure(3, 3)

for i in range(0, len(SubjectList), 1):
    subjectid = sPatient.loc[i, 'subjectid']
    subject_label = sPatient.loc[i, 'subject_label']
    # CTPath = SubjectList[SubjectList.subjectid == subjectid]['CT_Path']
    CTPath = SubjectList.loc[i, 'CT_Path']
    CTArray, meta = LoadImage()(CTPath)
    DosePath = SubjectList.loc[i, 'Dose_Path']
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
    print('No.{}:'.format(i) + str(subject_label))

    if len(series_uid) > 1:
        print(series_uid)
    else:
        ct_array = ct.array.squeeze()

        ni_ct = nib.Nifti1Image(np.double(ct_array), affine=ct.affine)
        dose_array = dose.array.squeeze()
        ni_dose = nib.Nifti1Image(np.double(dose_array), affine=dose.affine)
        spath = Path(path, subject_label)

        if not os.path.isdir(spath):
            os.mkdir(spath)
        nib.save(ni_ct, Path(spath, 'ct.nii.gz'))
        nib.save(ni_dose, Path(spath, 'dose.nii.gz'))

        # First define the ROI based on target
        RSPath = SubjectList[SubjectList.subjectid == subjectid]['Structs_Path']
        # FixRTSS(RSPath[0], list(CTPath)[0])
        RS = RTStructBuilder.create_from(dicom_series_path=list(CTPath)[0], rt_struct_path=list(RSPath)[0])
        # %%
        roi_names = RS.get_roi_names()
        strList = [x.lower() for x in roi_names]
        r1 = [r for r in strList if roi_name in r]

        mask = np.zeros_like(CTArray)

        for j in range(len(r1)):
            roi = r1[j]
            index = strList.index(roi.lower())
            mask_img = RS.get_roi_mask_by_name(roi_names[index])
            mask_img = np.rot90(mask_img)
            mask_img = np.flip(mask_img, 0)
            mask = mask + mask_img

        mask = ndimage.binary_dilation(mask, structure=se, iterations=3)  ## if needs expansion
        mask = MetaTensor(mask.copy(), meta=meta)
        mask = EnsureChannelFirst()(mask)
        mask = Spacing(pixdim=(1, 1, 3))(mask)
        mask_array = mask.array.squeeze()
        ni_mask = nib.Nifti1Image(mask_array.astype('int'), affine=mask.affine, dtype='uint8')
        nib.save(ni_mask, Path(spath, 'AI_target.nii.gz'))
