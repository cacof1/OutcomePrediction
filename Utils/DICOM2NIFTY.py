import torch
import sys, os
torch.cuda.empty_cache()
from DataGenerator.DataGenerator import *
import toml
from pathlib import Path
import nibabel as nib
from monai.transforms import Spacing, LoadImage, EnsureChannelFirst
from dcmrtstruct2nii import dcmrtstruct2nii, list_rt_structs
config = toml.load(sys.argv[1])
## First Connect to XNAT
session = xnat.connect(config['SERVER']['Address'], user=config['SERVER']['User'],password=config['SERVER']['Password'])


SubjectList = QuerySubjectList(config, session)
SynchronizeData(config, SubjectList)
## GeneratePath
for key in config['MODALITY'].keys():
    SubjectList[key+'_Path'] = ""
QuerySubjectInfo(config, SubjectList, session)
print(SubjectList)

save_path = '/home/dgs1/data/Nifty_Data/UCLH/'

for i in range(253, len(SubjectList)):
    sub = SubjectList.iloc[i]
    patient_label = sub['subject_label']
    out_folder = Path(save_path, patient_label)
    RSPath = sub['Structs_Path']
    CTPath = sub['CT_Path']
    DosePath = sub['Dose_Path']
    # command = f'plastimatch convert --input {CTPath} --output-img {out_folder}/CT.nii.gz --spacing "1 1 3"'
    # os.system(command)
    # DoseName = 'Dose'
    # command = f'plastimatch convert --input-dose-img {DosePath} --referenced-ct {CTPath} --resize-dose --output-dose-img {out_folder}/{DoseName}.nii.gz --fixed {out_folder}/CT.nii.gz'
    # os.system(command)

    # command = f'plastimatch convert --input {RSPath} --fixed {out_folder}/CT.nii.gz --output-prefix {out_folder}/struct_DICOM --referenced-ct {CTPath} --prefix-format nii.gz'
    # os.system(command)
    #
    #
    Struct_folder = Path(out_folder, 'struct_DICOM')
    print('before ', i, patient_label)
    if not os.path.exists(Struct_folder):
        os.system(f'mkdir  {Struct_folder}')
        # dcmrtstruct2nii(RSPath, CTPath, Path(out_folder, 'struct_DICOM'))
        CTArray, meta = LoadImage()(CTPath)
        RS = RTStructBuilder.create_from(dicom_series_path=CTPath, rt_struct_path=RSPath)
        # %%
        roi_names = RS.get_roi_names()

        for j in range(len(roi_names)):
            mask_img = RS.get_roi_mask_by_name(roi_names[j])
            mask_img = np.rot90(mask_img)
            mask_img = np.flip(mask_img, 0)
            mask = MetaTensor(mask_img.copy(), meta=meta)
            mask = EnsureChannelFirst()(mask)
            mask = Spacing(pixdim=(1, 1, 3))(mask)
            mask_array = mask.array.squeeze()
            ni_mask = nib.Nifti1Image(mask_array.astype('int'), affine=mask.affine, dtype='uint8')
            nib.save(ni_mask, Path(out_folder, 'struct_DICOM', roi_names[j]+'.nii.gz'))

        print(i)



