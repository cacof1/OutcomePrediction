import sys
import os
import csv
from pathlib import Path
from monai.transforms import LoadImage
import nibabel as nib
import numpy as np

base_path = '/home/dgs1/data/InnerEye/nii_root_folder/RTOG/'

subjects = os.listdir(base_path)
# roi = ['heart', 'oesophagus', 'spinal canal', 'prox bronch tree', 'proximal trachea', 'cwall & ribs', 'lt lung','rt lung']
# roi_name = ['heart', 'oesophagus', 'spinal_canal','prox_bronch_tree', 'proximal_trachea','cwall_ribs','lt_lung','rt_lung']
# roi = ['lung_cntr','lung_ipsi']
# roi_name = ['lung_cntr','lung_ipsi']
roi = ['esophagus', 'lung', 'heart', 'ptv']
header = ['subject', 'filePath', 'channel']
count = 0
rm_sub = []


# for i in range(len(subjects)):
#     sub = subjects[i]
#     subpath = Path(base_path, sub)
#     basename = os.listdir(subpath)
#     if os.path.exists(Path(subpath, 'lung_left.nii.gz')) and os.path.exists(Path(subpath, 'lung_right.nii.gz')):
#         LungL, meta = LoadImage()(Path(subpath, 'lung_left.nii.gz'))
#         LungR, meta = LoadImage()(Path(subpath, 'lung_right.nii.gz'))
#         Lung = LungL + LungR
#         ni_mask = nib.Nifti1Image(Lung.astype('int'), affine=meta['affine'],  dtype='uint8')
#         nib.save(ni_mask, Path(subpath, 'lung.nii.gz'))
#     else:
#         rm_sub.append(sub)
# print(rm_sub)

def BitSet(n, p, b):
    p = p.astype(int)
    n = n.astype(int)
    b = b.astype(int)
    mask = 1 << p
    bm = b << p
    return (n & ~mask) | bm


for i in range(len(subjects)):
    sub = subjects[i]
    subpath = Path(base_path, sub)
    basename = os.listdir(subpath)
    ct_name = ['ct.nii.gz', 'dose.nii.gz']
    oars = set(basename).difference(set(ct_name))
    oar_list = []

    for oar in oars:
        oar_name = oar.split(".")
        oar_list.append(oar_name[0].lower())

    if set(roi).issubset(set(oar_list)):
        count = count + 1
        for idx, r in enumerate(roi):
            try:
                data, meta = LoadImage()(Path(subpath, r+'.nii.gz'))
                if idx == 0:
                    masks_img = np.zeros_like(data)
            except:
                raise ValueError(sub + " has no ROI of name " + roi + " found in RTStruct")
            masks_img = BitSet(masks_img, idx * np.ones_like(masks_img), data)

        ni_img = nib.Nifti1Image(np.double(masks_img), affine=meta['affine'])
        nib.save(ni_img, Path(subpath, 'masks.nii.gz'))
    else:
        rm_sub.append(sub)


print('Counts: ', count)
print(rm_sub)