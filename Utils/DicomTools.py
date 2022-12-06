import os
import glob
#import cv2
import SimpleITK as sitk
import pydicom as dicom
from pydicom import dcmread
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.path import Path
from monai.data import ITKReader, PILReader
#import torchio as tio
from sklearn.preprocessing import KBinsDiscretizer

sitk.ProcessObject_SetGlobalWarningDisplay(False)
from rt_utils import RTStructBuilder
from scipy.ndimage import *


def get_bbox_from_mask(mask, img_shape):
    pos = np.where(mask)
    if pos[0].shape[0] == 0:
        bbox = np.zeros((0, 4))
    else:
        xmin = np.min(pos[3])
        xmax = np.max(pos[3])
        ymin = np.min(pos[2])
        ymax = np.max(pos[2])
        zmin = np.min(pos[1])
        zmax = np.max(pos[1])
        bbox = [zmin, zmax, ymin, ymax, xmin, xmax]
    return bbox


def ReadDicom(dicom_path, view_image=False):
    Reader = sitk.ImageSeriesReader()
    filenames = sorted(glob.glob(dicom_path + '/*.dcm'))
    Reader.SetFileNames(sorted(filenames))

    assert len(filenames) > 0
    Session = Reader.Execute()
    return Session


def ResamplingITK(Session, Reference, is_label=False, pad_value=0):
    resample = sitk.ResampleImageFilter()
    resample.SetOutputSpacing(Reference.GetSpacing())
    resample.SetSize(Reference.GetSize())
    resample.SetOutputDirection(Reference.GetDirection())
    resample.SetOutputOrigin(Reference.GetOrigin())
    resample.SetTransform(sitk.Transform())
    resample.SetDefaultPixelValue(Session.GetPixelIDValue())

    if is_label:
        resample.SetInterpolator(sitk.sitkNearestNeighbor)
    else:
        resample.SetInterpolator(sitk.sitkLinear)
    Resampled = resample.Execute(Session)
    return Resampled


def RStoContour(rs_path, targetROI='PTV'):
    rs_file = glob.glob(rs_path + '*.dcm')
    ds = dcmread(rs_file[0])
    for item in ds.StructureSetROISequence:
        if item.ROIName == targetROI:
            ROI = ds.ROIContourSequence[item.ROINumber - 1]
            contours = [contour for contour in ROI.ContourSequence]
            return contours


def poly_to_mask(polygon, img_shape):
    x, y = np.meshgrid(np.arange(img_shape[0]), np.arange(img_shape[1]))
    x, y = x.flatten(), y.flatten()
    points = np.vstack((x, y)).T
    path = Path(polygon)
    mask = path.contains_points(points)
    mask = mask.reshape(img_shape)

    return mask


def ViewROI(patient_id, img_array, mask_array, ROIbox, Inputbox):
    masked = np.ma.masked_where(mask_array == 0, mask_array)
    plt.subplot(1, 3, 1)
    plt.title('{} ROI mask'.format(patient_id))
    plt.imshow(img_array, cmap='gray')
    plt.imshow(masked, vmin=0, vmax=1, alpha=0.5)
    plt.subplot(1, 3, 2)
    plt.title('ROI Box')
    plt.imshow(ROIbox, cmap='gray')
    plt.subplot(1, 3, 3)
    plt.title('Input Box')
    plt.imshow(Inputbox, cmap='gray')
    plt.show()


def get_masked_img_voxel(ImageVoxel, mask_voxel):
    bbox = get_bbox_from_mask(mask_voxel, np.shape(ImageVoxel))
    assert len(mask_voxel) == ImageVoxel.shape[0]
    img_masked = ImageVoxel[:, bbox[0]:bbox[1], bbox[2]:bbox[3], bbox[4]:bbox[5]]
    return img_masked


def img_train_transform(img_dim):
    transform = tio.Compose([
        tio.transforms.ZNormalization(),
        tio.RandomAffine(),
        tio.RandomFlip(),
        tio.RandomNoise(),
        tio.RandomMotion(),
        tio.transforms.Resize(img_dim),
        tio.RescaleIntensity(out_min_max=(0, 1))
    ])
    return transform


def img_val_transform(img_dim):
    transform = tio.Compose([
        tio.transforms.ZNormalization(),
        tio.transforms.Resize(img_dim),
        tio.RescaleIntensity(out_min_max=(0, 1))
    ])
    return transform


def class_stratify(SubjectList, config):
    ptarget = SubjectList['xnat_subjectdata_field_map_' + config['DATA']['target']]
    kbins = KBinsDiscretizer(n_bins=5, encode='ordinal', strategy='uniform')
    ptarget = np.array(ptarget).reshape((len(ptarget), 1))
    data_trans = kbins.fit_transform(ptarget)
    return data_trans


def get_RS_masks(slabel, CTPath, mask_imgs, RSfile, mask_names):
     #RS = RTStructBuilder.create_from(dicom_series_path=CTPath, rt_struct_path=RSfile)
     #roi_names = RS.get_roi_names()
     #strList = [x.lower() for x in roi_names]
     #for idx, roi in enumerate(mask_names):
     #    if roi.lower() in strList:
     #        roi_s = roi_names[strList.index(roi.lower())]
     #        mask_img = RS.get_roi_mask_by_name(roi_s)
     #        # mask_img = distance_transform_edt(mask_img)
     #        mask_imgs = BitSet(mask_imgs, idx * np.ones_like(mask_imgs), mask_img)
     #    else:
     #        raise ValueError(slabel + " has no ROI of name " + roi + " found in RTStruct")
     #
     #return mask_imgs

    RS = RTStructBuilder.create_from(dicom_series_path=CTPath, rt_struct_path=RSfile)
    roi_names = RS.get_roi_names()
    strList = [x.lower() for x in roi_names]
    for idx, roi in enumerate(mask_names):
        if roi.lower() in strList:
            roi_s = roi_names[strList.index(roi.lower())]
            mask_img = RS.get_roi_mask_by_name(roi_s)
            mask_img = distance_transform_edt(mask_img)
            mask_imgs = mask_imgs + mask_img
        else:
            raise ValueError(slabel + " has no ROI of name " + roi + " found in RTStruct")
    
    return mask_imgs


def BitSet(n, p, b):
    p = p.astype(int)
    n = n.astype(int)
    mask = 1 << p
    bm = b << p
    return (n & ~mask) | bm
