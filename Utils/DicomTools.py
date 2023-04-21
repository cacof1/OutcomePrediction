import os
import glob
import SimpleITK as sitk
import pydicom as dicom
from pydicom import dcmread
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.preprocessing import KBinsDiscretizer
from monai.transforms import LoadImage
sitk.ProcessObject_SetGlobalWarningDisplay(False)
from rt_utils import RTStructBuilder
from scipy.ndimage import *
from concurrent.futures import ThreadPoolExecutor
import concurrent

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

def BitSet(n, p, b):
    p = p.astype(int)
    n = n.astype(int)
    b = b.astype(int)
    mask = 1 << p
    bm = b << p
    return (n & ~mask) | bm

def QuerySubjectInfo(config, SubjectList, session):
    if config['DATA']['Nifty']:
        for i in range(len(SubjectList)):
            subject_label = SubjectList.loc[i,'subject_label']
            for key in config['MODALITY'].keys():
                SubjectList.loc[i, key + '_Path'] = Path(config['DATA']['DataFolder'], subject_label)
    else:
        with ThreadPoolExecutor(max_workers=10) as executor:
            future_to_url = {executor.submit(get_subject_info, config, session, subjectid) for subjectid in
                             SubjectList['subjectid']}
            executor.shutdown(wait=True)
        for future in concurrent.futures.as_completed(future_to_url):
            subjectdata = future.result()
            subjectid = subjectdata["xnat:Subject"][0]["@ID"]
            for key in config['MODALITY'].keys():
                path = GeneratePath(subjectdata, Modality=key, config=config)
                if key == 'CT':
                    SubjectList.loc[SubjectList.subjectid == subjectid, key + '_Path'] = path
                else:
                    spath = glob.glob(path + '/*dcm')
                    SubjectList.loc[SubjectList.subjectid == subjectid, key + '_Path'] = spath[0]

def GeneratePath(subjectdata, Modality, config):
    subject = subjectdata['xnat:Subject'][0]
    subject_label = subject['@label']
    experiments = subject['xnat:experiments'][0]['xnat:experiment']

    ## Won't work with many experiments yet
    for experiment in experiments:
        experiment_label = experiment['@label']
        scans = experiment['xnat:scans'][0]['xnat:scan']
        for scan in scans:
            if (scan['@type'] in Modality):
                scan_label = scan['@ID'] + '-' + scan['@type']
                resources_label = scan['xnat:file'][0]['@label']
                if resources_label == 'SNAPSHOTS':
                    resources_label = scan['xnat:file'][1]['@label']
                path = os.path.join(config['DATA']['DataFolder'], subject_label, experiment_label, 'scans',
                                    scan_label, 'resources', resources_label, 'files')
                return path
