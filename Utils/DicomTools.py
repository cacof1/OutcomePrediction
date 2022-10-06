import os
import glob
import cv2
import SimpleITK as sitk
import pydicom as dicom
from pydicom import dcmread
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.path import Path


def get_bbox_from_mask(mask,img_shape):
    pos = np.where(mask)
    if pos[0].shape[0] == 0:
        bbox = np.zeros((0, 4))
    else:        
        xmin = np.min(pos[2])
        xmax = np.max(pos[2])
        ymin = np.min(pos[1])
        ymax = np.max(pos[1])
        zmin = np.min(pos[0])
        zmax = np.max(pos[0])
        bbox = [zmin, zmax, ymin, ymax, xmin, xmax]
    return bbox

def ReadDicom(dicom_path,view_image=False):
    Reader    = sitk.ImageSeriesReader()
    filenames = sorted(glob.glob(dicom_path+'/*.dcm'))
    print(len(filenames))
    Reader.SetFileNames(sorted(filenames))
    
    assert len(filenames) > 0
    Session = Reader.Execute()
    if view_image: ViewDicom(Session)

    return Session

def ResamplingITK(Session, Reference, is_label=False, pad_value=0):
    resample = sitk.ResampleImageFilter()
    resample.SetReferenceImage(Reference)
    resample.SetDefaultPixelValue(pad_value)
    if is_label:
        resample.SetInterpolator(sitk.sitkNearestNeighbor)
    else:
        resample.SetInterpolator(sitk.sitkLinear)
    Resampled = resample.Execute(Session)
    return Resampled

def RStoContour(rs_path, targetROI='PTV'):
    rs_file = glob.glob(rs_path + '*.dcm')
    ds      = dcmread(rs_file[0])
    for item in ds.StructureSetROISequence:
        if item.ROIName == targetROI:
            ROI = ds.ROIContourSequence[item.ROINumber - 1]
            contours = [contour for contour in ROI.ContourSequence]
            return contours

def FindMatchedImage(filenames,contour_coord):
    for i in range(len(filenames)):
        img = dicom.read_file(filenames[i])
        z = int(img.ImagePositionPatient[-1])
        if z == int(contour_coord[0][-1]): return filenames[i], img

def ContourtoROI(contour, CTPath):
    contour_coord = np.array(contour.ContourData)
    contour_coord = contour_coord.reshape(int(contour.NumberOfContourPoints), 3)
    filenames = glob.glob(CTPath + '*.dcm')
    img_path, img = FindMatchedImage(filenames, contour_coord)
    img_index = filenames.index(img_path)
    img_array = img.pixel_array
    x_spacing, y_spacing = float(img.PixelSpacing[0]), float(img.PixelSpacing[1])
    origin_x, origin_y, _ = img.ImagePositionPatient
    coords = [(abs(np.ceil(x - origin_x)) / x_spacing, abs(np.ceil(y - origin_y)) / y_spacing) for x, y, _ in contour_coord]

    return coords, img_array, img_index

def poly_to_mask(polygon, img_shape):
    x, y = np.meshgrid(np.arange(img_shape[0]), np.arange(img_shape[1]))
    x, y = x.flatten(), y.flatten()
    points = np.vstack((x, y)).T
    path = Path(polygon)
    mask = path.contains_points(points)
    mask = mask.reshape(img_shape)

    return mask

def ViewROI(patient_id,img_array,mask_array,ROIbox,Inputbox):
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

def get_ROI_voxel(contours, dicom_path, roi_range=[64,64,10]):
    mask_voxel  = []
    bbox_voxel  = []
    ROI_voxel   = []
    image_voxel = []
    img_indices = []
    N_slices = len(contours)
    for contour in contours:
        coords, img_array, img_index = ContourtoROI(contour, dicom_path)
        mask_array = poly_to_mask(coords, img_array.shape)
        bbox, ROI_region = get_bbox_from_mask(mask_array, img_array.shape, roi_range=roi_range[:-1])
        if len(bbox) < 3:
            continue
        elif (bbox[2] - bbox[0]) * (bbox[3] - bbox[1]) < 100:
            continue
        mask_voxel.append(mask_array)
        bbox_voxel.append(bbox)
        ROI_voxel.append(ROI_region)
        image_voxel.append(img_array)
        img_indices.append(img_index)

    mask_voxel  = mask_voxel[int((N_slices - roi_range[-1]) / 2): int((N_slices + roi_range[-1]) / 2)]
    bbox_voxel  = bbox_voxel[int((N_slices - roi_range[-1]) / 2): int((N_slices + roi_range[-1]) / 2)]
    ROI_voxel   = ROI_voxel[int((N_slices - roi_range[-1]) / 2): int((N_slices + roi_range[-1]) / 2)]
    image_voxel = image_voxel[int((N_slices - roi_range[-1]) / 2): int((N_slices + roi_range[-1]) / 2)]
    img_indices = img_indices[int((N_slices - roi_range[-1]) / 2): int((N_slices + roi_range[-1]) / 2)]

    return mask_voxel, bbox_voxel, ROI_voxel, image_voxel, img_indices

def get_masked_img_voxel(ImageVoxel, mask_voxel):
    bbox = get_bbox_from_mask(mask_voxel, np.shape(ImageVoxel))
    assert len(mask_voxel) == ImageVoxel.shape[0]
    img_masked = ImageVoxel[bbox[0]:bbox[1], bbox[2]:bbox[3], bbox[4]:bbox[5]]
    return img_masked








