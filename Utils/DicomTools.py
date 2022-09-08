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
from skimage.measure import regionprops
from monai.data import image_reader
from scipy.ndimage import map_coordinates
import torchio as tio


def get_bbox_from_mask(mask, img_shape, roi_range=[64, 64]):
    pos = np.where(mask)
    if pos[0].shape[0] == 0:
        bbox = np.zeros((0, 4))
        ROI_region = np.zeros((0, 4))
    else:
        xmin = np.min(pos[1])
        xmax = np.max(pos[1])
        ymin = np.min(pos[0])
        ymax = np.max(pos[0])
        bbox = [xmin, ymin, xmax, ymax]
        center = (int(0.5 * (xmax + xmin)), int(0.5 * (ymax + ymin)))
        x1 = max(0, int(center[0] - roi_range[0] / 2))
        x2 = x1 + roi_range[0]
        if x2 > img_shape[0]:
            x2 = img_shape[0]
            x1 = int(x2 - roi_range[0])
        y1 = max(0, int(center[1] - roi_range[1] / 2))
        y2 = y1 + roi_range[1]
        if y2 > img_shape[1]:
            y2 = img_shape[1]
            y1 = int(y2 - roi_range[1])
        ROI_region = (x1, y1, x2, y2)

    return bbox, ROI_region


def ViewDicom(itk_item, AppPath='E:/Apps/Fiji.app/ImageJ-win64.exe'):
    Viewer = sitk.ImageViewer()
    Viewer.SetApplication(AppPath)
    Viewer.Execute(itk_item)


def ReadDicom(dicom_path, view_image=False):
    Reader = sitk.ImageSeriesReader()
    sitk.ProcessObject_SetGlobalWarningDisplay(False)
    filenames = glob.glob(dicom_path + '*.dcm')
    Reader.SetFileNames(sorted(filenames))
    assert len(filenames) > 0
    Session = Reader.Execute()
    if view_image: ViewDicom(Session)

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
        resample.SetInterpolator(sitk.sitkBSpline)

    return resample.Execute(Session)


def RTSStoContour(rtss_path, targetROI='PTV'):
    rtss_file = glob.glob(rtss_path + '*.dcm')
    ds = dcmread(rtss_file[0])
    for item in ds.StructureSetROISequence:
        if item.ROIName == targetROI:
            ROI = ds.ROIContourSequence[item.ROINumber - 1]
            contours = [contour for contour in ROI.ContourSequence]
            return contours


def FindMatchedImage(filenames, contour_coord):
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
    coords = [(abs(np.ceil(x - origin_x)) / x_spacing, abs(np.ceil(y - origin_y)) / y_spacing) for x, y, _ in
              contour_coord]

    return coords, img_array, img_index


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


def get_ROI_voxel(contours, dicom_path, roi_range=[64, 64, 10]):
    mask_voxel = []
    bbox_voxel = []
    ROI_voxel = []
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

    mask_voxel = mask_voxel[int((N_slices - roi_range[-1]) / 2): int((N_slices + roi_range[-1]) / 2)]
    bbox_voxel = bbox_voxel[int((N_slices - roi_range[-1]) / 2): int((N_slices + roi_range[-1]) / 2)]
    ROI_voxel = ROI_voxel[int((N_slices - roi_range[-1]) / 2): int((N_slices + roi_range[-1]) / 2)]
    image_voxel = image_voxel[int((N_slices - roi_range[-1]) / 2): int((N_slices + roi_range[-1]) / 2)]
    img_indices = img_indices[int((N_slices - roi_range[-1]) / 2): int((N_slices + roi_range[-1]) / 2)]

    return mask_voxel, bbox_voxel, ROI_voxel, image_voxel, img_indices


def get_masked_img_voxel(ImageVoxel, mask_voxel, bbox_voxel, ROI_voxel, visImage=False, PatientID=None):
    assert len(mask_voxel) == ImageVoxel.shape[0]
    input_voxel = []
    for i in range(len(mask_voxel)):
        img_array = ImageVoxel[i]
        mask_array = mask_voxel[i]
        bbox = bbox_voxel[i]
        ROI_region = ROI_voxel[i]
        img_masked = np.where(mask_array == True, img_array, mask_array)
        img_croped_bbox = img_masked[bbox[1]:bbox[3], bbox[0]:bbox[2]]
        img_croped_roi = img_masked[ROI_region[1]:ROI_region[3], ROI_region[0]:ROI_region[2]]
        input_voxel.append(img_croped_roi.astype('float32'))
        if visImage: ViewROI(PatientID, img_masked, mask_array, img_croped_bbox, img_croped_roi)

    return np.array(input_voxel)


def interp3(x, y, z, v, xi, yi, zi, **kwargs):
    """Sample a 3D array "v" with pixel corner locations at "x","y","z" at the
    points in "xi", "yi", "zi" using linear interpolation. Additional kwargs
    are passed on to ``scipy.ndimage.map_coordinates``."""

    def index_coords(corner_locs, interp_locs):
        index = np.arange(len(corner_locs))
        if np.all(np.diff(corner_locs) < 0):
            corner_locs, index = corner_locs[::-1], index[::-1]
        return np.interp(interp_locs, corner_locs, index)

    orig_shape = np.asarray(xi).shape
    xi, yi, zi = np.atleast_1d(xi, yi, zi)
    for arr in [xi, yi, zi]:
        arr.shape = -1

    output = np.empty(xi.shape, dtype=float)
    coords = [index_coords(*item) for item in zip([x, y, z], [xi, yi, zi])]

    map_coordinates(v, coords, order=1, output=output, **kwargs)

    return output.reshape(orig_shape)


def DoseMatchCT(DoseObj, DoseVolume, CTObj):
    DoseVolume = DoseVolume.transpose(2, 1, 0)
    originD = DoseObj.GetOrigin()
    spaceD = DoseObj.GetSpacing()
    origin = CTObj.GetOrigin()
    space = CTObj.GetSpacing()

    dx = np.arange(0, DoseObj.GetSize()[0]) * spaceD[0] + originD[0]
    dy = np.arange(0, DoseObj.GetSize()[1]) * spaceD[1] + originD[1]
    dz = -np.arange(0, DoseObj.GetSize()[2]) * spaceD[2] + originD[2]
    dz.sort()

    cx = np.arange(0, CTObj.GetSize()[0]) * space[0] + origin[0]
    cy = np.arange(0, CTObj.GetSize()[1]) * space[1] + origin[1]
    cz = -np.arange(0, CTObj.GetSize()[2]) * space[2] + origin[2]
    cz.sort()

    cxv, cyv, czv = np.meshgrid(cx, cy, cz, indexing='ij')

    Vf = interp3(dx, dy, dz, DoseVolume, cxv, cyv, czv)
    Vf = Vf.transpose(2, 1, 0)
    return Vf


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
