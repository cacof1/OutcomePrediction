import os
import glob
import cv2
import SimpleITK as sitk
import pydicom as dicom
from pydicom import dcmread
from pydicom.dataset import Dataset
from pydicom.sequence import Sequence
import numpy as np

def FindMatchedImage(filenames,contour_coord):
    for i in range(len(filenames)):
        img = dicom.read_file(filenames[i])
        z = int(img.ImagePositionPatient[-1])
        if z == int(contour_coord[0][-1]): return filenames[i], img

def FixRTSS(rtss_path, CT_path):
    rtss_file = glob.glob(rtss_path + '*.dcm')
    ds = dcmread(rtss_file[0])
    for item in ds.StructureSetROISequence:
        ROI = ds.ROIContourSequence[item.ROINumber - 1]
        for i in range(len(ROI.ContourSequence)):
            contour = ds.ROIContourSequence[item.ROINumber - 1].ContourSequence[i]
            try:
                img_id = contour.ContourImageSequence[0].ReferencedSOPInstanceUID
            except:
                contour_coord = np.array(contour.ContourData)
                contour_coord = contour_coord.reshape(int(contour.NumberOfContourPoints), 3)
                filenames = glob.glob(CT_path + '*.dcm')
                img_path, img = FindMatchedImage(filenames, contour_coord)
                img_id = os.path.basename(img_path).split('_')[-1].split('.dcm')[0]
                CIS_ds = Dataset()
                CIS_ds.add_new([0x0008, 0x1150], 'UI', 'CT Image Storage')
                CIS_ds.add_new([0x0008, 0x1155], 'UI', img_id)
                ds.ROIContourSequence[item.ROINumber - 1].ContourSequence[i].add_new([0x3006, 0x0016],'SQ', Sequence([CIS_ds]))
            #print(ds.ROIContourSequence[item.ROINumber - 1].ContourSequence[i])
            
    ds.save_as(rtss_file[0])
