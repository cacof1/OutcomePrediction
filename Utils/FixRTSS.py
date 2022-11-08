from pathlib import Path
import os
import glob
import pydicom as dicom
from pydicom import dcmread
from pydicom.dataset import Dataset
from pydicom.sequence import Sequence
import numpy as np

def FindMatchedImage(filenames, contour_coord):
    for i in range(len(filenames)):
        img = dicom.read_file(filenames[i])
        z = int(img.ImagePositionPatient[-1])
        if z == int(contour_coord[0][-1]): return filenames[i], img


def FixRTSS(rtss_file, CT_path):
    ds = dcmread(rtss_file)
    for n, item in enumerate(ds.StructureSetROISequence):
        ROI = ds.ROIContourSequence[n]
        for i in range(len(ROI.ContourSequence)):
            contour = ds.ROIContourSequence[n].ContourSequence[i]
            contour_coord = np.array(contour.ContourData)
            contour_coord = contour_coord.reshape(int(contour.NumberOfContourPoints), 3)
            filenames = glob.glob(CT_path + '/*.dcm')
            img_path, img = FindMatchedImage(filenames, contour_coord)
            img_id = os.path.basename(img_path).split('_')[-1].split('.dcm')[0]
            CIS_ds = Dataset()
            CIS_ds.add_new([0x0008, 0x1150], 'UI', 'CT Image Storage')
            CIS_ds.add_new([0x0008, 0x1155], 'UI', img_id)
            ds.ROIContourSequence[n].ContourSequence[i].add_new([0x3006, 0x0016], 'SQ', Sequence([CIS_ds]))

            print(ds.ROIContourSequence[n].ContourSequence[i])

    ds.save_as(rtss_file)


bp = '/home/dgs1/data/UCLH/'
rtss_file = '/home/dgs1/data/OutcomePrediction/2d59d38f8e991100bcee7ffeb8d8fb656c78d067d49a717b57d490632aff3251/2d59d38f8e991100bcee7ffeb8d8fb656c78d067d49a717b57d490632aff3251/scans/602-Structs/resources/secondary/files/Unknown (0002).dcm'
CT_path = '/home/dgs1/data/OutcomePrediction/2d59d38f8e991100bcee7ffeb8d8fb656c78d067d49a717b57d490632aff3251/2d59d38f8e991100bcee7ffeb8d8fb656c78d067d49a717b57d490632aff3251/scans/601-CT/resources/DICOM/files'
FixRTSS(rtss_file, CT_path)


