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
    instant_record = []
    ds = dcmread(rtss_file)
    for n, item in enumerate(ds.StructureSetROISequence):
        ROI = ds.ROIContourSequence[n]
        try:
            len(ROI.ContourSequence)
            for i in range(len(ROI.ContourSequence)):
                contour = ds.ROIContourSequence[n].ContourSequence[i]
                contour_coord = np.array(contour.ContourData)
                contour_coord = contour_coord.reshape(int(contour.NumberOfContourPoints), 3)
                filenames = glob.glob(CT_path + '/*.dcm')
                img_path, img = FindMatchedImage(filenames, contour_coord)
                CIS_ds = Dataset()
                CIS_ds.add_new([0x0008, 0x1150], 'UI', 'CT Image Storage')
                CIS_ds.add_new([0x0008, 0x1155], 'UI', img.SOPInstanceUID)
                ds.ROIContourSequence[n].ContourSequence[i].add_new([0x3006, 0x0016], 'SQ', Sequence([CIS_ds]))
        except:
            del ds.ROIContourSequence[n]
            del ds.StructureSetROISequence[n]
            del ds.RTROIObservationsSequence[n]

            #print(ds.ROIContourSequence[item.ROINumber - 1].ContourSequence[i])

        filenames = sorted(glob.glob(CT_path + '/*.dcm'))

        for file in filenames:
            info = dicom.read_file(file)
            instant_record.append(info.InstanceNumber)

        info = ds.ReferencedFrameOfReferenceSequence
        RFUIDlist = info._list
        dinfo = RFUIDlist[0].RTReferencedStudySequence._list[0].RTReferencedSeriesSequence._list[0].ContourImageSequence._list

        for i in range(len(dinfo)):
            sliceuid = dinfo[i]
            if min(instant_record) == 0:
                n = i
            else:
                n = i + 1
            idx = instant_record.index(n)
            fileUID = dicom.read_file(filenames[idx]).SOPInstanceUID
            sliceuid.ReferencedSOPInstanceUID = fileUID

    ds.save_as(rtss_file)

def RenameCT(CTPath):
    instant_record = []
    filenames = sorted(glob.glob(CTPath + '/*.dcm'))
    for file in filenames:
        info = dicom.read_file(file)
        instant_record.append(info.InstanceNumber)
        UID = info.SOPInstanceUID
        oldpath = file.split('/')
        oldpath[-1] = UID + '.dcm'
        newpath = '/'.join(oldpath)
        os.rename(file,newpath)


