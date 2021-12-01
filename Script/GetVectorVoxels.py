import sys, os,glob
import SimpleITK as sitk
import numpy as np
import copy 
import matplotlib.pyplot as plt
from scipy import ndimage
from scipy import stats
from scipy import ndimage
basepath = sys.argv[1]
#database = {}
dX = 80; dY = 80; dZ = 20
patientdata  = np.zeros((1,2*dZ,2*dY,2*dX,2),dtype=np.float32)
database     = np.zeros((1,2*dZ,2*dY,2*dX,2),dtype=np.float32)
id_list      = []
NPatient = 0
for dirName, subdirList, fileList in os.walk(basepath):
    if("CT_masked.nrrd" in fileList and "dose_masked.nrrd" in fileList):
        patient_id = dirName.split("/")[-1]

        ## Masked
        img_path  = os.sep.join([dirName, "CT_masked.nrrd"])
        dose_path = os.sep.join([dirName, "dose_masked.nrrd"])        
        
        img_masked  = sitk.ReadImage(img_path)
        dose_masked = sitk.ReadImage(dose_path)
        
        img_masked  = sitk.GetArrayFromImage(img_masked).astype(np.float32)
        dose_masked = sitk.GetArrayFromImage(dose_masked).astype(np.float32)
        ids  = np.where(img_masked<-100)
        ids2  = np.where(img_masked>-100)

        print(img_masked.shape)
        ## Not masked
        try:
            img_path  = os.sep.join([dirName, "CT.nrrd"])
            dose_path = os.sep.join([dirName, "dose.nrrd"])        
            
            img  = sitk.ReadImage(img_path)
            dose = sitk.ReadImage(dose_path)
            
            img  = sitk.GetArrayFromImage(img).astype(np.float32)
            dose = sitk.GetArrayFromImage(dose).astype(np.float32)
        ## Not masked
        except:
            img_path  = os.sep.join([dirName, "CT_unmasked.nrrd"])
            dose_path = os.sep.join([dirName, "dose.nrrd"])        
            
            img  = sitk.ReadImage(img_path)
            dose = sitk.ReadImage(dose_path)
            
            img  = sitk.GetArrayFromImage(img).astype(np.float32)
            dose = sitk.GetArrayFromImage(dose).astype(np.float32)            

        
        ## Little trick to not mix image values at zero and cancer HU at zero
        img_masked += img_masked+1000
        img_masked[ids] = 0
        
        cm_z, cm_y, cm_x = list(map(int,ndimage.measurements.center_of_mass(img_masked))) 
        print(img.shape, cm_z, cm_y, cm_x)
        img  = img[cm_z-dZ:cm_z+dZ , cm_y-dY:cm_y+dY , cm_x-dX:cm_x+dX ] ## 160x160x20
        dose = dose[cm_z-dZ:cm_z+dZ , cm_y-dY:cm_y+dY , cm_x-dX:cm_x+dX ] ## 160x160x20
        
        img_masked   = img_masked[cm_z-dZ:cm_z+dZ , cm_y-dY:cm_y+dY , cm_x-dX:cm_x+dX ] ## 160x160x20
        dose_masked  = dose_masked[cm_z-dZ:cm_z+dZ , cm_y-dY:cm_y+dY , cm_x-dX:cm_x+dX ] ## 160x160x20        

        """
        plt.imshow(img_masked[10])
        plt.imshow(dose_masked[10],alpha=0.2,cmap='jet')
        plt.show()
        
        plt.imshow(img[10])
        plt.imshow(dose[10],alpha=0.2,cmap='jet')
        plt.show()
        """
        #img[np.where(img>0)]   = stats.zscore(img[np.where(img>0)])
        #dose[np.where(dose>0)] = stats.zscore(dose[np.where(dose>0)])
        #img  = stats.zscore(img)
        #dose = stats.zscore(dose)        
        print(img.shape)

        patientdata[0,:,:,:,0] = img ## Cheap and ugly but does the job
        patientdata[0,:,:,:,1] = dose
        if(NPatient==0):
            database[0] = patientdata
        else:
            database = np.append(database,patientdata,axis=0)
        print(database.shape)
        print(patient_id)
        id_list.append(patient_id)
        NPatient = NPatient +1
        print(NPatient)
        if(NPatient>350): break
    else:
        continue        
database = np.swapaxes(database,1,-1) ## Put channels in first
print(database.shape)
np.savez("database.npz",data = database, patid =id_list)





