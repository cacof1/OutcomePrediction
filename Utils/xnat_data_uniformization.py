from pyxnat import Interface
import pandas as pd
# set up connection with xnat by 'logging in'
interface= Interface(server='http://128.16.11.124:8080/xnat', user='yzhan', password='yzhan')
from pathlib import Path
# display the different projects
list(interface.select.projects())

# define the project of interest
pro= interface.select.projects().get()
prj=pro[6]

#define subjects
subjs= interface.select.projects(prj).subjects().get()
# print(dir(subjs))
#subjs.remove('XNAT01_S00032') #delete this subject from the 'testnlst' project because it does not have any scans and it sends an error whenever the code reaches it because of that

## this loop replaces the name of the type of each scan in each experiment for every subject by the name of the modality (which is more consistent)
# run the loop on all the subjects within the chosen project
# df = pd.read_excel('C:\\Users\\clara\\Documents\\patientdata.xlsx')
# id_list = df['id']

for i in range (len(subjs)):
  subj_i = interface.select.projects(prj).subjects(subjs[i])
  exps = subj_i.experiments().get()
  id = subj_i.get(['xnat:subjectData/SUBJECT_ID'])
  # item = df.loc[id_list == id[0].label()]
  # path = item['path']
  # path = Path(list(path)[0])
  # print(id[0].label())
  #run the loop on all the experiments of each subject
  for k in range(len(exps)):
    exp_k=exps[k]
    scns=interface.select.projects(prj).subjects(subjs[i]).experiments(exp_k).scans()
    #run the loop on all the scans present in each experiment
    for scn in scns:
      mod = scn.attrs.get('modality')
      typ=scn.attrs.get('type')
      if mod == 'RTSTRUCT':
        scn.attrs.set('type', 'Structs')
      elif mod == 'RTDOSE':
        # di = scn.dicom_dump()
        # for x in di:
        #   if x['tag1'] == '(0008,0060)':
        #     print(x['value'])
        #     scn.__setattr__('type', str(''))
        scn.attrs.set('type', 'Dose')
      else:
        scn.attrs.set('type',str(mod))
    # print('test')


# import xnat
# import numpy as np
# from pathlib import Path
# import nibabel as nib
# import os
# import pandas as pd
#
# session = xnat.connect('http://128.16.11.124:8080/xnat', user='yzhan', password='yzhan')
# project = session.projects['LUNG_IDEAL']
# subjectS = project.subjects
# print(subjectS)
# new_path = '/home/dgs1/data/InnerEye/Seg/'
# df = pd.read_excel('/home/dgs1/data/Overall.xlsx')
# id_list = df['TrialNo']
# for subject in subjectS.values():
#     # subject.fields['analysis_inclusion'] = 1
#     label = subject.label
#     No = label[-3:]
#     Os = df.loc[id_list == np.uint8(No)]['os_time']
#     try:
#         subject.fields['survival_months'] = list(Os)[0]
#     except:
#         print('error')
#     #   # path = item['path']

    # newpath = Path(new_path, label, 'ptv.nii.gz')
    # if os.path.exists(newpath):
    #     ptv_info = nib.load(newpath)
    #     ptv = ptv_info.get_fdata()
    #     ptv_volume = numpy.sum(ptv)*3/1000
    #     subject.fields['volume_ptv'] = ptv_volume

    # if 'patient_sex' in subject.fields:
    #     if subject.fields['patient_sex'] == 'Male':
    #         subject.fields['gender'] = 1
    #     else:
    #         subject.fields['gender'] = 2
    # if 'year_of_birth' in subject.fields:
    #     scan = subject.experiments[0].scans[0]
    #     di = scan.dicom_dump()
    #     for x in di:
    #         if x['tag1'] == '(0008,0012)':
    #             subject.fields['age'] = numpy.int16(x['value'][0:4]) - numpy.int16(subject.fields['year_of_birth'])
