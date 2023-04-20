import os
from pydicom import dcmread
from pathlib import Path
import shutil
from dcmrtstruct2nii import dcmrtstruct2nii, list_rt_structs

path = '/home/dgs1/data/IDEAL/IDEAL/'
subjects = os.listdir(path)
for i, sub in enumerate(subjects):
   print(i)
   subpath = Path(path, sub)
   if not str(subpath).endswith('.DS_Store'):
       plan_dir = Path(subpath, 'planning_data')
       ##
       sub_plan_dir = os.listdir(plan_dir)

       if len(sub_plan_dir) < 8:
           sub_plan_dir_planning = Path(plan_dir, 'planning')
           ddf = os.listdir(sub_plan_dir_planning)

           sub_structs_dir_planning = Path(plan_dir, 'structures')
           struct_dcm = os.listdir(sub_structs_dir_planning)[0]


