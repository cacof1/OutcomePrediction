import os
from pydicom import dcmread
from pathlib import Path
import shutil

path = '/home/dgs1/data/IDEAL/IDEAL/'
subjects = os.listdir(path)
for i, sub in enumerate(subjects):
   print(i)
   subpath = Path(path, sub)
   if not str(subpath).endswith('.DS_Store'):
       plan_dir = Path(subpath, 'planning_data')
       ##
       sub_plan_dir = os.listdir(plan_dir)

       # if len(sub_plan_dir) < 8:
       #     sub_plan_dir_planning = Path(plan_dir, 'planning')
       #     ddf = os.listdir(sub_plan_dir_planning)
       #     df = dcmread(Path(sub_plan_dir_planning, ddf[0]), force=True)
       #     sid_st = df.StudyInstanceUID
       #
       #     for sf in set(sub_plan_dir).difference(set(['planning','.DS_Store'])):
       #         splan_dir = Path(plan_dir, sf)
       #         ddf = os.listdir(splan_dir)
       #         filename = Path(splan_dir,ddf[0])
       #         df = dcmread(filename, force=True)
       #         df.StudyInstanceUID = sid_st
       #         df.save_as(filename)
       #
       # for dirpath, dirs, files in os.walk(subpath):
       #     for filename in files:
       #         fname = os.path.join(dirpath, filename)
       #         if not fname.endswith('.DS_Store'):
       #             ds = dcmread(fname, force=True)
       #             ds.PatientID = sub
       #             ds.PatientName = sub
       #             # print(ds.PatientID)
       #             ds.save_as(fname)


 # if not str(subpath).endswith('.DS_Store'):
       #     rmdir = os.listdir(subpath)
       #     plan_dir = Path(subpath, 'planning_data')
       #
       #     for rd in set(rmdir).difference(set(['planning_data'])):
       #         if not rd.endswith('.DS_Store'):
       #             shutil.rmtree(Path(subpath,rd))
       #             print(rd)
