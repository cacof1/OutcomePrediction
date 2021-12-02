import os
import sys
import numpy as np
import pandas as pd
from pathlib import Path

Mastersheet = pd.read_csv(sys.argv[2],index_col="patid") 
CTPath      = pd.Series(index=Mastersheet.index,dtype='object')
DosePath    = pd.Series(index=Mastersheet.index,dtype='object')
for root,dirs,files in os.walk(os.path.abspath(sys.argv[1])):

    if('CT.nrrd' in files):   CTPath[Path(root).name] = Path(root,"CT.nrrd")
    if('dose.nrrd' in files): DosePath[Path(root).name] = Path(root,"dose.nrrd")

Mastersheet["CTPath"] = CTPath
Mastersheet["DosePath"] = DosePath
Mastersheet.to_csv("Mastersheet.csv")    

