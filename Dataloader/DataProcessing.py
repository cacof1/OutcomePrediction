import pandas as pd
pd.options.mode.chained_assignment = None  # default='warn'
from collections import Counter
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.feature_selection import SelectKBest, mutual_info_classif, f_classif, chi2
from sklearn.feature_selection import GenericUnivariateSelect, SelectFromModel, VarianceThreshold
from sklearn.linear_model import LogisticRegression
from random import randint
import numpy as np
from imblearn.over_sampling import SMOTE

def CollectData(RANDOM_SEED, settings):
    basepath         = "/lustre/projects/ParticleCT/OutcomePrediction/Study/"+settings["Study"]+"/"    
    outcome_path     = basepath+"Outcome.csv"

    ## Load the data
    outcome          = pd.read_csv(outcome_path)
    if(settings["Classif"]):   classif   = pd.read_csv(basepath+"lobe_classification_TEMPLATE2PAT.txt",names=['patid','lobe'], skiprows=1, dtype={0:str, 1:int})    ## Lobe classification

    ## First we find the intersection between patient in the radiomics and the one in the outcome 
    ids_common = set.intersection(set(data['patid']), set(outcome['patid']))
    if(settings["Classif"]):   ids_common = set.intersection(set(ids_common), set(classif['patid']))

    ## Then we choose the data
    outcome   = outcome[outcome['patid'].isin(ids_common)]
    if(settings["Classif"]):   classif   = classif[classif['patid'].isin(ids_common)]

    ## Reset the index
    outcome   = outcome.reset_index()
    if(settings["Classif"]): classif   = classif.reset_index()
    
    ## One Hot Encoding
    if(settings["Classif"]): lobes = pd.get_dummies(classif['lobe'],prefix='lobe')
    if(settings["Clinical"]):
        smoke_hx = pd.get_dummies(outcome['smoke_hx'], prefix='smoke')    
        ajcc_stg = pd.get_dummies(outcome['ajcc_stage_grp'],prefix='ajcc')
        arm      = pd.get_dummies(outcome['arm'],prefix='arm')
    
    database     = outcome[settings["Label"]]
    if(settings["Classif"]): database  = database[classif['lobe']==settings["Lobe"]]

    ## Remove morphological based feature -- by MF request
    if(settings["morph"]):
        filter_col = [col for col in database if not col.startswith('original_shape')]
        database  = database[filter_col]

    ## Only keep N patients
    if(settings["NPatients"]>0): database = database.iloc[:settings["NPatients"],]
    database  = database.astype(np.float32)

    if(np.max(y_init)== 2.0): y_init[y_init ==2.0] = 0.0 ## Local Failure and Distant Failure
    if(np.any(y_init > 3)):   y_init = (y_init>24).astype('int16')  ## Months/Survival at 2 years as a boolean        
    
    return y_init
