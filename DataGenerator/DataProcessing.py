import pandas as pd
pd.options.mode.chained_assignment = None  # default='warn'
from collections import Counter
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler, OneHotEncoder
from sklearn.feature_selection import SelectKBest, mutual_info_classif, f_classif, chi2
from sklearn.feature_selection import GenericUnivariateSelect, SelectFromModel, VarianceThreshold
from sklearn.linear_model import LogisticRegression
from random import randint
import numpy as np

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

def LoadLabel(LabelName, FileName,):
    outcome          = pd.read_csv(FileName)
    label            = outcome[LabelName]
    if(np.max(label)== 2.0): label[label ==2.0] = 0.0 ## Local Failure and Distant Failure
    if(np.any(label > 3)):   label = (label>24).astype('int16')  ## Months/Survival at 2 years as a boolean
    return outcome['patid'], label


def LoadClincalData(MasterSheet):
    clinical_columns = ['arm', 'age', 'gender', 'race', 'ethnicity', 'zubrod',
                        'histology', 'nonsquam_squam', 'ajcc_stage_grp', 'rt_technique',
                        # 'egfr_hscore_200', 'received_conc_cetuximab','rt_compliance_physician',
                        'smoke_hx', 'rx_terminated_ae', 'rt_dose',
                        'volume_ptv', 'dmax_ptv', 'v100_ptv',
                        'v95_ptv', 'v5_lung', 'v20_lung', 'dmean_lung', 'v5_heart',
                        'v30_heart', 'v20_esophagus', 'v60_esophagus', 'Dmin_PTV_CTV_MARGIN',
                        'Dmax_PTV_CTV_MARGIN', 'Dmean_PTV_CTV_MARGIN',
                        'rt_compliance_ptv90', 'received_conc_chemo',
                        ]

    numerical_cols = ['age', 'volume_ptv', 'dmax_ptv', 'v100_ptv',
                      'v95_ptv', 'v5_lung', 'v20_lung', 'dmean_lung', 'v5_heart',
                      'v30_heart', 'v20_esophagus', 'v60_esophagus', 'Dmin_PTV_CTV_MARGIN',
                      'Dmax_PTV_CTV_MARGIN', 'Dmean_PTV_CTV_MARGIN']
    category_cols = list(set(clinical_columns).difference(set(numerical_cols)))

    numerical_data = MasterSheet[numerical_cols] #pd.DataFrame()
    category_data =  MasterSheet[category_cols]

    #for categorical in category_cols:
    #    temp_col = pd.get_dummies(MasterSheet[categorical], prefix=categorical)
    #    clinical_data = clinical_data.join(temp_col)

    return numerical_data, category_data