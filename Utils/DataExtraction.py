import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer, SimpleImputer
from sklearn.preprocessing import StandardScaler, RobustScaler, MinMaxScaler


def create_subject_list(config):
    patients = pd.read_csv(config['DATA']['patient_ids_file_path'])['PatientID']
    patient_paths = [Path(config['DATA']['data_folder']) / pat for pat in patients]
    data_columns = config['DATA']['clinical_cols'] + [config['DATA']['target']] + config['DATA']['additional_targets']
    if 'censor_label' in config['DATA'].keys():
        data_columns.append(config['DATA']['censor_label'])
    subject_list = pd.read_csv(config['DATA']['clinical_table_path'], index_col=config['DATA']['subject_label'])
    subject_list = subject_list.loc[patients, data_columns]
    # Certify censored value is 0 and event 1
    if 'censor_label' in config['DATA'].keys() and 'censored_value' in config['DATA'].keys():
        subject_list['Censored'] = (
                subject_list[config['DATA']['censor_label']] == config['DATA']['censored_value']).astype(float)
    elif 'censor_label' not in config['DATA'].keys():
        subject_list['Censored'] = 0

    # Add each patient's modality paths
    for modality in config['MODALITY']:
        if config['MODALITY'][modality]:
            filename = config['DATA'][f'{modality}_path']
            subject_list[f'{modality}_Path'] = [pat / filename for pat in patient_paths]

    if config['DATA']['impute_addit_target']:
        for i, col in enumerate(config['DATA']['additional_targets']):
            imputer = (SimpleImputer() if config['MODEL']['modes'][i + 1] == 'regression' else
                       SimpleImputer(strategy='most_frequent'))
            subject_list.loc[:, [col]] = imputer.fit_transform(subject_list.loc[:, [col]]).astype(np.float32)

    if config['DATA']['exclude_event_nan']:
        subject_list = subject_list.loc[
            subject_list[[config['DATA']['target']] +
                         config['DATA']['additional_targets']].sum(axis=1, skipna=False).notna()]

    if not config['DATA']['include_censored']:
        subject_list = subject_list.loc[subject_list.loc[:, 'Censored'] == 0]

    if 'threshold' in config['DATA']:  # it's classification
        # Exclude all patients with censored times before the threshold
        exclusion_cond = ((subject_list.loc[:, 'Censored'] == 1) &
                          (subject_list.loc[:, config['DATA']['target']] < config['DATA']['threshold']))
        subject_list = subject_list.loc[~exclusion_cond]

    if 'RECORDS' in config.keys() and config['RECORDS']['records'] and 'categorical_cols' in config['DATA']:
        subject_list_old = subject_list.copy()
        subject_list = pd.get_dummies(subject_list, columns=config['DATA']['categorical_cols'], drop_first=True,
                                      dtype=np.float32)
        new_categorical_cols = []
        for var in config['DATA']['categorical_cols']:
            cols = [c for c in list(subject_list.columns) if var in c]
            subject_list.loc[subject_list_old[var].isna(), cols] = np.nan
            new_categorical_cols += cols
        config['DATA']['given_categorical_cols'] = config['DATA']['categorical_cols']
        config['DATA']['given_clinical_cols'] = config['DATA']['clinical_cols']
        config['DATA']['categorical_cols'] = new_categorical_cols
        config['DATA']['clinical_cols'] = config['DATA']['continuous_cols'] + config['DATA']['categorical_cols']

    if config['DATA']['imputation'] == 'mice':
        imputer = IterativeImputer()
        subject_list.loc[:, config['DATA']['clinical_cols']] = imputer.fit_transform(
            subject_list.loc[:, config['DATA']['clinical_cols']]).astype(np.float32)

    return subject_list.reset_index()
