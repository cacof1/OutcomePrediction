import pandas as pd
import numpy as np
from prompt_toolkit import prompt
from prompt_toolkit.completion import PathCompleter
import os
from pathlib import Path


def create_subject_list(config):
    patients = pd.read_csv(config['DATA']['patient_ids_file_path'])['PatientID']
    patient_paths = [Path(config['DATA']['data_folder']) / pat for pat in patients]
    data_columns = config['DATA']['clinical_cols'] + [config['DATA']['target']]
    if 'censor_label' in config['DATA'].keys():
        data_columns.append(config['DATA']['censor_label'])
    subject_list = pd.read_csv(config['DATA']['clinical_table_path'], index_col=config['DATA']['subject_label'])
    subject_list = subject_list.loc[patients, data_columns]
    # Certify event value is 0 and censored 1
    if 'censor_label' in config['DATA'].keys() and config['DATA']['event_value']:
        subject_list[config['DATA']['censor_label']] = (
            ~subject_list[config['DATA']['censor_label']].astype(bool)).astype(float)
        subject_list.rename({config['DATA']['censor_label']: 'Censored'}, axis=1, inplace=True)
        config['DATA']['censor_label'] = 'Censored'
    # Add each patient's modality paths
    for modality in config['MODALITY']:
        if config['MODALITY'][modality]:
            filename = config['DATA'][f'{modality}_path']
            subject_list[f'{modality}_Path'] = [pat / filename for pat in patient_paths]
    return subject_list.reset_index()