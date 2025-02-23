import numpy as np
import pandas as pd
from itertools import chain


def inverse_transform_target(results, dataloader, config, include_binary=False):
    t_preproc = dataloader.target_preprocessing
    for i, col in enumerate(t_preproc):
        if not include_binary and config['MODEL']['modes'][i] == 'classification':
            continue
        results.loc[:, [f'Prediction_{col}']] = t_preproc[col].inverse_transform(results.loc[:, [f'Prediction_{col}']])
        results.loc[:, [f'Target_{col}']] = t_preproc[col].inverse_transform(results.loc[:, [f'Target_{col}']])
    return results


def get_results_table(results, dataloader, config):
    end_columns = ['Censored', 'Index'] if len(results[0]) == 4 else ['Index']
    pred_columns = [f'Prediction_{i}' for i in [config['DATA']['target']] + config['DATA']['additional_targets']]
    target_columns = [f'Target_{i}' for i in [config['DATA']['target']] + config['DATA']['additional_targets']]
    columns = pred_columns + target_columns + end_columns
    arr = [(np.array(list(chain(*[r[idx] for r in results])))) for idx in range(len(results[0]))]
    arr = np.concatenate([arr_[:, None] if arr_.ndim == 1 else arr_ for arr_ in arr], axis=1)
    tab = pd.DataFrame(arr, columns=columns)
    tab[config['DATA']['subject_label']] = dataloader.full_list.loc[tab['Index'], config['DATA']['subject_label']]
    tab['Train Set'] = tab[config['DATA']['subject_label']].isin(dataloader.train_list[config['DATA']['subject_label']])
    tab['Validation Set'] = tab[config['DATA']['subject_label']].isin(
        dataloader.val_list[config['DATA']['subject_label']])
    tab['Test Set'] = tab[config['DATA']['subject_label']].isin(dataloader.test_list[config['DATA']['subject_label']])
    tab = tab.set_index(config['DATA']['subject_label'], drop=True)
    return tab.drop('Index', axis=1)


def get_train_val_test_tab(dataloader, rd, config):
    tab = dataloader.full_list.loc[:, [config['DATA']['subject_label']]]
    tab['Train Set'] = tab[config['DATA']['subject_label']].isin(dataloader.train_list[config['DATA']['subject_label']])
    tab['Validation Set'] = tab[config['DATA']['subject_label']].isin(
        dataloader.val_list[config['DATA']['subject_label']])
    tab['Test Set'] = tab[config['DATA']['subject_label']].isin(dataloader.test_list[config['DATA']['subject_label']])
    tab = tab.set_index(config['DATA']['subject_label'], drop=True)
    tab.loc['random_seed', 'Train Set'] = rd
    return tab
