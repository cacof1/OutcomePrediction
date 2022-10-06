import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import convolve1d
from scipy.ndimage import gaussian_filter1d
from scipy.signal.windows import triang
from sksurv.metrics import cumulative_dynamic_auc
from torch import nn
from Models.Classifier3D import Classifier3D
from Models.Classifier2D import Classifier2D
from Models.Linear import Linear

def get_lds_kernel_window(kernel, ks, sigma):
    assert kernel in ['gaussian', 'triang', 'laplace']
    half_ks = (ks - 1) // 2
    if kernel == 'gaussian':
        base_kernel = [0.] * half_ks + [1.] + [0.] * half_ks
        kernel_window = gaussian_filter1d(base_kernel, sigma=sigma) / max(gaussian_filter1d(base_kernel, sigma=sigma))
    elif kernel == 'triang':
        kernel_window = triang(ks)
    else:
        laplace = lambda x: np.exp(-abs(x) / sigma) / (2. * sigma)
        kernel_window = list(map(laplace, np.arange(-half_ks, half_ks + 1))) / max(
            map(laplace, np.arange(-half_ks, half_ks + 1)))

    return kernel_window


def get_smoothed_label_distribution(SubjectList, config):
    label_all = get_train_label(SubjectList, config)
    range_max = np.max(label_all).astype(int) + 1
    range_min = np.min(label_all).astype(int)

    label_range = np.arange(range_min, range_max, 1)

    bin_index_per_label = np.histogram(label_all, bins=label_range)
    lds_kernel_window = get_lds_kernel_window(kernel='gaussian', ks=7, sigma=3)
    eff_label_dist = convolve1d(np.array(bin_index_per_label[0]), weights=lds_kernel_window, mode='constant')

    eff_num_per_label = [eff_label_dist[bin_idx] for bin_idx in np.arange(eff_label_dist.shape[0])]
    weights = [np.float32(1 / x) for x in eff_num_per_label]

    label_mean = np.mean(label_all)
    mse = ((label_all - label_mean) ** 2).mean()
    return weights, bin_index_per_label[1]


def get_train_label(SubjectList, config):
    train_label = []
    for patient in SubjectList:
        label = patient.fields[config['DATA']['target']]
        train_label.append(label)
    return train_label


def get_module(config):
    s_module = config['DATA']['module']
    module_dict = nn.ModuleDict()
    if config['MODEL']['Clinical_Backbone']:
        Clinical_backbone = Linear()
    for i, module in enumerate(s_module):
        if module == 'CT' or module == 'Dose' or module == 'PET':
            if config['MODEL']['spatial_dims'] == 3:
                Backbone = Classifier3D(config)
            else:
                Backbone = Classifier2D(config)
            module_dict[module] = Backbone
        else:
            module_dict[module] = Clinical_backbone

    return module_dict
