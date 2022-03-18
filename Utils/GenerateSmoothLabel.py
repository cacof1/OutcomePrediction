import sys
import pandas as pd
import torch
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import convolve1d
from scipy.ndimage import gaussian_filter1d
from scipy.signal.windows import triang
import torchvision
from sksurv.metrics import cumulative_dynamic_auc


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


def get_smoothed_label_distribution(MasterSheet, label):
    # MasterSheet = pd.read_csv(sys.argv[1], index_col='patid')
    # label = sys.argv[2]
    Label = [label]
    MasterSheet = MasterSheet.dropna(subset=[label])

    label_all = MasterSheet[Label].to_numpy()
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
    print('MSE:', mse)

    # _ = plt.hist(label_all, bins=label_range)
    # plt.show()
    #
    # plt.bar(label_range[0:-1], eff_label_dist)
    # plt.show()
    return weights, bin_index_per_label[1]


def generate_report(img):
    # img_batch = batch[0]['Anatomy']
    img_batch = img.view(img.shape[0] * img.shape[1], *[1, img.shape[2], img.shape[3]])
    grid = torchvision.utils.make_grid(img_batch)
    return grid


def generate_cumulative_dynamic_auc(y_train, y_test, risk_score):
    # va_times = label_range
    va_times = np.arange(int(y_test.cpu().min()) + 1, y_test.cpu().max(), 1)
    dtypes = np.dtype([('event', np.bool_), ('time', np.float)])
    construct_test = np.ndarray(shape=(len(y_test),), dtype=dtypes)
    for i in range(len(y_test)):
        construct_test[i] = (True, y_test[i].cpu().numpy())
    # construct_test = {'death': torch.ones(y_test.shape, dtype=torch.bool), 'time': y_test}
    construct_train = np.ndarray(shape=(len(y_train),), dtype=dtypes)
    for i in range(len(y_train)):
        construct_train[i] = (True, y_train.to_numpy()[i])

    cph_auc, cph_mean_auc = cumulative_dynamic_auc(
        construct_train, construct_test, risk_score.cpu(), va_times
    )
    fig = plt.figure()
    plt.plot(va_times, cph_auc, marker="o")
    plt.axhline(cph_mean_auc, linestyle="--")
    plt.xlabel("days from enrollment")
    plt.ylabel("time-dependent AUC")
    plt.grid(True)
    plt.show()
    return fig


def plot_AUROC(tpr, fpr):
    fig = plt.figure()
    lw = 2
    plt.plot(fpr.cpu(), tpr.cpu(), color='darkorange', lw=lw)
    plt.title('ROC curve')
    # x label
    plt.xlabel('False Positive Rate')
    # y label
    plt.ylabel('True Positive rate')
    plt.show()
    return fig
