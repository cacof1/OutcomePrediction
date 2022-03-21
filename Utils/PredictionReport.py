import numpy as np
import matplotlib.pyplot as plt
import torchvision
from sksurv.metrics import cumulative_dynamic_auc

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
