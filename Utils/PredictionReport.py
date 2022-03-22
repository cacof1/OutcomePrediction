import numpy as np
import matplotlib.pyplot as plt
import torchvision
from sksurv.metrics import cumulative_dynamic_auc
import torch
from torch import nn
from sksurv.metrics import concordance_index_censored
import torchmetrics


class PredictionReport():

    def __init__(self, prediction, label):
        super().__init__()
        self.prediction = prediction
        self.label = label

    def r2_index(self):
        loss = nn.MSELoss()
        MSE = loss(self.prediction, self.label)
        SSres = MSE * self.label.shape[0]
        SStotal = torch.sum(torch.square(self.label - torch.mean(self.label)))
        r2 = 1 - SSres / SStotal
        return r2

    def c_index(self):
        event_indicator = torch.ones(self.label.shape, dtype=torch.bool)
        risk = 1 / self.prediction.squeeze(dim=1)
        cindex = concordance_index_censored(event_indicator.cpu().detach().numpy(),
                                            event_time=self.label.cpu().detach().numpy(),
                                            estimate=risk.cpu().detach().numpy())
        return cindex

    def worst_case_show(self, config, validation_step_outputs):
        out = {}
        worst_AE = 0
        for i, data in enumerate(validation_step_outputs):
            loss = data['MAE']
            idx = torch.argmax(loss)
            if loss[idx] > worst_AE:
                if 'Anatomy' in config['DATA']['module']:
                    worst_img = data['img'][idx]
                if 'Dose' in config['DATA']['module']:
                    worst_dose = data['dose'][idx]
                worst_MAE = loss[idx]
        out['worst_AE'] = worst_AE
        if 'Anatomy' in config['DATA']['module']:
            out['worst_img'] = worst_img
        if 'Dose' in config['DATA']['module']:
            out['worst_dose'] = worst_dose
        return out

    def generate_report(self, img):
        # img_batch = batch[0]['Anatomy']
        img_batch = img.view(img.shape[0] * img.shape[1], *[1, img.shape[2], img.shape[3]])
        grid = torchvision.utils.make_grid(img_batch)
        return grid

    def generate_cumulative_dynamic_auc(self, y_train):
        # va_times = label_range
        y_test = self.label
        risk_score = 1 / self.prediction
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

    def classification_matrix(self):
        roc = torchmetrics.ROC()
        fpr, tpr, _ = roc(self.prediction, self.label)
        auc = torchmetrics.AUC(reorder=True)
        auc_value = auc(self.prediction, self.label)
        auroc = torchmetrics.AUROC()
        accuracy = auroc(self.prediction, self.label.int())
        specificity = torchmetrics.Specificity()
        spec = specificity(self.prediction.to('cpu'), self.label.int().to('cpu'))
        class_out = {'fpr': fpr, 'tpr': tpr, 'auc': auc_value, 'accuracy': accuracy, 'specificity': spec}
        return class_out

    def plot_AUROC(self, tpr, fpr):
        fig = plt.figure()
        # lw = 2
        # plt.plot(fpr.cpu(), tpr.cpu(), color='darkorange', lw=lw)
        plt.plot(fpr.cpu(), tpr.cpu(), color='darkorange')
        plt.title('ROC curve')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive rate')
        plt.show()
        return fig
