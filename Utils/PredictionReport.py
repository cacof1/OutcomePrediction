import numpy as np
import matplotlib.pyplot as plt
import torchvision
from sksurv.metrics import cumulative_dynamic_auc
import torch
from torch import nn
from sksurv.metrics import concordance_index_censored
import torchmetrics
from torch import Tensor


def r2_index(prediction, label):
    loss = nn.MSELoss()
    MSE = loss(prediction, label)
    SSres = MSE * label.shape[0]
    SStotal = torch.sum(torch.square(label - torch.mean(label)))
    r2 = 1 - SSres / SStotal
    return r2


def c_index(prediction, label):
    event_indicator = torch.ones(label.shape, dtype=torch.bool)
    # print('prediction:', prediction)
    risk = 1 / prediction.squeeze()
    cindex = concordance_index_censored(event_indicator.cpu().detach().numpy(),
                                        event_time=label.cpu().detach().numpy(),
                                        estimate=risk.cpu().detach().numpy())
    return cindex


def generate_report(img):
    # img_batch = batch[0]['Anatomy']
    img_batch = img.view(img.shape[0] * img.shape[1], *[1, img.shape[2], img.shape[3]])
    grid = torchvision.utils.make_grid(img_batch)
    return grid

#This function has a issue, needs improve

def generate_cumulative_dynamic_auc(y_train, prediction, y_test):
    # va_times = label_range
    risk_score = 1 / prediction
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


def classification_matrix(prediction, label):
    roc = torchmetrics.ROC()
    fpr, tpr, _ = roc(prediction, label)
    auc = torchmetrics.AUC(reorder=True)
    auc_value = auc(prediction, label)
    auroc = torchmetrics.AUROC()
    accuracy = auroc(prediction, label.int())
    specificity = torchmetrics.Specificity()
    spec = specificity(prediction.to('cpu'), label.int().to('cpu'))
    class_out = {'fpr': fpr, 'tpr': tpr, 'auc': auc_value, 'accuracy': accuracy, 'specificity': spec}
    return class_out


def plot_AUROC(tpr, fpr):
    fig = plt.figure()
    # lw = 2
    # plt.plot(fpr.cpu(), tpr.cpu(), color='darkorange', lw=lw)
    plt.plot(fpr.cpu(), tpr.cpu(), color='darkorange')
    plt.title('ROC curve')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive rate')
    plt.show()
    return fig


class PredictionReport():

    def __init__(self, config):
        super().__init__()
        self.config = config

    # def forward(self, log, prediction: Tensor, label: Tensor) -> Tensor:
    #
    #     return None

    def report_step(self, log, prediction, label):
        if self.config['MODEL']['Prediction_type'] == 'Regression':
            if 'cindex' in self.config['REPORT']['matrix']:
                cindex = c_index(prediction, label)
                log('train_cindex', cindex[0], on_step=False, on_epoch=True)
            if 'r2' in self.config['REPORT']['matrix']:
                r2 = r2_index(prediction, label)
                log('train_r2', r2, on_step=False, on_epoch=True)
        if self.config['MODEL']['Prediction_type'] == 'Classification':
            classification_out = classification_matrix(prediction.squeeze(), label)
            if 'AUROC' in self.config['REPORT']['matrix']:
                log('train_AUROC', classification_out['accuracy'], on_step=False,on_epoch=True)

    def validation_epoch(self, log, logger, prediction, label, train_label, validation_step_outputs, current_epoch):
        if self.config['MODEL']['Prediction_type'] == 'Regression':
            if 'cindex' in self.config['REPORT']['matrix']:
                cindex = c_index(prediction, label)
                log('validation_cindex', cindex[0])
            if 'r2' in self.config['REPORT']['matrix']:
                r2 = r2_index(prediction, label)
                log('validation_r2', r2)
            if 'AUC' in self.config['REPORT']['matrix']:
                fig = generate_cumulative_dynamic_auc(train_label, prediction, label)
                logger.experiment.add_figure("AUC", fig, current_epoch)
            if 'WorstCase' in self.config['REPORT']['matrix']:
                worst_record = self.worst_case_show(validation_step_outputs)
                log('worst_AE', worst_record['worst_AE'])
                if 'Anatomy' in self.config['DATA']['module']:
                    grid_img = generate_report(worst_record['worst_img'])
                    logger.experiment.add_image('validate_worst_case_img', grid_img, current_epoch)
                if 'Dose' in self.config['DATA']['module']:
                    grid_dose = generate_report(worst_record['worst_dose'])
                    logger.experiment.add_image('validate_worst_case_dose', grid_dose, current_epoch)

        if self.config['MODEL']['Prediction_type'] == 'Classification':
            classification_out = classification_matrix(prediction.squeeze(), label)
            if 'AUC' in self.config['REPORT']['matrix']:
                fig = plot_AUROC(classification_out['tpr'], classification_out['fpr'])
                logger.experiment.add_figure("AUC", fig, current_epoch)
            if 'Specificity' in self.config['REPORT']['matrix']:
                log('validation_specificity', classification_out['specificity'])
            if 'AUROC' in self.config['REPORT']['matrix']:
                log('validation_AUROC', classification_out['accuracy'])

    def worst_case_show(self, validation_step_outputs):
        out = {}
        worst_AE = 0
        for i, data in enumerate(validation_step_outputs):
            loss = data['MAE']
            idx = torch.argmax(loss)
            if loss[idx] > worst_AE:
                if 'Anatomy' in self.config['DATA']['module']:
                    worst_img = data['img'][idx]
                if 'Dose' in self.config['DATA']['module']:
                    worst_dose = data['dose'][idx]
                worst_AE = loss[idx]
        out['worst_AE'] = worst_AE
        if 'Anatomy' in self.config['DATA']['module']:
            out['worst_img'] = worst_img
        if 'Dose' in self.config['DATA']['module']:
            out['worst_dose'] = worst_dose
        return out
