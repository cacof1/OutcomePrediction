import argparse
from typing import Dict, Optional, Union
import os
from pytorch_lightning.loggers.base import rank_zero_experiment
from torch.utils.tensorboard import SummaryWriter
import numpy as np
from lifelines.utils import concordance_index
import matplotlib.pyplot as plt
import matplotlib
import torchvision
from pytorch_lightning.loggers import LightningLoggerBase
from sksurv.metrics import cumulative_dynamic_auc
import torch
from torch import nn
from sksurv.metrics import concordance_index_censored
import torchmetrics
from pytorch_lightning.utilities import rank_zero_only
from pytorch_lightning.core.saving import save_hparams_to_yaml
from pytorch_lightning.utilities.cloud_io import get_filesystem
import logging
from pytorch_lightning.loggers import TensorBoardLogger
from torchmetrics import ConfusionMatrix


class PredictionReports(TensorBoardLogger):
    def __init__(self, config,
                 save_dir: str,
                 name: Optional[str] = "default",
                 version: Optional[Union[int, str]] = None,
                 log_graph: bool = False,
                 default_hp_metric: bool = True,
                 prefix: str = "",
                 sub_dir: Optional[str] = None,
                 **kwargs, ):

        super().__init__(save_dir, name, version, log_graph, default_hp_metric, prefix, sub_dir, **kwargs)
        self.config = config

    def log_hyperparams(self, params: argparse.Namespace, *args, **kwargs):
        pass

    # @property
    # def version(self):
    #     description = ''
    #     for i, param in enumerate(self.config['CRITERIA'].keys()):
    #         clinical_criteria = str(self.config['CRITERIA'][param])
    #         if i > 0:
    #             description = description + '_'
    #         description = description + param + '+' + '+'.join(clinical_criteria)
    #     # Return the experiment version, int or str.
    #
    #     sub_str = description + '_' + 'modalities' + '+' + '+'.join(self.config['MODALITY'].keys())
    #     description = description + '_' + 'modalities' + '+' + '+'.join(self.config['MODALITY'].keys()) + '_' + str(self._get_next_version(sub_str))
    #     return description
    #
    # def _get_next_version(self, sub_str):
    #     root_dir = self.root_dir
    #     listdir_info = self._fs.listdir(root_dir)
    #     existing_versions = []
    #     for listing in listdir_info:
    #         d = listing["name"]
    #         bn = os.path.basename(d)
    #         if self._fs.isdir(d) and bn.startswith(sub_str):
    #             dir_ver = bn.split("_")[-1]
    #             existing_versions.append(int(dir_ver))
    #     if len(existing_versions) == 0:
    #         return 0
    #
    #     return max(existing_versions) + 1

    def log_metrics(self, metrics: Dict[str, float], step: Optional[int] = None):
        for k, v in metrics.items():
            if isinstance(v, torch.Tensor):
                v = v.item()
            self.experiment.add_scalar(k, v, step)

    def log_image(self, img, text, current_epoch=None):
        img = img.transpose(2, 0)
        img_batch = img.view(img.shape[0], *[1, img.shape[1], img.shape[2]])
        grid = torchvision.utils.make_grid(img_batch, normalize=True)
        self.experiment.add_image(text, grid, current_epoch)
        return grid

    def log_text(self) -> None:
        configurations = 'The modules included are ' + str(self.config['MODALITY'].keys())
        # configurations = 'The img_dim is ' + str(self.config['DATA']['dim']) + ' and the modules included are ' +
        # str(self.config['MODALITY'].keys())
        self.experiment.add_text('configurations:', configurations)

    def regression_matrix(self, prediction, censor_status, label, prefix):  ## matrix should be metrics
        r_out = {}
        if 'cindex' in self.config['CHECKPOINT']['matrix']:
            cindex = concordance_index(label.cpu().detach().numpy(), prediction.cpu().detach().numpy(),
                                       censor_status.cpu().detach().numpy())
            r_out[prefix + 'cindex'] = cindex
        if 'r2' in self.config['CHECKPOINT']['matrix']:
            r2 = r2_index(prediction.detach(), label.detach())
            r_out[prefix + 'r2'] = r2
        return r_out

    def classification_matrix(self, prediction, label, prefix):
        c_out = {}
        cm = ConfusionMatrix(num_classes=2)
        bcm = cm(prediction.round().to('cpu'), label.int().to('cpu'))
        tn = bcm[0][0]
        tp = bcm[1][1]
        fp = bcm[0][1]
        fn = bcm[1][0]
        if 'ROC' in self.config['CHECKPOINT']['matrix']:
            auroc = torchmetrics.AUROC()
            accuracy = auroc(prediction, label.int())
            c_out[prefix + 'roc'] = accuracy
        if 'Specificity' in self.config['CHECKPOINT']['matrix']:
            spec = tn / (tn + fp)
            c_out[prefix + 'specificity'] = spec

        if 'Sensitivity' in self.config['CHECKPOINT']['matrix']:
            sensitivity = tp / (tp + fn)
            c_out[prefix + 'sensitivity'] = sensitivity

        if 'Accuracy' in self.config['CHECKPOINT']['matrix']:
            acc = bcm.diag().sum() / bcm.sum()
            c_out[prefix + 'accuracy'] = acc

        if 'Precision' in self.config['CHECKPOINT']['matrix']:
            precision = tp / (tp + fp)
            c_out[prefix + 'precision'] = precision
        return c_out

    # def generate_cumulative_dynamic_auc(self, prediction, label, current_epoch, prefix) -> None:
    #     # this function has issues
    #     risk_score = 1 / prediction
    #     # risk_score = prediction
    #     # va_times = np.arange(int(label.cpu().min()) + 1, label.cpu().max(), 1)
    #     va_times = np.percentile(label.cpu(), np.linspace(5, 81, 20))
    #     dtypes = np.dtype([('event', np.bool_), ('time', np.float)])
    #     construct_test = np.ndarray(shape=(len(label),), dtype=dtypes)
    #     for i in range(len(label)):
    #         construct_test[i] = (True, label[i].cpu().numpy())
    #
    #     cph_auc, cph_mean_auc = cumulative_dynamic_auc(
    #         construct_test, construct_test, risk_score.cpu().squeeze(), va_times
    #     )
    #
    #     fig = plt.figure()
    #     plt.plot(va_times, cph_auc, marker="o")
    #     plt.axhline(cph_mean_auc, linestyle="--")
    #     plt.ylim([0, 1])
    #     plt.xlabel("survival months")
    #     plt.ylabel("time-dependent AUC")
    #     plt.grid(True)
    #     self.experiment.add_figure(prefix + "AUC", fig, current_epoch)
    #     plt.close(fig)

    def plot_AUROC(self, prediction, label, prefix, current_epoch=None) -> None:
        roc = torchmetrics.ROC()
        fpr, tpr, _ = roc(prediction, label)
        fig = plt.figure()
        # lw = 2
        # plt.plot(fpr.cpu(), tpr.cpu(), color='darkorange', lw=lw)
        plt.plot(fpr.cpu(), tpr.cpu(), color='darkorange')
        plt.title(prefix + '_roc_curve')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive rate')
        self.experiment.add_figure(prefix + "AUC", fig, current_epoch)
        plt.close(fig)

    def worst_case_show(self, validation_step_outputs, prefix):
        out = {}
        worst_AE = 0
        label = ''
        for i, data in enumerate(validation_step_outputs):
            loss = data['MAE']
            idx = torch.argmax(loss)
            if loss[idx] > worst_AE:
                if 'CT' in self.config['MODALITY'].keys():
                    worst_img = data['Image'][idx][0, :, :, :]
                if 'Dose' in self.config['MODALITY'].keys():
                    worst_dose = data['Image'][idx][1, :, :,
                                 :]  ### this index needs to be careful when adding pet image
                worst_AE = loss[idx]
                label = data['slabel'][idx]
        out[prefix + 'worst_AE'] = worst_AE
        if 'CT' in self.config['MODALITY'].keys():
            out[prefix + 'worst_img'] = worst_img
        if 'Dose' in self.config['MODALITY'].keys():
            out[prefix + 'worst_dose'] = worst_dose
        out[prefix + 'slabel'] = label
        return out

    def report_epoch(self, prediction, censor_status, label, validation_step_outputs,
                     current_epoch, prefix) -> None:
        if self.config['MODEL']['Prediction_type'] == 'Regression':
            regression_out = self.regression_matrix(prediction, censor_status, label, prefix)
            self.log_metrics(regression_out, current_epoch)

        if self.config['MODEL']['Prediction_type'] == 'Classification':
            classification_out = self.classification_matrix(prediction.squeeze(), label, prefix)
            self.log_metrics(classification_out, current_epoch)
            if 'AUROC' in self.config['CHECKPOINT']['matrix']:
                self.plot_AUROC(prediction.squeeze(), label, prefix, current_epoch)

        if 'WorstCase' in self.config['CHECKPOINT']['matrix']:
            worst_record = self.worst_case_show(validation_step_outputs, prefix)
            self.log_metrics({prefix + 'worst_AE': worst_record[prefix + 'worst_AE']}, current_epoch)
            self.log_metrics('worst_subject: ', str(worst_record[prefix + 'slabel']))
            if 'CT' in self.config['MODALITY'].keys():
                text = 'validate_worst_case_img'
                self.log_image(worst_record[prefix + 'worst_img'], text, current_epoch)
            if 'Dose' in self.config['MODALITY'].keys():
                text = 'validate_worst_case_dose'
                self.log_image(worst_record[prefix + 'worst_dose'], text, current_epoch)

    def report_test(self, config, outs, model, prediction_labels, validation_censor, validation_labels, prefix):
        if 'WorstCase' in config['CHECKPOINT']['matrix']:
            worst_record = self.worst_case_show(outs, prefix)
            self.experiment.add_text('worst_test_AE: ', str(worst_record[prefix + 'worst_AE']))
            self.experiment.add_text('worst_subject: ', str(worst_record[prefix + 'slabel']))
            if 'CT' in config['MODALITY'].keys():
                text = 'test_worst_case_img'
                self.log_image(worst_record[prefix + 'worst_img'], text)
            if 'Dose' in config['MODALITY'].keys():
                text = 'test_worst_case_dose'
                self.log_image(worst_record[prefix + 'worst_dose'], text)

        if config['MODEL']['Prediction_type'] == 'Regression':
            self.experiment.add_text('test loss: ', str(model.loss_fcn(prediction_labels, validation_labels)))
            regression_out = self.regression_matrix(prediction_labels, validation_censor, validation_labels, prefix)
            self.experiment.add_text('test_cindex: ', str(regression_out[prefix + 'cindex']))
            self.experiment.add_text('test_r2: ', str(regression_out[prefix + 'r2']))
            return regression_out

        if config['MODEL']['Prediction_type'] == 'Classification':
            classification_out = self.classification_matrix(prediction_labels.squeeze(), validation_labels, prefix)
            if 'AUROC' in config['CHECKPOINT']['matrix']:
                self.plot_AUROC(prediction_labels, validation_labels, prefix)
                self.experiment.add_text('test_AUROC: ', str(classification_out[prefix + 'roc']))
            if 'Specificity' in config['CHECKPOINT']['matrix']:
                self.experiment.add_text('Specificity:', str(classification_out[prefix + 'specificity']))
            if 'Sensitivity' in config['CHECKPOINT']['matrix']:
                self.experiment.add_text('Specificity:', str(classification_out[prefix + 'sensitivity']))
            if 'Accuracy' in config['CHECKPOINT']['matrix']:
                self.experiment.add_text('Specificity:', str(classification_out[prefix + 'accuracy']))
            if 'Precision' in config['CHECKPOINT']['matrix']:
                self.experiment.add_text('Specificity:', str(classification_out[prefix + 'precision']))
            if 'ROC' in config['CHECKPOINT']['matrix']:
                self.experiment.add_text('ROC:', str(classification_out[prefix + 'roc']))
            return classification_out


def r2_index(prediction, label):
    loss = nn.MSELoss(reduction='sum')
    SSres = loss(prediction, label)
    SStotal = torch.sum(torch.square(label - torch.mean(label)))
    r2 = 1 - SSres / SStotal
    return r2
