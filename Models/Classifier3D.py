import matplotlib.pyplot as plt
import torch
from pytorch_lightning import LightningModule
from torchinfo import summary
from torch import nn
import torchmetrics
from monai.networks import blocks, nets
from Models.UnetEncoder import UnetEncoder

## Model
class Classifier3D(LightningModule):
    def __init__(self, config):
        super().__init__()

        backbone_name = config['MODEL']['Backbone']
        parameters = config['MODEL_PARAMETERS']

        if backbone_name == '3DUnet':
            self.backbone = UnetEncoder(**parameters)
        else:
            model_str = 'nets.' + backbone_name + '(**parameters)'
            loaded_model = eval(model_str)
            # only use network for features
            self.backbone = loaded_model.features

        self.model = torch.nn.Sequential(
            self.backbone,
            torch.nn.Flatten(),
        )
        self.model.apply(self.weights_init)
        summary(self.model.to('cuda'), (config['MODEL']['batch_size'], 1, *config['DATA']['dim']),
                col_names=["input_size", "output_size"], depth=5)
        self.accuracy = torchmetrics.AUC(reorder=True)
        self.loss_fcn = torch.nn.BCEWithLogitsLoss()

    def forward(self, x):
        return self.model(x)

    def weights_init(self, m):
        if isinstance(m, nn.Conv3d) or isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight.data)



