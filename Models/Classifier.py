import matplotlib.pyplot as plt
import torch
import torchvision.models as models
from pytorch_lightning import LightningModule
from torch import nn
import torchmetrics
from monai.networks import blocks, nets
## Model
class Classifier(LightningModule):
    def __init__(self, config, module_str):
        super().__init__()

        model = config['MODEL']['Backbone']
        parameters = config['MODEL_PARAMETERS']
        # only use network for features
        if model == 'torchvision':
            model_name = config['MODEL'][module_str + '_model_name']
            model_str = 'models.' + model_name + '(pretrained=True)'
            self.backbone = eval(model_str)
        else:
            model_str = 'nets.' + model + '(**parameters)'
            self.backbone = eval(model_str)

        if 'out_channels' in config['MODEL_PARAMETERS'].keys():
            self.out_feat = config['MODEL_PARAMETERS']['out_channels']
        elif 'num_classes' in config['MODEL_PARAMETERS'].keys():
            self.out_feat = config['MODEL_PARAMETERS']['num_classes']


    def forward(self, x):
        return self.backbone(x)

    def weights_init(self, m):
        if isinstance(m, nn.Conv3d) or isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight.data)




