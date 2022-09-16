import pytorch_lightning as pl
import torch
import torch.nn as nn
from torchvision import models
from monai.networks import nets
from Models.UnetEncoder import UnetEncoder

class Classifier2D(pl.LightningModule):
    def __init__(self, config, module_str):
        super().__init__()
        self.config = config
        self.module_str = module_str
        model = config['MODEL'][module_str + '_Backbone']
        parameters = config[module_str + '_MODEL_PARAMETERS']
        if model == 'Vit':
            self.backbone = models.vit_b_16(pretrained=True).eval()
        elif model == 'Unet':
            self.backbone = UnetEncoder(**parameters)
        else:
            model_str = 'nets.' + model + '(**parameters)'
            self.backbone = eval(model_str)
            # self.backbone = loaded_model.features

        # self.backbone.eval()
        # for param in self.backbone.parameters():
        #     param.requires_grad = False

        # self.pool1 = nn.AdaptiveMaxPool2d((1, 1))
        self.pool2 = nn.AdaptiveAvgPool1d(256)

        layers = list(self.backbone.children())[:-1]
        self.feature_extractor = nn.Sequential(*layers)

        # self.feature_extractor.eval()
        # for param in self.feature_extractor[0:int(len(self.feature_extractor)/2)].parameters():
        #     param.requires_grad = False

    def convert2d(self, x):
        if self.config[self.module_str +'_MODEL_PARAMETERS']['in_channels'] == 3:
            x = x.repeat(1, 3, 1, 1)
        mid_layer = int(x.shape[0] / 2)
        x = x[mid_layer - 2:mid_layer + 2, :, :, :]
        # x = x[mid_layer, :, :, :].unsqueeze(0)
        features = self.feature_extractor(x)
        '''
        ft = self.addtional(features)
        ft = ft.transpose(0, 3)
        ft = ft.transpose(1, 2)
        f1 = self.addtional2(ft)
        features = f1.transpose(1, 2)
        '''
        features = features.flatten(1)
        # features = self.pool2(features)
        features = features.unsqueeze(0)
        features = features.unsqueeze(1)
        # features = features.permute(2, 3, 0, 1)
        return features

    def forward(self, x):
        features = torch.cat([self.convert2d(b.transpose(0, 1)) for i, b in enumerate(x)], dim=0)
        features = features.flatten(start_dim=1)
        return features

    def weights_init(self, m):
        if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight.data)


