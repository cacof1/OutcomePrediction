import monai.networks.nets
import torch
from torch import nn
from monai.networks import blocks, nets
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from pytorch_lightning import LightningDataModule, LightningModule, Trainer, seed_everything


class UnetEDcoder(LightningModule):
    def __init__(self, config):
        super().__init__()
        parameters = config['MODEL_PARAMETERS']
        model = eval('monai.networks.nets.BasicUNet(**parameters)')

        self.encoder = nn.Sequential(
            model.conv_0,
            model.down_1,
            model.down_2,
            model.down_3,
            model.down_4,
        )
        self.decoder = nn.Sequential(
            model.upcat_4,
            model.upcat_3,
            model.upcat_2,
            model.upcat_1,
            model.final_conv,
        )

    def forward(self, x):
        x0 = self.encoder[0](x)
        x1 = self.encoder[1](x0)
        x2 = self.encoder[2](x1)
        x3 = self.encoder[3](x2)
        x4 = self.encoder[4](x3)

        u4 = self.decoder[0](x4, x3)
        u3 = self.decoder[1](u4, x2)
        u2 = self.decoder[2](u3, x1)
        u1 = self.decoder[3](u2, x0)
        out = self.decoder[4](u1)
        return out
