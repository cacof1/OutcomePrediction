import torch
import torch.nn as nn
import torch.optim as optim
import pytorch_lightning as pl
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
import numpy as np
import os


# Define CNN model with optional residual connections and dropout
class SimpleCNN(pl.LightningModule):
    def __init__(self, config, use_residual=False, use_dropout=False):
        super().__init__()
        self.save_hyperparameters()

        self.use_residual = use_residual
        self.use_dropout = use_dropout

        self.dropout = nn.Dropout(0.5) if use_dropout else nn.Identity()

        # Define the CNN as a Sequential block
        self.cnn = nn.Sequential(
            nn.Conv3d(config['DATA']['n_channel'], 16, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool3d(kernel_size=2, stride=2),

            nn.Conv3d(16, 32, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.MaxPool3d(kernel_size=2, stride=2),

            nn.Conv3d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.MaxPool3d(kernel_size=2, stride=2),
        )

        # Define the feed-forward block as a Sequential block
        self.feed_forward = nn.Sequential(
            nn.Linear(256 * (config['DATA']['dim'][0] // 64 + 1) *
                      (config['DATA']['dim'][1] // 64 + 1) *
                      (config['DATA']['dim'][2] // 32), config['MODEL']['classifier_in']),
            nn.ReLU(),
            self.dropout
        )

    def forward(self, x):
        identity = x
        x = self.cnn(x)

        if self.use_residual:
            x += identity  # Add residual connection if enabled (only works if dimensions match)

        x = x.view(x.size(0), -1)  # Flatten
        x = self.feed_forward(x)
        return x

    def training_step(self, batch, batch_idx):
        # x, y = batch
        # y_hat = self.forward(x)
        # loss = F.cross_entropy(y_hat, y)
        # self.log('train_loss', loss)
        # return loss
        return

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self.forward(x)
        loss = F.cross_entropy(y_hat, y)
        acc = (y_hat.argmax(dim=1) == y).float().mean()
        self.log('val_loss', loss, prog_bar=True)
        self.log('val_acc', acc, prog_bar=True)

    def configure_optimizers(self):
        return optim.Adam(self.parameters(), lr=self.hparams.learning_rate)
