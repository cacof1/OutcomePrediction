import segmentation_models_pytorch as smp
import matplotlib.pyplot as plt
from pytorch_lightning import LightningDataModule, LightningModule
import numpy as np
import torch
from collections import Counter
import torchvision
from torchvision import datasets, models, transforms
from torchvision import transforms
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from pytorch_lightning import LightningDataModule, LightningModule, Trainer,seed_everything
from torchsummary import summary
import sys
import torchio as tio
import sklearn
from pytorch_lightning import loggers as pl_loggers
import torchmetrics

## Module - Dataloaders
from Dataloader.Dataloader import DataModule, DataGenerator, LoadSortDataLabel

## Model
class Classifier2D(LightningModule):
    def __init__(self):
        super().__init__()
        self.n_classes = 1
        self.backbone = models.resnet50(pretrained=True) 
        self.model= torch.nn.Sequential(
            self.unet_model,
            torch.nn.LazyLinear(128),
            torch.nn.LazyLinear(self.n_classes)            
        )
        summary(self.model.to('cuda'), (2,160,160,40))
        self.accuracy = torchmetrics.AUC(reorder=True)
        self.loss_fcn = torch.nn.BCEWithLogitsLoss()

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch,batch_idx):
        image,label = batch
        prediction  = self.forward(image)
        loss = self.loss_fcn(prediction.squeeze(), label)
        self.log("loss", loss)
        return {"loss":loss,"prediction":prediction.squeeze(),"label":label}

    def validation_step(self, batch,batch_idx):
        image,label = batch
        prediction  = self.forward(image)
        loss = self.loss_fcn(prediction.squeeze(), label)
        return {"loss":loss,"prediction":prediction.squeeze(),"label":label}
        
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)
        return [optimizer], [scheduler]

if __name__ == "__main__":

## Main
train_transform = tio.Compose([
    tio.RandomAffine(),
    # tio.RescaleIntensity(out_min_max=(0, 1))
])

val_transform = tio.Compose([
    tio.RandomAffine(),
    # tio.RescaleIntensity(out_min_max=(0, 1))
])
callbacks = [
    ModelCheckpoint(
        dirpath='./',
        monitor='val_loss',
        filename="model_DeepSurv",#.{epoch:02d}-{val_loss:.2f}.h5",                                                                                                                                                                                                                
        save_top_k=1,
        mode='min'),
    EarlyStopping(monitor='val_loss')
]

data_file    = np.load(sys.argv[1])
label_file   = sys.argv[2]
label_name   = sys.argv[3]

data,label   = LoadSortDataLabel(label_name, label_file, data_file)
trainer      = Trainer(gpus=1, max_epochs=20)#,callbacks=callbacks)
model        = DeepSurv()
dataloader   = DataModule(data, label, train_transform = train_transform, val_transform = val_transform, batch_size=4, inference=False)
trainer.fit(model, dataloader)
