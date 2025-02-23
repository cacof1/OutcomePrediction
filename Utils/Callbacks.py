from pytorch_lightning.callbacks import Callback, ModelCheckpoint, EarlyStopping, LearningRateMonitor
from pytorch_lightning import seed_everything
import pytorch_lightning as pl
import torch
import os
from Utils.Transformations import transform_pipeline


class StopCallback(pl.callbacks.Callback):
    def __init__(self, config: dict, on_validation_epoch_end: bool = True, rd: int = None):
        super().__init__()
        self.config = config
        self.local_epochs = config['RUN']['local_epochs']
        assert self.local_epochs >= 1
        self.on_val_epoch_end = on_validation_epoch_end
        self.rd = rd
        self.start_epoch = 0
        self.current_epoch = 0
        self.should_stop = False

    def on_train_epoch_start(self, trainer, model):
        self.my_function(trainer)
        seed_everything(self.rd, workers=True)

    def on_validation_epoch_end(self, trainer, pl_module):
        if self.on_val_epoch_end and not trainer.sanity_checking:
            self.check_stop()

    def on_train_epoch_end(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None:
        if not self.on_val_epoch_end and not trainer.sanity_checking:
            self.check_stop()

    def check_stop(self):
        self.current_epoch += 1
        if self.current_epoch - self.start_epoch == self.local_epochs:
            self.start_epoch = self.current_epoch
            self.should_stop = True

    def my_function(self, trainer):
        if trainer.train_dataloader is not None:
            trainer.train_dataloader.dataset.transform, __ = transform_pipeline(self.config, self.rd)


class ReseedCallback(Callback):
    def __init__(self, config, rd):
        self.config = config
        self.rd = rd

    def on_train_epoch_start(self, trainer, model):
        current_epoch = trainer.current_epoch
        # trainer.datamodule = self.dataloader
        # trainer.train_dataloader = self.dataloader.train_dataloader()
        # generator_2 = torch.Generator()
        # generator_2.manual_seed(int(self.rd))
        self.my_function(trainer)
        seed_everything(self.rd, workers=True)
        # trainer.train_dataloader.generator.manual_seed(int(self.rd))
        print('here')
        # ndarrays = get_parameters(model)
        # print(f'PARAMETERS SHAPE: {ndarrays[0][0][0]}')

    def my_function(self, trainer):
        if trainer.train_dataloader is not None:
            # print('Seeding transforms')
            # print(trainer.train_dataloader.dataset.transform.transforms[1].R._bit_generator.state['state']['key'][:2])
            # for j in [1, 2, 3, 4]:
            #     trainer.train_dataloader.dataset.transform.transforms[j].set_random_state(seed=self.rd)
            trainer.train_dataloader.dataset.transform, __ = transform_pipeline(self.config, self.rd)
            # print(trainer.train_dataloader.dataset.transform.transforms[1].R._bit_generator.state['state']['key'][:2])


class BestModelSaver(object):
    def __init__(self, pos, to_compare='current', filename='best_model'):
        self.score = None
        self.pos = pos
        self.to_compare = to_compare
        self.filename_init = filename
        self.full_filename = None

    def __call__(self, trainer, model, m_round):
        best_model_score = trainer.checkpoint_callbacks[self.pos].best_model_score.item()
        current_model_score = trainer.callback_metrics['val_loss'].item()
        mode = trainer.checkpoint_callbacks[self.pos].mode
        score = current_model_score if self.to_compare == 'current' else best_model_score
        print(f'SCORE: {score}')
        if (self.score is None) or (mode == 'min' and score < self.score) or (mode == 'max' and score > self.score):
            if self.score is not None:
                os.remove(self.full_filename)
            self.score = score
            save_dir = trainer.checkpoint_callbacks[self.pos].dirpath
            self.full_filename = f"{save_dir}/{self.filename_init}_round_{m_round}.ckpt"
            torch.save(model.state_dict(), self.full_filename)


def get_callbacks(config):
    lr_monitor = LearningRateMonitor(logging_interval='step')
    checkpoint_callback = ModelCheckpoint(
        monitor="val_loss",
        filename=f"{{model_name}}-epoch{{epoch:02d}}_lowest_val_loss",
        save_top_k=1,
        mode='min')
    checkpoint_callback_main_target = ModelCheckpoint(
        monitor="validation_mae_epoch" if config['MODEL']['modes'][0] == 'regression' else 'validation_auc_epoch',
        filename=f"{{model_name}}-epoch{{epoch:02d}}_best_main_target",
        save_top_k=1,
        mode='min' if config['MODEL']['modes'][0] == 'regression' else 'max')
    checkpoint_callback_last = ModelCheckpoint(
        save_last=True,
        save_top_k=0,
        save_on_train_epoch_end=False)
    return [lr_monitor, checkpoint_callback, checkpoint_callback_main_target, checkpoint_callback_last]
