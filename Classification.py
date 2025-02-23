import torch
import argparse
from torch import nn
import torch.distributed as dist
from pytorch_lightning import Trainer, seed_everything
import sys
from copy import deepcopy
from pytorch_lightning.strategies import DDPStrategy
import numpy as np
from typing import Literal
from collections import OrderedDict
from monai.utils import set_determinism
from pytorch_lightning.profilers import SimpleProfiler, AdvancedProfiler, PyTorchProfiler

## Module - Dataloaders
from DataGenerator.DataGenerator import DataModule
from Models.Classifier import Classifier
from Models.Linear import Linear
from Models.MixModel import MixModel
from Utils.DataExtraction import create_subject_list
from Utils.Transformations import transform_pipeline
from Utils.Callbacks import get_callbacks
from Utils.ProcessResults import inverse_transform_target, get_results_table, get_train_val_test_tab

## Main
import toml
from pathlib import Path
from pytorch_lightning.loggers import TensorBoardLogger, CSVLogger


def is_rank_zero():
    return not dist.is_initialized() or dist.get_rank() == 0

def threshold_at_one(x):
    return x > 2.1

def load_config():
    config = toml.load(sys.argv[1])
    return config


def update_nested_config(config, updates):
    """Update nested dictionary using dotted keys, e.g., RUN.random_state=42."""
    for k, v in updates.items():
        keys = k.split(".")
        d = config
        for key in keys[:-1]:
            d = d.setdefault(key, {})
        try:
            # Attempt to parse numeric types
            d[keys[-1]] = eval(v, {"__builtins__": None}, {})
        except:
            d[keys[-1]] = v


class GetDataLoader(object):
    def __init__(self, subject_list, **kwargs):
        self.subject_list = subject_list
        self.kwargs = kwargs

    def __call__(self):
        dataloader = DataModule(self.subject_list, **self.kwargs)
        return dataloader


class GetTrainer(object):
    def __init__(self, **kwargs):
        self.kwargs = kwargs

    def __call__(self):
        trainer = Trainer(**self.kwargs)
        return trainer


def build_model(config):
    module_dict = nn.ModuleDict()
    clinical_cols = config['DATA']['clinical_cols']
    if config['DATA']['multichannel']:  ## Single-Model Multichannel learning
        if config['MODALITY'].keys():
            module_dict['Image'] = Classifier(config, 'Image')
    else:
        for key in config['MODALITY'].keys():  # Multi-Model Single Channel learning
            if config['MODALITY'][key]:
                module_dict[key] = Classifier(config, key)
                if 'RTSTRUCT' in module_dict.keys():
                    module_dict.pop('RTSTRUCT')

    if 'RECORDS' in config.keys() and config['RECORDS']['records']:
        module_dict['records'] = Linear(config, in_feat=len(clinical_cols),
                                        out_feat=config['MODEL']['linear_out'])

    return module_dict


def get_logger(logger_folder, model_name, version=None):
    tb_logger = TensorBoardLogger(save_dir=logger_folder, name=model_name, version=version)
    csv_logger = CSVLogger(save_dir=logger_folder, name=model_name, version=tb_logger.version)
    return [tb_logger, csv_logger]


def get_parameters(model):
    return [val.cpu().numpy() for _, val in model.state_dict().items()]


def set_parameters(model, parameters):
    params_dict = zip(model.state_dict().keys(), parameters)
    state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
    model.load_state_dict(state_dict, strict=True)


def infer_and_save_results(ckpt_dict, log_dir, config, model, trainer, dataloader,
                           ckpt_type: Literal['ckpt', 's_dict'] = 'ckpt', module_dict=None):
    assert ckpt_type in ['ckpt', 's_dict']
    for ckpt_key in ckpt_dict:
        checkpoint_path = ckpt_dict[ckpt_key]
        h_param_path = Path(log_dir) / 'hparams.yaml'
        if ckpt_type == 'ckpt':
            best_model = model.__class__.load_from_checkpoint(checkpoint_path, hparams_file=h_param_path,
                                                              module_dict=module_dict, config=config)
        elif ckpt_type == 's_dict':
            best_model_state_dict = torch.load(checkpoint_path)
            best_model = deepcopy(model)
            best_model.load_state_dict(best_model_state_dict)
        results = trainer.predict(best_model, dataloader)
        results_table = get_results_table(results, dataloader, config)
        results_table = inverse_transform_target(results_table, dataloader, config)
        results_table.to_csv(Path(log_dir) / f'results_{ckpt_key}.csv')


def fit(trainer_getter, dataloader_getter, model):
    trainer = trainer_getter()
    dataloader = dataloader_getter()
    trainer.fit(model, dataloader)


def main(config, rd):
    seed_everything(rd, workers=True)
    SubjectList = create_subject_list(config)
    SubjectList.to_csv(Path(config['DATA']['log_folder'])/'data_table.csv', index=False)
    clinical_cols = config['DATA']['clinical_cols']
    logger = get_logger(config['DATA']['log_folder'], config['DATA']['model_name'])
    train_transform, val_transform = transform_pipeline(config, rd)
    module_dict = build_model(config)
    model = MixModel(module_dict, config)
    # model.apply(model.weights_reset)
    # ndarrays = get_parameters(model)
    # print(f'PARAMETERS SHAPE: {ndarrays[0][0]}')
    dataloader_getter = GetDataLoader(subject_list=SubjectList,
                                      config=config,
                                      keys=config['MODALITY'].keys(),
                                      train_transform=train_transform,
                                      val_transform=val_transform,
                                      clinical_cols=clinical_cols,
                                      rd=np.int16(rd),
                                      rd_worker=np.int16(config['RUN']['random_state_dataloader']),
                                      inference=False,
                                      num_workers=10,
                                      prefetch_factor=5,
                                      )
    dataloader = dataloader_getter()
    callbacks = get_callbacks(config)
    trainer_getter = GetTrainer(accelerator="gpu",
                                devices=config['RUN']['devices'],
                                # strategy=DDPStrategy(find_unused_parameters=True),
                                max_epochs=config['RUN']['max_epochs'],
                                logger=logger,
                                log_every_n_steps=1,
                                callbacks=callbacks,
                                # benchmark=True,
                                deterministic=True,  # added
                                # profiler=PyTorchProfiler(
                                #     on_trace_ready=torch.profiler.tensorboard_trace_handler("lightning_logs"),
                                #     record_shapes=True
                                )

    # Ensure the directory exists only on rank 0
    if is_rank_zero():
        if not Path(logger[0].log_dir).exists():
            Path(logger[0].log_dir).mkdir(parents=True)
        patient_list = get_train_val_test_tab(dataloader, rd, config)
        patient_list.to_csv(logger[0].log_dir + '/patient_list.csv')

        with open(logger[0].log_dir + "/Config.ini", "w+") as config_file:
            toml.dump(config, config_file)
            config_file.write("Train transform:\n")
            config_file.write(str(train_transform))
            config_file.write("Val/Test transform:\n")
            config_file.write(str(val_transform))

    # if config['MODEL']['model_path']:
    #     h_param_path = Path(config['MODEL']['model_path']) / 'hparams.yml'
    #     model = MixModel.load_from_checkpoint(config['MODEL']['model_path'], hparams_file=h_param_path,
    #                                           module_dict=module_dict, config=config)
    # trainer.fit(model, dataloader)
    fit(trainer_getter, dataloader_getter, model)
    ckpt_dict = {'lowest_val_loss': list((Path(logger[0].log_dir) / 'checkpoints').glob('*lowest_val_loss.ckpt'))[-1],
                 'best_main_target': list((Path(logger[0].log_dir) / 'checkpoints').glob('*best_main_target.ckpt'))[-1]}
    # ckpt_dict = {'lowest_val_loss': list((Path(log_dir) / 'checkpoints').glob('*best_model_round*'))[-1]}
    infer_and_save_results(ckpt_dict, logger[0].log_dir, config, model, trainer_getter(), dataloader,
                           module_dict=module_dict)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run training with optional config overrides.")
    parser.add_argument('--config', type=str, default="OPConfigurationSurvivalPredictionEfficientNetFedComparison.ini", help="Path to config file")
    parser.add_argument('--set', nargs='*', help="Override config values, e.g., --set RUN.random_state=42 DATA.model_name=CustomModel")
    args = parser.parse_args()

    config = toml.load(args.config)

    if args.set:
        # Convert list of key=value pairs into a dictionary
        overrides = dict(kv.split("=", 1) for kv in args.set)
        update_nested_config(config, overrides)

    y = range(config['RUN']['bootstrap_n'])
    if 'random_state' in config['RUN']:
        np.random.seed(config['RUN']['random_state'])

    random_seed_list = np.random.randint(10000, size=len(y))
    print("Random seeds:", random_seed_list)

    for i in y:
        main(config, random_seed_list[i])

    print("Random seeds:", random_seed_list)
