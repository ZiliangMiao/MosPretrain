import click
import yaml
import copy
import os
import numpy as np
import torch
from pytorch_lightning import Trainer
from pytorch_lightning import loggers as pl_loggers
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint

import mos4d.models.models as models
import mos4d.models.nusc_models as nusc_models
import mos4d.datasets.datasets as datasets
import mos4d.datasets.nusc_dataset as nusc_dataset

@click.command()
### Add your options here
@click.option(
    "--config",
    "-c",
    type=str,
    help="path to the config file (.yaml)",
    default="./config/gene_test_config.yaml",
)
@click.option(
    "--finetune",
    type=str,
    help="path to pretrained weights (.ckpt). Use this flag if you just want to load the weights from the checkpoint file without resuming training.",
    default=None,
)
@click.option(
    "--resume",
    type=str,
    help="path to checkpoint file (.ckpt) to resume training.",
    default=None,
)
def main(config, finetune, resume):
    if resume is None:  # training from scratch or finetuning
        cfg = yaml.safe_load(open(config))
    else:  # resume training
        cfg = torch.load(resume)["hyper_parameters"]

    dataset = cfg["TRAIN"]["DATASET"]
    if finetune is None:  # initialize the model with cfg
        if dataset == "NUSC":
            model = nusc_models.MOSNet(cfg)
        else:
            model = models.MOSNet(cfg)
    else:  # load the pretrained model
        if dataset == "NUSC":
            model = nusc_models.MOSNet.load_from_checkpoint(finetune, hparams=cfg)
        else:
            model = models.MOSNet.load_from_checkpoint(finetune, hparams=cfg)

    # Add callbacks
    lr_monitor = LearningRateMonitor(logging_interval="step")
    checkpoint_saver = ModelCheckpoint(
        monitor="val_moving_iou_step0",
        filename=cfg["EXPT"]["ID"] + "_{epoch:03d}_{val_moving_iou_step0:.3f}",
        mode="max",
        save_last=True,
    )

    # Logger
    log_dir = "./logs"
    os.makedirs(log_dir, exist_ok=True)
    tb_logger = pl_loggers.TensorBoardLogger(
        log_dir, name=cfg["EXPT"]["ID"], default_hp_metric=False
    )

    # Setup trainer
    trainer = Trainer(
        accelerator="gpu",
        strategy="ddp_spawn",
        devices=cfg["TRAIN"]["NUM_DEVICES"],
        logger=tb_logger,
        max_epochs=cfg["TRAIN"]["MAX_EPOCH"],
        accumulate_grad_batches=cfg["TRAIN"]["ACC_BATCHES"],  # accumulate batches, default=1
        callbacks=[lr_monitor, checkpoint_saver],
    )

    # load data from different datasets
    if dataset == "NUSC":
        data = nusc_dataset.NuscSequentialModule(cfg)
        data.setup()
    else:  # KITTI-like datasets
        data = datasets.KittiSequentialModule(cfg)
        data.setup()

    train_dataloader = data.train_dataloader()
    val_dataloader = data.val_dataloader()
    # Train!
    trainer.fit(model, train_dataloader, val_dataloader, ckpt_path=resume)
    # self, self._fit_impl, model, train_dataloaders, val_dataloaders, datamodule, ckpt_path

if __name__ == "__main__":
    main()
