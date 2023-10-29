import click
import yaml
import copy
import os
import numpy as np
import torch
from pytorch_lightning import Trainer
from pytorch_lightning import loggers as pl_loggers
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint

import mos4d.datasets.datasets as datasets
import mos4d.models.models as models

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
    "--weights",
    "-w",
    type=str,
    help="path to pretrained weights (.ckpt). Use this flag if you just want to load the weights from the checkpoint file without resuming training.",
    default=None,
)
@click.option(
    "--checkpoint",
    "-ckpt",
    type=str,
    help="path to checkpoint file (.ckpt) to resume training.",
    default=None,
)
def main(config, weights, checkpoint):
    # ckpt, default=None, training from scratch

    if checkpoint:
        cfg = torch.load(checkpoint)["hyper_parameters"]
    else:
        cfg = yaml.safe_load(open(config))

    # Load Model
    model = models.MOSNet(cfg)
    if weights is None:
        model = models.MOSNet(cfg)
    else:
        model = models.MOSNet.load_from_checkpoint(weights, hparams=cfg)

    # Add callbacks
    lr_monitor = LearningRateMonitor(logging_interval="step")
    checkpoint_saver = ModelCheckpoint(
        monitor="val_moving_iou_step0",
        filename=cfg["EXPERIMENT"]["ID"] + "_{epoch:03d}_{val_moving_iou_step0:.3f}",
        mode="max",
        save_last=True,
    )

    # Logger
    log_dir = "./logs"
    os.makedirs(log_dir, exist_ok=True)
    tb_logger = pl_loggers.TensorBoardLogger(
        log_dir, name=cfg["EXPERIMENT"]["ID"], default_hp_metric=False
    )

    # Setup trainer
    trainer = Trainer(
        accelerator="gpu",
        devices=cfg["TRAIN"]["NUM_DEVICES"],
        logger=tb_logger,
        max_epochs=cfg["TRAIN"]["MAX_EPOCH"],
        accumulate_grad_batches=cfg["TRAIN"]["ACC_BATCHES"],  # ?? what is ACC Baches
        callbacks=[lr_monitor, checkpoint_saver],
    )

    # Load data
    data = datasets.KittiSequentialModule(cfg)
    # train_set = datasets.KittiSequentialDataset(cfg, split="train")
    # val_set = datasets.KittiSequentialDataset(cfg, split="val")
    # test_set = datasets.KittiSequentialDataset(cfg, split="test")
    data.setup()
    train_dataloader = data.train_dataloader()
    val_dataloader = data.val_dataloader()
    # Train!
    trainer.fit(model, train_dataloader, val_dataloader, ckpt_path=checkpoint)
    # self, self._fit_impl, model, train_dataloaders, val_dataloaders, datamodule, ckpt_path

if __name__ == "__main__":
    main()
