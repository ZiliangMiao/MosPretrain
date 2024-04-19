import click
import yaml
import copy
import re
import os
import numpy as np
import torch
from pytorch_lightning import Trainer
from pytorch_lightning import loggers as pl_loggers
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint

from nuscenes.nuscenes import NuScenes
from tqdm import tqdm

import mos4d.models.models as models
import mos4d.models.nusc_models as nusc_models
import mos4d.datasets.kitti_dataset as datasets
import mos4d.datasets.nusc_dataset as nusc_dataset

def load_pretrained_encoder(ckpt_path, model):
    # if len(os.listdir(ckpt_dir)) > 0:
    #     pattern = re.compile(r"model_epoch_(\d+).pth")
    #     epochs = []
    #     for f in os.listdir(ckpt_dir):
    #         m = pattern.findall(f)
    #         if len(m) > 0:
    #             epochs.append(int(m[0]))
    #     resume_epoch = max(epochs)
    #     ckpt_path = f"{ckpt_dir}/model_epoch_{resume_epoch}.pth"

    print(f"Load pretrained encoder from checkpoint {ckpt_path}")

    checkpoint = torch.load(ckpt_path)
    pretrained_dict = checkpoint["state_dict"]
    model_dict = model.state_dict()

    # filter out unnecessary keys (generate new dict)
    pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
    pretrained_dict.pop('encoder.MinkUNet.final.kernel')
    pretrained_dict.pop('encoder.MinkUNet.final.bias')
    # overwrite finetune model dict
    model_dict.update(pretrained_dict)
    # load the pretrained model dict
    model.load_state_dict(model_dict)
    return model

@click.command()
### Add your options here
@click.option(
    "--config",
    "-c",
    type=str,
    help="path to the config file (.yaml)",
    default="../config/train_cfg.yaml",
)
@click.option(
    "--resume",
    type=str,
    help="path to checkpoint file (.ckpt) to resume training.",
    default=None,
)
def main(config, resume):
    if resume is None:  # training from scratch or finetuning
        cfg = yaml.safe_load(open(config))
    else:  # resume training
        cfg = torch.load(resume)["hyper_parameters"]

    # parameters
    mode = cfg["MODE"]
    assert mode != "test"
    dataset_name = cfg["DATA"]["dataset_name"]
    data_pct = cfg["TRAIN"]["DATA_PCT"]
    num_epoch = cfg["TRAIN"]["MAX_EPOCH"]
    batch_size = cfg["TRAIN"]["BATCH_SIZE"]
    voxel_size = cfg["DATA"]["VOXEL_SIZE"]
    time_interval = cfg["DATA"]["DELTA_T_PRED"]
    n_input = cfg["DATA"]["N_PAST_STEPS"]
    time = time_interval * n_input

    if mode == "finetune":  # load pretrained model
        pre_method = cfg["TRAIN"]["PRETRAIN_METHOD"]
        pre_dataset = cfg["TRAIN"]["pretrain_dataset"]
        pre_model = cfg["TRAIN"]["pretrain_model_name"]
        model_name = f"{pre_method}_{pre_dataset}_{pre_model}_vs-{voxel_size}_t-{time}_bs-{batch_size}_epo-{num_epoch}"

        pretrain_model_name = cfg["TRAIN"]["PRETRAIN_MODEL"]
        pretrain_ckpt_pth = "../ckpts/" + pretrain_model_name
        if dataset_name == "NUSC":
            pretrain_ckpt = torch.load(pretrain_ckpt_pth)
            # pretrain_ckpt['pytorch-lightning_version'] = '1.9.0'
            torch.save(pretrain_ckpt, pretrain_ckpt_pth)
            # model = nusc_models.MOSNet.load_from_checkpoint(pretrain_ckpt_pth, hparams=cfg)
            model = nusc_models.MOSNet(cfg)
            model = load_pretrained_encoder(pretrain_ckpt_pth, model)
        else:
            model = models.MOSNet.load_from_checkpoint(pretrain_ckpt_pth, hparams=cfg)
    elif mode == "train":  # init network with cfg
        model_name = f"vs-{voxel_size}_t-{time}_bs-{batch_size}_epo-{num_epoch}"
        if dataset_name == "NUSC":
            model = nusc_models.MOSNet(cfg)
        else:
            model = nusc_models.MOSNet(cfg)

    # Add callbacks
    lr_monitor = LearningRateMonitor(logging_interval="step")
    checkpoint_saver = ModelCheckpoint(
        monitor="epoch",
        verbose=True,
        save_top_k=cfg["TRAIN"]["MAX_EPOCH"],
        mode="max",
        filename=model_name + "_{epoch}",
        every_n_epochs=1,
        save_last=True,
    )

    # Logger
    log_dir = f"../logs/{mode}/{data_pct}%{dataset_name}"
    os.makedirs(log_dir, exist_ok=True)
    tb_logger = pl_loggers.TensorBoardLogger(log_dir, name=model_name, default_hp_metric=False)

    # load data from different datasets
    if dataset_name == "NUSC":
        nusc = NuScenes(dataroot=cfg["DATASET"]["NUSC"]["PATH"], version=cfg["DATASET"]["NUSC"]["VERSION"])
        data = nusc_dataset.NuscSequentialModule(cfg, nusc, "train")
        data.setup()
    else:  # KITTI-like datasets
        data = datasets.KittiSequentialModule(cfg)
        data.setup()

    train_dataloader = data.train_dataloader()
    val_dataloader = data.val_dataloader()

    # Setup trainer and fit
    trainer = Trainer(
        accelerator="gpu",
        strategy="ddp",
        devices=cfg["TRAIN"]["NUM_DEVICES"],
        logger=tb_logger,
        log_every_n_steps=1,
        max_epochs=cfg["TRAIN"]["MAX_EPOCH"],
        accumulate_grad_batches=cfg["TRAIN"]["ACC_BATCHES"],  # accumulate batches, default=1
        callbacks=[lr_monitor, checkpoint_saver],
        check_val_every_n_epoch=5,
        # val_check_interval=100,
    )
    trainer.fit(model, train_dataloader, val_dataloader, ckpt_path=resume)
    # self, self._fit_impl, model, train_dataloaders, val_dataloaders, datamodule, ckpt_path

def create_sekitti_mos_labels(dataset_path):
    semantic_config = yaml.safe_load(open("../config/semantic-kitti-mos.yaml"))
    def save_mos_labels(lidarseg_label_file, mos_label_file):
        """Load moving object labels from .label file"""
        if os.path.isfile(lidarseg_label_file):
            labels = np.fromfile(lidarseg_label_file, dtype=np.uint32)
            labels = labels.reshape((-1))
            labels = labels & 0xFFFF  # Mask semantics in lower half
            mapped_labels = copy.deepcopy(labels)
            for semantic_label, mos_label in semantic_config["learning_map"].items():
                mapped_labels[labels == semantic_label] = mos_label
            mos_labels = mapped_labels.astype(np.uint8)
            mos_labels.tofile(mos_label_file)
            num_unk = np.sum(mos_labels == 0)
            num_sta = np.sum(mos_labels == 1)
            num_mov = np.sum(mos_labels == 2)
            print(f"num of unknown pts: {num_unk}, num of static pts: {num_sta}, num of moving pts: {num_mov}")
            # Directly load saved mos labels and check if it is correct
            # mos_labels = np.fromfile(mos_label_file, dtype=np.uint32).reshape((-1)) & 0xFFFF
            # check_true = (mos_labels == mapped_labels).all()
            return None
        else:
            return torch.Tensor(1, 1).long()

    seqs_list = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    for seq_idx in tqdm(seqs_list):
        lidarseg_dir = os.path.join(dataset_path, str(seq_idx).zfill(4), "labels")
        mos_labels_dir = os.path.join(dataset_path, str(seq_idx).zfill(4), "mos_labels")
        os.makedirs(mos_labels_dir, exist_ok=True)
        for i, filename in enumerate(os.listdir(lidarseg_dir)):
            lidarseg_label_file = os.path.join(lidarseg_dir, filename)
            mos_label_file = os.path.join(mos_labels_dir, filename)
            save_mos_labels(lidarseg_label_file, mos_label_file)

def set_deterministic(random_seed=666):
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(random_seed)
    torch.manual_seed(random_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(random_seed)

if __name__ == "__main__":
    set_deterministic(666)
    # create_sekitti_mos_labels("/home/user/Datasets/SeKITTI/sequences")
    main()
