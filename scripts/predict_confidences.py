import os.path

import click
import yaml
from pytorch_lightning import Trainer
import torch
import numpy as np
import torch.nn.functional as F

import mos4d.models.models as models
import mos4d.datasets.datasets as datasets
import mos4d.datasets.nusc_dataset as nusc_dataset


from sklearn.metrics import confusion_matrix


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
    "--sequences",
    "-seq",
    type=int,
    help="run inference on a specific sequence, otherwise, default test split is used",
    default=None,
    multiple=True,
)
def main(config, sequences):
    # config
    cfg = yaml.safe_load(open(config))  # cfg = torch.load(weights)["hyper_parameters"]'
    assert cfg["EXPT"]["MODE"] == "TEST"
    num_device = cfg["TEST"]["NUM_DEVICES"]
    ckpt_path = cfg["TEST"]["CKPT"]
    # dataset
    dataset = cfg["TEST"]["DATASET"]
    if dataset == "NUSC":
        data = nusc_dataset.NuscSequentialModule(cfg)
        data.setup()
    else:
        cfg["DATASET"][dataset]["TRAIN"] = cfg["DATASET"][dataset]["TEST"]  # in test mode
        cfg["DATASET"][dataset]["VAL"] = cfg["DATASET"][dataset]["TEST"]  # in test mode
        if sequences:
            cfg["DATASET"][dataset]["TEST"] = list(sequences)
        # Load data and model for different datasets
        data = datasets.KittiSequentialModule(cfg)
        data.setup()

    # method params
    strategy = cfg["TEST"]["STRATEGY"]
    bayes_prior = cfg["TEST"]["BAYES_PRIOR"]
    delta_t = cfg["TEST"]["DELTA_T"]  # cfg["MODEL"]["DELTA_T_PREDICTION"]
    transform = cfg["DATA"]["TRANSFORM"]

    # Load ckeckpoint model
    ckpt = torch.load(ckpt_path)
    model = models.MOSNet(cfg)
    model.load_state_dict(ckpt["state_dict"])
    model = model.cuda()
    model.eval()
    model.freeze()
    # Setup trainer
    trainer = Trainer(accelerator="gpu", devices=num_device, logger=False)
    # Inference
    test_dataloader = data.test_dataloader()
    trainer.predict(model, test_dataloader)  # predict confidence scores (softmax output)

    if dataset == "NUSC":
        version = cfg["DATASET"][dataset]["VERSION"]
        # calculate IoU performance
        test_dataset = test_dataloader.dataset
        sample_data_tokens = test_dataset.sample_data_tokens

        gt_mos_labels_dir = os.path.join(cfg["DATASET"][dataset]["PATH"], "mos_labels", version)
        pred_mos_labels_dir = os.path.join(cfg["DATASET"][dataset]["PATH"], "4dmos_sekitti_pred", version)

        tp = 0
        fp = 0
        fn = 0
        for sample_data_token in sample_data_tokens:
            gt_mos_label_file = os.path.join(gt_mos_labels_dir, sample_data_token + "_mos.label")
            pred_mos_label_file = os.path.join(pred_mos_labels_dir, sample_data_token + "_mos_pred.label")
            gt_mos_label = np.fromfile(gt_mos_label_file, dtype=np.uint8)
            pred_mos_label = np.fromfile(pred_mos_label_file, dtype=np.uint8)

            cfs_mat = confusion_matrix(gt_mos_label, pred_mos_label, labels=[1, 2])
            tp_i, fp_i, fn_i = getStat(cfs_mat)  # stat of current sample
            tp += tp_i
            fp += fp_i
            fn += fn_i
        IoU = getIoU(tp, fp, fn)  # IoU of moving object (class 2)
        print("\n" + "Moving Object IoU: " + str(IoU))

def getStat(confusion_matrix):
    tp = np.diagonal(confusion_matrix)
    fp = np.sum(confusion_matrix, axis=1) - tp
    fn = np.sum(confusion_matrix, axis=0) - tp
    return tp[1], fp[1], fn[1]

def getIoU(tp, fp, fn):
    intersection = tp
    union = tp + fp + fn + 1e-15
    iou = intersection / union
    return iou

if __name__ == "__main__":
    main()
