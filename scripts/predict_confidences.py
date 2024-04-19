import logging
import os.path
import datetime
import sys
from tqdm import tqdm
import click
import yaml
from nuscenes import NuScenes
from pytorch_lightning import Trainer
import torch
import numpy as np
import torch.nn.functional as F

import mos4d.models.nusc_models as nusc_models
import mos4d.datasets.kitti_dataset as datasets
import mos4d.datasets.nusc_dataset as nusc_dataset
from mos4d.datasets.kitti_dataset import KittiSequentialDataset
from mos4d.datasets.nusc_dataset import NuscSequentialDataset


from sklearn.metrics import confusion_matrix


@click.command()
### Add your options here
@click.option(
    "--config",
    "-c",
    type=str,
    help="path to the config file (.yaml)",
    default="../config/test_cfg.yaml",
)
def main(config):
    # config
    cfg = yaml.safe_load(open(config))  # cfg = torch.load(weights)["hyper_parameters"]'
    num_device = cfg["TEST"]["NUM_DEVICES"]
    model_name = cfg["TEST"]["MODEL_NAME"]
    model_version = cfg["TEST"]["MODEL_VERSION"]
    model_dataset = cfg["TEST"]["model_dataset"]
    test_epoch = cfg["TEST"]["TEST_EPOCH"]
    model_dir = os.path.join("../logs", "finetune", model_dataset, model_name, model_version)
    ckpt_path = os.path.join(model_dir, "checkpoints", f"{model_name}_epoch={test_epoch}.ckpt")
    hparams_path = os.path.join(model_dir, "hparams.yaml")
    hparams = yaml.safe_load(open(hparams_path))

    # dataset
    test_dataset = cfg["DATA"]["DATASET"]
    if test_dataset == "NUSC":
        nusc = NuScenes(dataroot=cfg["DATASET"]["NUSC"]["PATH"], version=cfg["DATASET"]["NUSC"]["VERSION"])
        data = nusc_dataset.NuscSequentialModule(cfg, nusc, "test")
        data.setup()
    elif test_dataset == "SEKITTI":
        data = datasets.KittiSequentialModule(cfg)
        data.setup()
    else:
        raise ValueError("Not supported test dataset.")

    # Load ckeckpoint model
    ckpt = torch.load(ckpt_path)
    model = nusc_models.MOSNet(hparams)
    model.load_state_dict(ckpt["state_dict"])
    model = model.cuda()
    model.eval()
    model.freeze()

    # testing
    test_dataloader = data.test_dataloader()
    log_folder = os.path.join(model_dir, "results", f"epoch_{test_epoch}")
    os.makedirs(log_folder, exist_ok=True)

    date = datetime.date.today().strftime('%Y%m%d')
    log_file = os.path.join(log_folder, f"{model_name}_epoch-{test_epoch}_{date}.txt")
    logging.basicConfig(filename=log_file, level=logging.INFO,
                        format='[%(asctime)s.%(msecs)03d] %(message)s', datefmt='%H:%M:%S')
    logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
    logging.info(str(cfg))
    logging.info(log_file)

    # mos label directory
    pred_mos_labels_dir = os.path.join(log_folder, "predictions")
    os.makedirs(pred_mos_labels_dir, exist_ok=True)
    save_pred_wo_ego = True
    # loop batch
    TP_w, FP_w, FN_w = 0, 0, 0
    TP_wo, FP_wo, FN_wo = 0, 0, 0
    num_samples = 0
    for i, batch in tqdm(enumerate(test_dataloader)):
        meta, num_curr_pts, point_clouds, mos_labels = batch
        point_clouds = [point_cloud.cuda() for point_cloud in point_clouds]
        curr_coords_list, curr_feats_list = model(point_clouds)
        for batch_idx, (coords, logits) in enumerate(zip(curr_coords_list, curr_feats_list)):
            gt_label = mos_labels[batch_idx].cpu().detach().numpy()
            ignore_index = [0]
            logits[:, ignore_index] = -float("inf")  # ingore: 0, i.e., unknown or noise
            pred_softmax = F.softmax(logits, dim=1)
            pred_softmax = pred_softmax.detach().cpu().numpy()
            sum = np.sum(pred_softmax[:, 1:3], axis=1)
            assert np.isclose(sum, np.ones_like(sum)).all()
            moving_confidence = pred_softmax[:, 2]

            # directly output the mos label, without any bayesian strategy (don't need confidences_to_labels.py)
            pred_label = np.ones_like(moving_confidence, dtype=np.uint8)  # notice: dtype of mos labels is uint8
            pred_label[moving_confidence > 0.5] = 2

            # calculate iou w/ ego vehicle pts
            cfs_mat_w = confusion_matrix(gt_label, pred_label, labels=[1, 2])
            tp_w, fp_w, fn_w = getStat(cfs_mat_w)  # stat of current sample
            iou_w = getIoU(tp_w, fp_w, fn_w) * 100  # IoU of moving object (class 2)
            TP_w += tp_w
            FP_w += fp_w
            FN_w += fn_w
            # calculate iou w/o ego vehicle pts
            if test_dataset == "NUSC":
                # get ego mask
                curr_time_mask = point_clouds[batch_idx][:, -1] == 0.0
                ego_mask = NuscSequentialDataset.get_ego_mask(point_clouds[batch_idx][curr_time_mask]).cpu().numpy()
                # get pred mos label file name
                sample_data_token = meta[batch_idx]
                pred_label_file = os.path.join(pred_mos_labels_dir, f"{sample_data_token}_mos_pred.label")
            elif test_dataset == "SEKITTI":
                # get ego mask
                curr_time_mask = point_clouds[batch_idx][:, -1] == 0.0
                ego_mask = KittiSequentialDataset.get_ego_mask(point_clouds[batch_idx][curr_time_mask]).cpu().numpy()
                # get pred mos label file name
                seq_idx, scan_idx, _ = meta[batch_idx]
                pred_label_file = os.path.join(pred_mos_labels_dir, f"seq-{seq_idx}_scan-{scan_idx}_mos_pred.label")
            else:
                raise Exception("Not supported test dataset")
            cfs_mat_wo = confusion_matrix(gt_label[~ego_mask], pred_label[~ego_mask], labels=[1, 2])
            tp_wo, fp_wo, fn_wo = getStat(cfs_mat_wo)  # stat of current sample
            iou_wo = getIoU(tp_wo, fp_wo, fn_wo) * 100  # IoU of moving object (class 2)
            TP_wo += tp_wo
            FP_wo += fp_wo
            FN_wo += fn_wo
            # logging two iou
            num_samples += 1
            logging.info('Validation Sample Index %d, IoU w/ ego vehicle: %f' % (num_samples, iou_w))
            logging.info('Validation Sample Index %d, IoU w/o ego vehicle: %f' % (num_samples, iou_wo))
            # save predicted labels
            if save_pred_wo_ego:
                pred_label[ego_mask] = 0  # set ego vehicle points as unknown for visualization
            # save pred mos label
            pred_label.tofile(pred_label_file)
        torch.cuda.empty_cache()
    IOU_w = getIoU(TP_w, FP_w, FN_w)
    IOU_wo = getIoU(TP_wo, FP_wo, FN_wo)
    logging.info('Final Avg. IoU w/ ego vehicle: %f' % (IOU_w * 100))
    logging.info('Final Avg. IoU w/o ego vehicle: %f' % (IOU_wo * 100))

    # Use torch-lightning to test nuscenes mos, setup trainer:
    # 使用torch lightning就无法正常logging每一个sample的值了, 故直接循环, 不调用predict方法
    # trainer = Trainer(accelerator="gpu", devices=num_device, logger=False)
    # Inference
    # trainer.predict(model, test_dataloader)  # predict confidence scores (softmax output)

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

# def get_ego_mask(points):  # mask the points of ego vehicle: x [-0.8, 0.8], y [-1.5, 2.5]
#     # nuscenes car: length (4.084 m), width (1.730 m), height (1.562 m)
#     # https://forum.nuscenes.org/t/dimensions-of-the-ego-vehicle-used-to-gather-data/550
#     ego_mask = torch.logical_and(
#         torch.logical_and(-0.865 <= points[:, 0], points[:, 0] <= 0.865),
#         torch.logical_and(-1.5 <= points[:, 1], points[:, 1] <= 2.5),
#     )
#     return ego_mask.cpu().detach().numpy()

def set_deterministic(random_seed=666):
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(random_seed)
    torch.manual_seed(random_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(random_seed)

if __name__ == "__main__":
    set_deterministic(666)
    main()
