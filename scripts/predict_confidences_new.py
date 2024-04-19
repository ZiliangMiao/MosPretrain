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

from mos4d.models.metrics import ClassificationMetrics
import mos4d.models.nusc_models as nusc_models
import mos4d.datasets.kitti_dataset as datasets
import mos4d.datasets.nusc_dataset as nusc_dataset
from mos4d.datasets.kitti_dataset import KittiSequentialDataset
from mos4d.datasets.nusc_dataset import NuscSequentialDataset


import sklearn


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
    model_dir = os.path.join("../logs", "train", model_dataset, model_name, model_version)
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
    model = model.cuda()
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    model.freeze()

    # testing
    test_dataloader = data.test_dataloader()

    ##############################################################################
    # test_data_list = list(test_dataloader)
    # from torch.utils.data import DataLoader, Dataset
    # class PartialDataset(Dataset):
    #     def __init__(self, data_list):
    #         self.data_list = data_list
    #     def __len__(self):
    #         return len(self.data_list)
    #     def __getitem__(self, index):
    #         return self.data_list[index]
    # def collate_fn(batch):  # define how to merge a list of samples to from a mini-batch samples
    #     sample_data_token = [item[0][0] for item in batch]
    #     point_cloud = [item[1][0] for item in batch]
    #     mos_label = [item[2][0] for item in batch]
    #     return [sample_data_token, point_cloud, mos_label]
    # # 创建自定义的Dataset对象
    # partial_dataset = PartialDataset(test_data_list)
    # # 创建DataLoader对象
    # partial_dataloader = DataLoader(partial_dataset, batch_size=1, shuffle=False, collate_fn=collate_fn)
    # listss = list(partial_dataloader)
    ##############################################################################


    log_folder = os.path.join(model_dir, "results", f"epoch_{test_epoch}")
    os.makedirs(log_folder, exist_ok=True)

    date = datetime.date.today().strftime('%Y%m%d')
    log_file = os.path.join(log_folder, f"{model_name}_epoch-{test_epoch}_{date}.txt")
    logging.basicConfig(filename=log_file, level=logging.INFO,
                        format='[%(asctime)s.%(msecs)03d] %(message)s', datefmt='%H:%M:%S')
    logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
    logging.info(str(cfg))
    logging.info(log_file)

    # metrics
    metrics = ClassificationMetrics(n_classes=3, ignore_index=0)

    # mos label directory
    pred_mos_labels_dir = os.path.join(log_folder, "predictions")
    os.makedirs(pred_mos_labels_dir, exist_ok=True)
    save_pred_wo_ego = True

    # predict with pytorch-lightning
    trainer = Trainer(accelerator="gpu", strategy="ddp", devices=num_device, deterministic=True)
    pred_outputs = trainer.predict(model, dataloaders=test_dataloader, return_predictions=True, ckpt_path=ckpt_path)
    # pred iou
    conf_mat_list_pred = [output["confusion_matrix"] for output in pred_outputs]
    acc_conf_mat_pred = torch.zeros(3, 3)
    for conf_mat in conf_mat_list_pred:
        acc_conf_mat_pred = acc_conf_mat_pred.add(conf_mat)
    TP_pred, FP_pred, FN_pred = metrics.getStats(acc_conf_mat_pred)
    IOU_pred = metrics.getIoU(TP_pred, FP_pred, FN_pred)[2]
    logging.info('Final Avg. Moving Object IoU w/o ego vehicle: %f' % (IOU_pred.item() * 100))

    # # validate
    # valid_outputs = trainer.validate(model, dataloaders=test_dataloader, ckpt_path=ckpt_path)
    # # valid iou
    # conf_mat_list = model.validation_step_outputs
    # acc_conf_mat = torch.zeros(3, 3)
    # for conf_mat in conf_mat_list:
    #     acc_conf_mat = acc_conf_mat.add(conf_mat)
    # TP, FP, FN = metrics.getStats(acc_conf_mat)
    # IOU = metrics.getIoU(TP, FP, FN)[2]
    # logging.info('Final Avg. Moving Object IoU w/o ego vehicle: %f' % (IOU.item() * 100))
    # a = 1


    # # loop batch
    # num_classes = 3
    # ignore_class_idx = 0
    # moving_class_idx = 2
    # TP_mov, FP_mov, FN_mov = 0, 0, 0
    # acc_conf_mat = torch.zeros(num_classes, num_classes).cuda()
    # num_samples = 0
    # for i, batch in tqdm(enumerate(test_dataloader)):
    #     meta, point_clouds, mos_labels = batch
    #     point_clouds = [point_cloud.cuda() for point_cloud in point_clouds]
    #     curr_coords_list, curr_feats_list = model(point_clouds)
    #
    #     for batch_idx, (coords, logits) in enumerate(zip(curr_coords_list, curr_feats_list)):
    #         gt_label = mos_labels[batch_idx].cuda()
    #         if test_dataset == "NUSC":
    #             # get ego mask
    #             curr_time_mask = point_clouds[batch_idx][:, -1] == 0.0
    #             ego_mask = NuscSequentialDataset.get_ego_mask(point_clouds[batch_idx][curr_time_mask]).cpu().numpy()
    #             # get pred mos label file name
    #             sample_data_token = meta[batch_idx]
    #             pred_label_file = os.path.join(pred_mos_labels_dir, f"{sample_data_token}_mos_pred.label")
    #         elif test_dataset == "SEKITTI":
    #             # get ego mask
    #             curr_time_mask = point_clouds[batch_idx][:, -1] == 0.0
    #             ego_mask = KittiSequentialDataset.get_ego_mask(point_clouds[batch_idx][curr_time_mask]).cpu().numpy()
    #             # get pred mos label file name
    #             seq_idx, scan_idx, _ = meta[batch_idx]
    #             pred_label_file = os.path.join(pred_mos_labels_dir, f"seq-{seq_idx}_scan-{scan_idx}_mos_pred.label")
    #         else:
    #             raise Exception("Not supported test dataset")
    #
    #         # save predictions
    #         logits[:, ignore_class_idx] = -float("inf")  # ingore: 0, i.e., unknown or noise
    #         pred_confidence = F.softmax(logits, dim=1).detach().cpu().numpy()
    #         moving_confidence = pred_confidence[:, moving_class_idx]
    #         pred_label = np.ones_like(moving_confidence, dtype=np.uint8)  # notice: dtype of mos labels is uint8
    #         pred_label[moving_confidence > 0.5] = 2
    #
    #         # calculate iou w/o ego vehicle pts
    #         cfs_mat = metrics.compute_confusion_matrix(logits[~ego_mask], gt_label[~ego_mask])
    #         acc_conf_mat = acc_conf_mat.add(cfs_mat)
    #         tp, fp, fn = metrics.getStats(cfs_mat)  # stat of current sample
    #         iou_mov = metrics.getIoU(tp, fp, fn)[moving_class_idx] * 100  # IoU of moving object (class 2)
    #         TP_mov += tp[moving_class_idx]
    #         FP_mov += fp[moving_class_idx]
    #         FN_mov += fn[moving_class_idx]
    #         # logging two iou
    #         num_samples += 1
    #         logging.info('Validation Sample Index %d, Moving Object IoU w/o ego vehicle: %f' % (num_samples, iou_mov))
    #         # save predicted labels
    #         if save_pred_wo_ego:
    #             pred_label[ego_mask] = 0  # set ego vehicle points as unknown for visualization
    #         # save pred mos label
    #         pred_label.tofile(pred_label_file)
    #     torch.cuda.empty_cache()
    # IOU_mov = metrics.getIoU(TP_mov, FP_mov, FN_mov)
    #
    # # TP, FP, FN = metrics.getStats(acc_conf_mat)
    # # IOU = metrics.getIoU(TP, FP, FN)
    #
    # logging.info('Final Avg. Moving Object IoU w/o ego vehicle: %f' % (IOU_mov * 100))
    #
    # a = 1

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
