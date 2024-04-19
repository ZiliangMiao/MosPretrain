import click
import os
import yaml
import copy
from tqdm import tqdm
import multiprocessing
import numpy as np
from mos4d.datasets.utils import load_files
import torch
import torch.nn as nn
import torch.nn.functional as F

from sklearn.metrics import confusion_matrix

# @click.command()
# @click.option(
#     "--config",
#     "-c",
#     type=str,
#     help="path to the config file (.yaml)",
#     default="./config/train_cfg.yaml",
# )
# @click.option(
#     "--sequences",
#     "-seq",
#     type=int,
#     help="run inference on a specific sequence, otherwise, default test split is used",
#     default=None,
#     multiple=True,
# )
def gene_bench_test(seq_idx, cfg):
    dataset = cfg["TEST"]["DATASET"]
    # if dataset == "SEKITTI":
    #     semantic_config = yaml.safe_load(open(cfg["DATA"]["SEMANTIC_CONFIG_FILE"]))
    # else:
    #     semantic_config = None
    # method params
    strategy = cfg["TEST"]["STRATEGY"]
    bayes_prior = cfg["TEST"]["BAYES_PRIOR"]
    delta_t = cfg["TEST"]["DELTA_T"]
    # file path
    dataset_path = cfg["DATASET"][dataset]["PATH"]
    expt_path = os.path.join(cfg["TEST"]["PRED_PATH"], cfg["EXPT"]["ID"], dataset)

    # use non-overlapping or bayesian filter to predict the labels
    strategy_str = strategy
    if strategy == "bayes":  # use bayesian filter to combine the estimations at different timestamp
        strategy_str = strategy_str + "_{:.3e}".format(float(bayes_prior))
    strategy_str = strategy_str + "_" + str(delta_t)

    # Pred current scan, without Bayesian Fusion, do not use the future information
    tp = 0
    fp = 0
    fn = 0
    if strategy == "wo_bayes":
        pred_label_path = os.path.join(expt_path, "labels", strategy_str, str(seq_idx).zfill(4))
        os.makedirs(pred_label_path, exist_ok=True)
        seq_confidences_path = os.path.join(expt_path, "confidences", str(seq_idx).zfill(4))
        scan_indices = os.listdir(seq_confidences_path)  # scan indices inside current seq, start from 000009
        scan_indices.sort()
        for scan_idx in scan_indices:
            scan_confidences_path = os.path.join(seq_confidences_path, scan_idx)  # ./predictions/model_1024_1504/SeKITTI/confidences/11/000009
            pred_indices = os.listdir(scan_confidences_path)
            pred_indices.sort()
            scan_confidence_file = os.path.join(scan_confidences_path, pred_indices[-1])
            scan_confidence = np.load(scan_confidence_file)
            pred_label = conf_to_label(scan_confidence)
            pred_label_file = pred_label_path + "/" + str(scan_idx).zfill(6) + ".label"
            pred_label.tofile(pred_label_file)
            # directly use mapped mos labels
            mos_label_file = os.path.join(dataset_path, str(seq_idx).zfill(4), "labels", str(scan_idx).zfill(6) + ".label")
            mos_label = np.fromfile(mos_label_file, dtype=np.uint32).reshape((-1)) & 0xFFFF
            # use original semantic labels
            # gt_label_file = os.path.join(dataset_path, str(seq_idx).zfill(4), "labels", str(scan_idx).zfill(6) + ".label")
            # mos_label = load_gt_mos_label(gt_label_file)

            # calculate IoU of each sequences
            cfs_mat = confusion_matrix(mos_label, pred_label, labels=[1, 2])
            tp_i, fp_i, fn_i = getStat(cfs_mat)
            tp = tp + tp_i
            fp = fp + fp_i
            fn = fn + fn_i
            IoU_i = getIoU(tp_i, fp_i, fn_i)  # IoU of moving object
            print("IoU of current scan", scan_idx, ": ", IoU_i)
        IoU = getIoU(tp, fp, fn).reshape(-1, 1)
        print("IoU of current seq", seq_idx, ": ", IoU)
        np.savetxt(os.path.join(expt_path, "labels", strategy_str, str(seq_idx).zfill(4) + "_iou" + ".txt"), IoU)
    else:  # Original codes, contains bayes outputs
        pred_label_path = os.path.join(expt_path, "labels", strategy_str, str(seq_idx).zfill(4))
        os.makedirs(pred_label_path, exist_ok=True)
        seq_confidences_path = os.path.join(expt_path, "confidences", str(seq_idx).zfill(4))
        scan_indices = os.listdir(seq_confidences_path)  # scan indices inside current seq, start from 000009
        scan_indices.sort()
        seq_confidences = {}  # dictionary '000009': ['xxxx.npy, xxxx.npy'...]
        for scan_idx in scan_indices:
            scan_confidences_path = os.path.join(seq_confidences_path, scan_idx)  # ./predictions/model_1024_1504/SeKITTI/confidences/11/000009
            pred_indices = os.listdir(scan_confidences_path)
            pred_indices.sort()
            seq_confidences[scan_idx] = pred_indices  # several .npy files that contains both current scan confidence and previous scan confidences
        dict_confidences = {}  # store all the predictions follow scan sequences (each scan will have multiple predictions)
        for scan_idx, pred_confidences in seq_confidences.items():
            for pred_confidence in pred_confidences:
                pred_confidence_file = os.path.join(seq_confidences_path, scan_idx, pred_confidence)

                # Consider prediction if done with desired temporal resolution
                # each time will predict several previous scans
                pred_idx = pred_confidence.split("_")[0]
                temporal_resolution = float(pred_confidence.split("_")[-1].split(".")[0])
                if temporal_resolution == delta_t:
                    if pred_idx not in dict_confidences:
                        dict_confidences[pred_idx] = [pred_confidence_file]
                    else:  # multi predictions of a single scans
                        dict_confidences[pred_idx].append(pred_confidence_file)
                    dict_confidences[pred_idx].sort()
        # Non-overlapping, it is super stupid, nobody will do things like this
        if strategy == "non-overlapping":
            # all confidences predicted at different time of a single scan
            for pred_idx, confidences in tqdm(dict_confidences.items(), desc="Scans"):  # pred_inx = scan_idx, start from 09
                from_idx = int(pred_idx) % len(confidences)
                confidence = np.load(confidences[from_idx])
                pred_labels = conf_to_label(confidence)
                pred_labels.tofile(pred_label_path + "/" + pred_idx.split(".")[0] + ".label")
        # Bayesian Fusion
        elif strategy == "bayes":
            for pred_idx, confidences in tqdm(dict_confidences.items(), desc="Scans"):
                confidence = np.load(confidences[0])
                log_odds = prob_to_log_odds(confidence)
                for conf in confidences[1:]:
                    confidence = np.load(conf)
                    log_odds += prob_to_log_odds(confidence)
                    log_odds -= prob_to_log_odds(bayes_prior * np.ones_like(confidence))
                final_confidence = log_odds_to_prob(log_odds)
                pred_labels = conf_to_label(final_confidence)
                pred_labels.tofile(pred_label_path + "/" + pred_idx.split(".")[0] + ".label")

def load_gt_mos_label(gt_label_file):
    gt_label = np.fromfile(gt_label_file, dtype=np.uint32).reshape((-1)) & 0xFFFF
    # np.where(gt_labels.any() >= 251 & gt_labels.any() < 65535, 2, )  # moving
    # np.where(gt_labels.any() < 251 & gt_labels.any() < 65535, 1)  # static
    # np.where(gt_labels.any() == 65535, 0)  # invalid label
    # gt_label.reshape(1, -1)  # deleted at 10.29
    mos_label = np.zeros_like(gt_label)
    mos_label[gt_label >= 251] = 2  # moving
    mos_label[gt_label < 251] = 1  # static
    mos_label[gt_label >= 65535] = 0  # invalid label
    return mos_label.reshape((-1))

def prob_to_log_odds(prob):
    odds = np.divide(prob, 1 - prob + 1e-10)
    log_odds = np.log(odds)
    return log_odds

def log_odds_to_prob(log_odds):
    log_odds = np.clip(log_odds, -80, 80)
    odds = np.exp(log_odds)
    prob = np.divide(odds, odds + 1)
    return prob

def conf_to_label(confidence):  # previous: (confidence, semantic_config)
    pred_labels = np.ones_like(confidence)
    pred_labels[confidence > 0.5] = 2  # if mov confidence > 0.5 -> label it mov
    # pred_labels = to_original_labels(pred_labels, semantic_config)
    pred_labels = pred_labels.reshape((-1)).astype(np.uint32)
    return pred_labels

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

def getacc(self, confusion_matrix):
    tp, fp, fn = self.getStats(confusion_matrix)
    total_tp = tp.sum()
    total = tp.sum() + fp.sum() + 1e-15
    acc_mean = total_tp / total
    return acc_mean

if __name__ == '__main__':
    # config
    config = "./config/train_cfg.yaml"
    cfg = yaml.safe_load(open(config))
    dataset = cfg["TEST"]["DATASET"]
    seqs = cfg["DATASET"][dataset]["TEST"]

    pool = multiprocessing.Pool(processes=44)
    for seq_idx in seqs:
        pool.apply_async(func=gene_bench_test, args=(seq_idx, cfg,))
    pool.close()
    pool.join()
    print("multi-processing success!")


    # test
    waymo_path = "/home/mars/4DMOS/data/waymo_MOS/sequences"
    kitti_path = "/home/mars/4DMOS/data/KITTI_MOS/sequences"
    nuscenes_path = "/home/mars/4DMOS/data/nuScenes_MOS/sequences"
    seq_id = "0000"
    waymo_lidar_path = waymo_path + "/" + seq_id + "/lidar"
    waymo_label_path = waymo_path + "/" + seq_id + "/labels"
    label_file = "000000.label"
    waymo_filename = "/home/mars/4DMOS/data/waymo_MOS/sequences/0001/labels/000111.label"
    kitti_filename = "/home/mars/4DMOS/data/KITTI_MOS/sequences/0000/labels/000019.label"
    nuscenes_filename = "/home/mars/4DMOS/data/nuScenes_MOS/sequences/0001/labels/000010.label"
    labels_waymo = np.fromfile(waymo_filename, dtype=np.uint32).reshape((-1))
    labels_waymo = labels_waymo & 0xFFFF  # Mask semantics in lower half
    labels_kitti = np.fromfile(kitti_filename, dtype=np.uint32)
    labels_nuscenes = np.fromfile(nuscenes_filename, dtype=np.uint32) & 0xFFFF

    # & 0xFFFF
    a = -100
    a1 = bin(-100)
    a2 = a & 0xFFFF
    c = int('00000000000000001111111111111111', 2)
    d = int('0xFFFF', 16)
    a = 1