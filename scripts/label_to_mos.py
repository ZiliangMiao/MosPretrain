import os
import yaml
import copy
import click
import torch
import numpy as np
import multiprocessing

def save_mos_label(seq_idx):
    if dataset == "SEKITTI":
        semantic_config = yaml.safe_load(open(cfg["DATA"]["SEMANTIC_CONFIG_FILE"]))
        semantic_label_path = os.path.join(dataset_path, str(seq_idx).zfill(4), "labels")
        mos_label_path = os.path.join(dataset_path, str(seq_idx).zfill(4), "mos_labels")
        os.makedirs(mos_label_path, exist_ok=True)
        semantic_label_files = [os.path.join(dp, f) for dp, dn, fn in os.walk(os.path.expanduser(semantic_label_path)) for f in fn]
        semantic_label_files.sort()
        mos_label_files = [os.path.join(mos_label_path, os.path.split(i)[1]) for i in semantic_label_files]
        for semantic_label_file, mos_label_file in zip(semantic_label_files, mos_label_files):
            if os.path.isfile(semantic_label_file):
                labels = np.fromfile(semantic_label_file, dtype=np.uint32).reshape((-1)) & 0xFFFF
                mapped_labels = copy.deepcopy(labels)
                for semantic_label, mos_labels in semantic_config["learning_map"].items():
                    mapped_labels[labels == semantic_label] = mos_labels
                mapped_labels.tofile(mos_label_file)
                # Directly load saved mos labels and check if it is correct
                mos_labels = np.fromfile(mos_label_file, dtype=np.uint32).reshape((-1)) & 0xFFFF
                check_true = (mos_labels == mapped_labels).all()
                assert check_true
            else:
                print("semantic_label_file is not a file!")
                return torch.Tensor(1, 1).long()
        print("MOS label saved for SemanticKITTI sequence:", seq_idx)
    else:
        semantic_label_path = os.path.join(dataset_path, str(seq_idx).zfill(4), "labels")
        mos_label_path = os.path.join(dataset_path, str(seq_idx).zfill(4), "mos_labels")
        os.makedirs(mos_label_path, exist_ok=True)
        semantic_label_files = [os.path.join(dp, f) for dp, dn, fn in os.walk(os.path.expanduser(semantic_label_path)) for f in fn]
        semantic_label_files.sort()
        mos_label_files = [os.path.join(mos_label_path, os.path.split(i)[1]) for i in semantic_label_files]
        for semantic_label_file, mos_label_file in zip(semantic_label_files, mos_label_files):
            if os.path.isfile(semantic_label_file):
                labels = np.fromfile(semantic_label_file, dtype=np.uint32).reshape((-1)) & 0xFFFF
                mos_labels = np.zeros_like(labels)
                mos_labels[labels >= 251] = 2  # moving
                mos_labels[labels < 251] = 1  # static
                mos_labels[labels >= 65535] = 0  # invalid label
                mos_labels.tofile(mos_label_file)
                # double check if saved label is correct
                mos_label_check = np.fromfile(mos_label_file, dtype=np.uint32).reshape((-1)) & 0xFFFF
                check_true = (mos_labels == mos_label_check).all()
                assert check_true
            else:
                print("semantic_label_file is not a file!")
                return torch.Tensor(1, 1).long()
        print("MOS label saved for " + dataset + " sequence:", seq_idx)
    return None

if __name__ == '__main__':
    config = "./config/train_cfg.yaml"
    cfg = yaml.safe_load(open(config))
    dataset = cfg["TEST"]["DATASET"]
    dataset_path = cfg["DATASET"][dataset]["PATH"]
    train_seqs = cfg["DATASET"][dataset]["TRAIN"]
    val_seqs = cfg["DATASET"][dataset]["VAL"]
    test_seqs = cfg["DATASET"][dataset]["TEST"]
    seqs = train_seqs + val_seqs + test_seqs
    seqs.sort()

    pool = multiprocessing.Pool(processes=44)
    for seq_idx in seqs:
        pool.apply_async(func=save_mos_label, args=(seq_idx,))
    pool.close()
    pool.join()
    print("multi-processing success!")