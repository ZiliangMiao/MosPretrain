import numpy as np
import yaml
import os
import copy
import torch
from torch.utils.data import Dataset, DataLoader
from pytorch_lightning import LightningDataModule

from mos4d.datasets.utils import load_poses, load_calib, load_files
from mos4d.datasets.augmentation import (
    shift_point_cloud,
    rotate_point_cloud,
    jitter_point_cloud,
    random_flip_point_cloud,
    random_scale_point_cloud,
    rotate_perturbation_point_cloud,
)


class KittiSequentialModule(LightningDataModule):
    """A Pytorch Lightning module for Sequential KITTI data; Contains train, valid, test data"""

    def __init__(self, cfg):
        super(KittiSequentialModule, self).__init__()
        self.cfg = cfg

    def prepare_data(self):
        pass

    def setup(self, stage=None):
        """Dataloader and iterators for training, validation and test data"""

        ########## Point dataset splits
        train_set = KittiSequentialDataset(self.cfg, split="train")
        val_set = KittiSequentialDataset(self.cfg, split="val")
        test_set = KittiSequentialDataset(self.cfg, split="test")

        ########## Generate dataloaders and iterables

        self.train_loader = DataLoader(
            dataset=train_set,
            batch_size=self.cfg["TRAIN"]["BATCH_SIZE"],
            collate_fn=self.collate_fn,
            shuffle=self.cfg["DATA"]["SHUFFLE"],
            num_workers=self.cfg["DATA"]["NUM_WORKER"],  # num of multi-processing
            pin_memory=True,
            drop_last=False,  # drop the samples left from full batch
            timeout=0,
        )
        self.train_iter = iter(self.train_loader)

        self.valid_loader = DataLoader(
            dataset=val_set,
            batch_size=self.cfg["TRAIN"]["BATCH_SIZE"],
            collate_fn=self.collate_fn,
            shuffle=False,
            num_workers=self.cfg["DATA"]["NUM_WORKER"],
            pin_memory=True,
            drop_last=False,
            timeout=0,
        )
        self.valid_iter = iter(self.valid_loader)

        self.test_loader = DataLoader(
            dataset=test_set,
            batch_size=self.cfg["TRAIN"]["BATCH_SIZE"],
            collate_fn=self.collate_fn,
            shuffle=False,
            num_workers=self.cfg["DATA"]["NUM_WORKER"],
            pin_memory=True,
            drop_last=False,
            timeout=0,
        )
        self.test_iter = iter(self.test_loader)

        print(
            "Loaded {:d} training, {:d} validation and {:d} test samples.".format(
                len(train_set), len(val_set), (len(test_set))
            )
        )

    def train_dataloader(self):
        return self.train_loader

    def val_dataloader(self):
        return self.valid_loader

    def test_dataloader(self):
        return self.test_loader

    @staticmethod
    def collate_fn(batch):  # define how to merge a list of samples to from a mini-batch samples
        meta = [item[0] for item in batch]
        past_point_clouds = [item[1] for item in batch]
        past_labels = [item[2] for item in batch]
        return [meta, past_point_clouds, past_labels]


class KittiSequentialDataset(Dataset):
    """Semantic KITTI Dataset class"""
    def __init__(self, cfg, split):
        self.cfg = cfg
        self.mode = cfg["EXPT"]["MODE"]  # "TRAIN", "TEST"
        self.dataset = cfg[self.mode]["DATASET"]
        self.lidar_name = cfg["DATASET"][self.dataset]["LIDAR_NAME"]
        self.dataset_path = self.cfg["DATASET"][self.dataset]["PATH"]

        # Pose information
        self.transform = self.cfg["DATA"]["TRANSFORM"]
        self.poses = {}
        self.filename_poses = cfg["DATA"]["POSES"]

        # Semantic information
        self.semantic_config = yaml.safe_load(open(cfg["DATA"]["SEMANTIC_CONFIG_FILE"]))
        self.n_past_steps = self.cfg["MODEL"]["N_PAST_STEPS"]  # use how many past scans, default = 10

        self.split = split
        if self.split == "train":
            self.sequences = self.cfg["DATASET"][self.dataset]["TRAIN"]
        elif self.split == "val":
            self.sequences = self.cfg["DATASET"][self.dataset]["VAL"]
        elif self.split == "test":
            self.sequences = self.cfg["DATASET"][self.dataset]["TEST"]
        else:
            raise Exception("Split must be train/val/test")

        # Check if data and prediction frequency matches
        self.dt_pred = self.cfg["MODEL"]["DELTA_T_PREDICTION"]  # time resolution used for prediction
        self.dt_data = self.cfg["DATA"]["DELTA_T_DATA"]  # minimum time resolution of lidar scans
        assert (
            self.dt_pred >= self.dt_data
        ), "DELTA_T_PREDICTION needs to be larger than DELTA_T_DATA!"
        assert np.isclose(
            self.dt_pred / self.dt_data, round(self.dt_pred / self.dt_data), atol=1e-5
        ), "DELTA_T_PREDICTION needs to be a multiple of DELTA_T_DATA!"
        self.skip = round(self.dt_pred / self.dt_data)

        self.augment = self.cfg["TRAIN"]["AUGMENTATION"] and split == "train"  # data aug is only used for training
        self.dataset_size = 0
        self.filenames = {}  # dict: maps the sequence number to a list of scans file path
        self.idx_mapper = {}  # dict: maps a dataset idx to a seq number and the index of the current scan
        sample_idx = 0  # sample index of idx_mapper (a counter that crosses different sequences)
        for seq_idx in self.sequences:
            seqstr = "{0:04d}".format(int(seq_idx))
            path_to_seq = os.path.join(self.dataset_path, seqstr)
            scan_path = os.path.join(path_to_seq, self.lidar_name)
            self.filenames[seq_idx] = load_files(scan_path)  # load all files path in a folder and sort
            if self.transform:
                if self.dataset in {"SEKITTI", "KITTITRA", "KITTITRA_M", "APOLLO"}:
                    self.poses[seq_idx] = self.read_kitti_poses(path_to_seq)
                    # kitti pose: calib.txt (from lidar to camera), poses.txt (from current cam to previous cam)
                else:
                    self.poses[seq_idx] = self.read_poses(path_to_seq)
                assert len(self.poses[seq_idx]) == len(self.filenames[seq_idx])
            else:
                self.poses[seq_idx] = []

            # num of valid scans of current seq (we need 10 scans for prediction, so 00-08 are not valid), scan seqs begin at 09
            num_valid_scans = max(0, len(self.filenames[seq_idx]) - self.skip * (self.n_past_steps - 1))
            # Add to idx_mapper
            for idx in range(num_valid_scans):  # examp: sample index 00 -> scan index 09
                scan_idx = self.skip * (self.n_past_steps - 1) + idx
                self.idx_mapper[sample_idx] = (seq_idx, scan_idx)  # idx_mapper[sample_idx=0] = (seq_idx=00, scan_idx=09)
                sample_idx += 1
            self.dataset_size += num_valid_scans

    def __len__(self):  # return length of the whole dataset
        return self.dataset_size

    def __getitem__(self, sample_idx):  # define how to load each sample
        seq_idx, scan_idx = self.idx_mapper[sample_idx]
        # Load point clouds
        from_idx = scan_idx - self.skip * (self.n_past_steps - 1)  # included
        to_idx = scan_idx + 1  # not included
        scan_indices = list(range(from_idx, to_idx, self.skip))  # [from_idx, to_idx)
        scan_files = self.filenames[seq_idx][from_idx : to_idx : self.skip]
        if self.dataset in {"SEKITTI", "KITTITRA", "KITTITRA_M", "APOLLO"}:
            scans_list = [self.read_kitti_point_cloud(f) for f in scan_files]
        else:
            scans_list = [self.read_point_cloud(f) for f in scan_files]
        for i, pcd in enumerate(scans_list):
            if self.transform:  # transform to current pose
                from_pose = self.poses[seq_idx][scan_indices[i]]
                to_pose = self.poses[seq_idx][scan_indices[-1]]
                pcd = self.transform_point_cloud(pcd, from_pose, to_pose)
            rela_time_idx = i - self.n_past_steps + 1  # -9, -8 ... 0
            rela_timestamp = round(rela_time_idx * self.dt_pred, 3)  # -0.9, 0.8 ... 0.0 (relative timestamps)
            scans_list[i] = self.timestamp_tensor(pcd, rela_timestamp)
        point_clouds = torch.cat(scans_list, dim=0)  # 4D point cloud: [x y z rela_time]

        # Load labels
        label_files = [os.path.join(self.dataset_path, str(seq_idx).zfill(4), "labels", str(i).zfill(6) + ".label")
                       for i in scan_indices]
        labels_list = [self.read_labels(f) for f in label_files]
        # Save mos labels
        # mos_label_path = os.path.join(self.root_dir, str(seq_idx).zfill(4), "mos_labels")
        # os.makedirs(mos_label_path, exist_ok=True)
        # mos_label_files = [os.path.join(mos_label_path, str(i).zfill(6) + ".label") for i in scan_indices]
        # for semantic_label_file, mos_label_file in zip(label_files, mos_label_files):
        #     self.save_mos_labels(semantic_label_file, mos_label_file)

        # Load semantics labels and map to mos labels by semantic-kitti-mos.yaml
        for i, scan_label in enumerate(labels_list):
            rela_time_idx = i - self.n_past_steps + 1
            rela_timestamp = round(rela_time_idx * self.dt_pred, 3)
            labels_list[i] = self.timestamp_tensor(scan_label, rela_timestamp)
        labels = torch.cat(labels_list, dim=0)

        if self.augment:
            point_clouds, labels = self.augment_data(point_clouds, labels)

        meta = (seq_idx, scan_idx, scan_indices)
        return [meta, point_clouds, labels]  # [[index of current sequence, current scan, all scans], all scans, all labels]

    def transform_point_cloud(self, past_point_clouds, from_pose, to_pose):
        transformation = torch.Tensor(np.linalg.inv(to_pose) @ from_pose)
        NP = past_point_clouds.shape[0]
        xyz1 = torch.hstack([past_point_clouds, torch.ones(NP, 1)]).T
        past_point_clouds = (transformation @ xyz1).T[:, :3]
        return past_point_clouds

    def augment_data(self, past_point_clouds, past_labels):
        past_point_clouds = rotate_point_cloud(past_point_clouds)
        past_point_clouds = rotate_perturbation_point_cloud(past_point_clouds)
        past_point_clouds = jitter_point_cloud(past_point_clouds)
        past_point_clouds = shift_point_cloud(past_point_clouds)
        past_point_clouds = random_flip_point_cloud(past_point_clouds)
        past_point_clouds = random_scale_point_cloud(past_point_clouds)
        return past_point_clouds, past_labels

    def read_kitti_point_cloud(self, filename):
        """Load point clouds from .bin file"""
        point_cloud = np.fromfile(filename, dtype=np.float32)
        point_cloud = torch.tensor(point_cloud.reshape((-1, 4)))
        point_cloud = point_cloud[:, :3]
        return point_cloud

    def read_point_cloud(self, filename):
        """Load point clouds from .bin file"""
        point_cloud = np.fromfile(filename, dtype=np.float32)
        point_cloud = torch.tensor(point_cloud.reshape((-1, 3)))
        return point_cloud

    def read_labels(self, filename):
        """Load moving object labels from .label file"""
        if os.path.isfile(filename):
            labels = np.fromfile(filename, dtype=np.uint32)
            labels = labels.reshape((-1))
            labels = labels & 0xFFFF  # Mask semantics in lower half
            mapped_labels = copy.deepcopy(labels)
            for semantic_label, mos_label in self.semantic_config["learning_map"].items():
                mapped_labels[labels == semantic_label] = mos_label
            return_labels = torch.Tensor(mapped_labels.astype(np.float32)).long()
            return_labels = return_labels.reshape((-1, 1))
            return return_labels
        else:
            return torch.Tensor(1, 1).long()

    def save_mos_labels(self, semantic_label_file, mos_label_file):
        """Load moving object labels from .label file"""
        if os.path.isfile(semantic_label_file):
            labels = np.fromfile(semantic_label_file, dtype=np.uint32)
            labels = labels.reshape((-1))
            labels = labels & 0xFFFF  # Mask semantics in lower half
            mapped_labels = copy.deepcopy(labels)
            for semantic_label, mos_label in self.semantic_config["learning_map"].items():
                mapped_labels[labels == semantic_label] = mos_label
            mapped_labels.tofile(mos_label_file)
            # Directly load saved mos labels and check if it is correct
            # mos_labels = np.fromfile(mos_label_file, dtype=np.uint32).reshape((-1)) & 0xFFFF
            # check_true = (mos_labels == mapped_labels).all()
            return None
        else:
            return torch.Tensor(1, 1).long()

    @staticmethod
    def timestamp_tensor(tensor, time):
        """Add time as additional column to tensor"""
        n_points = tensor.shape[0]
        time = time * torch.ones((n_points, 1))
        timestamped_tensor = torch.hstack([tensor, time])
        return timestamped_tensor

    def read_kitti_poses(self, path_to_seq):
        pose_file = os.path.join(path_to_seq, self.filename_poses)
        calib_file = os.path.join(path_to_seq, "calib.txt")
        poses = np.array(load_poses(pose_file))
        inv_frame0 = np.linalg.inv(poses[0])

        # load calibrations
        T_cam_velo = load_calib(calib_file)
        T_cam_velo = np.asarray(T_cam_velo).reshape((4, 4))
        T_velo_cam = np.linalg.inv(T_cam_velo)

        # convert kitti poses from camera coord to LiDAR coord
        new_poses = []
        for pose in poses:
            new_poses.append(T_velo_cam.dot(inv_frame0).dot(pose).dot(T_cam_velo))
        poses = np.array(new_poses)
        return poses

    def read_poses(self, path_to_seq):
        pose_file = os.path.join(path_to_seq, self.filename_poses)
        poses = np.array(load_poses(pose_file))  # from current vehicle frame to global frame
        return poses