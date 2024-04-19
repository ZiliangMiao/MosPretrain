import os
from typing import List
import torch
import numpy as np
from pytorch_lightning import LightningDataModule
from torch.utils.data import Dataset, DataLoader, sampler

from nuscenes.nuscenes import NuScenes
from nuscenes.utils.splits import create_splits_logs
from nuscenes.utils.data_classes import LidarPointCloud
from pyquaternion import Quaternion
from nuscenes.utils.geometry_utils import transform_matrix

from mos4d.datasets.augmentation import (
    shift_point_cloud,
    rotate_point_cloud,
    jitter_point_cloud,
    random_flip_point_cloud,
    random_scale_point_cloud,
    rotate_perturbation_point_cloud,
)

class NuscSequentialModule(LightningDataModule):
    """A Pytorch Lightning module for Sequential Nusc data; Contains train, valid, test data"""

    def __init__(self, cfg, nusc, mode):
        super(NuscSequentialModule, self).__init__()
        self.cfg = cfg
        self.nusc = nusc
        self.mode = mode

    def prepare_data(self):
        pass

    def setup(self, stage=None):
        """Dataloader and iterators for training, validation and test data"""
        if self.mode == "train":
            train_set = NuscSequentialDataset(self.cfg, self.nusc, split=self.cfg["DATASET"]["NUSC"]["TRAIN"])
            val_set = NuscSequentialDataset(self.cfg, self.nusc, split=self.cfg["DATASET"]["NUSC"]["VAL"])
            ########## Generate dataloaders and iterables
            train_data_pct = self.cfg["TRAIN"]["DATA_PCT"] / 100
            self.train_loader = DataLoader(
                dataset=train_set,
                batch_size=self.cfg["TRAIN"]["BATCH_SIZE"],
                collate_fn=self.collate_fn,
                num_workers=self.cfg["TRAIN"]["NUM_WORKERS"],  # num of multi-processing
                pin_memory=True,
                drop_last=False,  # drop the samples left from full batch
                timeout=0,
                sampler=sampler.WeightedRandomSampler(weights=torch.ones(len(train_set)),
                                                      num_samples=int(train_data_pct * len(train_set))),
            )
            self.train_iter = iter(self.train_loader)
            self.valid_loader = DataLoader(
                dataset=val_set,
                batch_size=self.cfg["TRAIN"]["BATCH_SIZE"],
                collate_fn=self.collate_fn,
                shuffle=False,
                num_workers=self.cfg["TRAIN"]["NUM_WORKERS"],
                pin_memory=True,
                drop_last=False,
                timeout=0,
            )
            self.valid_iter = iter(self.valid_loader)
            print("Loaded {:d} training and {:d} validation samples.".format(len(train_set), len(val_set)))
        elif self.mode == "test":  # no test labels, use val set as test set
            test_set = NuscSequentialDataset(self.cfg, self.nusc, split=self.cfg["DATASET"]["NUSC"]["TEST"])
            ########## Generate dataloaders and iterables
            self.test_loader = DataLoader(
                dataset=test_set,
                batch_size=self.cfg["TEST"]["BATCH_SIZE"],
                collate_fn=self.collate_fn,
                shuffle=False,
                num_workers=self.cfg["TEST"]["NUM_WORKERS"],
                pin_memory=True,
                drop_last=False,
                timeout=0,
                # sampler=sampler.WeightedRandomSampler(weights=torch.ones(len(test_set)), num_samples=int(0.01 * len(test_set))),
            )
            self.test_iter = iter(self.test_loader)
            print("Loaded {:d} test samples.".format(len(test_set)))
        else:
            raise ValueError("Invalid Nusc Dataset Mode.")

    def train_dataloader(self):
        return self.train_loader

    def val_dataloader(self):
        return self.valid_loader

    def test_dataloader(self):
        return self.test_loader

    @staticmethod
    def collate_fn(batch):  # define how to merge a list of samples to from a mini-batch samples
        sample_data_token = [item[0] for item in batch]
        point_cloud = [item[1] for item in batch]
        mos_label = [item[2] for item in batch]
        return [sample_data_token, point_cloud, mos_label]

class NuscSequentialDataset(Dataset):
    def __init__(self, cfg, nusc, split):
        self.cfg = cfg
        self.version = cfg["DATASET"]["NUSC"]["VERSION"]
        self.data_dir = cfg["DATASET"]["NUSC"]["PATH"]

        self.split = split  # "train" "val" "mini_train" "mini_val" "test"

        self.nusc = nusc

        self.n_past_steps = self.cfg["DATA"]["N_PAST_STEPS"]  # use how many past scans, default = 10
        self.dt_data = self.cfg["DATASET"]["NUSC"]["DELTA_T"]  # minimum time resolution of lidar scans
        self.dt_pred = self.cfg["DATA"]["DELTA_T_PRED"]  # time resolution used for prediction

        split_logs = create_splits_logs(split, self.nusc)
        self.sample_tokens, self.sample_data_tokens = self._split_to_samples(split_logs)

        # sample token: 10 past lidar tokens; ignore the samples that have less than 10 past lidar scans
        self.sample_lidar_tokens_dict, self.valid_sample_data_tokens = self._get_sample_lidar_tokens_dict(self.sample_tokens)

        # self.gt_poses = self._load_poses()
        self.transform = self.cfg["DATA"]["TRANSFORM"]

    def __len__(self):
        return len(self.sample_lidar_tokens_dict)

    def __getitem__(self, sample_idx):  # define how to load each sample
        # sample
        sample_tokens = list(self.sample_lidar_tokens_dict.keys())
        sample_token = sample_tokens[sample_idx]
        sample = self.nusc.get("sample", sample_token)
        sample_data_token = sample['data']['LIDAR_TOP']
        sample_data = self.nusc.get('sample_data', sample_data_token)

        # reference pose (current timestamp)
        ref_pose_token = sample_data['ego_pose_token']
        ref_pose = self.nusc.get('ego_pose', ref_pose_token)
        ref_pose_mat_inv = transform_matrix(ref_pose['translation'], Quaternion(ref_pose['rotation']),
                                            inverse=True)  # from global to ref car

        # calib pose
        calib_token = sample_data['calibrated_sensor_token']
        calib = self.nusc.get('calibrated_sensor', calib_token)
        calib_mat = transform_matrix(calib['translation'], Quaternion(calib['rotation']))  # from lidar to car
        calib_mat_inv = transform_matrix(calib['translation'], Quaternion(calib['rotation']),
                                         inverse=True)  # from car to lidar

        # sample data: concat 4d point clouds
        lidar_tokens = self.sample_lidar_tokens_dict[sample_token]
        pts_with_rela_time = []  # 4D Point Cloud (relative timestamp)
        num_curr_pts = 0
        for ref_time_idx, lidar_token in enumerate(lidar_tokens):
            if lidar_token is None:
                lidar_data = self.nusc.get('sample_data', lidar_tokens[0])
                lidar_file = os.path.join(self.data_dir, lidar_data['filename'])
                points = LidarPointCloud.from_file(lidar_file).points.T  # [num_pts, 4]
                points_curr = points[:, :3]
                points_curr = np.zeros((points_curr.shape[0], points_curr.shape[1]))  # padded with zeros
            else:
                # from current scan to previous scans
                lidar_data = self.nusc.get('sample_data', lidar_token)
                lidar_file = os.path.join(self.data_dir, lidar_data['filename'])
                points = LidarPointCloud.from_file(lidar_file).points.T  # [num_pts, 4]
                points_curr = points[:, :3]
                if ref_time_idx == 0:
                    num_curr_pts = len(points_curr)
            if self.transform:
                # transform point cloud from curr pose to ref pose
                curr_pose_token = lidar_data['ego_pose_token']
                curr_pose = self.nusc.get('ego_pose', curr_pose_token)
                curr_pose_mat = transform_matrix(curr_pose['translation'],
                                                 Quaternion(curr_pose['rotation']))  # from curr car to global

                # transformation: curr_lidar -> curr_car -> global -> ref_car -> ref_lidar
                trans_mat = calib_mat @ curr_pose_mat @ ref_pose_mat_inv @ calib_mat_inv
                points_curr_homo = np.hstack([points_curr, np.ones((points_curr.shape[0], 1))]).T
                points_ref = torch.from_numpy((trans_mat @ points_curr_homo).T[:, :3])

                # 0 - 9, delta_t = 0.05s
                rela_timestamp = -round(ref_time_idx * self.dt_data, 3)
                point_cloud_with_time = self.timestamp_tensor(points_ref, rela_timestamp)
                pts_with_rela_time.append(point_cloud_with_time)
        point_cloud = torch.cat(pts_with_rela_time, dim=0)  # 4D point cloud: [x y z rela_time]
        point_cloud = point_cloud.float()  # point cloud has to be float32, otherwise MikEngine will get RunTimeError: in_feat.scalar_type() == kernel.scalar_type()

        # load labels
        mos_labels_dir = os.path.join(self.data_dir, "mos_labels", self.version)
        mos_label_file = os.path.join(mos_labels_dir, sample_data_token + "_mos.label")
        mos_label = torch.tensor(np.fromfile(mos_label_file, dtype=np.uint8))
        
        # mask ego vehicle point
        if self.cfg["MODE"] == "train" or self.cfg["MODE"] == "finetune":
            if self.cfg["TRAIN"]["AUGMENTATION"]:  # will not change the mapping from point to label
                point_cloud = self.augment_data(point_cloud)
            if self.cfg["TRAIN"]["EGO_MASK"]:
                ego_mask = self.get_ego_mask(point_cloud)
                time_mask = point_cloud[:, -1] == 0.0
                point_cloud = point_cloud[~ego_mask]
                mos_label = mos_label[~ego_mask[time_mask]]
        return [sample_data_token, point_cloud, mos_label]  # sample_data_token, past 4d point cloud, sample mos label
    
    @staticmethod
    def get_ego_mask(points):  # mask the points of ego vehicle: x [-0.8, 0.8], y [-1.5, 2.5]
    # nuscenes car: length (4.084 m), width (1.730 m), height (1.562 m)
    # https://forum.nuscenes.org/t/dimensions-of-the-ego-vehicle-used-to-gather-data/550
        ego_mask = torch.logical_and(
            torch.logical_and(-0.865 <= points[:, 0], points[:, 0] <= 0.865),
            torch.logical_and(-1.5 <= points[:, 1], points[:, 1] <= 2.5),
        )
        return ego_mask

    def augment_data(self, past_point_clouds):
        past_point_clouds = rotate_point_cloud(past_point_clouds)
        past_point_clouds = rotate_perturbation_point_cloud(past_point_clouds)
        past_point_clouds = jitter_point_cloud(past_point_clouds)
        past_point_clouds = shift_point_cloud(past_point_clouds)
        past_point_clouds = random_flip_point_cloud(past_point_clouds)
        past_point_clouds = random_scale_point_cloud(past_point_clouds)
        return past_point_clouds

    @staticmethod
    def timestamp_tensor(tensor, time):
        """Add time as additional column to tensor"""
        n_points = tensor.shape[0]
        time = time * torch.ones((n_points, 1))
        timestamped_tensor = torch.hstack([tensor, time])
        return timestamped_tensor

    def _split_to_samples(self, split_logs: List[str]) -> List[str]:
        sample_tokens = []  # store the sample tokens
        sample_data_tokens = []
        for sample in self.nusc.sample:
            sample_data_token = sample['data']['LIDAR_TOP']
            scene = self.nusc.get('scene', sample['scene_token'])
            log = self.nusc.get('log', scene['log_token'])
            logfile = log['logfile']
            if logfile in split_logs:
                sample_data_tokens.append(sample_data_token)
                sample_tokens.append(sample['token'])
        return sample_tokens, sample_data_tokens

    def _get_scene_tokens(self, split_logs: List[str]) -> List[str]:
        """
        Convenience function to get the samples in a particular split.
        :param split_logs: A list of the log names in this split.
        :return: The list of samples.
        """
        scene_tokens = []  # store the scene tokens
        for scene in self.nusc.scene:
            log = self.nusc.get('log', scene['log_token'])
            logfile = log['logfile']
            if logfile in split_logs:
                scene_tokens.append(scene['token'])
        return scene_tokens

    def _get_sample_lidar_tokens_dict(self, sample_tokens: List[str]):
        sample_lidar_dict = {}
        valid_sample_data_tokens = []  # samples that have 10 past scans (sample data token)
        for sample_token in sample_tokens:
            # Get records from DB.
            sample = self.nusc.get("sample", sample_token)
            sample_data_token = sample["data"]["LIDAR_TOP"]
            sample_data = self.nusc.get("sample_data", sample_data_token)

            lidar_tokens = []
            lidar_tokens.append(sample_data_token)  # lidar token of current scan
            lidar_prev_idx = 0
            sample_data_prev = sample_data
            while lidar_prev_idx < self.n_past_steps - 1:
                lidar_prev_idx += 1
                sample_data_curr = sample_data_prev
                if sample_data_curr["prev"] != "":
                    sample_data_prev_token = sample_data_curr["prev"]
                    sample_data_prev = self.nusc.get("sample_data", sample_data_prev_token)
                    lidar_tokens.append(sample_data_prev_token)
                else:
                    lidar_tokens.append(None)  # padded with zero point clouds
                    sample_data_prev = sample_data_curr
                    # break

            if len(lidar_tokens) == self.n_past_steps:
                sample_lidar_dict[sample_token] = lidar_tokens
                valid_sample_data_tokens.append(lidar_tokens[0])
        return sample_lidar_dict, valid_sample_data_tokens

    def _get_sample_pose_tokens_dict(self, sample_tokens: List[str]):
        sample_pose_dict = {}
        for sample_token in sample_tokens:
            sample = self.nusc.get("sample", sample_token)
            sample_data_token = sample["data"]["LIDAR_TOP"]
            sample_data = self.nusc.get("sample_data", sample_data_token)
            pose_token = sample_data['ego_pose_token']
            pose_tokens = []
            pose_tokens.append(pose_token)  # ego pose token of current scan

            lidar_prev_idx = 0
            while lidar_prev_idx < self.n_past_steps - 1:
                lidar_prev_idx += 1
                if sample_data["prev"] != "":
                    sample_data_prev_token = sample_data["prev"]
                    sample_data_prev = self.nusc.get("sample_data", sample_data_prev_token)
                    pose_token_prev = sample_data_prev['ego_pose_token']
                    pose_tokens.append(pose_token_prev)
                else:
                    break
            if len(pose_tokens) == self.n_past_steps:
                sample_pose_dict[sample_token] = pose_tokens
        return sample_pose_dict