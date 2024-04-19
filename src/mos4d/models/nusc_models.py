import os
import yaml

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from pytorch_lightning import LightningModule
import MinkowskiEngine as ME
from MinkowskiEngine.modules.resnet_block import BasicBlock
from mos4d.models.MinkowskiEngine.minkunet import MinkUNetBase
from mos4d.models.metrics import ClassificationMetrics
from mos4d.datasets.nusc_dataset import NuscSequentialDataset

from sklearn.metrics import confusion_matrix

#######################################
# Modules
#######################################

class MinkUNet14(MinkUNetBase):
    BLOCK = BasicBlock
    LAYERS = (1, 1, 1, 1, 1, 1, 1, 1)
    # PLANES = (8, 16, 32, 64, 64, 32, 16, 8)
    PLANES = (8, 32, 128, 256, 256, 128, 32, 8)
    INIT_DIM = 8

class MOSModel(nn.Module):
    def __init__(self, cfg: dict, n_classes: int):
        super().__init__()
        self.dt_prediction = cfg["DATA"]["DELTA_T_PRED"]
        ds = cfg["DATA"]["VOXEL_SIZE"]
        self.quantization = torch.Tensor([ds, ds, ds, self.dt_prediction])
        self.MinkUNet = MinkUNet14(in_channels=1, out_channels=n_classes, D=4)

    def forward(self, past_point_clouds):
        quantization = self.quantization.type_as(past_point_clouds[0])

        past_point_clouds = [
            torch.div(point_cloud, quantization) for point_cloud in past_point_clouds
        ]
        features = [
            0.5 * torch.ones(len(point_cloud), 1).type_as(point_cloud)
            for point_cloud in past_point_clouds
        ]

        coords, features = ME.utils.sparse_collate(past_point_clouds, features)
        tensor_field = ME.TensorField(features=features, coordinates=coords,
                                      quantization_mode=ME.SparseTensorQuantizationMode.RANDOM_SUBSAMPLE)
        sparse_tensor = tensor_field.sparse()

        # sinput = ME.SparseTensor(features=features,  # Convert to a tensor
        #                          coordinates=coords,
        #                          # coordinates must be defined in an integer grid.
        #                          quantization_mode=ME.SparseTensorQuantizationMode.UNWEIGHTED_AVERAGE)

        predicted_sparse_tensor = self.MinkUNet(sparse_tensor)

        out = predicted_sparse_tensor.slice(tensor_field)
        out.coordinates[:, 1:] = torch.mul(out.coordinates[:, 1:], quantization)
        return out

#######################################
# Lightning Module
#######################################

class MOSNet(LightningModule):
    def __init__(self, hparams: dict):
        super().__init__()
        self.save_hyperparameters(hparams)
        self.cfg = hparams

        self.dt_prediction = self.hparams["DATA"]["DELTA_T_PRED"]
        self.lr = self.hparams["TRAIN"]["LR"]
        self.lr_epoch = hparams["TRAIN"]["LR_EPOCH"]
        self.lr_decay = hparams["TRAIN"]["LR_DECAY"]
        self.weight_decay = hparams["TRAIN"]["WEIGHT_DECAY"]
        self.n_past_steps = hparams["DATA"]["N_PAST_STEPS"]

        self.n_classes = 3  # 0 -> unknown, 1 -> static, 2 -> moving
        self.ignore_class_idx = [0]  # ignore unknown class when calculating scores
        self.mov_class_idx = 2

        # need to change, kitti and nusc
        # self.dataset_name = self.hparams["DATA"]["dataset_name"]
        # self.poses = (
        #     self.hparams["DATASET"][self.dataset_name]["POSES"].split(".")[0]
        #     if self.hparams["DATA"]["TRANSFORM"]
        #     else "no_poses"
        # )

        self.encoder = MOSModel(hparams, self.n_classes)
        self.ClassificationMetrics = ClassificationMetrics(self.n_classes, self.ignore_class_idx)

        # init
        self.training_step_outputs = []
        self.validation_step_outputs = []

        # # Prediction
        # self.test_dataset = hparams["TEST"]["DATASET"]
        # self.test_datapath = hparams["DATASET"][self.test_dataset]["PATH"]
        # # NUSC
        # if self.test_dataset == "NUSC":
        #     self.version = hparams["DATASET"]["NUSC"]["VERSION"]

        # loss calculation
        self.softmax = nn.Softmax(dim=1)
        weight = [0.0 if i in self.ignore_class_idx else 1.0 for i in range(self.n_classes)]
        weight = torch.Tensor([w / sum(weight) for w in weight])  # ignore unknown class when calculate loss
        self.loss = nn.NLLLoss(weight=weight)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=self.lr_epoch, gamma=self.lr_decay)
        return [optimizer], [scheduler]

    def get_confusion_matrix(self, curr_feats_list, mos_labels):
        pred_logits = torch.cat(curr_feats_list, dim=0).detach().cpu()
        gt_labels = torch.cat(mos_labels, dim=0).detach().cpu()
        conf_mat = self.ClassificationMetrics.compute_confusion_matrix(pred_logits, gt_labels)
        return conf_mat

    def getLoss(self, curr_feats_list, mos_labels: list):
        # loop each batch data
        for curr_feats in curr_feats_list:
            curr_feats[:, self.ignore_class_idx] = -float("inf")
        logits = torch.cat(curr_feats_list, dim=0)
        softmax = self.softmax(logits)
        log_softmax = torch.log(softmax.clamp(min=1e-8))
        gt_labels = torch.cat(mos_labels, dim=0)
        assert len(gt_labels) == len(logits)
        loss = self.loss(log_softmax, gt_labels.long())  # dtype of label of torch.nllloss has to be torch.long
        return loss

    def forward(self, past_point_clouds: dict):
        out = self.encoder(past_point_clouds)
        # only output current timestamp
        curr_coords_list = []
        curr_feats_list = []
        for feats, coords in zip(out.decomposed_features, out.decomposed_coordinates):
            curr_time_mask = coords[:, -1] == 0.0
            curr_feats = feats[curr_time_mask]
            curr_coords = coords[curr_time_mask]
            curr_coords_list.append(curr_coords)
            curr_feats_list.append(curr_feats)
        return (curr_coords_list, curr_feats_list)

    def training_step(self, batch: tuple, batch_idx, dataloader_index=0):
        _, point_clouds, mos_labels = batch
        _, curr_feats_list = self.forward(point_clouds)
        loss = self.getLoss(curr_feats_list, mos_labels)

        # Logging metrics
        conf_mat = self.get_confusion_matrix(curr_feats_list, mos_labels)  # confusion matrix
        tp, fp, fn = self.ClassificationMetrics.getStats(conf_mat)  # stat of current sample
        iou = self.ClassificationMetrics.getIoU(tp, fp, fn)[self.mov_class_idx]
        self.log("train_loss_step", loss.item(), on_step=True, prog_bar=True, logger=True)
        self.log("train_iou_step", iou.item() * 100, on_step=True, prog_bar=True, logger=True)
        self.training_step_outputs.append({"train_loss": loss.item(), "confusion_matrix": conf_mat})
        torch.cuda.empty_cache()
        return loss

    def on_train_epoch_end(self):
        conf_mat_list = [output["confusion_matrix"] for output in self.training_step_outputs]
        acc_conf_mat = torch.zeros(self.n_classes, self.n_classes)
        for conf_mat in conf_mat_list:
            acc_conf_mat = acc_conf_mat.add(conf_mat)

        tp, fp, fn = self.ClassificationMetrics.getStats(acc_conf_mat)  # stat of current sample
        iou = self.ClassificationMetrics.getIoU(tp, fp, fn)[self.mov_class_idx]
        self.log("train_iou_epoch", iou.item() * 100, on_epoch=True, logger=True)

        # clean
        self.training_step_outputs = []
        torch.cuda.empty_cache()

    # 暂时修改， 为了测试predict和valid输出为什么不同
    # def validation_step(self, batch: tuple, batch_idx):
    #     # unfold batch data
    #     sample_data_tokens, point_clouds, mos_labels = batch
    #     batch_size = len(point_clouds)
    #     # network prediction
    #     curr_coords_list, curr_feats_list = self.forward(point_clouds)
    #     # loop batch data list
    #     acc_conf_mat = torch.zeros(self.n_classes, self.n_classes)
    #     for i, (curr_feats, mos_label) in enumerate(zip(curr_feats_list, mos_labels)):
    #         # get ego mask
    #         curr_time_mask = point_clouds[i][:, -1] == 0.0
    #         ego_mask = NuscSequentialDataset.get_ego_mask(point_clouds[i][curr_time_mask])
    #         # compute confusion matrix
    #         conf_mat = self.get_confusion_matrix([curr_feats[~ego_mask]], [mos_label[~ego_mask]])  # input is lists
    #         acc_conf_mat = acc_conf_mat.add(conf_mat)
    #         # compute iou metric
    #         tp, fp, fn = self.ClassificationMetrics.getStats(conf_mat)  # stat of current sample
    #         iou = self.ClassificationMetrics.getIoU(tp, fp, fn)[self.mov_class_idx]
    #         print(f"Validation Sample Index {i + batch_idx * batch_size}, Moving Object IoU w/o ego vehicle: {iou.item() * 100}")
    #     self.validation_step_outputs.append(acc_conf_mat.detach().cpu())
    #     torch.cuda.empty_cache()
    #     return {"confusion_matrix": acc_conf_mat.detach().cpu()}
    #
    # def on_validation_epoch_end(self):
    #     conf_mat_list = self.validation_step_outputs
    #     acc_conf_mat = torch.zeros(self.n_classes, self.n_classes)
    #     for conf_mat in conf_mat_list:
    #         acc_conf_mat = acc_conf_mat.add(conf_mat)
    #     tp, fp, fn = self.ClassificationMetrics.getStats(acc_conf_mat)  # stat of current sample
    #     iou = self.ClassificationMetrics.getIoU(tp, fp, fn)[self.mov_class_idx]
    #     self.log("val_iou", iou.item() * 100, on_epoch=True, logger=True)
    #     return self.validation_step_outputs

    def validation_step(self, batch: tuple, batch_idx):
        # unfold batch data
        _, point_clouds, mos_labels = batch
        _, curr_feats_list = self.forward(point_clouds)
        conf_mat = self.get_confusion_matrix(curr_feats_list, mos_labels)
        self.validation_step_outputs.append(conf_mat.detach().cpu())
        torch.cuda.empty_cache()
        return {"confusion_matrix": conf_mat.detach().cpu()}

    def on_validation_epoch_end(self):
        conf_mat_list = self.validation_step_outputs
        acc_conf_mat = torch.zeros(self.n_classes, self.n_classes)
        for conf_mat in conf_mat_list:
            acc_conf_mat = acc_conf_mat.add(conf_mat)
        tp, fp, fn = self.ClassificationMetrics.getStats(acc_conf_mat)  # stat of current sample
        iou = self.ClassificationMetrics.getIoU(tp, fp, fn)[self.mov_class_idx]
        self.log("val_iou", iou.item() * 100,  on_epoch=True, logger=True)

        # clean
        self.validation_step_outputs = []
        torch.cuda.empty_cache()

    # func "predict_step" is called by "trainer.predict"
    def predict_step(self, batch: tuple, batch_idx):
        # unfold batch data
        sample_data_tokens, point_clouds, mos_labels = batch
        batch_size = len(point_clouds)
        # network prediction
        curr_coords_list, curr_feats_list = self.forward(point_clouds)
        # loop batch data list
        acc_conf_mat = torch.zeros(self.n_classes, self.n_classes)
        for i, (curr_feats, mos_label) in enumerate(zip(curr_feats_list, mos_labels)):
            # get ego mask
            curr_time_mask = point_clouds[i][:, -1] == 0.0
            ego_mask = NuscSequentialDataset.get_ego_mask(point_clouds[i][curr_time_mask])
            # compute confusion matrix
            conf_mat = self.get_confusion_matrix([curr_feats[~ego_mask]], [mos_label[~ego_mask]])  # input is lists
            acc_conf_mat = acc_conf_mat.add(conf_mat)
            # compute iou metric
            tp, fp, fn = self.ClassificationMetrics.getStats(conf_mat)  # stat of current sample
            iou = self.ClassificationMetrics.getIoU(tp, fp, fn)[self.mov_class_idx]
            print(f"Validation Sample Index {i + batch_idx * batch_size}, Moving Object IoU w/o ego vehicle: {iou.item() * 100}")
        torch.cuda.empty_cache()
        return {"confusion_matrix": acc_conf_mat.detach().cpu()}



        # for batch_idx in range(len(batch[0])):
        #     sample_data_token = sample_data_tokens[batch_idx]
        #     mos_label = mos_labels[batch_idx].cpu().detach().numpy()
        #     step = 0  # only evaluate the performance of current timestamp
        #     coords = out.coordinates_at(batch_idx)
        #     logits = out.features_at(batch_idx)
        #
        #     t = round(-step * self.dt_prediction, 3)
        #     mask = coords[:, -1].isclose(torch.tensor(t))
        #     masked_logits = logits[mask]
        #     masked_logits[:, self.ignore_class_idx] = -float("inf")  # ingore: 0, i.e., unknown or noise
        #
        #     pred_softmax = F.softmax(masked_logits, dim=1)
        #     pred_softmax = pred_softmax.detach().cpu().numpy()
        #     assert pred_softmax.shape[1] == 3
        #     assert pred_softmax.shape[0] >= 0
        #     sum = np.sum(pred_softmax[:, 1:3], axis=1)
        #     assert np.isclose(sum, np.ones_like(sum)).all()
        #     moving_confidence = pred_softmax[:, 2]
        #
        #     # directly output the mos label, without any bayesian strategy (do not need confidences_to_labels.py file)
        #     pred_label = np.ones_like(moving_confidence, dtype=np.uint8)  # notice: dtype of nusc labels are always uint8
        #     pred_label[moving_confidence > 0.5] = 2
        #     pred_label_dir = os.path.join(self.test_datapath, "4dmos_sekitti_pred", self.version)
        #     os.makedirs(pred_label_dir, exist_ok=True)
        #     pred_label_file = os.path.join(pred_label_dir, sample_data_token + "_mos_pred.label")
        #     pred_label.tofile(pred_label_file)
        # torch.cuda.empty_cache()



