#!/usr/bin/env python3
# @file      loss.py
# @author    Benedikt Mersch     [mersch@igg.uni-bonn.de]
# Copyright (c) 2022 Benedikt Mersch, all rights reserved

import torch
import torch.nn as nn

import MinkowskiEngine as ME


class MOSLoss(nn.Module):
    def __init__(self, n_classes, ignore_index):
        super().__init__()
        self.n_classes = n_classes
        self.ignore_index = ignore_index
        self.softmax = nn.Softmax(dim=1)
        weight = [0.0 if i in ignore_index else 1.0 for i in range(n_classes)]
        weight = torch.Tensor([w / sum(weight) for w in weight])
        self.loss = nn.NLLLoss(weight=weight)

    def compute_loss(self, out: ME.TensorField, past_labels: list):
        # Get raw point-wise scores
        logits = out.features

        # Set ignored classes to -inf to not influence softmax
        logits[:, self.ignore_index] = -float("inf")

        softmax = self.softmax(logits)
        log_softmax = torch.log(softmax.clamp(min=1e-8))

        # Prepare ground truth labels
        gt_labels = torch.cat(past_labels, dim=0)[:, 0]

        loss = self.loss(log_softmax, gt_labels.long())
        return loss
