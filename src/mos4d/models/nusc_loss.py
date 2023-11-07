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
        weight = torch.Tensor([w / sum(weight) for w in weight])  # ignore unknown class when calculate loss
        self.loss = nn.NLLLoss(weight=weight)
    def compute_loss(self, out: ME.TensorField, num_curr_pts: list, mos_labels: list):
        logits = []
        for features, num_points in zip(out.decomposed_features, num_curr_pts):
            features[:, self.ignore_index] = -float("inf")
            logits.append(features[0:num_points, :])
        logits = torch.cat(logits, dim=0)
        softmax = self.softmax(logits)
        log_softmax = torch.log(softmax.clamp(min=1e-8))

        gt_labels = torch.cat(mos_labels, dim=0)
        loss = self.loss(log_softmax, gt_labels.long())  # dtype of label of torch.nllloss has to be torch.long
        return loss
