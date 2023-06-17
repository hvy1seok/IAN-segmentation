import numpy as np
import torch
from torch import nn
import torch.nn.functional as F


class BCEWithLogitsLoss(torch.nn.Module):
    def __init__(self, weights=None):
        super().__init__()
        self.weights = weights

    def forward(self, pred, gt):
        if pred.shape[1] == 1:
            pred = pred.squeeze()
            gt = gt.float()
            self.loss_fn = nn.BCEWithLogitsLoss(pos_weight=1/self.weights[0])
        else:
            # one hot encoding for cross entropy with digits. Bx1xHxW -> BxCxHxW
            B, C, Z, H, W = pred.shape
            gt_flat = gt.reshape(-1).unsqueeze(dim=1)  # 1xB*Z*H*W

            gt_onehot = torch.zeros(size=(B * Z * H * W, C), dtype=torch.float)  # 1xB*Z*H*W destination tensor
            gt_onehot.scatter_(1, gt_flat, 1)  # writing the conversion in the destination tensor

            gt = torch.squeeze(gt_onehot).reshape(B, Z, H, W, C)  # reshaping to the original shape
            pred = pred.permute(0, 2, 3, 4, 1)  # for BCE we want classes in the last axis
            self.loss_fn = nn.BCEWithLogitsLoss(pos_weight=self.weights)

        return self.loss_fn(pred, gt)


class CrossEntropyLoss(torch.nn.Module):
    def __init__(self, weights=None, apply_sigmoid=True):
        super().__init__()
        self.weights = weights
        self.apply_sigmoid = apply_sigmoid
        self.loss_fn = nn.CrossEntropyLoss(weight=self.weights)
        self.sigmoid = torch.nn.Sigmoid()

    def forward(self, pred, gt):
        pred = self.sigmoid(pred)
        return self.loss_fn(pred, gt)
    

    import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

class DiceLoss(nn.Module):
    # TODO: Check about partition_weights, see original code
    # what i didn't understand is that for dice loss, partition_weights gets
    # multiplied inside the forward and also in the factory_loss function
    # I think that this is wrong, and removed it from the forward
    def __init__(self, classes):
        super().__init__()
        self.eps = 1e-06
        self.classes = classes
        self.weights = partition_weights

    def forward(self, pred, gt):
        included = [v for k, v in self.classes.items() if k not in ['UNLABELED']]
        gt_onehot = torch.nn.functional.one_hot(gt.squeeze().long(), num_classes=len(self.classes))
        if gt.shape[0] == 1:  # we need to add a further axis after the previous squeeze()
            gt_onehot = gt_onehot.unsqueeze(0)

        gt_onehot = torch.movedim(gt_onehot, -1, 1)
        input_soft = F.softmax(pred, dim=1)
        dims = (2, 3, 4)

        intersection = torch.sum(input_soft * gt_onehot, dims)
        cardinality = torch.sum(input_soft + gt_onehot, dims)
        dice_score = 2. * intersection / (cardinality + self.eps)
        return 1. - dice_score[:, included]


class JaccardLoss(torch.nn.Module):
    def __init__(self, weight=None, size_average=True, per_volume=False, apply_sigmoid=False,
                 min_pixels=5):
        super().__init__()
        self.size_average = size_average
        self.weight = weight
        self.per_volume = per_volume
        self.apply_sigmoid = apply_sigmoid
        self.min_pixels = min_pixels

    def forward(self, pred, gt):
        assert pred.shape[1] == 1, 'this loss works with a binary prediction'
        if self.apply_sigmoid:
            pred = torch.sigmoid(pred)

        batch_size = pred.size()[0]
        eps = 1e-6
        if not self.per_volume:
            batch_size = 1
        dice_gt = gt.contiguous().view(batch_size, -1).float()
        dice_pred = pred.contiguous().view(batch_size, -1)
        intersection = torch.sum(dice_pred * dice_gt, dim=1)
        union = torch.sum(dice_pred + dice_gt, dim=1) - intersection
        loss = 1 - (intersection + eps) / (union + eps)
        return loss