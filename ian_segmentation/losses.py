import torch
from torch import nn


class LossFactory:
    def __init__(self, names, classes, weights=None):
        self.names = names
        if not isinstance(self.names, list):
            self.names = [self.names]

        print(f'Losses used: {self.names}')
        self.classes = classes
        self.weights = weights
        self.losses = {}
        for name in self.names:
            loss = self.get_loss(name)
            self.losses[name] = loss

    def get_loss(self, name):
        if name == 'CrossEntropyLoss':
            loss_fn = CrossEntropyLoss(self.weights, True)
        elif name == 'BCEWithLogitsLoss':
            loss_fn = BCEWithLogitsLoss(self.weights)
        elif name == 'Jaccard':
            loss_fn = JaccardLoss(weight=self.weights)
        else:
            raise Exception(f"Loss function {name} can't be found.")

        return loss_fn

    def __call__(self, pred, gt, partition_weights):
        """
        SHAPE MUST BE Bx1xHxW
        :param pred:
        :param gt:
        :return:
        """
        assert pred.device == gt.device
        assert gt.device != 'cpu'

        # print(f'pred has: {pred.view(2, -1).sum(-1)}')
        # print(f'gt has: {gt.view(2, -1).sum(-1)}')

        cur_loss = []
        for loss_name in self.losses.keys():
            loss = self.losses[loss_name](pred, gt)
            if torch.isnan(loss.sum()):
                raise ValueError(f'Loss {loss_name} has some NaN')
            # print(f'Loss {self.losses[loss_name].__class__.__name__}: {loss}')
            loss = loss * partition_weights
            cur_loss.append(loss.mean())
        return torch.sum(torch.stack(cur_loss))


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