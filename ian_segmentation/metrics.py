from abc import ABC, abstractmethod
from statistics import mean

import torch


class BaseMetric(ABC):
    def __init__(self) -> None:
        self.metric_per_batch = []

    def reset(self) -> None:
        self.metric_per_batch.clear()

    def collect(self, pred: torch.Tensor, gt: torch.Tensor) -> None:
        pred = pred.detach().to(torch.uint8).cuda()
        gt = gt.detach().to(torch.uint8).cuda()

        pred = pred[None, ...] if pred.ndim == 3 else pred
        gt = gt[None, ...] if gt.ndim == 3 else gt

        metric_score = self.compute(pred, gt)
        self.metric_per_batch.append(metric_score)

    @abstractmethod
    def compute(self, pred: torch.Tensor, gt: torch.Tensor) -> float:
        raise NotImplementedError

    def mean(self) -> float:
        metric_mean = 0 if len(self.metric_per_batch) == 0 else mean(self.metric_per_batch)

        self.reset()
        return metric_mean
    

class IoU(BaseMetric):
    def __init__(self) -> None:
        super().__init__()
        self.eps = 1e-6

    def compute(self, pred: torch.Tensor, gt: torch.Tensor) -> float:
        pred, gt = self.preprocess(pred, gt)

        intersection = (pred & gt).sum()
        dice_union = pred.sum() + gt.sum()
        iou_union = dice_union - intersection

        iou = (intersection + self.eps) / (iou_union + self.eps)

        return iou.item()
    
    def preprocess(self, pred: torch.Tensor, gt: torch.Tensor) -> tuple[torch.Tensor]:
        pred = (pred > 0.5).squeeze().detach()
        gt = gt.squeeze()

        return pred, gt
    

class Dice(BaseMetric):
    def __init__(self) -> None:
        super().__init__()
        self.eps = 1e-6

    def compute(self, pred: torch.Tensor, gt: torch.Tensor) -> float:
        pred, gt = self.preprocess(pred, gt)

        intersection = (pred & gt).sum()
        dice_union = pred.sum() + gt.sum()
        
        dice = (2 * intersection + self.eps) / (dice_union + self.eps)

        return dice.item()
    
    def preprocess(self, pred: torch.Tensor, gt: torch.Tensor) -> tuple[torch.Tensor]:
        pred = (pred > 0.5).squeeze().detach()
        gt = gt.squeeze()

        return pred, gt
