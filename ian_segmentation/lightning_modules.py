from typing import Any, Sequence

import lightning as L
import torch
import torchio as tio
from lightning.pytorch.utilities.types import STEP_OUTPUT, TRAIN_DATALOADERS
from torch import nn
from torchio import SubjectsDataset

from ian_segmentation.metrics import BaseMetric


class BaseModule(L.LightningModule):
    def __init__(
        self, 
        train_dataset: SubjectsDataset,
        model: nn.Module, 
        loss_fn: nn.Module, 
        optimizer: Any,
        metrics: dict[str, BaseMetric],
        val_dataset: SubjectsDataset | None = None,
        scheduler: Any | None = None,
        predict_dataset: SubjectsDataset | None = None
    ) -> None:
        super().__init__()
        self.train_dataset = train_dataset
        self.model = model
        self.loss_fn = loss_fn
        self.optimizer = optimizer
        self.metrics = metrics
        self.val_dataset = val_dataset
        self.scheduler = scheduler
        self.predict_dataset = predict_dataset

        self.eps = 1e-10

    def _compute_partition_weights(self, gt: torch.Tensor, stage: str = 'train') -> tuple[bool, torch.Tensor]:
        assert stage in ['train', 'val'], 'parameter `stage` have to be `train` or `val`'
        
        partition_weights = 1
        gt_count = torch.sum(gt == 1, dim=list(range(1, gt.ndim)))
        any_gt = torch.sum(gt_count) == 0

        if not any_gt:
            partition_weights = self.eps + gt_count
            divisor = torch.sum(gt_count)
            if stage == 'val':
                divisor += self.eps
            partition_weights /= divisor

        return any_gt, partition_weights

    def training_step(self, batch: torch.Tensor | Sequence[torch.Tensor], batch_idx: int) -> STEP_OUTPUT | None:
        images, gt, emb_codes = batch

        any_gt, partition_weights = self._compute_partition_weights(gt)
        if any_gt:
            return

        preds = self.model(images, emb_codes)

        assert preds.ndim == gt.ndim, \
            f'Gt and output dimensions are not the same before loss. {preds.ndim} vs {gt.ndim}'
        
        loss = self.loss_fn(preds, gt, partition_weights)
        
        for metric_name in self.metrics.keys():
            self.metrics[metric_name].collect(preds, gt)

        self.log('train_loss', loss.item(), prog_bar=True, logger=True, on_step=True, on_epoch=True)

        return loss

    def on_train_epoch_end(self) -> None:
        for metric_name in self.metrics.keys():
            self.log(f'train_{metric_name}', self.metrics[metric_name].mean())
            self.metrics[metric_name].reset()

    def validation_step(self, batch: torch.Tensor | Sequence[torch.Tensor], batch_idx: int) -> STEP_OUTPUT | None:
        sampler = tio.inference.GridSampler(batch, )

        preds = self.model(images, emb_codes)

    def preprocess_before_validation(self, subject):
        pass

    def configure_optimizers(self) -> dict[str, Any]:
        config = {'optimizer': self.optimizer}

        if self.scheduler is not None:
            config['lr_scheduler'] = self.scheduler

        return config
    
    def train_dataloader(self) -> TRAIN_DATALOADERS:
        pass