import torch
from torch import nn


class ModelFactory(nn.Module):
    def __init__(self, model_name, num_classes, in_ch, emb_shape=None):
        super(ModelFactory, self).__init__()
        self.model_name = model_name
        self.num_classes = num_classes
        self.in_ch = in_ch
        self.emb_shape = emb_shape

    def get(self):
        if self.model_name == 'PosPadUNet3D':
            from .PosPadUNet3D import PosPadUNet3D
            assert self.emb_shape is not None
            return PosPadUNet3D(self.num_classes, self.emb_shape, self.in_ch)
        elif self.model_name == 'DeepLabV3':
            from .deeplabv33d import DeepLabV3_3D
            return DeepLabV3_3D(self.num_classes, self.emb_shape, self.in_ch, 'resnet18_os8')
        elif self.model_name == 'SwinUNETR':
            from .SwinUNETR import SwinUNETR
            return SwinUNETR(self.num_classes, self.emb_shape, self.in_ch)
        else:
            raise ValueError(f'Model {self.model_name} not found')
