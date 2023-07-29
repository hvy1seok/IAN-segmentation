import torch
import torch.nn as nn
import torch.nn.functional as F

import os

from .resnet3d import ResNet18_OS16, ResNet34_OS16, ResNet50_OS16, ResNet101_OS16, ResNet152_OS16, ResNet18_OS8, ResNet34_OS8


import torch
import torch.nn as nn
import torch.nn.functional as F

class ASPP(nn.Module):
    def __init__(self, num_classes):
        super(ASPP, self).__init__()
        in_ch = 513

        self.conv_1x1_1 = nn.Conv3d(in_ch, 256, kernel_size=1)
        self.bn_conv_1x1_1 = nn.BatchNorm3d(256)

        self.conv_3x3_1 = nn.Conv3d(in_ch, 256, kernel_size=3, stride=1, padding=6, dilation=6)
        self.bn_conv_3x3_1 = nn.BatchNorm3d(256)

        self.conv_3x3_2 = nn.Conv3d(in_ch, 256, kernel_size=3, stride=1, padding=12, dilation=12)
        self.bn_conv_3x3_2 = nn.BatchNorm3d(256)

        self.conv_3x3_3 = nn.Conv3d(in_ch, 256, kernel_size=3, stride=1, padding=18, dilation=18)
        self.bn_conv_3x3_3 = nn.BatchNorm3d(256)

        self.avg_pool = nn.AdaptiveAvgPool3d(1)

        self.conv_1x1_2 = nn.Conv3d(in_ch, 256, kernel_size=1)
        self.bn_conv_1x1_2 = nn.BatchNorm3d(256)

        self.conv_1x1_3 = nn.Conv3d(1280, 256, kernel_size=1)
        self.bn_conv_1x1_3 = nn.BatchNorm3d(256)

        self.conv_1x1_4 = nn.Conv3d(256, num_classes, kernel_size=1)

    def forward(self, feature_map):
        feature_map_h = feature_map.size()[2]
        feature_map_w = feature_map.size()[3]
        feature_map_c = feature_map.size()[4]

        out_1x1 = F.relu(self.bn_conv_1x1_1(self.conv_1x1_1(feature_map)))
        out_3x3_1 = F.relu(self.bn_conv_3x3_1(self.conv_3x3_1(feature_map)))
        out_3x3_2 = F.relu(self.bn_conv_3x3_2(self.conv_3x3_2(feature_map)))
        out_3x3_3 = F.relu(self.bn_conv_3x3_3(self.conv_3x3_3(feature_map)))

        out_img = self.avg_pool(feature_map)
        out_img = F.relu(self.bn_conv_1x1_2(self.conv_1x1_2(out_img)))
        out_img = F.interpolate(out_img, size=(feature_map_h, feature_map_w, feature_map_c), mode='trilinear', align_corners=True)

        out = torch.cat([out_1x1, out_3x3_1, out_3x3_2, out_3x3_3, out_img], 1)
        out = F.relu(self.bn_conv_1x1_3(self.conv_1x1_3(out)))
        out = self.conv_1x1_4(out)

        return out

class ASPP_Bottleneck(nn.Module):
    def __init__(self, num_classes):
        super(ASPP_Bottleneck, self).__init__()
        in_ch = 512

        self.conv_1x1_1 = nn.Conv3d(4*in_ch+1, 256, kernel_size=1)
        self.bn_conv_1x1_1 = nn.BatchNorm3d(256)

        self.conv_3x3_1 = nn.Conv3d(4*in_ch+1, 256, kernel_size=3, stride=1, padding=6, dilation=6)
        self.bn_conv_3x3_1 = nn.BatchNorm3d(256)

        self.conv_3x3_2 = nn.Conv3d(4*in_ch+1, 256, kernel_size=3, stride=1, padding=12, dilation=12)
        self.bn_conv_3x3_2 = nn.BatchNorm3d(256)

        self.conv_3x3_3 = nn.Conv3d(4*in_ch+1, 256, kernel_size=3, stride=1, padding=18, dilation=18)
        self.bn_conv_3x3_3 = nn.BatchNorm3d(256)

        self.avg_pool = nn.AdaptiveAvgPool3d(1)

        self.conv_1x1_2 = nn.Conv3d(4*in_ch+1, 256, kernel_size=1)
        self.bn_conv_1x1_2 = nn.BatchNorm3d(256)

        self.conv_1x1_3 = nn.Conv3d(1280, 256, kernel_size=1) # (1280 = 5*256)
        self.bn_conv_1x1_3 = nn.BatchNorm3d(256)

        self.conv_1x1_4 = nn.Conv3d(256, num_classes, kernel_size=1)

    def forward(self, feature_map):
        feature_map_h = feature_map.size()[2]
        feature_map_w = feature_map.size()[3]
        feature_map_c = feature_map.size()[4]

        out_1x1 = F.relu(self.bn_conv_1x1_1(self.conv_1x1_1(feature_map)))
        out_3x3_1 = F.relu(self.bn_conv_3x3_1(self.conv_3x3_1(feature_map)))
        out_3x3_2 = F.relu(self.bn_conv_3x3_2(self.conv_3x3_2(feature_map)))
        out_3x3_3 = F.relu(self.bn_conv_3x3_3(self.conv_3x3_3(feature_map)))

        out_img = self.avg_pool(feature_map)
        out_img = F.relu(self.bn_conv_1x1_2(self.conv_1x1_2(out_img)))
        out_img = F.interpolate(out_img, size=(feature_map_h, feature_map_w, feature_map_c), mode='trilinear', align_corners=True)
        
        out = torch.cat([out_1x1, out_3x3_1, out_3x3_2, out_3x3_3, out_img], 1)
        out = F.relu(self.bn_conv_1x1_3(self.conv_1x1_3(out)))
        out = self.conv_1x1_4(out)

        return out


class DeepLabV3_3D(nn.Module):
    def __init__(self, num_classes, emb_shape, input_channels, resnet, last_activation = 'sigmoid'):
        super(DeepLabV3_3D, self).__init__()
        self.num_classes = num_classes
        self.last_activation = last_activation
        self.emb_shape = torch.as_tensor(emb_shape)

        self.pos_emb_layer = nn.Linear(6, torch.prod(self.emb_shape).item())

        if resnet.lower() == 'resnet18_os16':
            self.resnet = ResNet18_OS16(input_channels)
        
        elif resnet.lower() == 'resnet34_os16':
            self.resnet = ResNet34_OS16(input_channels)
        
        elif resnet.lower() == 'resnet50_os16':
            self.resnet = ResNet50_OS16(input_channels)
        
        elif resnet.lower() == 'resnet101_os16':
            self.resnet = ResNet101_OS16(input_channels)
        
        elif resnet.lower() == 'resnet152_os16':
            self.resnet = ResNet152_OS16(input_channels)
        
        elif resnet.lower() == 'resnet18_os8':
            self.resnet = ResNet18_OS8(input_channels)
        
        elif resnet.lower() == 'resnet34_os8':
            self.resnet = ResNet34_OS8(input_channels)

        if resnet.lower() in ['resnet50_os16', 'resnet101_os16', 'resnet152_os16']:
            self.aspp = ASPP_Bottleneck(num_classes=self.num_classes)
        else:
            self.aspp = ASPP(num_classes=self.num_classes)

    def forward(self, x, emb_codes):

        h = x.size()[2]
        w = x.size()[3]
        c = x.size()[4]

        feature_map = self.resnet(x)
        emb_pos = self.pos_emb_layer(emb_codes).view(-1, 1, *self.emb_shape)
        feature_map = torch.cat((feature_map, emb_pos), dim=1)

        output = self.aspp(feature_map)

        output = F.interpolate(output, size=(h, w, c), mode='trilinear', align_corners=True)

        if self.last_activation == None: return output
        
        if self.last_activation.lower() == 'sigmoid':
            output = nn.Sigmoid()(output)
        
        elif self.last_activation.lower() == 'softmax':
            output = nn.Softmax()(output)
        
        return output