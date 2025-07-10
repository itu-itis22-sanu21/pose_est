'''
# with_mobilenet.py
import torch
import torch.nn as nn
import torch.nn.functional as F

class Model(nn.Module):
    def __init__(self, num_refinement_stages=2):
        super().__init__()
        # Lightweight MobileNet-like feature extractor
        self.backbone = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
        )

        self.initial_stage = nn.Sequential(
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 13, kernel_size=1),  # 13 heatmaps for keypoints
        )

        self.refinement_stages = nn.ModuleList()
        for _ in range(num_refinement_stages):
            self.refinement_stages.append(
                nn.Sequential(
                    nn.Conv2d(13, 128, kernel_size=7, padding=3),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(128, 13, kernel_size=1),
                )
            )

    def forward(self, x):
        x = self.backbone(x)
        out = []
        initial = self.initial_stage(x)
        out.append(initial)
        for stage in self.refinement_stages:
            x = stage(out[-1])
            out.append(x)
        return out  # list of heatmaps per stage
'''
import torch
import torch.nn as nn
import torch.nn.functional as F
from modules.conv import conv, conv_dw, conv_dw_no_bn

# Assuming conv, conv_dw, conv_dw_no_bn are your conv helpers from modules.conv

class Cpm(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.align = conv(in_channels, out_channels, kernel_size=1, padding=0, bn=False)
        self.trunk = nn.Sequential(
            conv_dw_no_bn(out_channels, out_channels),
            conv_dw_no_bn(out_channels, out_channels),
            conv_dw_no_bn(out_channels, out_channels)
        )
        self.conv = conv(out_channels, out_channels, bn=False)

    def forward(self, x):
        x = self.align(x)
        x = self.conv(x + self.trunk(x))
        return x


class InitialStage(nn.Module):
    def __init__(self, num_channels, num_heatmaps):
        super().__init__()
        self.trunk = nn.Sequential(
            conv(num_channels, num_channels, bn=False),
            conv(num_channels, num_channels, bn=False),
            conv(num_channels, num_channels, bn=False)
        )
        self.heatmaps = nn.Sequential(
            conv(num_channels, 512, kernel_size=1, padding=0, bn=False),
            conv(512, num_heatmaps, kernel_size=1, padding=0, bn=False, relu=False)
        )

    def forward(self, x):
        trunk_features = self.trunk(x)
        heatmaps = self.heatmaps(trunk_features)
        return heatmaps


class RefinementStageBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.initial = conv(in_channels, out_channels, kernel_size=1, padding=0, bn=False)
        self.trunk = nn.Sequential(
            conv(out_channels, out_channels),
            conv(out_channels, out_channels, dilation=2, padding=2)
        )

    def forward(self, x):
        initial_features = self.initial(x)
        trunk_features = self.trunk(initial_features)
        return initial_features + trunk_features


class RefinementStage(nn.Module):
    def __init__(self, in_channels, out_channels, num_heatmaps):
        super().__init__()
        self.trunk = nn.Sequential(
            RefinementStageBlock(in_channels, out_channels),
            RefinementStageBlock(out_channels, out_channels),
            RefinementStageBlock(out_channels, out_channels),
            RefinementStageBlock(out_channels, out_channels),
            RefinementStageBlock(out_channels, out_channels)
        )
        self.heatmaps = nn.Sequential(
            conv(out_channels, out_channels, kernel_size=1, padding=0, bn=False),
            conv(out_channels, num_heatmaps, kernel_size=1, padding=0, bn=False, relu=False)
        )

    def forward(self, x):
        trunk_features = self.trunk(x)
        heatmaps = self.heatmaps(trunk_features)
        return heatmaps


class PoseEstimationWithMobileNet(nn.Module):
    def __init__(self, num_refinement_stages=1, num_channels=128, num_heatmaps=13):
        super().__init__()
        self.model = nn.Sequential(
            conv(3, 32, stride=2, bias=False),
            conv_dw(32, 64),
            conv_dw(64, 128, stride=2),
            conv_dw(128, 128),
            conv_dw(128, 256, stride=2),
            conv_dw(256, 256),
            conv_dw(256, 512),  # conv4_2
            conv_dw(512, 512, dilation=2, padding=2),
            conv_dw(512, 512),
            conv_dw(512, 512),
            conv_dw(512, 512),
            conv_dw(512, 512)  # conv5_5
        )
        self.cpm = Cpm(512, num_channels)

        self.initial_stage = InitialStage(num_channels, num_heatmaps)
        self.refinement_stages = nn.ModuleList()
        for _ in range(num_refinement_stages):
            # input channels to refinement = num_channels + num_heatmaps (concat backbone + heatmaps)
            self.refinement_stages.append(
                RefinementStage(num_channels + num_heatmaps, num_channels, num_heatmaps)
            )

    def forward(self, x):
        backbone_features = self.model(x)
        backbone_features = self.cpm(backbone_features)

        heatmaps = self.initial_stage(backbone_features)
        outputs = [heatmaps]

        for refinement_stage in self.refinement_stages:
            # concat backbone features with last heatmaps output
            concat_input = torch.cat([backbone_features, outputs[-1]], dim=1)
            heatmaps = refinement_stage(concat_input)
            outputs.append(heatmaps)

        return outputs  # list of heatmaps at each stage
