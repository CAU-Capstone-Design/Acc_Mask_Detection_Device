import os
import time
import copy

import xml.etree.ElementTree as ET
from xml.etree.ElementTree import Element, dump, ElementTree

import numpy as np
from PIL import Image, ImageDraw
import cv2
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from tqdm import tqdm

from collections import OrderedDict

import torch
from torch import nn
import torch.utils.data
from torch.utils.data import Dataset, DataLoader
from torch.utils.data import random_split

import torchvision
from torchvision import datasets, transforms
import torchvision.models as models
from torchsummary import summary

device = "cuda" if torch.cuda.is_available() else "cpu"

import warnings
warnings.filterwarnings("ignore")


## Basic Covn Block in YOLO v3 Model
class BasicConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding):
        super().__init__()

        self.conv_block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding),
            nn.BatchNorm2d(out_channels),
            nn.ReLU()
        )

    def forward(self, input):
        return self.conv_block(input)


## Residual Block in YOLO v3 Model
class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels

        self.conv_1 = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0)
        self.conv_2 = nn.Conv2d(out_channels, in_channels, kernel_size=3, stride=1, padding=1)
        self.relu = nn.ReLU()
        self.batchnorm_1 = nn.BatchNorm2d(in_channels)
        self.batchnorm_2 = nn.BatchNorm2d(out_channels)

    def forward(self, input):  # Architecture of Full Pre-Activation
        temp1 = self.batchnorm_1(input)  # Batch-Normalization first
        temp2 = self.relu(temp1)  # ReLU Activation second
        temp3 = self.conv_1(temp2)  # Pass Convolutional Layer next, and iterate twice

        temp4 = self.batchnorm_2(temp3)
        temp5 = self.relu(temp4)
        out_temp = self.conv_2(temp5)

        output = input + out_temp

        return output


## FPN For Top Down Method in YOLO v3 Model
class FPNTopDown(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()

        self.topdown_block = nn.Sequential(
            BasicConv(in_channels, out_channels, kernel_size=1, stride=1, padding=0),
            BasicConv(out_channels, out_channels * 2, kernel_size=3, stride=1, padding=1),
            BasicConv(out_channels * 2, out_channels, kernel_size=1, stride=1, padding=0),
            BasicConv(out_channels, out_channels * 2, kernel_size=3, stride=1, padding=1),
            BasicConv(out_channels * 2, out_channels, kernel_size=1, stride=1, padding=0),
        )

    def forward(self, input):
        return self.topdown_block(input)


## 3-Scale Feature Extraction in YOLO v3 Model
class YOLOExt(nn.Module):
    def __init__(self, channels, anchors, n_class=3, img_size=416):
        super().__init__()

        self.anchors = anchors  # small 3개 + medium 3개 + large 3개
        self.n_anchor = len(anchors)
        self.n_class = n_class
        self.img_size = img_size
        self.grid_size = 0

        self.conv_block = nn.Sequential(
            BasicConv(channels, channels * 2, kernel_size=3, stride=1, padding=1),
            nn.Conv2d(channels * 2, 24, 1, stride=1, padding=0)
            # 24 = 3(# of scale) X (4=bbox coordinates + 1=conf score + 3=# of class)
        )

    def forward(self, input):
        input = self.conv_block(input)

        batch_size = input.size(0)
        grid_size = input.size(2)  # 13 or 26 or 52
        device = input.device

        pred = input.view(batch_size, self.n_anchor, self.n_class + 5, grid_size,
                          grid_size)  # [batch_size, 3, 8(3+4+1), grid_size, grid_size]
        pred = pred.permute(0, 1, 3, 4, 2).contiguous()  # [batch_size, 3, grid_size, grid_size, 8(4+1+3)] 로 변환
        obj_score = torch.sigmoid(pred[..., 4])  # object conf score
        pred_class = torch.sigmoid(pred[..., 5:])  # class prediction

        if grid_size != self.grid_size:
            self.compute_grid_cells(grid_size)

        pred_boxes = self.compute_pred_bbox(pred)
        output = torch.cat((pred_boxes.view(batch_size, -1, 4), obj_score.view(batch_size, -1, 1),
                            pred_class.view(batch_size, -1, self.n_class)), -1)

        return output

    def compute_grid_cells(self, grid_size):
        self.grid_size = grid_size
        self.stride = self.img_size / self.grid_size  # stride = 1개의 행/열 당 몇개의 Grid Cell이 존재하는지

        self.grid_x = torch.arange(grid_size, device="cpu").repeat(1, 1, grid_size, 1).type(torch.float32)
        self.grid_y = torch.arange(grid_size, device="cpu").repeat(1, 1, grid_size, 1).transpose(3, 2).type(
            torch.float32)  # 원래는 transpose(3, 2)가 있음

        scaled_anchors = [(aw / self.stride, ah / self.stride) for aw, ah in
                          self.anchors]  # anchor 가로/세로를 grid size와 동일하게 맞춤(정규화)
        self.scaled_anchors = torch.tensor(scaled_anchors, device="cpu")

        self.anchor_w = self.scaled_anchors[:, 0:1].view((1, self.n_anchor, 1, 1))
        self.anchor_h = self.scaled_anchors[:, 1:2].view((1, self.n_anchor, 1, 1))

    def compute_pred_bbox(self, pred):
        device = pred.device

        cx = torch.sigmoid(pred[..., 0])
        cy = torch.sigmoid(pred[..., 1])
        w = pred[..., 2]
        h = pred[..., 3]

        pred_boxes = torch.zeros_like(pred[..., :4]).to(device)
        pred_boxes[..., 0] = cx.data + self.grid_x
        pred_boxes[..., 1] = cy.data + self.grid_y
        pred_boxes[..., 2] = torch.exp(w.data) * self.anchor_w
        pred_boxes[..., 3] = torch.exp(h.data) * self.anchor_h

        return pred_boxes * self.stride  # 원래는 pred_boxes * self.stride(?)


## DarkNet-53 in YOLO v3 Model
class DarkNet(nn.Module):
    def __init__(self, anchors, blocks=[1, 2, 8, 8, 4], n_class=3):
        super().__init__()

        self.conv_block = BasicConv(3, 32, 3, stride=1, padding=1)

        self.res_block_1 = self._make_residual(64, blocks[0])
        self.res_block_2 = self._make_residual(128, blocks[1])
        self.res_block_3 = self._make_residual(256, blocks[2])  # Top-Down for 52 X 52 Feature map
        self.res_block_4 = self._make_residual(512, blocks[3])  # Top-Down for 26 X 26 Feature map
        self.res_block_5 = self._make_residual(1024, blocks[4])  # Top-Down for 13 X 13 Feature map

        self.fpn_topdown_1 = FPNTopDown(1024, 512)
        self.fpn_topdown_2 = FPNTopDown(768, 256)
        self.fpn_topdown_3 = FPNTopDown(384, 128)

        self.before_upsample_1 = BasicConv(512, 256, 1, stride=1, padding=0)
        self.before_upsample_2 = BasicConv(256, 128, 1, stride=1, padding=0)

        self.yolo_large = YOLOExt(512, anchors=anchors[2])  # 이후에 1024 X 13 X 13
        self.yolo_medium = YOLOExt(256, anchors=anchors[1])  # 이후에 512 X 26 X 26
        self.yolo_small = YOLOExt(128, anchors=anchors[0])  # 이후에 256 X 52 X 52

        self.upsample = nn.Upsample(scale_factor=2)

    def _make_residual(self, channels, num_blocks):
        res_blocks = []
        res_blocks.append(BasicConv(channels // 2, channels, 3, stride=2, padding=1))

        for i in range(num_blocks):
            res_blocks.append(ResidualBlock(channels, channels // 2))

        return nn.Sequential(*res_blocks)

    def forward(self, input):
        input = self.conv_block(input)

        c1 = self.res_block_1(input)
        c2 = self.res_block_2(c1)
        c3 = self.res_block_3(c2)
        c4 = self.res_block_4(c3)
        c5 = self.res_block_5(c4)

        p5 = self.fpn_topdown_1(c5)  # 2022-11-21 기준 수정된 아키텍쳐
        p4 = self.fpn_topdown_2(
            torch.cat((self.upsample(self.before_upsample_1(p5)), c4), 1))  # channel dimension 기준으로 concetenation
        p3 = self.fpn_topdown_3(
            torch.cat((self.upsample(self.before_upsample_2(p4)), c3), 1))  # channel dimension 기준으로 concetenation

        large_feature = self.yolo_large(p5)
        medium_feature = self.yolo_medium(p4)
        small_feature = self.yolo_small(p3)

        return torch.cat((large_feature, medium_feature, small_feature), 1), [large_feature, medium_feature,
                                                                              small_feature]


anchors = [[(10, 13), (16, 30), (33, 23)], [(30, 61), (62, 45), (59, 119)], [(116, 90), (156, 198), (373, 326)]]
sample = torch.randn(1, 3, 416, 416).to(device)
with torch.no_grad():
    yolov3 = DarkNet(anchors).to(device)
    output_cat, output = yolov3(sample)
