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


class People_Dataset(Dataset):
    def __init__(self, root, imgs, transforms=None):
        self.root = root
        self.imgs = imgs                # 이미지 파일명 list
        self.img_path_list = []         # 이미지 파일 경로 list
        self.box_list = []              # Ground Truth B-Box list
        self.num_obj_list = []          # 모든 이미지의 Object 갯수 list
        self.label_list = []            # 모든 이미지 내의 Object Class Label list

        self.transforms = transforms
        self.max_obj_num = 0

        for i in tqdm(range(len(self.imgs)), desc='get data...'):
            img = Image.open(os.path.join(self.root, self.imgs[i])).convert("RGB")
            num_objs, boxes, labels = self._get_infos(i)
            boxes = torch.as_tensor(boxes, dtype=torch.float32)
            labels = torch.as_tensor(labels, dtype=torch.int64)

            if num_objs != 0:

                if num_objs > self.max_obj_num:
                    self.max_obj_num = num_objs

                self.img_path_list.append(os.path.join(self.root, self.imgs[i]))
                self.num_obj_list.append(num_objs)
                self.box_list.append(boxes)
                self.label_list.append(labels)

    def __getitem__(self, idx):  # arg로 들어온 idx에 해당하는 이미지 불러오기
        img = Image.open(self.img_path_list[idx]).convert("RGB")
        img = transforms.ToTensor()(img)
        boxes = self.box_list[idx]
        num_objs = self.num_obj_list[idx]
        labels = self.label_list[idx]
        iscrowd = torch.zeros((num_objs,), dtype=torch.int64)

        if num_objs != 1:
            area = torch.as_tensor(boxes[:, 2] * boxes[:, 3], dtype=torch.float32)
        else:
            area = torch.as_tensor(boxes[0, 2] * boxes[0, 3], dtype=torch.float32).unsqueeze(0)

        boxes = torch.cat((boxes, torch.zeros(self.max_obj_num - num_objs, boxes.size()[1])),
                          dim=0)  # max_obj_num 만큼 tensor 크기 단일화를 위한 padding
        labels = torch.cat((labels, torch.zeros(self.max_obj_num - num_objs).type(torch.IntTensor) + 3), dim=0)
        iscrowd = torch.cat((iscrowd, torch.zeros(self.max_obj_num - num_objs).type(torch.IntTensor) + 2), dim=0)
        area = torch.cat((area, torch.zeros(self.max_obj_num - num_objs)), dim=0)

        target = {}
        target["boxes"] = boxes
        target["labels"] = labels
        target["iscrowd"] = iscrowd
        target["area"] = area

        return img, target  # return 튜플 => (img, target => boxes, labels, area, iscrowd)

    def __len__(self):
        return len(self.img_path_list)

    def _get_infos(self, idx):
        num_objs = 0
        boxes = []
        labels = []

        file_name = self.imgs[idx][:-4] + '.xml'

        with open(os.path.join(self.root, file_name)) as xml_file:
            doc = ET.parse(xml_file)
            annots = doc.getroot()

            for thing in annots.iter("object"):
                xmin = int(thing.find("bndbox").findtext("xmin"))
                xmax = int(thing.find("bndbox").findtext("xmax"))
                ymin = int(thing.find("bndbox").findtext("ymin"))
                ymax = int(thing.find("bndbox").findtext("ymax"))

                if xmax > xmin and ymax > ymin and xmax <= 416 and xmin >= 0 and ymax <= 416 and ymin >= 0:
                    num_objs += 1  # 해당 이미지의 object 갯수 1 증가

                    cx = ((xmin + xmax) / 2) / 416  # Original Image size = 256 X 256 => convert to YOLO B-Box
                    cy = ((ymin + ymax) / 2) / 416
                    w = (xmax - xmin) / 416
                    h = (ymax - ymin) / 416

                    boxes.append([cx, cy, w, h])  # 현재 확인하고 object의 bbox 추가

                    label = thing.findtext("name")
                    if label == "mask":
                        label = 1
                    elif label == "no-mask":
                        label = 2
                    elif label == "improper_mask":
                        label = 0
                    labels.append(label)  # 현재 확인하고 있는 object의 label 추가

        return num_objs, boxes, labels


train_root = os.path.join(os.getcwd(), "datasets", "train")  ## Train Image data 경로 설정
train_list = list(os.listdir(train_root))

train_imgs = []
train_annots = []
for a in range(len(train_list)):
    if train_list[a][-4:] == ".jpg":
        train_imgs.append(train_list[a])
        continue

    if train_list[a][-4:] == ".xml":
        train_annots.append(train_list[a])

valid_root = os.path.join(os.getcwd(), "datasets", "valid")  ## Valid Image data 경로 설정
valid_list = list(os.listdir(valid_root))

valid_imgs = []
valid_annots = []
for a in range(len(valid_list)):
    if valid_list[a][-4:] == ".jpg":
        valid_imgs.append(valid_list[a])
        continue

    if valid_list[a][-4:] == ".xml":
        valid_annots.append(valid_list[a])

test_root = os.path.join(os.getcwd(), "datasets", "test")  ## Test Image data 경로 설정
test_list = list(os.listdir(test_root))

test_imgs = []
test_annots = []
for a in range(len(test_list)):
    if test_list[a][-4:] == ".jpg":
        test_imgs.append(test_list[a])
        continue

    if test_list[a][-4:] == ".xml":
        test_annots.append(test_list[a])

train_data = People_Dataset(root=train_root, imgs=train_imgs, transforms=None)
valid_data = People_Dataset(root=valid_root, imgs=valid_imgs, transforms=None)
test_data = People_Dataset(root=test_root, imgs=test_imgs, transforms=None)

del train_data, valid_data

def get_test_data():
    return test_data
