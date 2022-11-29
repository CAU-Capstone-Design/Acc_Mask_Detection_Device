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
import torchvision.transforms.functional as F
from torchsummary import summary

device = "cuda" if torch.cuda.is_available() else "cpu"

import warnings
warnings.filterwarnings("ignore")

from dataclass import get_test_data
from yolov3_model import *

# bluetooth part
from bluetooth import *

# LED
import RPi.GPIO as GPIO

def led_on(pin) :
    GPIO.setmode(GPIO.BCM)
    GPIO.setup(pin, GPIO.OUT)
    GPIO.output(pin, GPIO.HIGH)

def led_off(pin) :
    GPIO.setmode(GPIO.BCM)
    GPIO.setup(pin, GPIO.OUT)
    GPIO.output(pin, GPIO.LOW)

# bluetooth
socket = BluetoothSocket( RFCOMM )
socket.connect(( "98:D3:31:F6:1A:A9",1))

# torch.cuda.empty_cache()

root = "./best_epoch"

for file_name in os.listdir(root):
    if file_name[-4:] == ".pth":
        best_yolo_model = file_name

best_yolo_path = os.path.join(root, best_yolo_model)

# train_loader, valid_loader, test_loader = get_loader()
test_data = get_test_data()

yolov3_model = DarkNet(anchors).to(device)
output_cat, output = yolov3_model(sample)
yolov3_model.load_state_dict(torch.load(best_yolo_path, map_location=torch.device('cpu')))
yolov3_model.eval()

anchors = [[(10, 13), (16, 30), (33, 23)], [(30, 61), (62, 45), (59, 119)], [(116, 90), (156, 198), (373, 326)]]
sample = torch.randn(1, 3, 416, 416).to(device)
with torch.no_grad():
    output_cat, output = yolov3_model(sample)

# print("========== YOLOv3 Test ==========")
# yolov3_tester = YOLOv3_Actuator(
#     root=root,
#     train_loader=train_loader,
#     valid_loader=valid_loader,
#     test_loader=test_loader,
#     model=yolov3_model,
#     opt="adam",
#     lr=0.001,
#     has_scheduler=True,
#     device=device).to(device)
# yolov3_tester.test()


def is_belong(box_a, box_b):
    box_a_xmin = box_a[0] - (box_a[2] / 2)
    box_a_xmax = box_a[0] + (box_a[2] / 2)
    box_a_ymin = box_a[1] - (box_a[3] / 2)
    box_a_ymax = box_a[1] + (box_a[3] / 2)

    box_b_xmin = box_b[0] - (box_b[2] / 2)
    box_b_xmax = box_b[0] + (box_b[2] / 2)
    box_b_ymin = box_b[1] - (box_b[3] / 2)
    box_b_ymax = box_b[1] + (box_b[3] / 2)

    if (box_a_xmin < box_b_xmin) and (box_a_ymin < box_b_ymin) and (box_a_xmax > box_b_xmax) and (
            box_a_ymax > box_b_ymax):
        return True
    elif (box_a_xmin > box_b_xmin) and (box_a_ymin > box_b_ymin) and (box_a_xmax < box_b_xmax) and (
            box_a_ymax < box_b_ymax):
        return True
    else:
        return False


def iou_infer(max_box, comp_box):
    max_box_xmin = max_box[0] - (max_box[2] / 2)
    max_box_xmax = max_box[0] + (max_box[2] / 2)
    max_box_ymin = max_box[1] - (max_box[3] / 2)
    max_box_ymax = max_box[1] + (max_box[3] / 2)

    comp_box_xmin = comp_box[0] - (comp_box[2] / 2)
    comp_box_xmax = comp_box[0] + (comp_box[2] / 2)
    comp_box_ymin = comp_box[1] - (comp_box[3] / 2)
    comp_box_ymax = comp_box[1] + (comp_box[3] / 2)

    xmin = max(max_box_xmin, comp_box_xmin)
    xmax = min(max_box_xmax, comp_box_xmax)
    ymin = max(max_box_ymin, comp_box_ymin)
    ymax = min(max_box_ymin, comp_box_ymax)

    area_inter = (xmax - xmin) * (ymax - ymin)
    area_union = (max_box[2] * max_box[3]) + (comp_box[2] * comp_box[3]) - area_inter

    return area_inter / area_union


def nms_infer(output, conf_thres=0.3, iou_thres=0.5):
    nms_index = (output[:, 4] > 0.8).nonzero(as_tuple=True)[0]
    output = output[nms_index, :].to('cpu').detach()

    after_nms = []
    bbox_filter = []
    bbox_sort = sorted(output, reverse=True, key=lambda x: x[4])

    for bbox in bbox_sort:
        if bbox[4] > conf_thres:
            bbox_filter.append(bbox)

    survive_idx = torch.ones(len(bbox_filter), dtype=torch.bool)

    for i in range(len(bbox_filter)):
        for j in range(len(bbox_filter)):
            if i != j:
                iou = iou_infer(bbox_filter[i][:4], bbox_filter[j][:4])
                if (iou > 0.6) or (is_belong(bbox_filter[i][:4], bbox_filter[j][:4]) == True):
                    if bbox_filter[i][4] >= bbox_filter[j][4]:
                        survive_idx[j] = False
                    else:
                        survive_idx[i] = False
    if len(bbox_filter) > 0:
        bbox_filter = torch.stack(bbox_filter, dim=0)
        return bbox_filter[survive_idx, :]
    else:
        return None

colors = np.random.randint(0, 255, size=(3, 3), dtype="uint8")

capture = cv2.VideoCapture(0)

if not capture.isOpened():
        print("Could not open webcam")
        exit()
        
cv2.namedWindow('detect_mask')
cv2.resizeWindow(winname = 'detect_mask', width = 640, height=480)

while True:
    status, frame = capture.read()
    frame = Image.fromarray(frame).resize((416, 416))
    if not status:
        break
    elif cv2.waitKey(1) == ord('q'):           # q 누르면 탈출
        break
    img_np = copy.deepcopy(frame)
    img = F.pil_to_tensor(frame).float()
    # img = img.permute(2,0,1).contiguous()
    
    start_time = time.time()
    input_img = img.unsqueeze(0).to('cpu')
    output = yolov3_model(input_img)[0][0]
    after_nms = nms_infer(output)
    end_time = time.time()
    
    running_time = end_time - start_time
    infer_time = str(1.0/running_time) + "fps"
    
    print(f"Inference Time = {infer_time}")

    if after_nms != None :
        labels = torch.argmax(after_nms[:, 5:], dim=1)
        boxes = after_nms[:, :4]

    # width, height = img.size
    draw = ImageDraw.Draw(img_np)

    if after_nms != None:
        for label, box in zip(labels, boxes):
            if label == 0:
                name = "Improper Mask"
            elif label == 1:
                name = "Mask"
            elif label == 2:
                name = "No Mask"

            if label <= 2:
                xmin = int(box[0] - box[2] / 2)
                ymin = int(box[1] - box[3] / 2)

                xmax = int(box[0] + box[2] / 2)
                ymax = int(box[1] + box[3] / 2)

                color = [int(c) for c in colors[label]]

                draw.rectangle([(xmin, ymin), (xmax, ymax)], outline=tuple(color), width=3)
                draw.text((xmin, ymin), name, fill=(255, 255, 255, 0))
    
    img_np = img_np.resize((640, 480))
    
    cv2.imshow('detect_mask', np.array(img_np))
    cv2.waitKey(1) 
    
    if after_nms != None :
        if (0 in labels) or (2 in labels):          # 0 : improper mask, 1 : mask, 2 : no mask
            socket.send("1")
            led_on(18)
            led_off(24)
        else:
            socket.send("0")
            led_on(24)
            led_off(18)
