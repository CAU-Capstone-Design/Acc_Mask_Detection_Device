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

# from dataclass import get_loader
from yolov3_model import *

# train_loader, valid_loader, test_loader = get_loader()

class YOLOv3_Actuator(nn.Module):
    def __init__(self, root, train_loader, valid_loader, test_loader, model, opt="adam", lr=0.001, has_scheduler=False,
                 device="cpu"):
        super().__init__()

        self.root = root  # Best Epoch Model 저장할 디렉토리 full path

        self.train_loader = train_loader
        self.valid_loader = valid_loader
        self.test_loader = test_loader

        self.model = model
        self.mse_loss = nn.MSELoss(reduction="sum")  # 논문과의 차이점 : SSE 대신 MSE 적용
        self.bce_loss = nn.BCELoss(reduction="sum")

        self.scaled_anchors = [
            self.model.yolo_large.scaled_anchors,
            self.model.yolo_medium.scaled_anchors,
            self.model.yolo_small.scaled_anchors,
        ]

        self.param_loss = {
            "scaled_anchors": self.scaled_anchors,
            "ignore_thres": 0.5,
            "n_yolo": 3,  # scaled feature map 갯수 = 3
            "n_anchor": 3,  # 각각의 scaled feature map당 anchor box의 갯수 = 3
            "obj_scale": 5,
            "noobj_scale": 0.5,
        }

        self._get_optimizer(opt=opt.lower(), lr=lr)
        self.has_scheduler = has_scheduler
        if self.has_scheduler:
            self._get_scheduler()

        self.device = device

    def _batch_loss(self, output, targets, param_loss, is_train=False):
        ignore_thres = param_loss["ignore_thres"]
        scaled_anchors = param_loss["scaled_anchors"]
        n_yolo = param_loss["n_yolo"]
        n_anchor = param_loss["n_anchor"]
        obj_scale = param_loss["obj_scale"]
        noobj_scale = param_loss["noobj_scale"]

        loss = 0.0

        for yolo_id in range(n_yolo):  # large(13 X 13), medium(26 X 26), small(52 X 52) 순서로 대해 반복
            yolo_out = output[yolo_id]  # yolo_out = batch, num_boxes, coord(4) + 1(conf) + 3(class)
            batch_size, n_box, _ = yolo_out.shape
            grid_size = int(np.sqrt(n_box / n_anchor))

            yolo_out = yolo_out.view(batch_size, n_anchor, grid_size, grid_size, -1)
            pred_boxes = yolo_out[:, :, :, :, :4]
            pred_conf = yolo_out[:, :, :, :, 4]
            pred_class_prob = yolo_out[:, :, :, :, 5:]
            x, y, w, h = self._change_bbox(pred_boxes,
                                           scaled_anchors[yolo_id])  # pred box에 대해 grid cell 내에서의 x, y, w, h로 변환

            yolo_targets = self._yolo_targets({
                "pred_boxes": pred_boxes,
                "pred_class_prob": pred_class_prob,
                "targets": targets,
                "anchors": scaled_anchors[yolo_id],
                "ignore_thres": ignore_thres
            })

            obj_mask = yolo_targets["obj_mask"]
            noobj_mask = yolo_targets["noobj_mask"]
            tx = yolo_targets["tx"]
            ty = yolo_targets["ty"]
            tw = yolo_targets["tw"]
            th = yolo_targets["th"]
            tclass = yolo_targets["tclass"]
            tconf = yolo_targets["tconf"]

            x_loss = self.mse_loss(x[obj_mask].to(device),
                                   tx[obj_mask].to(device))  # 예측 Bbox Coordinates와 Anchor Box 간의 MSE Loss 계산
            y_loss = self.mse_loss(y[obj_mask].to(device), ty[obj_mask].to(device))
            w_loss = self.mse_loss(w[obj_mask].to(device), tw[obj_mask].to(device))
            h_loss = self.mse_loss(h[obj_mask].to(device), th[obj_mask].to(device))

            conf_obj_loss = self.bce_loss(pred_conf[obj_mask],
                                          tconf[obj_mask])  # object confidence score에 대해 예측 & 실제와의 BCE Loss 계산
            conf_noobj_loss = self.bce_loss(pred_conf[noobj_mask], tconf[noobj_mask])
            conf_loss = (obj_scale * conf_obj_loss) + (noobj_scale * conf_noobj_loss)

            class_prob_loss = self.bce_loss(pred_class_prob[obj_mask],
                                            tclass[obj_mask])  # object가 특정 class label일 확률에 대한 예측 & 실제와의 BCE Loss 계산

            loss += x_loss + y_loss + w_loss + h_loss + conf_loss + class_prob_loss  # 모든 Loss 총합 계산

        loss = loss / x.size()[0]

        if (self.optimizer is not None) and (is_train == True):
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

        return loss.item()

    def _change_bbox(self, box, anchors):  # 전체 이미지의 x, y, w, h => Grid Cell 내의 x, y, w, h로 변환
        x = box[:, :, :, :, 0]
        y = box[:, :, :, :, 1]
        w = box[:, :, :, :, 2]
        h = box[:, :, :, :, 3]

        anchor_w = anchors[:, 0].view((1, 3, 1, 1))
        anchor_h = anchors[:, 1].view((1, 3, 1, 1))

        x -= x.floor()  # 전체 이미지에서의 x좌표 => 특정 grid cell 내에서의 x좌표로 변환
        y -= y.floor()  # 전체 이미지에서의 y좌표 => 특정 grid cell 내에서의 y좌표로 변환
        w = torch.log(w / anchor_w + 1e-16)  # 예측 bbox의 width / anchor box의 width로 변환
        h = torch.log(h / anchor_h + 1e-16)  # 예측 bbox의 height / anchor box의 height로 변환

        return x, y, w, h

    def _yolo_targets(self, arguments):
        pred_boxes = arguments["pred_boxes"]
        pred_class_prob = arguments["pred_class_prob"]
        target = arguments["targets"]
        anchors = arguments["anchors"]
        ignore_thres = arguments["ignore_thres"]

        batch_size = pred_boxes.size(0)
        n_anchor = pred_boxes.size(1)
        grid_size = pred_boxes.size(2)
        n_class = pred_class_prob.size(-1)

        # target 분해
        boxes = target["boxes"]  # [batch_size, 100, 4]
        labels = target["labels"]  # [batch_size, 100]
        iscrowd = target["iscrowd"]  # [batch_size, 100]
        area = target["area"]  # [batch_size, 100]

        sizeT = batch_size, n_anchor, grid_size, grid_size  # (batch_size, anchor box 갯수, grid_size, grid_sze) 튜플화
        obj_mask = torch.zeros(sizeT, device="cpu", dtype=torch.uint8)
        noobj_mask = torch.ones(sizeT, device="cpu", dtype=torch.uint8)
        tx = torch.zeros(sizeT, device="cpu", dtype=torch.float32)
        ty = torch.zeros(sizeT, device="cpu", dtype=torch.float32)
        tw = torch.zeros(sizeT, device="cpu", dtype=torch.float32)
        th = torch.zeros(sizeT, device="cpu", dtype=torch.float32)

        sizeT = batch_size, n_anchor, grid_size, grid_size, n_class
        tclass = torch.zeros(sizeT, device="cpu", dtype=torch.float32)

        target_boxes = boxes * grid_size  # [batch_size, 100, 4] * grid_size

        # 각 pred_bbox의 답지를 구하는 과정
        for i in range(target_boxes.size()[0]):  # Iteration의 각각은 batch_size, 원래는 iteration 없이 한번에 처리

            valid_val_index = (labels[i] < 3).nonzero(as_tuple=True)[0]

            boxes = target_boxes[i, valid_val_index, :]  # i번째 batch의 valid_val_index에 해당하는 object의 bbox 좌표

            t_xy = boxes[:, :2]
            t_wh = boxes[:, 2:]
            t_x, t_y = t_xy.t()  # bbox x좌표값들의 tensor
            t_w, t_h = t_wh.t()  # bbox y좌표값들의 tensor

            grid_i, grid_j = t_xy.long().t()

            iou_anchors = [self._get_IoU_WH(anchor, t_wh) for anchor in anchors]  # Anchor와 Ground Truth간의 IoU 계산
            iou_anchors = torch.stack(iou_anchors)
            best_iou_anchor, best_anchor_id = iou_anchors.max(0)  # iou_anchors의 열을 기준으로 (최댓값, 최댓값의 index) return

            target_labels = labels[i, valid_val_index].long()
            obj_mask[i, best_anchor_id, grid_j, grid_i] = 1  # Object가 있는 곳에 대해 IoU가 가장 높은 Anchor를 Responsible
            noobj_mask[i, best_anchor_id, grid_j, grid_i] = 0  # Object가 없는 곳에 대해 IoU가 가장 높은 Anchor를 Non-Responsible

            for id, iou_anchor in enumerate(iou_anchors.t()):
                noobj_mask[i, iou_anchor > ignore_thres, grid_j[id], grid_i[id]] = 0

            tx[i, best_anchor_id, grid_j, grid_i] = t_x - t_x.float()
            ty[i, best_anchor_id, grid_j, grid_i] = t_y - t_y.float()

            anchor_w = anchors[best_anchor_id][:, 0]
            tw[i, best_anchor_id, grid_j, grid_i] = torch.log(t_w / anchor_w + 1e-16)
            anchor_h = anchors[best_anchor_id][:, 1]
            th[i, best_anchor_id, grid_j, grid_i] = torch.log(t_h / anchor_h + 1e-16)

            tclass[i, best_anchor_id, grid_j, grid_i, target_labels] = 1

        output = {
            "obj_mask": obj_mask,
            "noobj_mask": noobj_mask,
            "tx": tx,
            "ty": ty,
            "tw": tw,
            "th": th,
            "tclass": tclass[:, :, :, :, :],  # tclass[:, :, :, :, :-1]
            "tconf": obj_mask.float(),
        }

        return output

    def _get_IoU_WH(self, wh1, wh2):  # Anchor와 Ground Truth간의 IoU 계산
        wh2 = wh2.t()
        w1, h1 = wh1[0], wh1[1]
        w2, h2 = wh2[0], wh2[1]

        area_of_intersection = torch.min(w1, w2) * torch.min(h1, h2)
        area_of_union = (w1 * h1 + 1e-16) + (w2 * h2) - area_of_intersection

        return area_of_intersection / area_of_union

    def _get_optimizer(self, opt, lr=0.001):
        if opt == "sgd":
            self.optimizer = torch.optim.SGD(params=self.model.parameters(), lr=lr)
        elif opt == "adagrad":
            self.optimizer = torch.optim.Adagrad(params=self.model.parameters(), lr=lr)
        elif opt == "adadelta":
            self.optimizer = torch.optim.Adadelta(params=self.model.parameters(), lr=lr)
        elif opt == "rmsprop":
            self.optimizer = torch.optim.RMSprop(params=self.model.parameters(), lr=lr)
        elif opt == "adam":
            self.optimizer = torch.optim.Adam(params=self.model.parameters(), lr=lr)
        else:
            raise ValueError(f"Inputted Optimizer {opt} is not Supported")

    def _get_scheduler(self):
        # self.scheduler = torch.optim.lr_scheduler.StepLR(optimizer=self.optimizer, step_size=5, gamma=0.5, verbose=True)
        # self.scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer=self.optimizer, lr_lambda=lambda epoch: 0.95**epoch)
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer=self.optimizer, mode="min", factor=0.5,
                                                                    patience=20, verbose=1)

    def _get_current_lr(self):
        for param_group in self.optimizer.param_groups:
            return param_group["lr"]

    def train_and_valid(self, max_epochs=10, disp_epochs=1):
        print("========== Train Start ==========")
        start_time = time.time()

        history = {"Train Loss": [], "Valid Loss": []}
        best_epoch = 0
        best_model = copy.deepcopy(self.model.state_dict())
        best_loss = float("inf")

        for epoch in tqdm(range(max_epochs), desc='train...'):
            current_lr = self._get_current_lr()

            train_loss = self._train_epoch()
            valid_loss = self._valid_epoch()

            history["Train Loss"].append(train_loss)
            history["Valid Loss"].append(valid_loss)

            if valid_loss < best_loss:
                best_epoch = epoch + 1
                best_loss = valid_loss
                best_model = copy.deepcopy(self.model.state_dict())
                print("Best YOLOv3 Model is Updated")

            if self.has_scheduler:
                self.scheduler.step(valid_loss)

            if epoch % disp_epochs == 0:
                print(
                    f"{epoch + 1}/{max_epochs} Epoch ==> LR : {current_lr}, Train Loss : {train_loss:>6f}, Valid Loss : {valid_loss:>6f}, Time : {time.time() - start_time:>3f}")

        best_name = "YOLOv3 Best of " + str(best_epoch) + "Epoch.pth"
        torch.save(self.model.state_dict(), os.path.join(self.root, best_name))
        self.plot_history(history)

    def _train_epoch(self):
        train_epoch_loss = 0
        self.model.train()

        for (x, y) in self.train_loader:
            x = x.to(self.device)

            y["boxes"] = y["boxes"].to(self.device)
            y["labels"] = y["labels"].to(self.device)
            y["iscrowd"] = y["iscrowd"].to(self.device)
            y["area"] = y["area"].to(self.device)

            _, output = self.model(x)

            temp_loss = self._batch_loss(output, y, self.param_loss, is_train=True)
            train_epoch_loss += temp_loss

        train_epoch_loss /= len(self.train_loader)
        return train_epoch_loss

    def _valid_epoch(self):
        valid_epoch_loss = 0
        self.model.eval()

        for (x, y) in self.valid_loader:
            x = x.to(self.device)

            y["boxes"] = y["boxes"].to(self.device)
            y["labels"] = y["labels"].to(self.device)
            y["iscrowd"] = y["iscrowd"].to(self.device)
            y["area"] = y["area"].to(self.device)

            _, output = self.model(x)

            temp_loss = self._batch_loss(output, y, self.param_loss, is_train=False)
            valid_epoch_loss += temp_loss

        valid_epoch_loss /= len(self.valid_loader)
        return valid_epoch_loss

    def plot_history(self, history):
        fig = plt.figure(figsize=(10, 10))

        plt.plot(history["Train Loss"], color="red", label="Train Loss")
        plt.plot(history["Valid Loss"], color="orange", label="Valid Loss")
        plt.title("Loss of Object Detection")
        plt.legend()

        fig.show()

    def test(self):
        print("========== Test Start ==========")
        start_time = time.time()

        test_loss = 0
        self.model.eval()

        for (x, y) in self.test_loader:
            x = x.to(self.device)

            y["boxes"] = y["boxes"].to(self.device)
            y["labels"] = y["labels"].to(self.device)
            y["iscrowd"] = y["iscrowd"].to(self.device)
            y["area"] = y["area"].to(self.device)
            _, output = self.model(x)

            temp_loss = self._batch_loss(output, y, self.param_loss)
            test_loss += temp_loss

        test_loss /= len(self.test_loader)
        print(f"Test Loss : {test_loss:>6f}, Time : {time.time() - start_time:>3f}")


device = "cpu"
root = os.path.join(os.getcwd(),"best_epoch")
anchors = [[(10, 13), (16, 30), (33, 23)], [(30, 61), (62, 45), (59, 119)], [(116, 90), (156, 198), (373, 326)]]

yolov3_actuator = YOLOv3_Actuator(
    root=root,
    train_loader=train_loader,
    valid_loader=valid_loader,
    test_loader=test_loader,
    model=yolov3,
    opt="adam",
    lr=0.001,
    has_scheduler=True,
    device=device).to(device)

start_time = time.time()
yolov3_actuator.train_and_valid(max_epochs=30, disp_epochs=1)
print(f"Total Training Time : {time.time() - start_time:>3f}")
