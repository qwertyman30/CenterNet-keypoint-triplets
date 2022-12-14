#!/usr/bin/env python
# coding: utf-8

from tqdm import tqdm
import numpy as np
import os

os.environ['CUDA_VISIBLE_DEVICES'] = "0"
os.environ['CUDA_LAUNCH_BLOCKING'] = "1"

import torch
from torch.utils.data import DataLoader
from mmcv.runner import build_optimizer

from dataset import DatasetFactory
from dataset.utils.collate import collate

from model.dense_heads.pycenternet_head import PyCenterNetHead
from model.neck.fpn import FPN
from model.detectors.pycenternet_detector import PyCenterNetDetector
from model.utils.utils import save_model, update_lr
from model_helpers import step, plot
from config import *

print(torch.cuda.is_available())
print(torch.version.cuda)
torch.cuda.empty_cache()

seed = 15921
torch.manual_seed(seed)
np.random.seed(seed)

# Dataset and loader
dataset_train = DatasetFactory(opts, split="train")
dataset_val = DatasetFactory(opts, split="val")

train_loader = DataLoader(dataset_train,
                          batch_size=opts["batch_size"],
                          shuffle=True,
                          num_workers=opts["num_workers"],
                          collate_fn=collate,
                          pin_memory=True)

val_loader = DataLoader(dataset_val,
                        batch_size=opts["batch_size"],
                        shuffle=True,
                        num_workers=opts["num_workers"],
                        collate_fn=collate,
                        pin_memory=True)

# Model
b_name = opts["backbone"]
if "dla" in b_name:
    pretrained = backbone_cfg[b_name]["pretrained"]
    backbone = backbone_cfg[b_name]["model"]().cuda()
elif "resnet" in b_name:
    pretrained = backbone_cfg[b_name]["pretrained"]
    backbone = backbone_cfg[opts["backbone"]]["model"]
    backbone_cfg[b_name].pop("model")
    backbone = backbone(**backbone_cfg[b_name]).cuda()
neck = FPN(**neck_cfg).cuda()
bbox_head = PyCenterNetHead(**bbox_head_cfg).cuda()
detector = PyCenterNetDetector(backbone,
                               neck,
                               bbox_head,
                               train_cfg=train_cfg,
                               test_cfg=test_cfg,
                               pretrained=pretrained).cuda()
optimizer = build_optimizer(detector, optimizer_cfg)

# pytorch_total_params = sum(p.numel() for p in detector.parameters() if p.requires_grad)

# Training
f = open("tqdm_log.txt", "w") if opts["nohup"] is True else None
progress = tqdm(range(1, opts["num_epochs"] + 1), file=f)
LOSS_CLS_TRAIN, LOSS_PTS_INIT_TRAIN, LOSS_PTS_REFINE_TRAIN, LOSS_HEATMAP_TRAIN = [], [], [], []
LOSS_OFFSET_TRAIN, LOSS_SEM_TRAIN, LOSSES_TRAIN = [], [], []
LOSS_CLS_VAL, LOSS_PTS_INIT_VAL, LOSS_PTS_REFINE_VAL, LOSS_HEATMAP_VAL = [], [], [], []
LOSS_OFFSET_VAL, LOSS_SEM_VAL, LOSSES_VAL = [], [], []
for epoch in progress:
    print(f"TRAIN STEP {epoch}")
    # Training
    loss_means_train = step(detector,
                            train_loader,
                            progress,
                            optimizer=optimizer,
                            train=True)
    LOSSES_TRAIN.append(loss_means_train["loss"])
    LOSS_CLS_TRAIN.append(loss_means_train["loss_cls"])
    LOSS_PTS_INIT_TRAIN.append(loss_means_train["loss_pts_init"])
    LOSS_PTS_REFINE_TRAIN.append(loss_means_train["loss_pts_refine"])
    LOSS_HEATMAP_TRAIN.append(loss_means_train["loss_heatmap"])
    LOSS_OFFSET_TRAIN.append(loss_means_train["loss_offset"])
    LOSS_SEM_TRAIN.append(loss_means_train["loss_sem"])

    # lr scheduler
    if epoch in opts["lr_step"]:
        lr = opts["lr"] * (0.1**(opts["lr_step"].index(epoch) + 1))
        update_lr(optimizer, lr)
    
    # Validation
    if epoch % opts["save_interval"] == 0 or epoch == opts["num_epochs"]:
        with torch.no_grad():
            print(f"VAL STEP {epoch}")
            loss_means_val = step(detector,
                                  val_loader,
                                  progress,
                                  optimizer=None,
                                  train=False)
            LOSSES_VAL.append(loss_means_val["loss"])
            LOSS_CLS_VAL.append(loss_means_val["loss_cls"])
            LOSS_PTS_INIT_VAL.append(loss_means_val["loss_pts_init"])
            LOSS_PTS_REFINE_VAL.append(loss_means_val["loss_pts_refine"])
            LOSS_HEATMAP_VAL.append(loss_means_val["loss_heatmap"])
            LOSS_OFFSET_VAL.append(loss_means_val["loss_offset"])
            LOSS_SEM_VAL.append(loss_means_val["loss_sem"])
            save_model(detector, optimizer, epoch, LOSSES_TRAIN,
                       LOSS_CLS_TRAIN, LOSS_PTS_INIT_TRAIN,
                       LOSS_PTS_REFINE_TRAIN, LOSS_HEATMAP_TRAIN,
                       LOSS_OFFSET_TRAIN, LOSS_SEM_TRAIN, LOSSES_VAL,
                       LOSS_CLS_VAL, LOSS_PTS_INIT_VAL, LOSS_PTS_REFINE_VAL,
                       LOSS_HEATMAP_VAL, LOSS_OFFSET_VAL, LOSS_SEM_VAL, opts)

# Plotting losses
plot(LOSSES_TRAIN, f"LOSSES_TRAIN_{b_name}")
plot(LOSS_CLS_TRAIN, f"LOSS_CLS_TRAIN_{b_name}")
plot(LOSS_PTS_INIT_TRAIN, f"LOSS_PTS_INIT_TRAIN_{b_name}")
plot(LOSS_PTS_REFINE_TRAIN, f"LOSS_PTS_REFINE_TRAIN_{b_name}")
plot(LOSS_HEATMAP_TRAIN, f"LOSS_HEATMAP_TRAIN_{b_name}")
plot(LOSS_OFFSET_TRAIN, f"LOSS_OFFSET_TRAIN_{b_name}")
plot(LOSS_SEM_TRAIN, f"LOSS_SEM_TRAIN_{b_name}")
plot(LOSSES_VAL, f"LOSSES_VAL_{b_name}")
plot(LOSS_CLS_VAL, f"LOSS_CLS_VAL_{b_name}")
plot(LOSS_PTS_INIT_VAL, f"LOSS_PTS_INIT_VAL_{b_name}")
plot(LOSS_PTS_REFINE_VAL, f"LOSS_PTS_REFINE_VAL_{b_name}")
plot(LOSS_HEATMAP_VAL, f"LOSS_HEATMAP_VAL_{b_name}")
plot(LOSS_OFFSET_VAL, f"LOSS_OFFSET_VAL_{b_name}")
plot(LOSS_SEM_VAL, f"LOSS_SEM_VAL_{b_name}")
