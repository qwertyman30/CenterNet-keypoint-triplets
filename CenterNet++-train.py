#!/usr/bin/env python
# coding: utf-8

from tqdm import tqdm
import numpy as np
import os

os.environ['CUDA_VISIBLE_DEVICES'] = "0"
os.environ['CUDA_LAUNCH_BLOCKING'] = "1"

import torch
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from mmcv.runner import build_optimizer
from mmcv.parallel.data_parallel import scatter_kwargs

from dataset import DatasetFactory
from dataset.utils.collate import collate

from model.dense_heads.pycenternet_head import PyCenterNetHead
from model.neck.fpn import FPN
from model.detectors.pycenternet_detector import PyCenterNetDetector
from model.utils.utils import clip_grads, save_model, update_lr
from config import *

print(torch.cuda.is_available())
print(torch.version.cuda)
torch.cuda.empty_cache()

seed = 15921
torch.manual_seed(seed)
np.random.seed(seed)


def step(detector, data_loader, progress, optimizer=None, train=True):
    if train:
        detector.train()
        func = detector.train_step
    else:
        detector.eval()
        func = detector.val_step
    loss_cls_, loss_pts_init_, loss_pts_refine_, loss_heatmap_, loss_offset_, loss_sem_, loss_ = [], [], [], [], [], [], []
    for batch in data_loader:
        batch, _ = scatter_kwargs(batch, None, [0])
        LOSS = func(batch[0])

        log_vars = LOSS["log_vars"]
        loss_cls = log_vars['loss_cls']
        loss_pts_init = log_vars['loss_pts_init']
        loss_pts_refine = log_vars['loss_pts_refine']
        loss_heatmap = log_vars['loss_heatmap']
        loss_offset = log_vars['loss_offset']
        loss_sem = log_vars['loss_sem']
        loss = log_vars['loss']

        # backprop
        if train:
            optimizer.zero_grad()
            LOSS["loss"].backward()
            if grad_clip is not None:
                clip_grads(detector.parameters(), grad_clip)
            optimizer.step()

        progress.set_description(
            "LOSS: {}, LOSS_CLS: {} LOSS_PTS_INIT: {} LOSS_PTS_REFINE: {} LOSS_HEATMAP: {} LOSS_OFFSET: {} LOSS_SEM: {}"
            .format(log_vars['loss'], loss_cls, loss_pts_init, loss_pts_refine,
                    loss_heatmap, loss_offset, loss_sem, loss))

        loss_cls_.append(loss_cls)
        loss_pts_init_.append(loss_pts_init)
        loss_pts_refine_.append(loss_pts_refine)
        loss_heatmap_.append(loss_heatmap)
        loss_offset_.append(loss_offset)
        loss_sem_.append(loss_sem)
        loss_.append(loss)

    loss_cls_mean = np.mean(loss_cls_)
    loss_pts_init_mean = np.mean(loss_pts_init_)
    loss_pts_refine_mean = np.mean(loss_pts_refine_)
    loss_heatmap_mean = np.mean(loss_heatmap_)
    loss_offset_mean = np.mean(loss_offset_)
    loss_sem_mean = np.mean(loss_sem_)
    loss_mean = np.mean(loss_)
    loss_means = {}
    loss_means['loss_cls'] = loss_cls_mean
    loss_means['loss_pts_init'] = loss_pts_init_mean
    loss_means['loss_pts_refine'] = loss_pts_refine_mean
    loss_means['loss_heatmap'] = loss_heatmap_mean
    loss_means['loss_offset'] = loss_offset_mean
    loss_means['loss_sem'] = loss_sem_mean
    loss_means['loss'] = loss_mean
    print(
        f"\nLOSS_CLS: {loss_cls_mean}, LOSS_PTS_INIT: {loss_pts_init_mean}, LOSS_PTS_REFINE: {loss_pts_refine_mean}, "
        f"LOSS_HEATMAP: {loss_heatmap_mean}, LOSS_OFFSET: {loss_offset_mean}, LOSS_SEM: {loss_sem_mean}, "
        f"LOSS: {loss_mean}\n")
    return loss_means


def plot(losses, title, log_scale=False):
    if log_scale:
        plt.yscale("log")
    plt.plot(losses)
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.title(f"{title}")
    plt.savefig(f"{title}.png")
    plt.show()
    plt.clf()
    plt.cla()
    plt.close()


# Dataset and loader
dataset_train = DatasetFactory(opts, train=True)
dataset_val = DatasetFactory(opts, train=False)

train_loader = DataLoader(
    dataset_train,
    batch_size=opts["batch_size"],
    shuffle=True,
    num_workers=0,
    collate_fn=collate,
    pin_memory=True)

val_loader = DataLoader(
    dataset_train,
    batch_size=opts["batch_size"],
    shuffle=True,
    num_workers=0,
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
detector = PyCenterNetDetector(backbone, neck, bbox_head, train_cfg=train_cfg, test_cfg=test_cfg,
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
    print("TRAIN STEP")
    loss_means_train = step(detector, train_loader, progress, optimizer=optimizer, train=True)
    LOSSES_TRAIN.append(loss_means_train["loss"])
    LOSS_CLS_TRAIN.append(loss_means_train["loss_cls"])
    LOSS_PTS_INIT_TRAIN.append(loss_means_train["loss_pts_init"])
    LOSS_PTS_REFINE_TRAIN.append(loss_means_train["loss_pts_refine"])
    LOSS_HEATMAP_TRAIN.append(loss_means_train["loss_heatmap"])
    LOSS_OFFSET_TRAIN.append(loss_means_train["loss_offset"])
    LOSS_SEM_TRAIN.append(loss_means_train["loss_sem"])

    if epoch in opts["lr_step"]:
        lr = opts["lr"] * (0.1 ** (opts["lr_step"].index(epoch) + 1))
        update_lr(optimizer, lr)

    if epoch % opts["save_interval"] == 0 or epoch == opts["num_epochs"]:
        with torch.no_grad():
            print("VAL STEP")
            loss_means_val = step(detector, val_loader, progress, optimizer=None, train=False)
            LOSSES_VAL.append(loss_means_val["loss"])
            LOSS_CLS_VAL.append(loss_means_val["loss_cls"])
            LOSS_PTS_INIT_VAL.append(loss_means_val["loss_pts_init"])
            LOSS_PTS_REFINE_VAL.append(loss_means_val["loss_pts_refine"])
            LOSS_HEATMAP_VAL.append(loss_means_val["loss_heatmap"])
            LOSS_OFFSET_VAL.append(loss_means_val["loss_offset"])
            LOSS_SEM_VAL.append(loss_means_val["loss_sem"])
            save_model(detector, optimizer, epoch, LOSSES_TRAIN, LOSS_CLS_TRAIN, LOSS_PTS_INIT_TRAIN,
                       LOSS_PTS_REFINE_TRAIN, LOSS_HEATMAP_TRAIN, LOSS_OFFSET_TRAIN, LOSS_SEM_TRAIN, LOSSES_VAL,
                       LOSS_CLS_VAL, LOSS_PTS_INIT_VAL, LOSS_PTS_REFINE_VAL, LOSS_HEATMAP_VAL, LOSS_OFFSET_VAL,
                       LOSS_SEM_VAL)

# Plotting losses
plot(LOSSES_TRAIN, "LOSSES_TRAIN")
plot(LOSS_CLS_TRAIN, "LOSS_CLS_TRAIN")
plot(LOSS_PTS_INIT_TRAIN, "LOSS_PTS_INIT_TRAIN")
plot(LOSS_PTS_REFINE_TRAIN, "LOSS_PTS_REFINE_TRAIN")
plot(LOSS_HEATMAP_TRAIN, "LOSS_HEATMAP_TRAIN")
plot(LOSS_OFFSET_TRAIN, "LOSS_OFFSET_TRAIN")
plot(LOSS_SEM_TRAIN, "LOSS_SEM_TRAIN")
plot(LOSSES_VAL, "LOSSES_VAL")
plot(LOSS_CLS_VAL, "LOSS_CLS_VAL")
plot(LOSS_PTS_INIT_VAL, "LOSS_PTS_INIT_VAL")
plot(LOSS_PTS_REFINE_VAL, "LOSS_PTS_REFINE_VAL")
plot(LOSS_HEATMAP_VAL, "LOSS_HEATMAP_VAL")
plot(LOSS_OFFSET_VAL, "LOSS_OFFSET_VAL")
plot(LOSS_SEM_VAL, "LOSS_SEM_VAL")
