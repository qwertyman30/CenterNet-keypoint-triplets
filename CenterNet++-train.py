#!/usr/bin/env python
# coding: utf-8

from tqdm import tqdm
import numpy as np
import os

os.environ['CUDA_VISIBLE_DEVICES'] = "0"
os.environ['CUDA_LAUNCH_BLOCKING'] = "1"

import torch
import matplotlib.pyplot as plt
from mmcv.runner import build_optimizer

from dataset.coco import COCO
from dataset.kitti import KITTI
from dataset.utils.collate import collate

from model.dense_heads.pycenternet_head import PyCenterNetHead
from model.backbone.resnet import ResNet
from model.neck.fpn import FPN
from model.detectors.pycenternet_detector import PyCenterNetDetector
from model.utils.utils import clip_grads, save_model, update_lr
from model.model_config import *
from mmcv.parallel.data_parallel import scatter_kwargs

print(torch.cuda.is_available())
print(torch.version.cuda)
torch.cuda.empty_cache()

seed = 42
torch.manual_seed(seed)
np.random.seed(seed)

# Dataset and loader
if opts["dataset"] == "coco":
    dataset = COCO(opts)
elif opts["dataset"] == "kitti":
    dataset = KITTI(opts)
train_loader = torch.utils.data.DataLoader(
    dataset,
    batch_size=opts["batch_size"],
    shuffle=True,
    num_workers=2,
    collate_fn=collate,
    pin_memory=True)

# Model
backbone = ResNet(**backbone_cfg).cuda()
neck = FPN(**neck_cfg).cuda()
bbox_head = PyCenterNetHead(**bbox_head_cfg).cuda()
detector = PyCenterNetDetector(backbone, neck, bbox_head, train_cfg=train_cfg, test_cfg=test_cfg).cuda()
optimizer = build_optimizer(detector, optimizer_cfg)

# Training
progress = tqdm(range(1, opts["num_epochs"] + 1))
LOSS_CLS, LOSS_PTS_INIT, LOSS_PTS_REFINE, LOSS_HEATMAP, LOSS_OFFSET, LOSS_SEM, LOSSES = [], [], [], [], [], [], []
curr_iter = 0
keys = []
for epoch in progress:
    loss_cls_, loss_pts_init_, loss_pts_refine_, loss_heatmap_, loss_offset_, loss_sem_, loss_ = [], [], [], [], [], [], []
    for results in train_loader:
        # LOSS = model.train_step(results, optimizer)
        results, _ = scatter_kwargs(results, None, [0])
        LOSS = detector.train_step(results[0])

        log_vars = LOSS["log_vars"]
        loss_cls = log_vars['loss_cls']
        loss_pts_init = log_vars['loss_pts_init']
        loss_pts_refine = log_vars['loss_pts_refine']
        loss_heatmap = log_vars['loss_heatmap']
        loss_offset = log_vars['loss_offset']
        loss_sem = log_vars['loss_sem']
        loss = log_vars['loss']
        progress.set_description(
            "LOSS: {}, LOSS_CLS: {} LOSS_PTS_INIT: {} LOSS_PTS_REFINE: {} LOSS_HEATMAP: {} LOSS_OFFSET: {} LOSS_SEM: {}"
            .format(log_vars['loss'], loss_cls, loss_pts_init, loss_pts_refine,
                    loss_heatmap, loss_offset, loss_sem, loss))

        # backprop
        optimizer.zero_grad()
        LOSS["loss"].backward()
        if grad_clip is not None:
            clip_grads(detector.parameters(), grad_clip)
        optimizer.step()

        loss_cls_.append(loss_cls)
        loss_pts_init_.append(loss_pts_init)
        loss_pts_refine_.append(loss_pts_refine)
        loss_heatmap_.append(loss_heatmap)
        loss_offset_.append(loss_offset)
        loss_sem_.append(loss_sem)
        loss_.append(loss)

        curr_iter += 1
        # keep lr low when iters less than warmup period and then increase
        if curr_iter == 500:
            update_lr(optimizer, opts["post_warmup_lr"])

    loss_cls_mean = np.mean(loss_cls_)
    loss_pts_init_mean = np.mean(loss_pts_init_)
    loss_pts_refine_mean = np.mean(loss_pts_refine_)
    loss_heatmap_mean = np.mean(loss_heatmap_)
    loss_offset_mean = np.mean(loss_offset_)
    loss_sem_mean = np.mean(loss_sem_)
    loss_mean = np.mean(loss_)

    LOSSES.append(loss_mean)
    LOSS_CLS.append(loss_cls_mean)
    LOSS_PTS_INIT.append(loss_pts_init_mean)
    LOSS_PTS_REFINE.append(loss_pts_refine_mean)
    LOSS_HEATMAP.append(loss_heatmap_mean)
    LOSS_OFFSET.append(loss_offset_mean)
    LOSS_SEM.append(loss_sem_mean)

    if epoch in opts["lr_step"]:
        lr = opts["post_warmup_lr"] * (0.1 ** (opts["lr_step"].index(epoch) + 1))
        update_lr(optimizer, lr)

    if epoch % 10 or epoch == opts["num_epochs"]:
        save_model(detector, optimizer, epoch,
                   LOSSES, LOSS_CLS, LOSS_PTS_INIT, LOSS_PTS_REFINE,
                   LOSS_HEATMAP, LOSS_OFFSET, LOSS_SEM)
    print(
        f"LOSS_CLS: {loss_cls_mean}, LOSS_PTS_INIT: {loss_pts_init_mean}, LOSS_PTS_REFINE: {loss_pts_refine_mean}, LOSS_HEATMAP: {loss_heatmap_mean}, LOSS_OFFSET: {loss_offset_mean}, LOSS_SEM: {loss_sem_mean}, LOSS: {loss_mean}\n")

# Plotting losses
plt.plot(LOSSES)
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.title("LOSS")
plt.savefig("LOSS.png")
plt.show()
plt.clf()
plt.cla()
plt.close()

plt.plot(LOSS_CLS)
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.title("Losses_CLS")
plt.savefig("losses_cls.png")
plt.show()
plt.clf()
plt.cla()
plt.close()

plt.plot(LOSS_PTS_INIT)
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.title("LOSS_PTS_INIT")
plt.savefig("LOSS_PTS_INIT.png")
plt.show()
plt.clf()
plt.cla()
plt.close()

plt.plot(LOSS_PTS_REFINE)
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.title("LOSS_PTS_REFINE")
plt.savefig("LOSS_PTS_REFINE.png")
plt.show()
plt.clf()
plt.cla()
plt.close()

plt.plot(LOSS_HEATMAP)
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.title("LOSS_HEATMAP")
plt.savefig("LOSS_HEATMAP.png")
plt.show()
plt.clf()
plt.cla()
plt.close()

plt.plot(LOSS_OFFSET)
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.title("LOSS_OFFSET")
plt.savefig("LOSS_OFFSET.png")
plt.show()
plt.clf()
plt.cla()
plt.close()

plt.plot(LOSS_SEM)
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.title("LOSS_SEM")
plt.savefig("LOSS_SEM.png")
plt.show()
plt.clf()
plt.cla()
plt.close()
