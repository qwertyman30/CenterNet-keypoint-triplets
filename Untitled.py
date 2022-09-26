from functools import partial
from tqdm import tqdm
import numpy as np
import os

os.environ['CUDA_VISIBLE_DEVICES'] = "0"
os.environ['CUDA_LAUNCH_BLOCKING'] = "1"

import torch
from mmcv.runner import build_optimizer
from mmcv.parallel import MMDataParallel

from dataset.coco import COCO
from dataset.utils.collate import collate
from dataset.utils.group_sampler import GroupSampler

from model.dense_heads.pycenternet_head import PyCenterNetHead
from model.backbone.resnet import ResNet
from model.neck.fpn import FPN
from model.detectors.pycenternet_detector import PyCenterNetDetector
from model.utils.utils import clip_grads
from model.model_config import *

print(torch.cuda.is_available())
print(torch.version.cuda)
torch.cuda.empty_cache()

seed = 42
torch.manual_seed(seed)
np.random.seed(seed)

opts = dict()
opts["img_scale"] = (1333, 800)
opts["keep_ratio"] = True
opts["flip_ratio"] = 0.5
opts["mean"] = [123.675, 116.28, 103.53]
opts["std"] = [58.395, 57.12, 57.375]
opts["to_rgb"] = True
opts["size_divisor"] = 32
opts["data_root"] = "Data/"
opts["batch_size"] = 2
opts["ann_file"] = "Data/COCO/annotations/instances_train2017.json"
opts["img_prefix"] = "Data/COCO/images/train2017"
opts["seg_prefix"] = None
opts["num_epochs"] = 72

dataset = COCO(opts)
sampler = GroupSampler(dataset, 2)

train_loader = torch.utils.data.DataLoader(
    dataset,
    batch_size=opts["batch_size"],
    sampler=sampler,
    num_workers=2,
    collate_fn=partial(collate, samples_per_gpu=2),
    pin_memory=False,
    worker_init_fn=None)

backbone = ResNet(**backbone_cfg).cuda()
neck = FPN(**neck_cfg).cuda()
bbox_head = PyCenterNetHead(**bbox_head_cfg).cuda()
detector = PyCenterNetDetector(backbone, neck, bbox_head, train_cfg=train_cfg, test_cfg=test_cfg).cuda()
optimizer = build_optimizer(detector, optimizer_cfg)

model = MMDataParallel(detector.cuda(0), device_ids=range(0, 1))

progress = tqdm(range(1, opts["num_epochs"] + 1))
LOSS_CLS, LOSS_PTS_INIT, LOSS_PTS_REFINE, LOSS_HEATMAP, LOSS_OFFSET, LOSS_SEM, LOSS = [], [], [], [], [], [], []
for epochs in progress:
    loss_cls_, loss_pts_init_, loss_pts_refine_, loss_heatmap_, loss_offset_, loss_sem_, loss_ = [], [], [], [], [], [], []
    for results in train_loader:
        LOSS = model.train_step(results, optimizer)
        log_vars = LOSS["log_vars"]
        loss_cls = log_vars['loss_cls']
        loss_pts_init = log_vars['loss_pts_init']
        loss_pts_refine = log_vars['loss_pts_refine']
        loss_heatmap = log_vars['loss_heatmap']
        loss_offset = log_vars['loss_offset']
        loss_sem = log_vars['loss_sem']
        loss = log_vars['loss']
        progress.set_description(
            "LOSS_CLS: {}, LOSS_PTS_INIT: {}, LOSS_PTS_REFINE: {} LOSS_HEATMAP: {}, LOSS_OFFSET: {}, LOSS_SEM: {}, LOSS: {}"
            .format(loss_cls, loss_pts_init, loss_pts_refine, loss_heatmap,
                    loss_offset, loss_sem, loss, log_vars['loss']))
        optimizer.zero_grad()
        LOSS["loss"].backward()
        if grad_clip is not None:
            clip_grads(model.parameters(), grad_clip)
        optimizer.step()
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

    LOSS_CLS.append(loss_cls_mean)
    LOSS_PTS_INIT.append(loss_pts_init_mean)
    LOSS_PTS_REFINE.append(loss_pts_refine_mean)
    LOSS_HEATMAP.append(loss_heatmap_mean)
    LOSS_OFFSET.append(loss_offset_mean)
    LOSS_SEM.append(loss_sem_mean)
    LOSS.append(loss_mean)
    print(
        f"LOSS_CLS: {loss_cls_mean}, LOSS_PTS_INIT: {loss_pts_init_mean}, LOSS_PTS_REFINE: {loss_pts_refine_mean}, LOSS_HEATMAP: {loss_heatmap_mean}, LOSS_OFFSET: {loss_offset_mean}, LOSS_SEM: {loss_sem_mean}, LOSS: {loss_mean}")

