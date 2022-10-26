#!/usr/bin/env python
# coding: utf-8

from tqdm import tqdm
import numpy as np
import os

os.environ['CUDA_VISIBLE_DEVICES'] = "0"
os.environ['CUDA_LAUNCH_BLOCKING'] = "1"

import torch
from torch.utils.data import DataLoader
from mmcv.parallel.data_parallel import scatter_kwargs

from dataset import DatasetFactory
from dataset.utils.collate import collate

from model.dense_heads.pycenternet_head import PyCenterNetHead
from model.neck.fpn import FPN
from model.detectors.pycenternet_detector import PyCenterNetDetector
from config import *

print(torch.cuda.is_available())
print(torch.version.cuda)
torch.cuda.empty_cache()

seed = 42
torch.manual_seed(seed)
np.random.seed(seed)

# Dataset and loader
dataset = DatasetFactory(opts, split="val")
val_loader = DataLoader(dataset,
                        batch_size=1,
                        shuffle=True,
                        num_workers=2,
                        collate_fn=collate,
                        pin_memory=True)

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

checkpoint = torch.load(
    "saved_models/kitti_resnet50/CenterNet_pp_resnet50_1500.pth")
state_dict = checkpoint["model_state_dict"]
detector.load_state_dict(state_dict)

CLASSES = {0: "Pedestrian", 1: "Car", 2: "Cyclist"}

# evaluation = dict(interval=1, metric='bbox')

# results = []
# detector.eval()
# for batch in tqdm(val_loader):
#     with torch.no_grad():
#         batch, _ = scatter_kwargs(batch, None, [0])
#         result = detector.simple_test(batch[0]["img"], batch[0]["img_metas"], rescale=True)[0]

#         # result = detector.forward_test(batch[0]["img"], batch[0]["img_metas"], rescale=True)
#     results.append(result)
# dataset.evaluate(results, "results")

for batch in tqdm(val_loader):
    batch, _ = scatter_kwargs(batch, None, [0])
    f_name, _ = os.path.splitext(
        os.path.basename(batch[0]["img_metas"][0]["filename"]))
    f_name += ".txt"
    res_path = os.path.join("results/results/", f_name)
    ann, cls = detector.simple_test(batch[0]["img"],
                                    batch[0]["img_metas"],
                                    rescale=True)[0]
    cls = cls.view(-1, 1)
    preds = torch.cat([ann, cls], dim=1)
    results = []
    for bbox in preds:
        conf = round(bbox[4].item(), 2)
        # if conf < 0.3:
        #     continue
        left = round(bbox[0].item(), 2)
        top = round(bbox[1].item(), 2)
        right = round(bbox[2].item(), 2)
        bottom = round(bbox[3].item(), 2)
        cls = CLASSES[bbox[5].item()]
        rem = ["0.00"] * 7
        rem = " ".join(rem)
        f_str = f"{cls}" + " 0.0 0 0.00 " + f"{left} {top} {right} {bottom} {rem} {conf}"
        results.append(f_str)
    file = open(res_path, "w")
    for result in results:
        file.writelines(result + '\n')
    file.close()
