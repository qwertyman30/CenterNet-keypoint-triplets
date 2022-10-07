#!/usr/bin/env python
# coding: utf-8

from tqdm import tqdm
import numpy as np
import os

os.environ['CUDA_VISIBLE_DEVICES'] = "0"
os.environ['CUDA_LAUNCH_BLOCKING'] = "1"

import torch
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
dataset = DatasetFactory(opts, train=False)
val_loader = torch.utils.data.DataLoader(dataset,
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

checkpoint = torch.load("saved_models/KITTI_train_dla169/CenterNet_pp_1000.pth")
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
    anno = {
        'name': [], 'truncated': [], 'occluded': [], 'alpha': [], 'bbox': [], 'dimensions': [],
        'location': [], 'rotation_y': [], 'score': []
    }
    nums = 0
    for bbox in preds:
        bbox = bbox.cpu()
        score = bbox[4]
        if score < 0.05:
            continue
        cls_name = CLASSES[bbox[5].item()]
        bbox_2d_pred = bbox[:4]
        anno["score"].append(score)
        anno["name"].append(cls_name)
        anno["truncated"].append(0.0)
        anno["occluded"].append(0)
        anno["bbox"].append(bbox_2d_pred)
        anno["alpha"].append(0.)
        anno["dimensions"].append(np.array([0, 0, 0], dtype=np.float32))
        anno["location"].append(np.array([-1000, -1000, -1000], dtype=np.float32))
        anno["rotation_y"].append(0.)
        nums += 1
    if nums != 0:
        anno = {k: np.stack(v) for k, v in anno.items()}
    else:
        anno = {
            'name': np.array([]), 'truncated': np.array([]), 'occluded': np.array([]),
            'alpha': np.array([]), 'bbox': np.zeros([0, 4]), 'dimensions': np.zeros([0, 3]),
            'location': np.zeros([0, 3]), 'rotation_y': np.array([]), 'score': np.array([])
        }
    with open(res_path, 'w') as f:
        bbox = anno['bbox']
        loc = anno['location']
        dims = anno['dimensions']  # lhw -> hwl

        for idx in range(len(bbox)):
            print('%s 0.0 0 %.4f %.4f %.4f %.4f %.4f %.4f %.4f %.4f %.4f %.4f %.4f %.4f %.4f'
            % (anno['name'][idx], anno['alpha'][idx], bbox[idx][0], bbox[idx][1], bbox[idx][2], bbox[idx][3],
            dims[idx][1], dims[idx][2], dims[idx][0], loc[idx][0], loc[idx][1], loc[idx][2],
            anno['rotation_y'][idx], anno['score'][idx]), file=f)

    # for bbox in preds:
    #     conf = round(bbox[4].item(), 2)
    #     if conf < 0.3:
    #         continue
    #     left = round(bbox[0].item(), 2)
    #     top = round(bbox[1].item(), 2)
    #     right = round(bbox[2].item(), 2)
    #     bottom = round(bbox[3].item(), 2)
    #     cls = CLASSES[bbox[5].item()]
    #     rem = ["0.00"] * 7
    #     rem = " ".join(rem)
    #     f_str = f"{cls}" + " 0.00 0 0.00 " + f"{left} {top} {right} {bottom} {rem} {conf}"
    #     results.append(f_str)
    # file = open(res_path, "w")
    # for result in results:
    #     file.writelines(result + '\n')
    # file.close()
