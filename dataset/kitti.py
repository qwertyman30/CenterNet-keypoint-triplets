import pycocotools.coco as coco
import numpy as np
import mmcv
import cv2

import os.path as osp

import numpy as np
from torch.utils.data import Dataset

# from mmdet.core import eval_map, eval_recalls
from .utils.loaders import LoadAnnotations, LoadImageFromFile
from .utils.transforms import Resize, RandomFlip, Normalize, ColorTransform
from .utils.formatting import Collect, ImageToTensor, RPDV2FormatBundle, LoadRPDV2Annotations


class KITTI(Dataset):
    CLASSES = ['Pedestrian', 'Car', 'Cyclist']

    def __init__(self, opts,
                 test_mode=False,
                 filter_empty_gt=True):
        self.ann_file = opts["ann_file"]
        self.data_root = opts["data_root"]
        self.img_prefix = opts["img_prefix"]
        self.seg_prefix = opts["seg_prefix"]
        self.split = opts["split"]
        self.test_mode = test_mode
        self.filter_empty_gt = filter_empty_gt

        self.mean = opts["mean"]
        self.std = opts["std"]

        self.alpha_in_degree = False

        # load annotations
        self.data_infos = self.load_annotations(self.ann_file)

        # Loading pipeline
        self.LoadImageFromFile = LoadImageFromFile(to_float=opts["to_float"])
        self.LoadAnnotations = LoadAnnotations(with_bbox=True)

        # transformation pipeline training
        self.Resize_train = Resize(img_scale=opts["img_scale"], multiscale_mode='range', keep_ratio=True)
        self.RandomFlip_train = RandomFlip(flip_ratio=opts["flip_ratio"])
        self.ColorTransform = ColorTransform(level=5.)
        self.Normalize = Normalize(mean=self.mean, std=self.std, to_rgb=True)

        # formatting pipeline
        self.LoadRPDV2Annotations = LoadRPDV2Annotations(num_classes=3)
        self.RPDV2FormatBundle = RPDV2FormatBundle()
        self.Collect_train = Collect(keys=['img', 'gt_bboxes', 'gt_labels', 'gt_sem_map', 'gt_sem_weights'])

        # test transforms and pipeline
        self.ImageToTensor = ImageToTensor(keys=['img'])
        self.Collect_test = Collect(keys=['img'])

    def load_annotations(self, ann_file):
        self.annot_path = osp.join(ann_file)
        self.coco = coco.COCO(self.annot_path)
        self.cat_ids = self.coco.getCatIds(catNms=self.CLASSES)
        self.cat2label = {cat_id: i for i, cat_id in enumerate(self.cat_ids)}
        self.img_ids = self.coco.getImgIds()
        self.num_samples = len(self.img_ids)
        print('Loaded {} samples'.format(self.num_samples))

        data_infos = []
        for i in self.img_ids:
            info = self.coco.loadImgs([i])[0]
            info['filename'] = info['file_name']
            data_infos.append(info)
        return data_infos

    def __len__(self):
        """Total number of samples of data."""
        return len(self.data_infos)

    def get_ann_info(self, idx):
        """Get COCO annotation by index.

        Args:
            idx (int): Index of data.

        Returns:
            dict: Annotation info of specified index.
        """

        img_id = self.data_infos[idx]['id']
        ann_ids = self.coco.getAnnIds(imgIds=[img_id])
        ann_info = self.coco.loadAnns(ann_ids)
        return self._parse_ann_info(self.data_infos[idx], ann_info)

    def _parse_ann_info(self, img_info, ann_info):
        """Parse bbox and mask annotation.

        Args:
            ann_info (list[dict]): Annotation info of an image.
            with_mask (bool): Whether to parse mask annotations.

        Returns:
            dict: A dict containing the following keys: bboxes, bboxes_ignore,\
                labels, masks, seg_map. "masks" are raw annotations and not \
                decoded into binary masks.
        """
        gt_bboxes = []
        gt_labels = []
        gt_bboxes_ignore = []
        img_path = osp.join(self.img_prefix, img_info['filename'])
        img = cv2.imread(img_path)
        width = img.shape[1]
        height = img.shape[0]
        for ann in ann_info:
            if ann.get('ignore', False):
                continue
            x1, y1, w, h = ann['bbox']
            inter_w = max(0, min(x1 + w, width) - max(x1, 0))
            inter_h = max(0, min(y1 + h, height) - max(y1, 0))
            if inter_w * inter_h == 0:
                continue
            if ann['truncated'] == 2 and w < 1 or h < 1:
                continue
            if ann['category_id'] not in self.cat_ids:
                continue
            bbox = [x1, y1, x1 + w, y1 + h]
            if ann['occluded'] == 2 and ann['truncated'] == 1:
                gt_bboxes_ignore.append(bbox)
            else:
                gt_bboxes.append(bbox)
                gt_labels.append(self.cat2label[ann['category_id']])

        if gt_bboxes:
            gt_bboxes = np.array(gt_bboxes, dtype=np.float32)
            gt_labels = np.array(gt_labels, dtype=np.int64)
        else:
            gt_bboxes = np.zeros((0, 4), dtype=np.float32)
            gt_labels = np.array([], dtype=np.int64)

        if gt_bboxes_ignore:
            gt_bboxes_ignore = np.array(gt_bboxes_ignore, dtype=np.float32)
        else:
            gt_bboxes_ignore = np.zeros((0, 4), dtype=np.float32)

        seg_map = img_info['filename'].replace('jpg', 'png')

        ann = dict(
            bboxes=gt_bboxes,
            labels=gt_labels,
            bboxes_ignore=gt_bboxes_ignore,
            seg_map=seg_map)

        return ann

    def prepare_train_img(self, idx):
        """Get training data and annotations after pipeline.

        Args:
            idx (int): Index of data.

        Returns:
            dict: Training data and annotation after pipeline with new keys \
                introduced by pipeline.
        """

        img_info = self.data_infos[idx]
        ann_info = self.get_ann_info(idx)
        results = dict(img_info=img_info, ann_info=ann_info)

        results['img_prefix'] = self.img_prefix
        results['seg_prefix'] = self.seg_prefix
        results['bbox_fields'] = []
        results['mask_fields'] = []
        results['seg_fields'] = []

        # pipeline of transforms
        results = self.LoadImageFromFile(results)
        results = self.LoadAnnotations(results)
        results = self.Resize_train(results)
        results = self.RandomFlip_train(results)
        results = self.ColorTransform(results)
        results = self.Normalize(results)
        results = self.LoadRPDV2Annotations(results)
        results = self.RPDV2FormatBundle(results)
        results = self.Collect_train(results)
        return results

    def prepare_test_img(self, idx):
        """Get testing data  after pipeline.

        Args:
            idx (int): Index of data.

        Returns:
            dict: Testing data after pipeline with new keys introduced by \
                pipeline.
        """
        img_info = self.data_infos[idx]
        results = dict(img_info=img_info)
        results['img_prefix'] = self.img_prefix
        results['bbox_fields'] = []

        results = self.LoadImageFromFile(results)
        results = self.Normalize(results)
        results = self.ImageToTensor(results)
        results = self.Collect_test(results)

        return results

    def __getitem__(self, idx):
        if self.test_mode:
            return self.prepare_test_img(idx)
        while True:
            data = self.prepare_train_img(idx)
            if data is None:
                idx = self._rand_another(idx)
                continue
            return data
