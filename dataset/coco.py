import pycocotools.coco as coco
import mmcv

import os.path as osp

import numpy as np
from torch.utils.data import Dataset

# from mmdet.core import eval_map, eval_recalls
from .utils.loaders import LoadAnnotations, LoadImageFromFile
from .utils.transforms import Resize, RandomFlip, Normalize, Pad
from .utils.formatting import Collect, RPDV2FormatBundle, LoadRPDV2Annotations
from .utils.test_augs import MultiScaleFlipAug


class COCO(Dataset):
    CLASSES = [
        'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train',
        'truck', 'boat', 'traffic light', 'fire hydrant', 'stop sign',
        'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep',
        'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella',
        'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard',
        'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard',
        'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup', 'fork',
        'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich', 'orange',
        'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair',
        'couch', 'potted plant', 'bed', 'dining table', 'toilet', 'tv',
        'laptop', 'mouse', 'remote', 'keyboard', 'cell phone', 'microwave',
        'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase',
        'scissors', 'teddy bear', 'hair drier', 'toothbrush'
    ]

    def __init__(self, opts, split="train", train=True, filter_empty_gt=True):
        if split == "train":
            self.ann_file = opts["ann_file_train"]
        else:
            self.ann_file = opts["ann_file_val"]
        self.data_root = opts["data_root"]
        self.img_prefix = opts["img_prefix"]
        self.seg_prefix = opts["seg_prefix"]
        self.test_mode = not train
        self.filter_empty_gt = filter_empty_gt

        # load annotations
        self.data_infos = self.load_annotations(self.ann_file)

        # filter images too small and containing no annotations
        if not self.test_mode:
            valid_inds = self._filter_imgs()
            self.data_infos = [self.data_infos[i] for i in valid_inds]
            # set group flag for the sampler
            self._set_group_flag()

        # Loading pipeline
        self.LoadImageFromFile = LoadImageFromFile(to_float=opts["to_float"])
        self.LoadAnnotations = LoadAnnotations(with_bbox=True)

        # transformation pipeline training
        back = "resnet50"
        if "dla" in opts["backbone"]:
            back = "dla"
        self.Resize_train = Resize(img_scale=opts["img_scale"],
                                   multiscale_mode='range',
                                   keep_ratio=opts["keep_ratio"],
                                   backbone=back)
        self.RandomFlip_train = RandomFlip(flip_ratio=opts["flip_ratio"])
        self.Normalize = Normalize(mean=opts["mean"],
                                   std=opts["std"],
                                   to_rgb=opts["to_rgb"])
        self.Pad = Pad(size_divisor=opts["size_divisor"])

        # formatting pipeline
        self.LoadRPDV2Annotations = LoadRPDV2Annotations()
        self.RPDV2FormatBundle = RPDV2FormatBundle()
        self.Collect_train = Collect(keys=[
            'img', 'gt_bboxes', 'gt_labels', 'gt_sem_map', 'gt_sem_weights'
        ])

        # test transforms
        self.MultiScaleFlipAug = MultiScaleFlipAug(opts,
                                                   img_scale=(736, 512),
                                                   flip=False)

    def _set_group_flag(self):
        """Set flag according to image aspect ratio.

        Images with aspect ratio greater than 1 will be set as group 1,
        otherwise group 0.
        """
        self.flag = np.zeros(len(self), dtype=np.uint8)
        for i in range(len(self)):
            img_info = self.data_infos[i]
            if img_info['width'] / img_info['height'] > 1:
                self.flag[i] = 1

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
        results = self.Normalize(results)
        results = self.Pad(results)
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
        results = self.MultiScaleFlipAug(results)

        return results

    def _rand_another(self, idx):
        """Get another random index from the same group as the given index."""
        pool = np.where(self.flag == self.flag[idx])[0]
        return np.random.choice(pool)

    def __getitem__(self, idx):
        """Get training/test data after pipeline.

        Args:
            idx (int): Index of data.

        Returns:
            dict: Training/test data (with annotation if `test_mode` is set \
                True).
        """

        if self.test_mode:
            return self.prepare_test_img(idx)
        while True:
            data = self.prepare_train_img(idx)
            if data is None:
                idx = self._rand_another(idx)
                continue
            return data

    def __len__(self):
        """Total number of samples of data."""
        return len(self.data_infos)

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
        gt_masks_ann = []
        for i, ann in enumerate(ann_info):
            if ann.get('ignore', False):
                continue
            x1, y1, w, h = ann['bbox']
            inter_w = max(0, min(x1 + w, img_info['width']) - max(x1, 0))
            inter_h = max(0, min(y1 + h, img_info['height']) - max(y1, 0))
            if inter_w * inter_h == 0:
                continue
            if ann['area'] <= 0 or w < 1 or h < 1:
                continue
            if ann['category_id'] not in self.cat_ids:
                continue
            bbox = [x1, y1, x1 + w, y1 + h]
            if ann.get('iscrowd', False):
                gt_bboxes_ignore.append(bbox)
            else:
                gt_bboxes.append(bbox)
                gt_labels.append(self.cat2label[ann['category_id']])
                gt_masks_ann.append(ann.get('segmentation', None))

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

        ann = dict(bboxes=gt_bboxes,
                   labels=gt_labels,
                   bboxes_ignore=gt_bboxes_ignore,
                   masks=gt_masks_ann,
                   seg_map=seg_map)

        return ann

    def load_annotations(self, ann_file):
        """Load annotation from COCO style annotation file.

        Args:
            ann_file (str): Path of annotation file.

        Returns:
            list[dict]: Annotation info from COCO api.
        """
        self.coco = coco.COCO(ann_file)
        self.cat_ids = self.coco.getCatIds(catNms=self.CLASSES)
        self.cat2label = {cat_id: i for i, cat_id in enumerate(self.cat_ids)}
        self.img_ids = self.coco.getImgIds()

        data_infos = []
        for i in self.img_ids:
            info = self.coco.loadImgs([i])[0]
            info['filename'] = info['file_name']
            data_infos.append(info)
        return data_infos

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

    def get_cat_ids(self, idx):
        """Get COCO category ids by index.

        Args:
            idx (int): Index of data.

        Returns:
            list[int]: All categories in the image of specified index.
        """

        img_id = self.data_infos[idx]['id']
        ann_ids = self.coco.getAnnIds(imgIds=[img_id])
        ann_info = self.coco.loadAnns(ann_ids)
        return [ann['category_id'] for ann in ann_info]

    def _filter_imgs(self, min_size=32):
        """Filter images too small or without ground truths."""
        valid_inds = []
        # obtain images that contain annotation
        ids_with_ann = set(_['image_id'] for _ in self.coco.anns.values())
        # obtain images that contain annotations of the required categories
        ids_in_cat = set()
        for i, class_id in enumerate(self.cat_ids):
            ids_in_cat |= set(self.coco.catToImgs[class_id])
        # merge the image id sets of the two conditions and use the merged set
        # to filter out images if self.filter_empty_gt=True
        ids_in_cat &= ids_with_ann

        valid_img_ids = []
        for i, img_info in enumerate(self.data_infos):
            img_id = self.img_ids[i]
            if self.filter_empty_gt and img_id not in ids_in_cat:
                continue
            if min(img_info['width'], img_info['height']) >= min_size:
                valid_inds.append(i)
                valid_img_ids.append(img_id)
        self.img_ids = valid_img_ids
        return valid_inds

    def xyxy2xywh(self, bbox):
        """Convert ``xyxy`` style bounding boxes to ``xywh`` style for COCO
        evaluation.

        Args:
            bbox (numpy.ndarray): The bounding boxes, shape (4, ), in
                ``xyxy`` order.

        Returns:
            list[float]: The converted bounding boxes, in ``xywh`` order.
        """

        _bbox = bbox.tolist()
        return [
            _bbox[0],
            _bbox[1],
            _bbox[2] - _bbox[0],
            _bbox[3] - _bbox[1],
        ]

    def _det2json(self, results):
        """Convert detection results to COCO json style."""
        json_results = []
        for idx in range(len(self)):
            img_id = self.img_ids[idx]
            result = results[idx]
            for label in range(len(result)):
                bboxes = result[label]
                for i in range(bboxes.shape[0]):
                    data = dict()
                    data['image_id'] = img_id
                    data['bbox'] = self.xyxy2xywh(bboxes[i])
                    data['score'] = float(bboxes[i][4])
                    data['category_id'] = self.cat_ids[label]
                    json_results.append(data)
        return json_results

    def results2json(self, results, outfile_prefix):
        """Dump the detection results to a COCO style json file.

        There are 3 types of results: proposals, bbox predictions, mask
        predictions, and they have different data types. This method will
        automatically recognize the type, and dump them to json files.

        Args:
            results (list[list | tuple | ndarray]): Testing results of the
                dataset.
            outfile_prefix (str): The filename prefix of the json files. If the
                prefix is "somepath/xxx", the json files will be named
                "somepath/xxx.bbox.json", "somepath/xxx.segm.json",
                "somepath/xxx.proposal.json".

        Returns:
            dict[str: str]: Possible keys are "bbox", "segm", "proposal", and \
                values are corresponding filenames.
        """
        result_files = dict()
        if isinstance(results[0], list):
            json_results = self._det2json(results)
            result_files['bbox'] = f'{outfile_prefix}.bbox.json'
            result_files['proposal'] = f'{outfile_prefix}.bbox.json'
            mmcv.dump(json_results, result_files['bbox'])
        elif isinstance(results[0], tuple):
            json_results = self._segm2json(results)
            result_files['bbox'] = f'{outfile_prefix}.bbox.json'
            result_files['proposal'] = f'{outfile_prefix}.bbox.json'
            result_files['segm'] = f'{outfile_prefix}.segm.json'
            mmcv.dump(json_results[0], result_files['bbox'])
            mmcv.dump(json_results[1], result_files['segm'])
        elif isinstance(results[0], np.ndarray):
            json_results = self._proposal2json(results)
            result_files['proposal'] = f'{outfile_prefix}.proposal.json'
            mmcv.dump(json_results, result_files['proposal'])
        else:
            raise TypeError('invalid type of results')
        return result_files

    def _segm2json(self, results):
        """Convert instance segmentation results to COCO json style."""
        bbox_json_results = []
        segm_json_results = []
        for idx in range(len(self)):
            img_id = self.img_ids[idx]
            det, seg = results[idx]
            for label in range(len(det)):
                # bbox results
                bboxes = det[label]
                for i in range(bboxes.shape[0]):
                    data = dict()
                    data['image_id'] = img_id
                    data['bbox'] = self.xyxy2xywh(bboxes[i])
                    data['score'] = float(bboxes[i][4])
                    data['category_id'] = self.cat_ids[label]
                    bbox_json_results.append(data)

                # segm results
                # some detectors use different scores for bbox and mask
                if isinstance(seg, tuple):
                    segms = seg[0][label]
                    mask_score = seg[1][label]
                else:
                    segms = seg[label]
                    mask_score = [bbox[4] for bbox in bboxes]
                for i in range(bboxes.shape[0]):
                    data = dict()
                    data['image_id'] = img_id
                    data['bbox'] = self.xyxy2xywh(bboxes[i])
                    data['score'] = float(mask_score[i])
                    data['category_id'] = self.cat_ids[label]
                    if isinstance(segms[i]['counts'], bytes):
                        segms[i]['counts'] = segms[i]['counts'].decode()
                    data['segmentation'] = segms[i]
                    segm_json_results.append(data)
        return bbox_json_results, segm_json_results

    # def _load_bboxes(self, results):
    #     """Private function to load bounding box annotations.
    #     """
    #
    #     ann_info = results['ann_info']
    #     results['gt_bboxes'] = ann_info['bboxes'].copy()
    #
    #     gt_bboxes_ignore = ann_info.get('bboxes_ignore', None)
    #     if gt_bboxes_ignore is not None:
    #         results['gt_bboxes_ignore'] = gt_bboxes_ignore.copy()
    #         results['bbox_fields'].append('gt_bboxes_ignore')
    #     results['bbox_fields'].append('gt_bboxes')
    #     return results
    #
    # def _load_labels(self, results):
    #     """Private function to load label annotations.
    #
    #     Args:
    #         results (dict): Result dict from :obj:`mmdet.CustomDataset`.
    #
    #     Returns:
    #         dict: The dict contains loaded label annotations.
    #     """
    #
    #     results['gt_labels'] = results['ann_info']['labels'].copy()
    #     return results
    #
    # def pre_pipeline(self, results):
    #     """Prepare results dict for pipeline."""
    #     results['img_prefix'] = self.img_dir
    #     results['bbox_fields'] = []
    #     img_path = os.path.join(results["img_prefix"], results["img_info"]["file_name"])
    #     img = cv2.imread(img_path)
    #     results["img"] = img
    #
    #     return results

    # def _to_float(self, x):
    #     return float("{:.2f}".format(x))
    #
    # def convert_eval_format(self, all_bboxes):
    #     # import pdb; pdb.set_trace()
    #     detections = []
    #     for image_id in all_bboxes:
    #         for cls_ind in all_bboxes[image_id]:
    #             category_id = self._valid_ids[cls_ind - 1]
    #             for bbox in all_bboxes[image_id][cls_ind]:
    #                 bbox[2] -= bbox[0]
    #                 bbox[3] -= bbox[1]
    #                 score = bbox[4]
    #                 bbox_out = list(map(self._to_float, bbox[0:4]))
    #
    #                 detection = {
    #                     "image_id": int(image_id),
    #                     "category_id": int(category_id),
    #                     "bbox": bbox_out,
    #                     "score": float("{:.2f}".format(score))
    #                 }
    #                 if len(bbox) > 5:
    #                     extreme_points = list(map(self._to_float, bbox[5:13]))
    #                     detection["extreme_points"] = extreme_points
    #                 detections.append(detection)
    #     return detections
    #
    # def __len__(self):
    #     return self.num_samples
    #
    # def save_results(self, results, save_dir):
    #     json.dump(self.convert_eval_format(results),
    #               open('{}/results.json'.format(save_dir), 'w'))
    #
    # def run_eval(self, results, save_dir):
    #     # result_json = os.path.join(save_dir, "results.json")
    #     # detections  = self.convert_eval_format(results)
    #     # json.dump(detections, open(result_json, "w"))
    #     self.save_results(results, save_dir)
    #     coco_dets = self.coco.loadRes('{}/results.json'.format(save_dir))
    #     coco_eval = COCOeval(self.coco, coco_dets, "bbox")
    #     coco_eval.evaluate()
    #     coco_eval.accumulate()
    #     coco_eval.summarize()
