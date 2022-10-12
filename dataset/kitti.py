import cv2
import numpy as np
import mmcv
from torch.utils.data import Dataset
import pycocotools.coco as coco
import os.path as osp
# from mmdet.core import eval_map, eval_recalls
from .utils.loaders import LoadAnnotations, LoadImageFromFile
from .utils.transforms import Resize, RandomFlip, Normalize, ColorTransform, Pad
from .utils.test_augs import MultiScaleFlipAug
from .utils.formatting import Collect, RPDV2FormatBundle, LoadRPDV2Annotations
from collections import OrderedDict


class KITTI(Dataset):
    CLASSES = ['Pedestrian', 'Car', 'Cyclist']

    def __init__(self, opts, split="train", train=True):
        if split == "train":
            self.ann_file = opts["ann_file_train"]
        else:
            self.ann_file = opts["ann_file_val"]
        self.data_root = opts["data_root"]
        self.img_prefix = opts["img_prefix"]
        self.seg_prefix = opts["seg_prefix"]
        self.test_mode = not train

        self.alpha_in_degree = False

        # load annotations
        self.data_infos = self.load_annotations(self.ann_file)

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
        # self.ColorTransform = ColorTransform(level=5.)
        self.Normalize = Normalize(mean=opts["mean"],
                                   std=opts["std"],
                                   to_rgb=opts["to_rgb"])
        self.Pad = Pad(size_divisor=opts["size_divisor"])
        # formatting pipeline
        self.LoadRPDV2Annotations = LoadRPDV2Annotations(num_classes=3)
        self.RPDV2FormatBundle = RPDV2FormatBundle()
        self.Collect_train = Collect(keys=[
            'img', 'gt_bboxes', 'gt_labels', 'gt_sem_map', 'gt_sem_weights'
        ])

        # test transforms and pipeline
        self.MultiScaleFlipAug = MultiScaleFlipAug(
            opts, img_scale=opts["img_scale_test"])

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
            info["annos"] = self.coco.loadAnns(
                self.coco.getAnnIds(imgIds=[i], catIds=self.cat_ids))
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

        ann = dict(bboxes=gt_bboxes,
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
        # results = self.ColorTransform(results)
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

    def __getitem__(self, idx):
        if self.test_mode:
            return self.prepare_test_img(idx)
        while True:
            return self.prepare_train_img(idx)

    def format_results(self,
                       results,
                       save_dir=None,
                       save_debug_dir=None,
                       **kwargs):
        """Format the results to txt (standard format for Kitti evaluation).
        Args:
            results (list): Testing results of the dataset.
            save_dir (str | None): The prefix of txt files. It includes
                the file path and the prefix of filename, e.g., "a/b/prefix".
                If not specified, a temp file will be created. Default: None.
        Returns:
            tuple: (result_files, tmp_dir), result_files is a dict containing
                the txt filepaths, tmp_dir is the temporal directory created
                for saving txt files when txtfile_prefix is not specified.
        """
        assert isinstance(results, list), 'results must be a list'
        assert len(results) == len(self), (
            'The length of results is not equal to the dataset len: {} != {}'.
            format(len(results), len(self)))

        det_annos = []
        for idx, bbox_list in enumerate(results):
            sample_idx = self.data_infos[idx]['image_idx']
            num_example = 0

            anno = {
                'name': [],
                'truncated': [],
                'occluded': [],
                'alpha': [],
                'bbox': [],
                'dimensions': [],
                'location': [],
                'rotation_y': [],
                'score': []
            }
            for cls_idx, cls_bbox in enumerate(bbox_list):
                cls_name = self.cat_ids[cls_idx]
                bbox_2d_preds = cls_bbox[:, :4]
                scores = cls_bbox[:, 4]
                # TODO: 3d bbox prediction
                for bbox_2d_pred, score in zip(bbox_2d_preds, scores):
                    # TODO: out of range detection, should be filtered in the network part
                    anno["score"].append(score)
                    anno["name"].append(cls_name)
                    anno["truncated"].append(0.0)
                    anno["occluded"].append(0)
                    anno["bbox"].append(bbox_2d_pred)
                    anno["alpha"].append(0.)
                    anno["dimensions"].append(
                        np.array([0, 0, 0], dtype=np.float32))
                    anno["location"].append(
                        np.array([-1000, -1000, -1000], dtype=np.float32))
                    anno["rotation_y"].append(0.)
                    num_example += 1

            if num_example != 0:
                anno = {k: np.stack(v) for k, v in anno.items()}
            else:
                anno = {
                    'name': np.array([]),
                    'truncated': np.array([]),
                    'occluded': np.array([]),
                    'alpha': np.array([]),
                    'bbox': np.zeros([0, 4]),
                    'dimensions': np.zeros([0, 3]),
                    'location': np.zeros([0, 3]),
                    'rotation_y': np.array([]),
                    'score': np.array([])
                }

            anno["sample_idx"] = np.array([sample_idx] * num_example,
                                          dtype=np.int64)
            det_annos.append(anno)

            if save_dir is not None:
                cur_det_file = osp.join(save_dir, 'results',
                                        '%06d.txt' % sample_idx)
                print("saving results to", cur_det_file)

                # dump detection results into txt files
                with open(cur_det_file, 'w') as f:
                    bbox = anno['bbox']
                    loc = anno['location']
                    dims = anno['dimensions']  # lhw -> hwl

                    for idx in range(len(bbox)):
                        print(
                            '%s -1 -1 %.4f %.4f %.4f %.4f %.4f %.4f %.4f %.4f %.4f %.4f %.4f %.4f %.4f'
                            % (anno['name'][idx], anno['alpha'][idx],
                               bbox[idx][0], bbox[idx][1], bbox[idx][2],
                               bbox[idx][3], dims[idx][1], dims[idx][2],
                               dims[idx][0], loc[idx][0], loc[idx][1],
                               loc[idx][2], anno['rotation_y'][idx],
                               anno['score'][idx]),
                            file=f)

            if save_debug_dir is not None:
                # dump debug infos into pkl files
                cur_debug_file = osp.join(save_debug_dir,
                                          'det_info_%06d.pkl' % sample_idx)
                debug_infos = {
                    'anno': anno,
                    'gt_anno': self.data_infos[idx]["annos"],
                }
                with open(cur_debug_file, 'wb') as f:
                    mmcv.dump(debug_infos, f)

        return det_annos

    def evaluate(self, results, txtfile_prefix=None):
        if 'annos' not in self.data_infos[0]:
            print('The testing results of the whole dataset is empty.')
            raise ValueError(
                "annotations not available for the test set of KITTI")

        print('Evaluating KITTI object detection \n')
        det_annos = self.format_results(results, txtfile_prefix)
        gt_annos = [x['annos'] for x in self.data_infos]

        from kitti_object_eval_python.eval import get_official_eval_result
        eval_results = get_official_eval_result(gt_annos, det_annos,
                                                self.CLASSES)

        return_results = OrderedDict()
        for cls_name, ret in eval_results.items():
            for k, v in ret.items():
                msg = ','.join([f"{vv:.3f}" for vv in v])
                return_results[f"{cls_name}_{k}"] = msg
                print(f"{cls_name}_{k}:  {msg}")

        return return_results
