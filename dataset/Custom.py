import warnings
# from collections import OrderedDict
import os.path as osp

import mmcv
import numpy as np
# from mmcv.utils import print_log
from torch.utils.data import Dataset

# from mmdet.core import eval_map, eval_recalls
from .utils.loaders import LoadAnnotations, LoadImageFromFile
from .utils.transforms import Resize, RandomFlip, Normalize, Pad
from .utils.formatting import Collect, ImageToTensor, RPDV2FormatBundle, LoadRPDV2Annotations
from .utils.test_augs import MultiScaleFlipAug


class CustomDataset(Dataset):
    """Custom dataset for detection.

    The annotation format is shown as follows. The `ann` field is optional for
    testing.

    .. code-block:: none

        [
            {
                'filename': 'a.jpg',
                'width': 1280,
                'height': 720,
                'ann': {
                    'bboxes': <np.ndarray> (n, 4) in (x1, y1, x2, y2) order.
                    'labels': <np.ndarray> (n, ),
                    'bboxes_ignore': <np.ndarray> (k, 4), (optional field)
                    'labels_ignore': <np.ndarray> (k, 4) (optional field)
                }
            },
            ...
        ]

    Args:
        ann_file (str): Annotation file path.
        pipeline (list[dict]): Processing pipeline.
        classes (str | Sequence[str], optional): Specify classes to load.
            If is None, ``cls.CLASSES`` will be used. Default: None.
        data_root (str, optional): Data root for ``ann_file``,
            ``img_prefix``, ``seg_prefix``, ``proposal_file`` if specified.
        test_mode (bool, optional): If set True, annotation will not be loaded.
        filter_empty_gt (bool, optional): If set true, images without bounding
            boxes of the dataset's classes will be filtered out. This option
            only works when `test_mode=False`, i.e., we never filter images
            during tests.
    """

    CLASSES = None

    def __init__(self,
                 opts,
                 classes=None,
                 test_mode=False,
                 filter_empty_gt=True):
        self.ann_file = opts["ann_file"]
        self.data_root = opts["data_root"]
        self.img_prefix = opts["img_prefix"]
        self.seg_prefix = opts["seg_prefix"]
        self.test_mode = test_mode
        self.filter_empty_gt = filter_empty_gt
        self.CLASSES = self.get_classes(classes)

        # load annotations
        self.data_infos = self.load_annotations(self.ann_file)

        # filter images too small and containing no annotations
        if not test_mode:
            valid_inds = self._filter_imgs()
            self.data_infos = [self.data_infos[i] for i in valid_inds]
            # set group flag for the sampler
            self._set_group_flag()

        # Loading pipeline
        self.LoadImageFromFile = LoadImageFromFile(to_float=opts["to_float"])
        self.LoadAnnotations = LoadAnnotations(with_bbox=True)

        # transformation pipeline training
        self.Resize_train = Resize(img_scale=opts["img_scale"], multiscale_mode='range',
                                   keep_ratio=opts["keep_ratio"])
        self.RandomFlip_train = RandomFlip(flip_ratio=opts["flip_ratio"])
        self.Normalize = Normalize(mean=opts["mean"], std=opts["std"], to_rgb=True)
        self.Pad = Pad(size_divisor=opts["size_divisor"])

        # formatting pipeline
        self.LoadRPDV2Annotations = LoadRPDV2Annotations()
        self.RPDV2FormatBundle = RPDV2FormatBundle()
        self.Collect_train = Collect(keys=['img', 'gt_bboxes', 'gt_labels', 'gt_sem_map', 'gt_sem_weights'])

        # test transforms
        self.MultiScaleFlipAug = MultiScaleFlipAug(img_scale=(736, 512), flip=False)

    def __len__(self):
        """Total number of samples of data."""
        return len(self.data_infos)

    def load_annotations(self, ann_file):
        """Load annotation from annotation file."""
        return mmcv.load(ann_file)

    def get_ann_info(self, idx):
        """Get annotation by index.

        Args:
            idx (int): Index of data.

        Returns:
            dict: Annotation info of specified index.
        """

        return self.data_infos[idx]['ann']

    def get_cat_ids(self, idx):
        """Get category ids by index.

        Args:
            idx (int): Index of data.

        Returns:
            list[int]: All categories in the image of specified index.
        """

        return self.data_infos[idx]['ann']['labels'].astype(np.int).tolist()

    def _filter_imgs(self, min_size=32):
        """Filter images too small."""
        if self.filter_empty_gt:
            warnings.warn(
                'CustomDataset does not support filtering empty gt images.')
        valid_inds = []
        for i, img_info in enumerate(self.data_infos):
            if min(img_info['width'], img_info['height']) >= min_size:
                valid_inds.append(i)
        return valid_inds

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

    @classmethod
    def get_classes(cls, classes=None):
        """Get class names of current dataset.

        Args:
            classes (Sequence[str] | str | None): If classes is None, use
                default CLASSES defined by builtin dataset. If classes is a
                string, take it as a file name. The file contains the name of
                classes where each line contains one class name. If classes is
                a tuple or list, override the CLASSES defined by the dataset.

        Returns:
            tuple[str] or list[str]: Names of categories of the dataset.
        """
        if classes is None:
            return cls.CLASSES

        if isinstance(classes, str):
            # take it as a file path
            class_names = mmcv.list_from_file(classes)
        elif isinstance(classes, (tuple, list)):
            class_names = classes
        else:
            raise ValueError(f'Unsupported type {type(classes)} of classes.')

        return class_names

    # def evaluate(self,
    #              results,
    #              metric='mAP',
    #              logger=None,
    #              proposal_nums=(100, 300, 1000),
    #              iou_thr=0.5,
    #              scale_ranges=None):
    #     """Evaluate the dataset.
    #
    #     Args:
    #         results (list): Testing results of the dataset.
    #         metric (str | list[str]): Metrics to be evaluated.
    #         logger (logging.Logger | None | str): Logger used for printing
    #             related information during evaluation. Default: None.
    #         proposal_nums (Sequence[int]): Proposal number used for evaluating
    #             recalls, such as recall@100, recall@1000.
    #             Default: (100, 300, 1000).
    #         iou_thr (float | list[float]): IoU threshold. Default: 0.5.
    #         scale_ranges (list[tuple] | None): Scale ranges for evaluating mAP.
    #             Default: None.
    #     """
    #
    #     if not isinstance(metric, str):
    #         assert len(metric) == 1
    #         metric = metric[0]
    #     allowed_metrics = ['mAP', 'recall']
    #     if metric not in allowed_metrics:
    #         raise KeyError(f'metric {metric} is not supported')
    #     annotations = [self.get_ann_info(i) for i in range(len(self))]
    #     eval_results = OrderedDict()
    #     iou_thrs = [iou_thr] if isinstance(iou_thr, float) else iou_thr
    #     if metric == 'mAP':
    #         assert isinstance(iou_thrs, list)
    #         mean_aps = []
    #         for iou_thr in iou_thrs:
    #             print_log(f'\n{"-" * 15}iou_thr: {iou_thr}{"-" * 15}')
    #             mean_ap, _ = eval_map(
    #                 results,
    #                 annotations,
    #                 scale_ranges=scale_ranges,
    #                 iou_thr=iou_thr,
    #                 dataset=self.CLASSES,
    #                 logger=logger)
    #             mean_aps.append(mean_ap)
    #             eval_results[f'AP{int(iou_thr * 100):02d}'] = round(mean_ap, 3)
    #         eval_results['mAP'] = sum(mean_aps) / len(mean_aps)
    #     elif metric == 'recall':
    #         gt_bboxes = [ann['bboxes'] for ann in annotations]
    #         recalls = eval_recalls(
    #             gt_bboxes, results, proposal_nums, iou_thr, logger=logger)
    #         for i, num in enumerate(proposal_nums):
    #             for j, iou in enumerate(iou_thrs):
    #                 eval_results[f'recall@{num}@{iou}'] = recalls[i, j]
    #         if recalls.shape[1] > 1:
    #             ar = recalls.mean(axis=1)
    #             for i, num in enumerate(proposal_nums):
    #                 eval_results[f'AR@{num}'] = ar[i]
    #     return eval_results
