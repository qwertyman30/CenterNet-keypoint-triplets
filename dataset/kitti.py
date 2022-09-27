import pycocotools.coco as coco
import numpy as np
import mmcv

from .Custom import CustomDataset


class COCO(CustomDataset):
    CLASSES = ['Pedestrian', 'Car', 'Cyclist']

