from .coco import COCO
from .kitti import KITTI


def DatasetFactory(opts, train=True):
    if opts["dataset"] == "coco":
        dataset = COCO(opts, train)
    elif opts["dataset"] == "kitti":
        dataset = KITTI(opts, train)
    return dataset


__all__ = ["DatasetFactory"]
