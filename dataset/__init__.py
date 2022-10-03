from .coco import COCO
from .kitti import KITTI


def DatasetFactory(opts):
    if opts["dataset"] == "coco":
        dataset = COCO(opts)
    elif opts["dataset"] == "kitti":
        dataset = KITTI(opts)
    return dataset


__all__ = ["DatasetFactory"]
