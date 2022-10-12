from .coco import COCO
from .kitti import KITTI


def DatasetFactory(opts, split="train", train=True):
    if opts["dataset"] == "coco":
        dataset = COCO(opts, split, train)
    elif opts["dataset"] == "kitti":
        dataset = KITTI(opts, split, train)
    return dataset


__all__ = ["DatasetFactory"]
