# [CenterNet++ for Object Detection](https://arxiv.org/abs/2204.08394)

Extend CenterNet++-RT to replace backbone ResNet-50 with DLA-34.

Extend CenterNet++ to work with KITTI dataset

## Installation

### Data

```mkdir data```

```cd data```

#### COCO

```mkdir coco && cd coco```

```mkdir images && cd images```

```wget http://images.cocodataset.org/zips/train2017.zip http://images.cocodataset.org/zips/val2017.zip http://images.cocodataset.org/zips/test2017.zip```

```unzip train2017.zip```

```unzip val2017.zip```

```unzip test2017.zip```

```cd ..```

```mkdir annotations```

```wget http://images.cocodataset.org/annotations/annotations_trainval2017.zip```

```unzip annotations_trainval2017.zip```

#### KITTI

```mkdir kitti && cd kitti```

```mkdir images && cd images```

```wget https://s3.eu-central-1.amazonaws.com/avg-kitti/data_object_image_2.zip https://s3.eu-central-1.amazonaws.com/avg-kitti/data_object_calib.zip https://s3.eu-central-1.amazonaws.com/avg-kitti/data_object_label_2.zip```

```unzip data_object_image_2```

```unzip data_object_label_2```

```unzip data_object_calib```

The dataset directory should be like this:

```plain
├── data
│   ├── coco
│   │   ├── annotations
│   │   ├── images
            ├── train2017
            ├── val2017
            ├── test2017
│   ├── kitti
│   │   ├── annotations
│   │   ├── images
            ├── training
            ├── testing

```

##### 1. Install pytorch

- ``` conda install pytorch==1.11.0 torchvision==0.12.0 torchaudio==0.11.0 cudatoolkit=11.3 -c pytorch```

##### 2. Install requirements

- ```python -m pip install matplotlib mmpycocotools tqdm mmcv-full terminaltables```

##### 3. Install mmdet

- ```python setup.py develop```




