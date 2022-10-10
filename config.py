from model.backbone.dla import dla34, dla46_c, dla46x_c, dla60, dla60x, dla60x_c, dla102, dla102x, dla102x2, dla169
from model.backbone.resnet import ResNet

# Configuration
opts = dict()
# for debugging purposes. if set to True, the output would be redirected
# to another file in order to filter out the noise in the nohup output
opts["nohup"] = False
opts["train"] = True
# transforms
opts["dataset"] = "kitti"
opts["keep_ratio"] = True
opts["flip_ratio"] = 0.5
opts["to_rgb"] = True
opts["size_divisor"] = 32
opts["to_float"] = False
# learning hyperparams
opts["batch_size"] = 16
opts["num_workers"] = 4
opts["num_epochs"] = 72 if opts["dataset"] == "coco" else 1500
opts["lr"] = 1e-4
opts["lr_step"] = [63, 69] if opts["dataset"] == "coco" else [625, 1200]
opts["save_interval"] = 10 if opts["dataset"] == "coco" else 150
# dataset config
opts["data_root"] = "data/"
opts["seg_prefix"] = None
if opts["to_float"]:
    opts["mean"] = [0.485, 0.456, 0.406]
    opts["std"] = [0.229, 0.224, 0.225]
else:
    opts["mean"] = [123.675, 116.28, 103.53]
    opts["std"] = [58.395, 57.12, 57.375]
opts["img_scale"] = [(900, 256), (900, 608)]
opts["img_scale_test"] = (736, 512)
if opts["dataset"] == "kitti":
    opts["num_classes"] = 3
    opts["ann_file_train"] = "data/kitti/annotations/kitti_3dop_train.json"
    opts["ann_file_val"] = "data/kitti/annotations/kitti_3dop_val.json"
    opts["img_prefix"] = "data/kitti/images/training/image_2"
elif opts["dataset"] == "coco":
    opts["num_classes"] = 80
    opts["ann_file_train"] = "data/coco/annotations/instances_train2017.json"
    opts["ann_file_val"] = "data/coco/annotations/instances_val2017.json"
    opts["img_prefix"] = "data/coco/images/train2017"
opts["backbone"] = "resnet50"
assert opts["backbone"] in [
    "resnet50", "dla34", "dla46_c", "dla46x_c", "dla60", "dla60x", "dla60x_c",
    "dla102", "dla102x", "dla102x2", "dla169"
], "backbone not supported"

norm_cfg = dict(type='GN', num_groups=32, requires_grad=True)

grad_clip = {'max_norm': 35, 'norm_type': 2}

# model settings
backbone_cfg = {
    "resnet50":
    dict(model=ResNet,
         depth=50,
         num_stages=4,
         out_indices=(0, 1, 2, 3),
         frozen_stages=1,
         norm_cfg=dict(type='BN', requires_grad=True),
         norm_eval=True,
         style='pytorch',
         pretrained="pretrained/resnet50-19c8e357.pth"),
    "dla34":
    dict(
        model=dla34,
        levels=[1, 1, 1, 2, 2, 1],
        channels=[16, 32, 64, 128, 256, 512],
        num_classes=opts["num_classes"],
        pretrained="pretrained/dla34-ba72cf86.pth" if opts["train"] else None),
    "dla46_c":
    dict(model=dla46_c,
         levels=[1, 1, 1, 2, 2, 1],
         channels=[16, 32, 64, 64, 128, 256],
         num_classes=opts["num_classes"],
         pretrained="pretrained/dla46_c-2bfd52c3.pth"
         if opts["train"] else None),
    "dla46x_c":
    dict(model=dla46x_c,
         levels=[1, 1, 1, 2, 2, 1],
         channels=[16, 32, 64, 64, 128, 256],
         num_classes=opts["num_classes"],
         pretrained="pretrained/dla46x_c-d761bae7.pth"
         if opts["train"] else None),
    "dla60":
    dict(
        model=dla60,
        levels=[1, 1, 1, 2, 3, 1],
        channels=[16, 32, 128, 256, 512, 1024],
        num_classes=opts["num_classes"],
        pretrained="pretrained/dla60-24839fc4.pth" if opts["train"] else None),
    "dla60x":
    dict(model=dla60x,
         levels=[1, 1, 1, 2, 2, 1],
         channels=[16, 32, 128, 256, 512, 1024],
         num_classes=opts["num_classes"],
         pretrained="pretrained/dla60x-d15cacda.pth"
         if opts["train"] else None),
    "dla60x_c":
    dict(model=dla60x_c,
         levels=[1, 1, 1, 2, 3, 1],
         channels=[16, 32, 64, 64, 128, 256],
         num_classes=opts["num_classes"],
         pretrained="pretrained/dla60x_c-b870c45c.pth"
         if opts["train"] else None),
    "dla102":
    dict(model=dla102,
         levels=[1, 1, 1, 3, 4, 1],
         channels=[16, 32, 128, 256, 512, 1024],
         num_classes=opts["num_classes"],
         pretrained="pretrained/dla102-d94d9790.pth")
    if opts["train"] else None,
    "dla102x":
    dict(model=dla102x,
         levels=[1, 1, 1, 3, 4, 1],
         channels=[16, 32, 128, 256, 512, 1024],
         num_classes=opts["num_classes"],
         pretrained="pretrained/dla102x-ad62be81.pth"
         if opts["train"] else None),
    "dla102x2":
    dict(model=dla102x2,
         levels=[1, 1, 1, 3, 4, 1],
         channels=[16, 32, 128, 256, 512, 1024],
         num_classes=opts["num_classes"],
         pretrained="pretrained/dla102x2-262837b6.pth"
         if opts["train"] else None),
    "dla169":
    dict(
        model=dla169,
        levels=[1, 1, 2, 3, 5, 1],
        channels=[16, 32, 128, 256, 512, 1024],
        num_classes=opts["num_classes"],
        pretrained="pretrained/dla169-0914e092.pth" if opts["train"] else None)
}

neck_cfg = dict(in_channels=backbone_cfg[opts["backbone"]]["channels"]
                if "dla" in opts["backbone"] else [256, 512, 1024, 2048],
                out_channels=256,
                start_level=3 if "dla" in opts["backbone"] else 1,
                add_extra_convs='on_input',
                num_outs=5,
                norm_cfg=norm_cfg)

bbox_head_cfg = dict(
    num_classes=opts["num_classes"],
    in_channels=256,
    feat_channels=256,
    point_feat_channels=256,
    stacked_convs=1,
    shared_stacked_convs=1,
    first_kernel_size=3,
    kernel_size=1,
    corner_dim=64,
    num_points=9,
    gradient_mul=0.1,
    point_strides=[8, 16, 32, 64, 128],
    point_base_scale=4,
    norm_cfg=norm_cfg,
    conv_module_style='dcn',  # norm or dcn, norm is faster
    loss_cls=dict(use_sigmoid=True, gamma=2.0, alpha=0.25, loss_weight=1.0),
    loss_bbox_init=dict(loss_weight=1.0),
    loss_bbox_refine=dict(loss_weight=2.0),
    loss_heatmap=dict(alpha=2.0, gamma=4.0, loss_weight=0.25),
    loss_offset=dict(beta=1.0 / 9.0, loss_weight=1.0),
    loss_sem=dict(gamma=2.0, alpha=0.25, loss_weight=0.1))

# Detector cfg
train_cfg = dict(init=dict(assigner=dict(type='PointAssignerV2',
                                         scale=4,
                                         pos_num=1),
                           allowed_border=-1,
                           pos_weight=-1,
                           debug=False),
                 heatmap=dict(assigner=dict(type='PointHMAssignerV2',
                                            gaussian_bump=True,
                                            gaussian_iou=0.7),
                              allowed_border=-1,
                              pos_weight=-1,
                              debug=False),
                 refine=dict(assigner=dict(type='ATSSAssigner', topk=9),
                             allowed_border=-1,
                             pos_weight=-1,
                             debug=False))

test_cfg = dict(distance_threshold=0.5,
                nms_pre=1000,
                min_bbox_size=0,
                score_thr=0.05,
                nms=dict(type='nms', iou_threshold=0.6),
                max_per_img=100)

optimizer_cfg = dict(
    type='AdamW',
    lr=opts["lr"],
    betas=(0.9, 0.999),
    weight_decay=0.05,
    paramwise_cfg=dict(
        custom_keys={
            'absolute_pos_embed': dict(decay_mult=0.),
            'relative_position_bias_table': dict(decay_mult=0.),
            'norm': dict(decay_mult=0.)
        }))
