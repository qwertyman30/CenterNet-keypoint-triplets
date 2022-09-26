norm_cfg = dict(type='GN', num_groups=32, requires_grad=True)

backbone_cfg = dict(depth=50,
                    num_stages=4,
                    out_indices=(0, 1, 2, 3),
                    frozen_stages=1,
                    norm_cfg=dict(type='BN', requires_grad=True),
                    norm_eval=True,
                    style='pytorch')
neck_cfg = dict(in_channels=[256, 512, 1024, 2048],
                out_channels=256,
                start_level=1,
                add_extra_convs='on_input',
                num_outs=5,
                norm_cfg=norm_cfg)
bbox_head_cfg = dict(num_classes=80,
                     in_channels=256,
                     feat_channels=256,
                     point_feat_channels=256,
                     stacked_convs=3,
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
                     loss_cls=dict(
                         use_sigmoid=True,
                         gamma=2.0,
                         alpha=0.25,
                         loss_weight=1.0),
                     loss_bbox_init=dict(loss_weight=1.0),
                     loss_bbox_refine=dict(loss_weight=2.0),
                     loss_heatmap=dict(
                         alpha=2.0,
                         gamma=4.0,
                         loss_weight=0.25),
                     loss_offset=dict(beta=1.0 / 9.0, loss_weight=1.0),
                     loss_sem=dict(
                         gamma=2.0,
                         alpha=0.25,
                         loss_weight=0.1))
# Pycenternet_detector cfg
train_cfg = dict(
    init=dict(
        assigner=dict(type='PointAssignerV2', scale=4, pos_num=1),
        allowed_border=-1,
        pos_weight=-1,
        debug=False),
    heatmap=dict(
        assigner=dict(type='PointHMAssignerV2', gaussian_bump=True, gaussian_iou=0.7),
        allowed_border=-1,
        pos_weight=-1,
        debug=False),
    refine=dict(
        assigner=dict(type='ATSSAssigner', topk=9),
        allowed_border=-1,
        pos_weight=-1,
        debug=False))
test_cfg = dict(
    distance_threshold=0.5,
    nms_pre=1000,
    min_bbox_size=0,
    score_thr=0.05,
    nms=dict(type='nms', iou_threshold=0.6),
    max_per_img=100)

optimizer_cfg = dict(type='AdamW', lr=1e-7, betas=(0.9, 0.999), weight_decay=0.05,
                     paramwise_cfg=dict(custom_keys={'absolute_pos_embed': dict(decay_mult=0.),
                                                     'relative_position_bias_table': dict(decay_mult=0.),
                                                     'norm': dict(decay_mult=0.)}))

grad_clip = {'max_norm': 35, 'norm_type': 2}
