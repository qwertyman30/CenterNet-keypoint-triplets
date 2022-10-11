import numpy as np
import matplotlib.pyplot as plt
from mmcv.parallel.data_parallel import scatter_kwargs
from model.utils.utils import clip_grads
from config import grad_clip


def step(detector, data_loader, progress, optimizer=None, train=True):
    if train:
        detector.train()
        func = detector.train_step
    else:
        detector.eval()
        func = detector.val_step
    loss_cls_, loss_pts_init_, loss_pts_refine_, loss_heatmap_, loss_offset_, loss_sem_, loss_ = [], [], [], [], [], [], []
    for batch in data_loader:
        batch, _ = scatter_kwargs(batch, None, [0])
        LOSS = func(batch[0])

        log_vars = LOSS["log_vars"]
        loss_cls = log_vars['loss_cls']
        loss_pts_init = log_vars['loss_pts_init']
        loss_pts_refine = log_vars['loss_pts_refine']
        loss_heatmap = log_vars['loss_heatmap']
        loss_offset = log_vars['loss_offset']
        loss_sem = log_vars['loss_sem']
        loss = log_vars['loss']

        # backprop
        if train:
            optimizer.zero_grad()
            LOSS["loss"].backward()
            if grad_clip is not None:
                clip_grads(detector.parameters(), grad_clip)
            optimizer.step()

        progress.set_description(
            'LOSS: %.4f, CLS: %.4f PTS_INIT: %.4f PTS_REFINE: %.4f HEATMAP: %.4f OFFSET: %.4f SEM: %.4f'
            % (loss, loss_cls, loss_pts_init, loss_pts_refine, loss_heatmap,
               loss_offset, loss_sem))

        loss_cls_.append(loss_cls)
        loss_pts_init_.append(loss_pts_init)
        loss_pts_refine_.append(loss_pts_refine)
        loss_heatmap_.append(loss_heatmap)
        loss_offset_.append(loss_offset)
        loss_sem_.append(loss_sem)
        loss_.append(loss)

    loss_cls_mean = np.mean(loss_cls_)
    loss_pts_init_mean = np.mean(loss_pts_init_)
    loss_pts_refine_mean = np.mean(loss_pts_refine_)
    loss_heatmap_mean = np.mean(loss_heatmap_)
    loss_offset_mean = np.mean(loss_offset_)
    loss_sem_mean = np.mean(loss_sem_)
    loss_mean = np.mean(loss_)
    loss_means = {}
    loss_means['loss_cls'] = loss_cls_mean
    loss_means['loss_pts_init'] = loss_pts_init_mean
    loss_means['loss_pts_refine'] = loss_pts_refine_mean
    loss_means['loss_heatmap'] = loss_heatmap_mean
    loss_means['loss_offset'] = loss_offset_mean
    loss_means['loss_sem'] = loss_sem_mean
    loss_means['loss'] = loss_mean
    print(
        f"\nLOSS_CLS: {loss_cls_mean}, LOSS_PTS_INIT: {loss_pts_init_mean}, LOSS_PTS_REFINE: {loss_pts_refine_mean}, "
        f"LOSS_HEATMAP: {loss_heatmap_mean}, LOSS_OFFSET: {loss_offset_mean}, LOSS_SEM: {loss_sem_mean}, "
        f"LOSS: {loss_mean}\n")
    return loss_means


def plot(losses, title, log_scale=False):
    if log_scale:
        plt.yscale("log")
    plt.plot(losses)
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.title(f"{title}")
    plt.savefig(f"{title}.png")
    plt.show()
    plt.clf()
    plt.cla()
    plt.close()
