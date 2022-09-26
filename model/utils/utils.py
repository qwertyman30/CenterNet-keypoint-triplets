import torch
from torch.nn.utils import clip_grad


def clip_grads(params, grad_clip):
    params = list(
        filter(lambda p: p.requires_grad and p.grad is not None, params))
    if len(params) > 0:
        return clip_grad.clip_grad_norm_(params, **grad_clip)


def save_model(model, optimizer, epochs,
               LOSS, LOSS_CLS, LOSS_PTS_INIT, LOSS_PTS_REFINE,
               LOSS_HEATMAP, LOSS_OFFSET, LOSS_SEM):
    model_name = "centernet_pp_{}.pth".format(epochs)
    torch.save({
        'epoch': epochs,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'train_loss_history': LOSS,
        'train_loss_history_cls': LOSS_CLS,
        'train_loss_history_pts_init': LOSS_PTS_INIT,
        'train_loss_history_pts_refine': LOSS_PTS_REFINE,
        'train_loss_history_heatmap': LOSS_HEATMAP,
        'train_loss_history_offset': LOSS_OFFSET,
        'train_loss_history_sem': LOSS_SEM,
    }, model_name)
