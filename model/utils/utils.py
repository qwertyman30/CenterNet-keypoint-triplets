import torch
from torch.nn.utils import clip_grad


def clip_grads(params, grad_clip):
    params = list(
        filter(lambda p: p.requires_grad and p.grad is not None, params))
    if len(params) > 0:
        return clip_grad.clip_grad_norm_(params, **grad_clip)


def update_lr(optimizer, lr):
    print('Drop LR to', lr)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def save_model(model, optimizer, epochs,
               loss_train, loss_cls_train, loss_pts_init_train, loss_pts_refine_train,
               loss_heatmap_train, loss_offset_train, loss_sem_train,
               loss_val, loss_cls_val, loss_pts_init_val, loss_pts_refine_val,
               loss_heatmap_val, loss_offset_val, loss_sem_val):
    model_name = "CenterNet_pp_{}.pth".format(epochs)
    torch.save({
        'epoch': epochs,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'train_loss_history': loss_train,
        'train_loss_history_cls': loss_cls_train,
        'train_loss_history_pts_init': loss_pts_init_train,
        'train_loss_history_pts_refine': loss_pts_refine_train,
        'train_loss_history_heatmap': loss_heatmap_train,
        'train_loss_history_offset': loss_offset_train,
        'train_loss_history_sem': loss_sem_train,
        'val_loss_history': loss_val,
        'val_loss_history_cls': loss_cls_val,
        'val_loss_history_pts_init': loss_pts_init_val,
        'val_loss_history_pts_refine': loss_pts_refine_val,
        'val_loss_history_heatmap': loss_heatmap_val,
        'val_loss_history_offset': loss_offset_val,
        'val_loss_history_sem': loss_sem_val
    }, model_name)
