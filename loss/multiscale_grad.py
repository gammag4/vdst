import torch
import torch.nn as nn
import torch.nn.functional as F


# TODO convert to module


def compute_grad_sobel(rendered, target, target_mask):
    kernel_shape = (1, 1, 3, 3)
    kernel_x = torch.tensor([[-1., 0, 1], [-2, 0, 2], [-1, 0, 1]], device=rendered.device).expand(kernel_shape)
    kernel_y = torch.tensor([[-1., -2, -1], [0, 0, 0], [1, 2, 1]], device=rendered.device).expand(kernel_shape)
    kernel_block = torch.ones(kernel_shape, dtype=torch.float32, device=rendered.device)
    
    m = F.conv2d(target_mask, kernel_block, padding=0) > 8.9
    rx, tx = [F.conv2d(t, kernel_x, padding=0)[m] for t in (rendered, target)]
    ry, ty = [F.conv2d(t, kernel_y, padding=0)[m] for t in (rendered, target)]

    return rx, tx, ry, ty


def compute_grad_difference(images, target_mask):
    is_diff = type(images) is torch.Tensor
    if is_diff:
        images = (images,)
    
    mx = target_mask[..., :, 1:] * target_mask[..., :, :-1]
    my = target_mask[..., 1:, :] * target_mask[..., :-1, :]
    x_grads = [(t[..., :, 1:] - t[..., :, :-1])[mx] for t in images]
    y_grads = [(t[..., 1:, :] - t[..., :-1, :])[my] for t in images]
    
    if is_diff:
        x_grads, y_grads = x_grads[0], y_grads[0]
    
    return x_grads, y_grads


def grad_loss_diff_abs(rendered, target, target_mask, compute_grads=compute_grad_difference):
    diff = rendered - target
    gx, gy = compute_grads(diff, target_mask)
    
    return (gx.abs().sum() + gy.abs().sum()) / target_mask.sum()


def multiscale_grad_loss(rendered, target, target_mask, n_scales=4, grad_loss=grad_loss_diff_abs, compute_grads=compute_grad_difference):
    loss = 0.0
    for s_level in range(n_scales):
        s = 2 ** s_level
        i, gt, m = [k[..., ::s, ::s] for k in (rendered, target, target_mask)]
        
        loss = loss + grad_loss(i, gt, m, compute_grads)

    return loss / n_scales
