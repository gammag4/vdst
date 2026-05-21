import torch
import torch.nn as nn
import torch.nn.functional as F


class SobelGradComputer(nn.Module):
    def __init__(self):
        super().__init__()
        
        kernel_shape = (1, 1, 3, 3)
        self.kernel_x = nn.Buffer(torch.tensor([[-1., 0, 1], [-2, 0, 2], [-1, 0, 1]]).expand(kernel_shape))
        self.kernel_y = nn.Buffer(torch.tensor([[-1., -2, -1], [0, 0, 0], [1, 2, 1]]).expand(kernel_shape))
        self.kernel_block = nn.Buffer(torch.ones(kernel_shape, dtype=torch.float32))
    
    def forward(self, images, target_mask):
        m = F.conv2d(target_mask, self.kernel_block, padding=0) > 8.9
        x_grads = [F.conv2d(t, self.kernel_x, padding=0)[m] for t in images]
        y_grads = [F.conv2d(t, self.kernel_y, padding=0)[m] for t in images]
        
        return x_grads, y_grads


class DifferenceGradComputer(nn.Module):
    def forward(self, images, target_mask):
        mx = target_mask[..., :, 1:] * target_mask[..., :, :-1]
        my = target_mask[..., 1:, :] * target_mask[..., :-1, :]
        x_grads = [(t[..., :, 1:] - t[..., :, :-1])[mx] for t in images]
        y_grads = [(t[..., 1:, :] - t[..., :-1, :])[my] for t in images]
        
        return x_grads, y_grads


class MultiScaleGradLoss(nn.Module):
    def __init__(self, n_scales=4, grad_computer=DifferenceGradComputer(), dist_fn=torch.abs):
        super().__init__()
        
        self.n_scales = n_scales
        self.grad_computer = grad_computer
        self.dist_fn = dist_fn
    
    def grad_loss(self, rendered, target, target_mask):
        diff = rendered - target
        images = (diff,)
        gx, gy = self.grad_computer(images, target_mask)
        gx, gy = gx[0], gy[0]

        return (self.dist_fn(gx).sum() + self.dist_fn(gy).sum()) / target_mask.sum()
    
    def forward(self, rendered, target, target_mask):
        loss = 0.0
        for s_level in range(self.n_scales):
            s = 2 ** s_level
            i, gt, m = [k[..., ::s, ::s] for k in (rendered, target, target_mask)]

            loss = loss + self.grad_loss(i, gt, m)

        return loss / self.n_scales
