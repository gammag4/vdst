import torch
import torch.nn as nn
import einx
import torch.nn.functional as F
from easydict import EasyDict as edict

from .multiscale_grad import multiscale_grad_loss
from .perceptual import PerceptualLoss


class Loss(nn.Module):
    def __init__(self, model_config, loss_config):
        super().__init__()
        
        self.model_config = model_config
        self.config = loss_config
        # perceptual_weights = torch.tensor([1.6, 1.0, 0.8, 0.7, 0.5, 0.5, 0.5, 0.5, 0.5, 0.2])
        perceptual_weights = torch.full((9,), 1.0)
        self.perceptual = PerceptualLoss(perceptual_weights)
        
        self.eval()
        for param in self.parameters():
            param.requires_grad = False
    
    def forward(self, gen_targets, targets):
        images, depths = gen_targets.images, gen_targets.depths
        images_gt, depths_gt = targets.images, targets.depths
        depths_gt_masks = targets.depth_masks
        
        image_mse_loss = F.mse_loss(images, images_gt)
        
        images_clamped = images
        # if self.model_config.model.standardize_inputs: #TODO check standardized with this
        #     images_clamped = images.clamp(0, 1)
        image_perceptual_loss = self.perceptual(images_clamped, images_gt)
        
        # these need to be masked later to prevent biases
        depths_log = torch.where(depths_gt_masks, depths.log(), 0.0)  # always valid bc output is exp
        depths_gt_log = torch.where(depths_gt_masks, depths_gt.log(), 0.0)
        
        log_diff = (depths_log - depths_gt_log)[depths_gt_masks]
        
        ldsm = (log_diff ** 2).mean()
        ldms = log_diff.mean() ** 2
        
        depth_silog_loss_train = ldsm - 0.85 * ldms
        
        depth_multiscale_grad_loss = multiscale_grad_loss(depths_log, depths_gt_log, depths_gt_masks)
        
        # TODO maybe use network trained specifically for depth
        rlog, tlog = depths_log, depths_gt_log
        rlog, tlog = [(t - tlog.mean()) / tlog.std() for t in (rlog, tlog)]
        rlog, tlog = [einx.rearrange('... c2 h w -> (... c2) c h w', t, c=3) for t in (rlog, tlog)]
        depth_perceptual_loss = self.perceptual(rlog, tlog, use_raw_distance=False)
        
        # TODO add SSI error
        
        # TODO adaptive weights with weighted average over time of losses proportional to how much of each there is
        # do something like beta * last + (1 - beta) * current
        weights = self.config.weights
        losses = [
            image_mse_loss,
            image_perceptual_loss,
            depth_silog_loss_train,
            depth_multiscale_grad_loss,
            depth_perceptual_loss
        ]
        weighted_losses = torch.stack([w * l for w, l in zip(weights, losses)])
        loss = weighted_losses.sum() / sum(weights)
        
        res = edict(
            loss=loss,
            weighted_losses=weighted_losses,
            # image_mse_loss=image_mse_loss,
            # image_perceptual_loss=image_perceptual_loss,
            # depth_silog_loss_train=depth_silog_loss_train,
            # depth_multiscale_grad_loss=depth_multiscale_grad_loss,
            # depth_perceptual_loss=depth_perceptual_loss
        )
        return res
