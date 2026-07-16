import torch
import torch.nn as nn
import einx
import torch.nn.functional as F
import torchvision.transforms.v2 as T
from easydict import EasyDict as edict
import math

from .multiscale_grad import MultiScaleGradLoss
from .perceptual import PerceptualLoss
from utils.data import create_input_normalizer


class Loss(nn.Module):
    def __init__(self, model_config, loss_config):
        super().__init__()
        
        self.model_config = model_config
        self.config = loss_config
        
        self.dmin, self.dmax = self.model_config.d_range
        self.dmin_log, self.dmax_log = math.log(self.dmin), math.log(self.dmax)
        self.should_normalize_weights = self.config.should_normalize_weights
        self.weights = nn.Buffer(torch.tensor(self.config.weights))
        
        self.silog_ms_lambda = 1.0 - self.config.silog.scale_lambda
        
        self.multiscale_grad_loss = MultiScaleGradLoss(n_scales=4)
        
        self.perceptual_image = PerceptualLoss(config=self.config.perceptual.image, dist_fn_raw=torch.square, dist_fn=torch.abs)
        self.perceptual_depth = PerceptualLoss(config=self.config.perceptual.depth, dist_fn_raw=torch.square, dist_fn=torch.abs)
        self.depth_perceptual_type = self.config.perceptual.depth.input_type
        
        self.normalize_image = create_input_normalizer(self.config.perceptual.image.input_type, self.config.perceptual.image.is_diff)
        self.normalize_depth = create_input_normalizer(self.config.perceptual.depth.input_type, self.config.perceptual.depth.is_diff)
        
        self.eval()
        for param in self.parameters():
            param.requires_grad = False
    
    def forward(self, gen_targets, targets):
        images, depths = gen_targets.images, gen_targets.depths
        images_gt, depths_gt = targets.images, targets.depths
        depths_gt_masks = targets.depth_masks
        
        normalize_before_all_losses = True #TODO move to loss config
        
        if normalize_before_all_losses:
            images, images_gt = [self.normalize_image(t) for t in (images, images_gt)]
        
        image_mse_loss = F.mse_loss(images, images_gt)
        
        image_perceptual_loss, weighted_image_perceptual_losses, _ = self.perceptual_image(images, images_gt, use_raw_distance=False, should_normalize=not normalize_before_all_losses)
        
        depths_log, depths_gt_log = depths.log(), depths_gt.log()
        depths_log, depths_gt_log = [self.normalize_depth(t) for t in (depths_log, depths_gt_log)]
        
        depths_log, depths_gt_log = [torch.where(depths_gt_masks, t, 0.0) for t in (depths_log, depths_gt_log)]
        depths_mask = depths_log.isinf() | depths_log.isnan()
        depths_log = torch.where(depths_mask, depths_gt_log, depths_log)
        
        log_diff = depths_log - depths_gt_log
        log_diff_masked = log_diff[depths_gt_masks]
        
        ldsm = (log_diff_masked ** 2).mean()
        ldms = log_diff_masked.mean() ** 2
        
        depth_silog_loss_train = ldsm - self.silog_ms_lambda * ldms
        
        depth_multiscale_grad_loss = self.multiscale_grad_loss(depths_log, depths_gt_log, depths_gt_masks)
        
        # TODO maybe use network trained specifically for depth
        # normalization not needed bc the perceptual transforms already normalizes the images
        _, weighted_depth_perceptual_losses, depth_perceptual_per_image_losses = self.perceptual_depth(depths_log, depths_gt_log, depths_gt_masks, use_raw_distance=False, should_normalize=not normalize_before_all_losses)
        
        # TODO one problem is that its hard to normalize the losses only for valid pixels bc you cant know in the middle layers which pixels are valid
        # This is the best way i found to normalize them only for valid pixels TODO check if its better than not normalizing
        # Since it divides by numel, we multiply by numel and divide by number of valid pixels
        # Then we assume this ratio in distances will statistically propagate to the features in the middle
        total_to_valid_pixels_ratio = depths_log[-3:].numel() / einx.sum('... h w -> ...', depths_gt_masks)
        depth_perceptual_loss = (depth_perceptual_per_image_losses * total_to_valid_pixels_ratio).mean()
        
        # TODO add SSI error
        
        # TODO adaptive weights with weighted average over time of losses proportional to how much of each there is
        # do something like beta * last + (1 - beta) * current
        weights = self.weights
        losses = torch.stack([
            image_mse_loss,
            image_perceptual_loss,
            depth_silog_loss_train,
            depth_multiscale_grad_loss,
            depth_perceptual_loss
        ])
        weighted_losses = weights * losses
        loss = weighted_losses.sum()
        if self.should_normalize_weights:
            loss = loss / weights.sum()
        
        res = edict(
            loss=loss,
            loss_weights=weights.data,
            raw_losses=losses,
            weighted_losses=weighted_losses,
            weighted_image_perceptual_losses=weighted_image_perceptual_losses,
            weighted_depth_perceptual_losses=weighted_depth_perceptual_losses,
            # image_mse_loss=image_mse_loss,
            # image_perceptual_loss=image_perceptual_loss,
            # depth_silog_loss_train=depth_silog_loss_train,
            # depth_multiscale_grad_loss=depth_multiscale_grad_loss,
            # depth_perceptual_loss=depth_perceptual_loss
        )
        return res
