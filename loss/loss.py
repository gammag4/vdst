import torch
import torch.nn as nn
import einx
import torch.nn.functional as F
from easydict import EasyDict as edict

from .multiscale_grad import MultiScaleGradLoss
from .perceptual import PerceptualLoss
from utils.data import normalize_depths


class Loss(nn.Module):
    def __init__(self, model_config, loss_config):
        super().__init__()
        
        self.model_config = model_config
        self.config = loss_config
        self.silog_ms_lambda = 1.0 - self.config.silog.scale_lambda
        self.weights = nn.Buffer(torch.tensor(self.config.weights))
        self.perceptual_image = PerceptualLoss(config=self.config.perceptual.image, dist_fn_raw=torch.square, dist_fn=torch.abs)
        self.perceptual_depth = PerceptualLoss(config=self.config.perceptual.depth, dist_fn_raw=torch.square, dist_fn=torch.abs)
        self.multiscale_grad_loss = MultiScaleGradLoss(n_scales=4)
        self.depth_perceptual_type = self.config.perceptual.depth.input_type
        self.dmin, self.dmax = self.model_config.d_range
        
        self.eval()
        for param in self.parameters():
            param.requires_grad = False
    
    def forward(self, gen_targets, targets):
        images, depths = gen_targets.images, gen_targets.depths
        images_gt, depths_gt = targets.images, targets.depths
        depths_gt_masks = targets.depth_masks
        
        image_mse_loss = F.mse_loss(images, images_gt)
        
        image_perceptual_loss, weighted_image_perceptual_losses, _ = self.perceptual_image(images, images_gt, use_raw_distance=False)
        
        # these need to be masked later to prevent biases
        depths_log = torch.where(depths_gt_masks, depths.log(), 0.0)  # always valid bc output is exp
        depths_gt_log = torch.where(depths_gt_masks, depths_gt.log(), 0.0)
        
        log_diff = depths_log - depths_gt_log
        log_diff_masked = log_diff[depths_gt_masks]
        
        ldsm = (log_diff_masked ** 2).mean()
        ldms = log_diff_masked.mean() ** 2
        
        depth_silog_loss_train = ldsm - self.silog_ms_lambda * ldms
        
        depth_multiscale_grad_loss = self.multiscale_grad_loss(depths_log, depths_gt_log, depths_gt_masks)
        
        if self.depth_perceptual_type == 'log_depth':
            # TODO
            # We hypothesize this already imitates grad loss but the problem is that here we cant use the mask
            depths_log2 = torch.where(depths_log.isinf() | depths_log.isnan(), depths_gt_log, depths_log)
            d1, d2 = depths_log2, depths_gt_log
        elif self.depth_perceptual_type == 'norm_log_depth':
            depths_norm, depths_gt_norm = [normalize_depths(t, self.dmin, self.dmax) for t in (depths, depths_gt)]
            depths_norm, depths_gt_norm = [torch.where(depths_gt_masks, t, 0.0) for t in (depths_norm, depths_gt_norm)]
            depths_norm2 = torch.where(depths_norm.isinf() | depths_norm.isnan(), depths_gt_norm, depths_norm)
            d1, d2 = depths_norm2, depths_gt_norm
        else:
            assert False, f'Invalid perceptual type "{self.depth_perceptual_type}"'
        
        # TODO maybe use network trained specifically for depth
        # normalization not needed bc the perceptual transforms already normalizes the images
        d1_img, d2_img = [einx.id('... c2 h w -> (... c2) c h w', t, c=3) for t in (d1, d2)]
        _, weighted_depth_perceptual_losses, depth_perceptual_per_image_losses = self.perceptual_depth(d1_img, d2_img, use_raw_distance=False)
        
        # TODO one problem is that its hard to normalize the losses only for valid pixels bc you cant know in the middle layers which pixels are valid
        # This is the best way i found to normalize them only for valid pixels TODO check if its better than not normalizing
        # Since it divides by numel, we multiply by numel and divide by number of valid pixels
        # Then we assume this ratio in distances will statistically propagate to the features in the middle
        total_to_valid_pixels_ratio = d1[-3:].numel() / einx.sum('... h w -> ...', depths_gt_masks)
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
        loss = weighted_losses.sum() / weights.sum()
        
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
