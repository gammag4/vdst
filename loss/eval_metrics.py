import torch
import torch.nn as nn
import einx
import torch.nn.functional as F
from easydict import EasyDict as edict
from torchmetrics.image import StructuralSimilarityIndexMeasure as SSIM
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity as LPIPS


class EvalMetrics(nn.Module):
    def __init__(self):
        super().__init__()
        
        self.ssim = SSIM(gaussian_kernel=True, kernel_size=11, reduction=None, data_range=1.0)
        self.lpips = LPIPS(net_type='vgg', reduction=None, normalize=True)
        
        self.eval()
        for param in self.parameters():
            param.requires_grad = False
    
    def forward(self, gen_targets, targets, valid_depth_range=(0.001, 20)):
        images, depths = gen_targets.images.detach(), gen_targets.depths.detach()
        images_gt, depths_gt = targets.images.detach(), targets.depths.detach()
        depths_gt_masks = targets.depth_masks.detach()
        
        valid_depth_masks = depths_gt_masks & (depths_gt > valid_depth_range[0]) & (depths_gt < valid_depth_range[1])
        
        def reduce(op, t):
            return op('... c h w -> ...', t)
        
        def compute_batch_metric(t1, t2, metric):
            r = metric(*[einx.id('... c h w -> (...) c h w', t) for t in (t1, t2)])
            return r.reshape(t1.shape[:-3])
        
        valid_depth_count = reduce(einx.sum, valid_depth_masks.float())

        def reduce_mean_depth(t):
            return reduce(einx.sum, torch.where(valid_depth_masks, t, 0.0)) / valid_depth_count
        
        diff = images - images_gt
        
        images_clamped = images.clamp(0, 1)
        
        images_mse = reduce(einx.mean, diff ** 2)
        images_psnr = -10.0 * torch.log10(images_mse)
        images_ssim = compute_batch_metric(images_clamped, images_gt, self.ssim)
        images_lpips = compute_batch_metric(images_clamped, images_gt, self.lpips) # normalize=True already normalizes to (-1, 1) range
        
        # these need to be masked later
        depths_log = depths.log()
        depths_gt_log = depths_gt.log()
        
        diff = depths - depths_gt
        log_diff = depths_log - depths_gt_log
        
        threshold = torch.max((depths_gt / depths), (depths / depths_gt))
        
        depths_abs_rel = reduce_mean_depth(diff.abs() / depths_gt)
        depths_sq_rel = reduce_mean_depth((diff ** 2) / depths_gt)
        
        depths_mse = reduce_mean_depth(diff ** 2)
        depths_rmse = depths_mse.sqrt()
        depths_rmse_log = reduce_mean_depth(log_diff ** 2).sqrt()
        
        depths_delta_1_25 = reduce_mean_depth((threshold < 1.25).float())
        depths_delta_1_25_2 = reduce_mean_depth((threshold < 1.25 ** 2).float())
        depths_delta_1_25_3 = reduce_mean_depth((threshold < 1.25 ** 3).float())
        
        ldsm = reduce_mean_depth(log_diff ** 2)
        ldm = reduce_mean_depth(log_diff)
        ldms = ldm ** 2
        depths_silog_raw = ldsm - ldms
        depths_silog = 100.0 * depths_silog_raw.sqrt()
        # SNR = mean^2 / std^2
        # If this metric is falling, that means that the error of the absolute component is falling faster than the relative one
        # (and the converse is true if it is rising)
        # This can help choosing the right weights for absolute and relative losses
        depths_snr_log = ldms / depths_silog_raw
        depths_sqrt_snr_log = depths_snr_log.sqrt()
        
        depths_log_10 = depths.log10()
        depths_gt_log_10 = depths_gt.log10()
        depths_mean_log10 = reduce_mean_depth((depths_log_10 - depths_gt_log_10).abs())
        
        image_metrics = edict(
            mse=images_mse,
            psnr=images_psnr,
            ssim=images_ssim,
            lpips=images_lpips
        )
        depth_metrics = edict(
            abs_rel=depths_abs_rel,
            sq_rel=depths_sq_rel,
            
            mse=depths_mse,
            rmse=depths_rmse,
            rmse_log=depths_rmse_log,
            
            delta_1_25=depths_delta_1_25,
            delta_1_25_2=depths_delta_1_25_2,
            delta_1_25_3=depths_delta_1_25_3,
            
            silog=depths_silog,
            snr_log=depths_snr_log,
            sqrt_snr_log=depths_sqrt_snr_log,
            mean_log10=depths_mean_log10
        )
        
        for t in (image_metrics, depth_metrics):
            for k in t.keys():
                t[k] = t[k].cpu()
        
        eval_metrics = edict(
            images=image_metrics,
            depths=depth_metrics,
        )
        
        return eval_metrics
