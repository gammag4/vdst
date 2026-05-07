from typing import Callable
from easydict import EasyDict as edict
import torch
import torch.nn as nn
import einx

from .pose_encoder import PoseEncoder
from .transformer import Encoder


class VDST(nn.Module):
    # not specified: H, W, C, N_{context}
    # n_heads should divide d_model
    # p should divide H and W (padding, cropping and resizing)
    def __init__(self, config, loss):
        super().__init__()
        
        self.config = config
        
        self.transformer = Encoder(
            self.config.n_layers,
            self.config.d_model,
            self.config.d_attn,
            self.config.n_heads,
            self.config.e_ff,
            self.config.qk_norm.enabled,
            self.config.qk_norm.eps,
            self.config.train.dropout,
            nn.GELU,
            self.config.attn_op,
            self.config.use_activation_checkpointing
        )
        
        self.pose_encoder_source = PoseEncoder(is_query_encoder=False, config=self.config)
        self.pose_encoder_query = PoseEncoder(is_query_encoder=True, config=self.config)
        
        image_decoder = [
            nn.Linear(
                in_features=self.config.d_model,
                out_features=self.config.C * self.config.p ** 2
            )
        ]
        depth_decoder = [
            nn.Linear(
                in_features=self.config.d_model,
                out_features=self.config.p ** 2
            )
        ]
        if self.config.dec_layer_norm:
            image_decoder = [nn.LayerNorm(self.config.d_model)] + image_decoder
            depth_decoder = [nn.LayerNorm(self.config.d_model)] + depth_decoder
        self.image_decoder = nn.Sequential(*image_decoder)
        self.depth_decoder = nn.Sequential(*depth_decoder)
        
        self.loss = loss
    
    def normalize_sources(self, images, depths):
        eps = 1e-8
        
        depths = (depths + eps).log()
        
        sumdims = (-4, -3, -2, -1)
        (imean, istd), (dmean, dstd) = [(t.mean(dim=sumdims, keepdim=True), t.std(dim=sumdims, keepdim=True)) for t in (images, depths)]
        
        if self.config.standardize_inputs:
            images = (images - imean) / (istd + eps)
            depths = (depths - dmean) / (dstd + eps)
        else:
            images = images * 2.0 - 1.0
        
        normalization_factors = imean, istd, dmean, dstd
        
        return images, depths, normalization_factors
    
    def denormalize_targets(self, images, depths, normalization_factors):
        eps = 1e-8
        
        imean, istd, dmean, dstd = normalization_factors

        images = torch.sigmoid(images)
        
        if self.config.standardize_inputs:
            images = (istd + eps) * images + imean
            depths = (dstd + eps) * depths + dmean
        
        depths = depths.exp()
        
        return images, depths
    
    # Shape: (B, F, C, H, W)
    def forward(self, scene):
        sources, targets = scene.sources, scene.targets
        targets_hw = targets.images.shape[-2:]
        sources_images, sources_depths, normalization_factors = self.normalize_sources(sources.images, sources.depths)
        sources = edict(
            K=sources.K,
            R=sources.R,
            t=sources.t,
            images=sources_images,
            depths=sources_depths,
            depth_masks=sources.depth_masks
        )
        queries = edict(
            K=targets.K,
            R=targets.R,
            t=targets.t,
            hw=targets_hw
        )

        source_embeds, source_depth_mask, _ = self.pose_encoder_source(sources) # TODO do something with source depth mask
        query_embeds, _, pad = self.pose_encoder_query(queries)
        targets_hw_padded = targets_hw[0] + pad[2] + pad[3], targets_hw[1] + pad[0] + pad[1]

        orig_query_shape = query_embeds.shape
        source_embeds, query_embeds = [einx.rearrange('... v n d -> (...) (v n) d', t) for t in (source_embeds, query_embeds)]
        in_embeds = torch.concat([source_embeds, query_embeds], dim=-2)
        out_embeds = self.transformer(in_embeds)
        out_embeds = out_embeds[..., -query_embeds.shape[-2]:, :]
        out_embeds = out_embeds.reshape(orig_query_shape)
        
        out_image_embeds, out_depth_embeds = self.image_decoder(out_embeds), self.depth_decoder(out_embeds)
        out_images_padded, out_depths_padded = [
            einx.rearrange(
                '... (h w) (c p1 p2) -> ... c (h p1) (w p2)',
                t,
                h=targets_hw_padded[0] // self.config.p,
                w=targets_hw_padded[1] // self.config.p,
                p1=self.config.p,
                p2=self.config.p,
                c=c
            )
            for t, c in ((out_image_embeds, 3), (out_depth_embeds, 1))
        ]
        out_images, out_depths = [t[..., pad[2]:t.shape[-2]-pad[3], pad[0]:t.shape[-1]-pad[1]] for t in (out_images_padded, out_depths_padded)]
        
        gen_images, gen_depths = self.denormalize_targets(out_images, out_depths, normalization_factors)
        gen_targets = edict(
            K=targets.K,
            R=targets.R,
            t=targets.t,
            hw=targets_hw,
            images=gen_images,
            depths=gen_depths,
            # depth_masks=gen_depth_masks # TODO maybe gen this
        )
        
        evaluate_targets = True # TODO check if eval mode or loss is none
        loss = self.loss(gen_targets, targets) if evaluate_targets else None
        
        return edict(
            scene_name=scene.scene_name,
            sources=scene.sources,
            targets=scene.targets,
            gen_targets=gen_targets,
            loss=loss
        )
