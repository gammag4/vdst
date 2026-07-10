from typing import Callable
from easydict import EasyDict as edict
import torch
import torch.nn as nn
import einx
import math

from .pose_encoder import PoseEncoder
from .transformer import Encoder
from utils.data import normalize_depths, denormalize_depths, normalize_depths2, denormalize_depths2


class VDST(nn.Module):
    # not specified: H, W, C, N_{context}
    # n_heads should divide d_model
    # p should divide H and W (padding, cropping and resizing)
    def __init__(self, config, loss=None):
        super().__init__()
        
        self.config = config
        
        assert self.config.depth_normalization_type in [None, 'standardize', 'min_max'], f'Invalid depth normalization "{self.config.depth_normalization_type}"'
        assert self.config.image_output_type in ['linear', 'sigmoid'], f'Invalid image output type "{self.config.image_output_type}"'
        assert self.config.norm_depth_output_type in ['linear', 'sigmoid'], f'Invalid norm depth output type "{self.config.norm_depth_output_type}"'
        
        self.dmin, self.dmax = self.config.d_range
        self.dmin_log, self.dmax_log = math.log(self.dmin), math.log(self.dmax)
        
        if self.config.depth_normalization_type == 'standardize':
            # self.imean, self.istd = [nn.Parameter(torch.tensor(t)) for t in (0.0, 1.0)]
            self.dmean, self.dstd = [nn.Parameter(torch.tensor(t)) for t in (0.0, 1.0)]
        
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
        
        if self.config.dec_layer_norm:
            self.image_decoder_norm = nn.Sequential(
                nn.LayerNorm(self.config.d_model),
                nn.Linear(in_features=self.config.d_model, out_features=self.config.d_model, bias=False)
            ) if self.config.dec_residual_layer_norm else nn.LayerNorm(self.config.d_model)
            self.depth_decoder_norm = nn.Sequential(
                nn.LayerNorm(self.config.d_model),
                nn.Linear(in_features=self.config.d_model, out_features=self.config.d_model, bias=False)
            ) if self.config.dec_residual_layer_norm else nn.LayerNorm(self.config.d_model)
        
        self.image_decoder_linear = nn.Linear(
            in_features=self.config.d_model,
            out_features=self.config.C * self.config.p ** 2,
            bias=False
        )
        self.depth_decoder_linear = nn.Linear(
            in_features=self.config.d_model,
            out_features=self.config.p ** 2,
            bias=False
        )
        
        self.loss = loss
    
    def state_dict(self):
        # doesnt need to store loss weights since it is not learned
        state_dict = super().state_dict()
        state_dict = {k: v for k, v in state_dict.items() if not k.startswith('loss.')}
        
        return state_dict
    
    def load_state_dict(self, state_dict):
        # doesnt need to store loss weights since it is not learned
        current_state_dict = {k: v for k, v in super().state_dict().items() if k.startswith('loss.')}
        state_dict = {k: v for k, v in state_dict.items() if not k.startswith('loss.')}
        state_dict = {**state_dict, **current_state_dict}
        
        return super().load_state_dict(state_dict)
    
    def normalize_sources(self, images, depths, depth_masks):
        eps = 1e-8
        
        # if self.config.depth_normalization_type != 'standardize':
        images = images * 2.0 - 1.0
        
        depth_masks = depth_masks.float() * 2.0 - 1.0
        
        if self.config.depth_normalization_type == 'min_max':
            depths = normalize_depths(depths, self.dmin_log, self.dmax_log)
            depths = depths * 2.0 - 1.0
        else:
            depths = (depths + eps).log()
            
            if self.config.depth_normalization_type == 'standardize':
                dmean, dstd = self.dmean, self.dstd
                depths = (depths - dmean) / (dstd + eps)
                # images = (images - self.imean) / (self.istd + eps)
        
        return images, depths, depth_masks
    
    def denormalize_targets(self, images, depths):
        norm_depths = None
        
        if self.config.image_output_type == 'sigmoid':
            images = torch.sigmoid(images)
        else:
            # if self.config.depth_normalization_type != 'standardize':
            images = (images + 1.0) * 0.5
        
        if self.config.depth_normalization_type == 'min_max':
            if self.config.norm_depth_output_type == 'sigmoid':
                # Seems to be worse with sigmoid
                norm_depths = torch.sigmoid(depths)
            else:
                norm_depths = (depths + 1.0) * 0.5
            
            depths = denormalize_depths(norm_depths, self.dmin_log, self.dmax_log)
        else:
            if self.config.depth_normalization_type == 'standardize':
                dmean, dstd = self.dmean, self.dstd
                depths = dstd * depths + dmean
                # images = self.istd * images + self.imean
            
            depths = depths.exp()
        
        return images, depths, norm_depths
    
    # Shape: (B, F, C, H, W)
    def forward(self, scene):
        sources, targets = scene.sources, scene.targets
        targets_hw = (targets.images if targets.images is not None else sources.images).shape[-2:]
        sources_depth_masks, targets_depth_masks = [(t.depth_masks & ((t.depths > self.dmin) & (t.depths < self.dmax))) for t in (sources, targets)]
        sources_depths, targets.depths = [torch.clamp(t, min=self.dmin, max=self.dmax) for t in (sources.depths, targets.depths)]
        sources_images, sources_depths, norm_sources_depth_masks = self.normalize_sources(sources.images, sources_depths, sources_depth_masks)
        
        sources = edict(
            K=sources.K,
            R=sources.R,
            t=sources.t,
            images=sources_images,
            depths=sources_depths,
            depth_masks=sources_depth_masks,
            norm_depth_masks=norm_sources_depth_masks
        )
        queries = edict(
            K=targets.K,
            R=targets.R,
            t=targets.t,
            hw=targets_hw
        )
        targets = edict(
            K=targets.K,
            R=targets.R,
            t=targets.t,
            images=targets.images,
            depths=targets.depths,
            depth_masks=targets_depth_masks
        )

        source_embeds, _ = self.pose_encoder_source(sources)
        query_embeds, pad = self.pose_encoder_query(queries)
        targets_hw_padded = targets_hw[0] + pad[2] + pad[3], targets_hw[1] + pad[0] + pad[1]

        orig_query_shape = query_embeds.shape
        source_embeds, query_embeds = [einx.id('... v n d -> (...) (v n) d', t) for t in (source_embeds, query_embeds)]
        in_embeds = torch.concat([source_embeds, query_embeds], dim=-2)
        out_embeds = self.transformer(in_embeds)
        out_embeds = out_embeds[..., -query_embeds.shape[-2]:, :]
        out_embeds = out_embeds.reshape(orig_query_shape)

        if self.config.dec_layer_norm:
            norm_out_img_embeds, norm_out_depth_embeds = self.image_decoder_norm(out_embeds), self.depth_decoder_norm(out_embeds)
            if self.config.dec_residual_layer_norm:
                norm_out_img_embeds, norm_out_depth_embeds = norm_out_img_embeds + out_embeds, norm_out_depth_embeds + out_embeds
        else:
            norm_out_img_embeds, norm_out_depth_embeds = out_embeds, out_embeds
        out_image_embeds, out_depth_embeds = self.image_decoder_linear(norm_out_img_embeds), self.depth_decoder_linear(norm_out_depth_embeds)
        
        out_images_padded, out_depths_padded = [
            einx.id(
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
        
        gen_images, gen_depths, gen_norm_depths = self.denormalize_targets(out_images, out_depths)
        gen_targets = edict(
            K=targets.K,
            R=targets.R,
            t=targets.t,
            hw=targets_hw,
            images=gen_images,
            depths=gen_depths,
            norm_depths=gen_norm_depths,
            # depth_masks=gen_depth_masks # TODO maybe gen this
        )
        
        loss = self.loss(gen_targets, targets) if self.loss is not None else None
        
        return edict(
            scene_name=scene.scene_name,
            sources=scene.sources,
            targets=scene.targets,
            gen_targets=gen_targets,
            loss=loss
        )
