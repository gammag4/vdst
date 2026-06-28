import einx
import torch
import torch.nn as nn
import torch.nn.functional as F


def compute_pad(hw: tuple[int] | torch.Size, p: int):
    # Pads the input so that it is divisible by 'p'
    # hw: (2,), p: (1)

    pad_raw = [((p - i) % p) for i in hw]
    pad_s = [i // 2 for i in pad_raw]
    pad = (pad_s[1], pad_raw[1] - pad_s[1], pad_s[0], pad_raw[0] - pad_s[0])
    hw_padded = [i + d for i, d in zip(hw, pad_raw)]

    # pad: (pad_width_start, pad_width_end, pad_height_start, pad_height_end) (starts from last dimension to pad)
    # hw_padded: (2,), pad: (4,)
    return hw_padded, pad


def compute_view_rays_from_vecs(vecs: torch.Tensor, K: torch.Tensor, R: torch.Tensor, t: torch.Tensor):
    # Computes view rays (o, d)
    # R, T should be c2w transformations
    # vecs: meshgrid vecs, first dim is (x, y, z)
    # vecs: (3, h, w), K: (3, 3), R: (B, 3, 3), t: (B, 3)

    h, w = vecs.shape[-2:]

    o = t  # c2w: t
    # o = -einx.dot('... h w, ... h -> ... w', R, t)  # w2c: -R^T t
    o = o.view(o.shape + (1, 1)) # allows broadcasting over h w later

    d = einx.dot('... c2 x1, ... x1 c, c h w -> ... c2 h w', R, K.inverse(), vecs)  # c2w: R K^-1 x_ij,cam + t - o = R K^-1 x_ij,cam
    # d = einx.dot('... x1 c2, ... x1 c, c h w -> ... c2 h w', R, K.inverse(), vecs)  # w2c: R^T K^-1 x_ij,cam - R^T t - o = R^T K^-1 x_ij,cam

    d = d / einx.sum('... [c] h w -> ... 3 h w', d ** 2).sqrt()  # normalize d

    # o: (B, 3, 1, 1), d: (B, 3, H, W)
    return o, d


def compute_view_rays(K: torch.Tensor, R: torch.Tensor, t: torch.Tensor, pad: tuple[int], hw: tuple[int] | torch.Size):
    # The forward function was split into two to display the view rays layer

    device = R.device
    pad_s = pad[-2::-2]

    # Creates vectors for each pixel in screen
    # No need to unflip y axis since it being flipped does not affect the topological structure of the representation TODO is it true?
    ranges = [torch.arange(l, dtype=torch.float32, device=device) - o + 0.5 for o, l in zip(pad_s, hw)]
    # Used torch.ones since it seems to be used by most of the vision models similar to this
    # The torch.ones is used bc the convention is that the theoretical sensor plane has focal length 1 (it maps to coordinates (u, v, 1), which would be equivalent to (f u, f v, f) = f(u, v, 1))
    vecs = torch.meshgrid(*ranges, indexing='ij')
    vecs = torch.concat([torch.stack([*vecs[::-1]]), torch.ones((1, *vecs[0].shape), device=device)], dim=-3)

    o, d = compute_view_rays_from_vecs(vecs, K, R, t)
    return o, d


def compute_plucker_rays(o: torch.Tensor, d: torch.Tensor, ray_embedding):
    # o, d: (B, 3, H, W)

    if ray_embedding == 'plucker':
        l = o.cross(d, dim=-3)
        rays = torch.concat([d, l], dim=-3)
    elif ray_embedding == 'origin_direction':
        rays = torch.concat([o, d], dim=-3)
    elif ray_embedding == 'origin_only':
        rays = o
    elif ray_embedding == 'direction_only':
        rays = d
    elif ray_embedding is None:
        return None
    else:
        assert False, f'Invalid ray embedding "{ray_embedding}"'
    
    # rays: (B, ray_C, H, W)
    return rays


# def compute_octaves(v: torch.Tensor, n_oct: int, dim=-1):
#     assert dim < 0, 'No positive dim allowed'

#     v = v * torch.pi
#     tensors = [torch.sin(v), torch.cos(v)]
#     last = v
#     for _ in range(n_oct - 1):
#         last = last * 2
#         tensors.append(torch.sin(last))
#         tensors.append(torch.cos(last))

#     return torch.stack(tensors, dim=dim).flatten(dim - 1, dim)


class PoseEncoder(nn.Module):
    def __init__(self, is_query_encoder, config):
        super().__init__()
        
        self.is_query_encoder = is_query_encoder
        self.config = config
        self.d_model = self.config.d_model
        self.C = self.config.C
        self.p = self.config.p
        self.ray_embedding = self.config.ray_embedding_query if is_query_encoder else self.config.ray_embedding_source
        
        assert not is_query_encoder or self.ray_embedding in ['plucker', 'origin_direction'], f'Invalid ray embedding for query pose encoder "{self.ray_embedding}"'
        
        assert self.config.depth_input_type in ['depth', 'spatial', None], f'Invalid depth input type "{self.config.depth_input_type}"'
        
        c = 0
        if not self.is_query_encoder:
            c = self.C
            
            if self.config.depth_input_type in ['depth', 'spatial']:
                if self.config.depth_input_type == 'depth':
                    c += 1
                elif self.config.depth_input_type == 'spatial':
                    c += 3
                
                if self.config.has_input_depth_masks:
                    c += 1
        
        ray_c = 0 if self.ray_embedding is None else 3 if self.ray_embedding in ['origin_only', 'direction_only'] else 6
        d_input = (ray_c + c) * self.p ** 2
        
        if self.config.enc_layer_norm:
            self.embed_encoder_norm = nn.Sequential(
                nn.LayerNorm(d_input),
                nn.Linear(in_features=d_input, out_features=d_input, bias=False)
            ) if self.config.enc_residual_layer_norm else nn.LayerNorm(d_input)
        
        self.embed_encoder_linear = nn.Linear(
            in_features=d_input,
            out_features=self.d_model,
            bias=False
        )
        self.d_input = d_input

    # HW = tuple with height and width
    # Set both if image has been resized, specifying original image height and width in HW
    # We assume images are already resized (always resize them maintaining aspect ratio)
    # We assume images are already padded so that p divides H and W
    # We assume that the K matrix uses xy mapping instead of uv (sensor area is real in range [(0, 0), (h, w)], not [(0, 0), (1, 1)])
    # We assume images are in type float with colors in range 0-1
    def create_embeds(self, batch):
        # images, depths, depth_masks: (...B, C, H, W), K: (3, 3), R: (...B, 3, 3), t: (...B, 3), hw: (2,)
        if self.is_query_encoder:
            K, R, t, hw = batch.K, batch.R, batch.t, batch.hw
            depth_masks = None
        else:
            K, R, t, images, depths, depth_masks = batch.K, batch.R, batch.t, batch.images, batch.depths, batch.depth_masks

            hw = images.shape[-2:]

        # Pads the input so that it is divisible by 'p'
        hw, pad = compute_pad(hw, self.p)
        if not self.is_query_encoder:
            images = F.pad(images, pad, 'constant', 0)
            if self.config.depth_input_type is not None:
                depths = F.pad(depths, pad, 'constant', 0)
                depth_masks = F.pad(depth_masks, pad, 'constant', False)
        
        o, d = compute_view_rays(K, R, t, pad, hw)
        
        if self.config.depth_input_type == 'spatial':
            # TODO check if this works
            # TODO also check if scale is right (the scale of o, d generated from K R t should be in meters too, check if o is in meters)
            depths = einx.id('b 1 h w -> b c h w', depths, c=3)
            depths = d * depths
        
        plucker_rays = compute_plucker_rays(o, d, self.ray_embedding)  # (B, ray_C, H, W)
        
        exp_inputs = [plucker_rays] if plucker_rays is not None else []
        
        if not self.is_query_encoder:
            exp_inputs.append(images)
            
            if self.config.depth_input_type is not None:
                exp_inputs.append(depths)
                
                if self.config.has_input_depth_masks:
                    exp_inputs.append(depth_masks.float())
        
        left_exp = ', '.join([f'... c{i} (h p1) (w p2)' for i in range(len(exp_inputs))])
        right_exp = ' + '.join([f'c{i}' for i in range(len(exp_inputs))])
        exp = f'{left_exp} -> ... (h w) (({right_exp}) p1 p2)'

        # Concatenating image with rays and rearranging into embeddings
        # (B, HW/p^2, (ray_C + C) * p^2)
        embeds = einx.id(
            # Full exp: '... c0 (h p1) (w p2), ... c1 (h p1) (w p2), ... c2 (h p1) (w p2), ... c3 (h p1) (w p2) -> ... (h w) ((c0 + c1 + c2 + c3) p1 p2)',
            exp,
            *exp_inputs,
            p1=self.p,
            p2=self.p
        )

        # (B, HW/p^2, (ray_C + C) * p^2), (4,)
        return embeds, pad

    # HW = tuple with height and width
    # Set both if image has been resized, specifying original image height and width in HW
    # We assume images are already resized (always resize them maintaining aspect ratio)
    # We assume images are already padded so that p divides H and W
    # We assume that the K matrix uses xy mapping instead of uv (sensor area is real in range [(0, 0), (h, w)], not [(0, 0), (1, 1)])
    # We assume images are in type float with colors in range 0-1
    def forward(self, batch):
        embeds, pad = self.create_embeds(batch)
        
        if self.config.enc_layer_norm:
            out_embeds = self.embed_encoder_norm(embeds)
            if self.config.enc_residual_layer_norm:
                out_embeds = out_embeds + embeds
        else:
            out_embeds = embeds
        out_embeds = self.embed_encoder_linear(out_embeds)

        # (B, n_lat, d_model), (...B, C, H, W), (4,)
        return out_embeds, pad
