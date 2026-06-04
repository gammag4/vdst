import copy
import torch
import torch.nn as nn
import einx
import torchvision.transforms.v2 as T
from torchvision.models import convnext_tiny, ConvNeXt_Tiny_Weights, vgg19, VGG19_Weights
import torchvision.transforms.functional as VF
from easydict import EasyDict as edict


class PerceptualLoss(nn.Module):
    cnn_archs = edict()
    
    def __init__(self, config, dist_fn_raw=torch.square, dist_fn=torch.abs):
        super().__init__()
        
        model_type = config.type
        layer_weights = config.weights
        self.transforms_type = config.transforms
        input_type = config.input_type
        self.is_diff = config.is_diff
        interpolation = config.interpolation
        
        self.dist_fn_raw = dist_fn_raw
        self.dist_fn = dist_fn
        
        assert interpolation in ['default', 'nearest'], f'Invalid interpolation "{interpolation}"'
        assert model_type in ['convnext', 'vgg', 'vgg2', 'vgg3'], f'Invalid model type "{model_type}"'
        
        # The stats from CO3D were estimated from the processed dataset, so they are different from the ones from the full raw dataset
        if input_type == 'image':
            # Gets defaults for imagenet1k
            data_mean, data_std = None, None
        elif input_type == 'log_depth':
            # From CO3D
            data_mean, data_std = -0.2749, 0.9187
        elif input_type == 'norm_log_depth':
            # From CO3D
            # TODO these values were computed for d_min = 0.01 and d_max = 1000.0
            # if the range changes, these need to be recomputed for the new range
            # fix this so that it can be normalized for range shifts (they were computed from the formula in loss.py)
            data_mean, data_std = 0.0020, 0.0018
        else:
            assert False, f'Invalid input type "{input_type}"'
        
        if self.is_diff and data_mean is not None:
            data_mean, data_std = 0.0, data_std * (2 ** 0.5)
        
        norm_dict = dict(mean=[data_mean] * 3, std=[data_std] * 3) if data_mean is not None else None
        
        arch = self.cnn_archs.get(model_type, None)
        if arch is not None:
            weights, self.model, self.layers = arch.weights, arch.model, arch.layers
        
        if model_type == 'convnext':
            if arch is None:
                weights = ConvNeXt_Tiny_Weights.DEFAULT
                self.model = convnext_tiny(weights=weights)
                self.layers = list(self.model.features)
                self.cnn_archs[model_type] = edict(weights=weights, model=self.model, layers=self.layers)
            
            layer_weights = torch.ones(9, dtype=torch.float32) if layer_weights is None else torch.tensor(layer_weights, dtype=torch.float32)
            
        elif model_type in ['vgg', 'vgg2', 'vgg3']:
            if arch is None:
                weights = VGG19_Weights.DEFAULT
                self.model = vgg19(weights=weights)
            
            # AvgPool allows better differentiability
            features = [nn.AvgPool2d(kernel_size=2, stride=2) if isinstance(i, nn.MaxPool2d) else i for i in self.model.features]
            if model_type == 'vgg3':
                indices = [0] + [i + 1 for i, e in enumerate(features) if isinstance(e, nn.ReLU)]
            elif model_type == 'vgg2':
                indices = [0] + [i for i, e in enumerate(features) if isinstance(e, nn.AvgPool2d)]
            else:
                indices = [0, 4, 9, 14, 23, 32] # layers conv1_2 ... conv5_2 from https://arxiv.org/pdf/1707.09405 p. 5
            
            if arch is None:
                self.layers = [nn.Sequential(*features[s:e]) for s, e in zip(indices[:-1], indices[1:])]
                self.cnn_archs[model_type] = edict(weights=weights, model=self.model, layers=self.layers)
            
            layer_weights = torch.ones(len(indices), dtype=torch.float32) if layer_weights is None else layer_weights
        
        # TODO test with classifier layer
        # self.classifier_layer = lambda x: self.model.classifier(self.model.avgpool(x))
        
        if self.transforms_type in ['default', 'default_size_224', 'default_size_236']:
            self.base_transforms = weights.transforms(**norm_dict) if norm_dict is not None else weights.transforms()
            
            # resizing and cropping at same size to not lose information
            if self.transforms_type == 'default_size_224':
                self.base_transforms.resize_size = 224
            if self.transforms_type == 'default_size_236':
                self.base_transforms.crop_size = 236
            
            if interpolation == 'nearest':
                self.base_transforms.interpolation = T.InterpolationMode.NEAREST
            
        elif self.transforms_type == 'normalize':
            if norm_dict is None:
                raw_transforms = weights.transforms()
                self.base_transforms = T.Normalize(mean=raw_transforms.mean, std=raw_transforms.std)
            else:
                self.base_transforms = T.Normalize(**norm_dict)
            
        elif self.transforms_type == 'standardize':
            # Using transforms in pairs allows it to standardize in comparison to ground truth, making it dependent on scale discrepancies
            def base_t(x, y):
                eps = 1e-5
                (m1, s1), (m2, s2) = [(einx.mean('... v c h w -> ... 1 1 1 1', t), einx.std('... v c h w -> ... 1 1 1 1', t)) for t in (x, y)]
                m = (m1 + m2) * 0.5
                s = ((s1 ** 2 + s2 ** 2) * 0.5 + ((m1 - m2) * 0.5) ** 2).sqrt() + eps
                
                return tuple((t - m) / s for t in (x, y))
            
            self.transforms = base_t
            
        else:
            assert False, f'Invalid transforms type "{self.transforms_type}"'
        
        if self.transforms_type != 'standardize':
            self.transforms = lambda x, y: (self.base_transforms(x), self.base_transforms(y))
        
        self.layer_weights = nn.Buffer(layer_weights)
        
        self.eval()
        for param in self.parameters():
            param.requires_grad = False
    
    def distance(self, x1, x2, dist_fn=torch.abs):
        # TODO check which is better l1 or l2 (l1 is more like a mean of the per-pixel errors and l2 is more like a distance in the space of possible images)
        # Using l1 for now so that it is equivalent to a score where 1 is furthest image possible (image of zeros vs image of ones) and 0 is closest possible (exact match)
        # Then that score is used to know how well it is still maintaining information from previous frames in current latent embeds
        # This also sums over batch dim, unlike other models, to allow loss to be proportional to batch size
        # return torch.norm(x1 - x2, p=1, dim=-1).sum() / x1.shape[-1] # norm / C * H * W
        return dist_fn(x1 - x2).mean(dim=-1) # norm / C * H * W
        # return ((x1 - x2) ** 2).mean(dim=-1) # norm / C * H * W
    
    def forward_layer(self, x1, x2, dist_fn=torch.abs):
        x1, x2 = [einx.id('... c h w -> ... (c h w)', k) for k in (x1, x2)]
        
        return self.distance(x1, x2, dist_fn=dist_fn)
    
    def forward(self, input, target, use_raw_distance=True):
        losses = []

        # This is not the best for depth bc it cant propagate to the middle layers
        if self.is_diff:
            input = input - target
            target = torch.zeros_like(input)
        
        input, target = [einx.id('... c h w -> (...) c h w', k) for k in (input, target)]
        
        x1, x2 = input, target
        if use_raw_distance:
            losses.append(self.forward_layer(x1, x2, dist_fn=self.dist_fn_raw))
        
        if 'default' in self.transforms_type:
            # Pads with zeros to make it square
            x1, x2 = [VF.center_crop(t, max(t.shape[-1], t.shape[-2])) for t in (input, target)]
        x1, x2 = self.transforms(x1, x2)
        
        for l in self.layers:
            x1, x2 = l(x1), l(x2)
            losses.append(self.forward_layer(x1, x2, dist_fn=self.dist_fn))
        
        # TODO test with classifier layer
        # losses.append(self.distance(self.classifier_layer(x1), self.classifier_layer(x2), dist_fn=self.dist_fn))
        
        weights = self.layer_weights if use_raw_distance else self.layer_weights[1:]
        weights = weights / weights.sum() # Normalizes weights
        
        losses = torch.stack(losses, dim=-1) * weights
        per_image_losses = einx.sum('... d -> ...', losses)
        weighted_losses = einx.mean('... d -> d', losses)
        loss = weighted_losses.sum()
        
        return loss, weighted_losses, per_image_losses
