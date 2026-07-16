import copy
import torch
import torch.nn as nn
import einx
import torchvision.transforms.v2 as T
import torchvision.transforms.functional as VF
from torchvision.models import convnext_tiny, ConvNeXt_Tiny_Weights, vgg19, VGG19_Weights
from easydict import EasyDict as edict

from utils.data import create_input_normalizer


def normalize_conv2d_layer(layer):
    layer.bias.zero_()
    layer.weight.fill_(1.0 / layer.weight.shape[-3:].numel())


class PerceptualLoss(nn.Module):
    cnn_archs = edict()
    
    def __init__(self, config, dist_fn_raw=torch.square, dist_fn=torch.abs):
        super().__init__()
        
        self.mask_threshold = config.get('mask_threshold', None)
        model_type = config.type
        layer_weights = config.weights
        self.transforms_type = config.transforms
        self.input_type = config.input_type
        self.is_diff = config.is_diff
        interpolation = config.interpolation
        
        self.dist_fn_raw = dist_fn_raw
        self.dist_fn = dist_fn
        
        assert interpolation in ['default', 'nearest'], f'Invalid interpolation "{interpolation}"'
        assert model_type in ['convnext', 'vgg', 'vgg2', 'vgg3'], f'Invalid model type "{model_type}"'
        
        arch = self.cnn_archs.get(model_type, None)
        mask_arch = self.cnn_archs.get(model_type + '_mask', None) if self.input_type != 'image' else None
        if arch is not None:
            weights, self.model, self.layers = arch.weights, arch.model, arch.layers
        if mask_arch is not None:
            self.mask_model, self.mask_layers = arch.model, arch.layers
        
        if model_type == 'convnext':
            if arch is None:
                weights = ConvNeXt_Tiny_Weights.DEFAULT
                self.model = convnext_tiny(weights=weights)
                self.layers = list(self.model.features)
                self.cnn_archs[model_type] = edict(weights=weights, model=self.model, layers=self.layers)
            
            # Creates depth mask network
            if self.input_type != 'image' and mask_arch is None:
                self.mask_model = convnext_tiny()
                self.mask_model.eval()
                for param in self.mask_model.parameters():
                    param.requires_grad = False
                
                self.mask_model.features[0] = self.mask_model.features[0][0]
                for i in range(1, 8, 2):
                    for j, e in enumerate(self.mask_model.features[i]):
                        self.mask_model.features[i][j] = e.block[0]
                        normalize_conv2d_layer(self.mask_model.features[i][j])
                for i in range(2, 8, 2):
                    self.mask_model.features[i] = self.mask_model.features[i][1]
                for i in range(0, 8, 2):
                    normalize_conv2d_layer(self.mask_model.features[i])
                
                self.mask_layers = list(self.mask_model.features)
                self.cnn_archs[model_type + '_mask'] = edict(model=self.mask_model, layers=self.mask_layers)
            
            layer_weights = torch.ones(9, dtype=torch.float32) if layer_weights is None else torch.tensor(layer_weights, dtype=torch.float32)
            
        elif model_type in ['vgg', 'vgg2', 'vgg3']:
            if arch is None:
                weights = VGG19_Weights.DEFAULT
                self.model = vgg19(weights=weights)
            
            # AvgPool allows better differentiability
            features = [nn.AvgPool2d(kernel_size=2, stride=2) if isinstance(i, nn.MaxPool2d) else i for i in self.model.features]
            
            if self.input_type != 'image' and mask_arch is None:
                self.mask_model = vgg19()
                self.mask_model.eval()
                for param in self.mask_model.parameters():
                    param.requires_grad = False
                
                mask_features = [nn.AvgPool2d(kernel_size=2, stride=2) if isinstance(i, nn.MaxPool2d) else i for i in self.mask_model.features]
            
            if model_type == 'vgg3':
                indices = [0] + [i + 1 for i, e in enumerate(features) if isinstance(e, nn.ReLU)]
            elif model_type == 'vgg2':
                indices = [0] + [i for i, e in enumerate(features) if isinstance(e, nn.AvgPool2d)]
            else:
                indices = [0, 4, 9, 14, 23, 32] # layers conv1_2 ... conv5_2 from https://arxiv.org/pdf/1707.09405 p. 5
            
            if arch is None:
                self.layers = [nn.Sequential(*features[s:e]) for s, e in zip(indices[:-1], indices[1:])]
                self.cnn_archs[model_type] = edict(weights=weights, model=self.model, layers=self.layers)
            
            if self.input_type != 'image' and mask_arch is None:
                self.mask_layers = [nn.Sequential(*mask_features[s:e]) for s, e in zip(indices[:-1], indices[1:])]
                for l in self.mask_layers:
                    for i in range(len(l) - 1, -1, -1):
                        if isinstance(l[i], nn.ReLU):
                            l.pop(i)
                        elif isinstance(l[i], nn.Conv2d):
                            normalize_conv2d_layer(l[i])
                self.cnn_archs[model_type + '_mask'] = edict(weights=weights, model=self.mask_model, layers=self.mask_layers)
            
            layer_weights = torch.ones(len(indices), dtype=torch.float32) if layer_weights is None else layer_weights
        
        if self.transforms_type in ['default', 'default_size_224', 'default_size_236']:
            self.transforms = weights.transforms(mean=[0.0] * 3, std=[1.0] * 3)
            
            # resizing and cropping at same size to not lose information
            if self.transforms_type == 'default_size_224':
                self.transforms.resize_size = 224
            if self.transforms_type == 'default_size_236':
                self.transforms.crop_size = 236
            
            if interpolation == 'nearest':
                self.transforms.interpolation = T.InterpolationMode.NEAREST
            
        elif self.transforms_type == 'normalize':
            self.transforms = None
            
        else:
            assert False, f'Invalid transforms type "{self.transforms_type}"'
        
        self.normalize_transform = create_input_normalizer(self.input_type, self.is_diff)
        
        self.layer_weights = nn.Buffer(layer_weights)
        
        self.eval()
        for param in self.parameters():
            param.requires_grad = False
    
    def distance(self, x1, x2, m, dist_fn=torch.abs):
        # TODO check which is better l1 or l2 (l1 is more like a mean of the per-pixel errors and l2 is more like a distance in the space of possible images)
        # Using l1 for now so that it is equivalent to a score where 1 is furthest image possible (image of zeros vs image of ones) and 0 is closest possible (exact match)
        # Then that score is used to know how well it is still maintaining information from previous frames in current latent embeds
        # This also sums over batch dim, unlike other models, to allow loss to be proportional to batch size
        # return torch.norm(x1 - x2, p=1, dim=-1).sum() / x1.shape[-1] # norm / C * H * W
        if m is None or self.mask_threshold is None:
            return dist_fn(x1 - x2).mean()  # norm / C * H * W
        
        eps = 1e-12
        mask_threshold = self.mask_threshold
        if mask_threshold == 'mean_max':
            mask_threshold = 0.4 * m.mean() + 0.6 * m.max()
        elif mask_threshold == 'mean_std':
            mask_threshold = m.mean() + m.std()
        
        # print(0, m.mean(), m.std())
        m = (m > mask_threshold).float()
        # print(1, m.mean(), m.std())
        return (dist_fn(x1 - x2) * m).sum(dim=-1) / (m.sum(dim=-1) + eps)  # norm / C * H * W
        
        # return ((x1 - x2) ** 2).mean(dim=-1)  # norm / C * H * W
    
    def forward_layer(self, x1, x2, m, dist_fn=torch.abs):
        x1, x2, m = [einx.id('... c h w -> ... (c h w)', k) if k is not None else None for k in (x1, x2, m)]
        
        return self.distance(x1, x2, m, dist_fn=dist_fn)
    
    def forward(self, input, target, mask=None, use_raw_distance=True, should_normalize=True):
        losses = []
        
        mask = mask.float() if mask is not None else None
        
        if self.input_type != 'image':
            input, target, mask = [einx.id('... c2 h w -> (... c2) c h w', t, c=3) for t in (input, target, mask)]
        
        # This is not the best for depth bc it cant propagate to the middle layers
        if self.is_diff:
            input = input - target
            target = torch.zeros_like(input)
        
        if self.mask_threshold is None:
            mask = None
        
        input, target, mask = [einx.id('... c h w -> (...) c h w', k) if k is not None else None for k in (input, target, mask)]
        
        x1, x2, m = input, target, mask
        if use_raw_distance:
            losses.append(self.forward_layer(x1, x2, m, dist_fn=self.dist_fn_raw))
        
        if 'default' in self.transforms_type:
            # Pads with zeros to make it square
            x1, x2, m = [VF.center_crop(t, max(t.shape[-1], t.shape[-2])) if t is not None else None for t in (x1, x2, m)]
        
        x1, x2, m = [(self.normalize_transform(t) if should_normalize else t) if t is not None else None for t in (x1, x2, m)]
        x1, x2, m = [(self.transforms(t) if self.transforms is not None else t) if t is not None else None for t in (x1, x2, m)]
        
        for i, l in enumerate(self.layers):
            x1, x2 = l(x1), l(x2)
            m = self.mask_layers[i](m) if mask is not None else None
            losses.append(self.forward_layer(x1, x2, m, dist_fn=self.dist_fn))
        
        weights = self.layer_weights if use_raw_distance else self.layer_weights[1:]
        weights = weights / weights.sum() # Normalizes weights
        
        losses = torch.stack(losses, dim=-1) * weights
        per_image_losses = einx.sum('... d -> ...', losses)
        weighted_losses = einx.mean('... d -> d', losses)
        loss = weighted_losses.sum()
        
        return loss, weighted_losses, per_image_losses
