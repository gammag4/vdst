import torch
import torch.nn as nn
import einx
import torchvision.transforms.v2 as T
from torchvision.models import convnext_tiny, ConvNeXt_Tiny_Weights, vgg19, VGG19_Weights
import torchvision.transforms.functional as VF


class PerceptualLoss(nn.Module):
    # model_type: convnext, vgg, vgg2, vgg3
    # transforms: default, default_no_crop, normalize_only
    def __init__(self, config, dist_fn_raw=torch.square, dist_fn=torch.abs):
        super().__init__()
        model_type = config.type
        layer_weights = config.weights
        transforms = config.transforms
        
        self.dist_fn_raw = dist_fn_raw
        self.dist_fn = dist_fn
        
        if model_type == 'convnext':
            weights = ConvNeXt_Tiny_Weights.DEFAULT
            self.model = convnext_tiny(weights=weights)
            self.layers = list(self.model.features)
            
            layer_weights = torch.ones(9, dtype=torch.float32) if layer_weights is None else layer_weights
        elif model_type in ['vgg', 'vgg2', 'vgg3']:
            weights = VGG19_Weights.DEFAULT
            self.model = vgg19(weights=weights)
            features = [nn.AvgPool2d(kernel_size=2, stride=2) if isinstance(i, nn.MaxPool2d) else i for i in self.model.features]
            if model_type == 'vgg3':
                indices = [0] + [i + 1 for i, e in enumerate(features) if isinstance(e, nn.ReLU)]
            elif model_type == 'vgg2':
                indices = [0] + [i for i, e in enumerate(features) if isinstance(e, nn.AvgPool2d)]
            else:
                indices = [0, 4, 9, 14, 23, 32] # layers conv1_2 ... conv5_2 from https://arxiv.org/pdf/1707.09405 p. 5
            self.layers = [nn.Sequential(features[s:e]) for s, e in zip(indices[:-1], indices[1:])]
            
            layer_weights = torch.ones(len(indices), dtype=torch.float32) if layer_weights is None else layer_weights
        else:
            assert False, f'Invalid model type "{model_type}"'
        
        # TODO test with classifier layer
        # self.classifier_layer = lambda x: self.model.classifier(self.model.avgpool(x))
        
        if transforms in ['default', 'default_no_crop']:
            self.transforms = weights.transforms()
            if transforms == 'default_no_crop':
                self.transforms.resize_size=224 # resizing to same cropping size to not lose information
        elif transforms == 'normalize_only':
            # Following standard settings for ImageNet 1K from:
            # https://docs.pytorch.org/vision/main/models/generated/torchvision.models.convnext_tiny.html
            # https://docs.pytorch.org/vision/main/models/generated/torchvision.models.vgg19.html
            self.transforms = T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        else:
            assert False, f'Invalid transforms type "{transforms}"'
        
        self.layer_weights = nn.Buffer(layer_weights)
    
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
        
        input, target = [einx.id('... c h w -> (...) c h w', k) for k in (input, target)]
        
        x1, x2 = input, target
        if use_raw_distance:
            losses.append(self.forward_layer(x1, x2, dist_fn=self.dist_fn_raw))
        
        x1, x2 = [self.transforms(VF.center_crop(t, max(t.shape[-1], t.shape[-2]))) for t in (input, target)]
        for l in self.layers:
            x1, x2 = l(x1), l(x2)
            losses.append(self.forward_layer(x1, x2, dist_fn=self.dist_fn))
        
        # TODO test with classifier layer
        # losses.append(self.distance(self.classifier_layer(x1), self.classifier_layer(x2), dist_fn=self.dist_fn))
        
        weights = self.layer_weights if use_raw_distance else self.layer_weights[1:]
        weights = weights / weights.sum() # Normalizes weights
        
        losses = torch.stack(losses, dim=-1) * weights
        losses = einx.mean('... d -> d', losses)
        loss = losses.sum()
        
        return loss, losses
