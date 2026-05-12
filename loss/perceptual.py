import torch
import torch.nn as nn
import einx
from torchvision.models import convnext_tiny, ConvNeXt_Tiny_Weights
import torchvision.transforms.functional as VF


class PerceptualLoss(nn.Module):
    def __init__(self, layer_weights=torch.ones(9, dtype=torch.float32)):
        super().__init__()
        
        weights = ConvNeXt_Tiny_Weights.DEFAULT
        self.model = convnext_tiny(weights=weights)  # TODO also test with vgg
        
        self.transforms = weights.transforms()
        self.transforms.resize_size=224
        self.layers = list(self.model.features)
        # TODO test without classifier layer
        # self.classifier_layer = lambda x: self.model.classifier(self.model.avgpool(x)) # TODO
        
        self.layer_weights = nn.Buffer(layer_weights)
    
    def distance(self, x1, x2):
        # TODO check which is better l1 or l2 (l1 is more like a mean of the per-pixel errors and l2 is more like a distance in the space of possible images)
        # Using l1 for now so that it is equivalent to a score where 1 is furthest image possible (image of zeros vs image of ones) and 0 is closest possible (exact match)
        # Then that score is used to know how well it is still maintaining information from previous frames in current latent embeds
        # This also sums over batch dim, unlike other models, to allow loss to be proportional to batch size
        # return torch.norm(x1 - x2, p=1, dim=-1).sum() / x1.shape[-1] # norm / C * H * W
        return (x1 - x2).abs().mean(dim=-1).sum() # norm / C * H * W
        # return ((x1 - x2) ** 2).mean(dim=-1).sum() # norm / C * H * W
    
    def forward_layer(self, x1, x2):
        x1, x2 = [einx.rearrange('... c h w -> ... (c h w)', k) for k in (x1, x2)]
        
        return self.distance(x1, x2)
    
    def forward(self, input, target, use_raw_distance=True):
        losses = []
        
        input, target = [einx.rearrange('... c h w -> (...) c h w', k) for k in (input, target)]

        x1, x2 = input, target
        if use_raw_distance:
            losses.append(self.forward_layer(x1, x2)) # TODO check with image distance
        
        x1, x2 = [self.transforms(VF.center_crop(t, max(t.shape[-1], t.shape[-2]))) for t in (input, target)]
        for l in self.layers:
            x1, x2 = l(x1), l(x2)
            losses.append(self.forward_layer(x1, x2))
        
        # losses.append(self.distance(self.classifier_layer(x1), self.classifier_layer(x2))) # TODO check w classifier layer

        weights = self.layer_weights if use_raw_distance else self.layer_weights[1:]
        weights = weights / weights.sum() # Normalizes weights
        
        losses = torch.stack(losses) * weights
        loss = losses.sum()
        
        return loss, losses
