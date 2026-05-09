import torch
import torch.amp as amp


# Manages gradient scaling and skipping, doing gradient skipping even with GradScaler disabled
class GradScaler(amp.GradScaler):
    def step(self, optimizer, *args, **kwargs):
        if self._enabled:
            return super().step(optimizer, *args, **kwargs)
        
        params = [p for pg in optimizer.param_groups for p in pg['params']]
        should_step = True
        for param in params:
            if param.grad is not None:
                if torch.isnan(param.grad).any() or torch.isinf(param.grad).any():
                    should_step = False
                    break
        
        if should_step:
            return optimizer.step()
