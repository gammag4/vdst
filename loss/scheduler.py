from abc import ABC, abstractmethod
from typing import Callable
import torch
import math

from .loss import PerceptualLoss


class LossScheduler(ABC):
    def __init__(self, loss, n_iter: int):
        self.loss = loss
        self.n_iter = n_iter
        self.iter = 0
        self._update_loss()
    
    @abstractmethod
    def _update_loss(self):
        pass
    
    def step(self):
        if self.iter >= self.n_iter:
            return
        
        self.iter += 1
        self._update_loss()
    
    def state_dict(self):
        return {
            'n_iter': self.n_iter,
            'iter': self.iter
        }
    
    def load_state_dict(self, state_dict):
        self.n_iter = state_dict['n_iter']
        self.iter = state_dict['iter']


class LambdaLossScheduler(LossScheduler):
    def __init__(self, loss, n_iter: int, loss_lambda: Callable[[any, int, int], None]):
        super().__init__(loss, n_iter)
        self.loss_lambda = loss_lambda
        self.loss_lambda(self.loss, self.n_iter, self.iter)
    
    def _update_loss(self):
        self.loss_lambda(self.loss, self.n_iter, self.iter)


# beta tells which minimum value a weight can have (the maximum is 1), its range is [0, 1]
# n_iter is the number of iterations
# i is the current iteration, its range is [0, n_iter - 1]
# perc_idx are the indices of each layer from the cnn where the output is extracted from (the original image also counts as the output of the layer 0)
def h(n_iter, i, perc_idx: torch.Tensor):
    return (i / (n_iter - 1) - 0.5) * torch.cos((torch.pi * perc_idx) / (perc_idx.shape[0] - 1)) + 0.5


def r1(beta, n_iter, i, perc_idx):
    return (1 - beta) * h(n_iter, i, perc_idx) + beta


def r2(beta, n_iter, i, perc_idx):
    return torch.ones(perc_idx.shape, device=perc_idx.device)


def r3(beta, n_iter, i, perc_idx):
    return (beta - 1) * h(n_iter, i, perc_idx) + 1


# TODO
class PerceptualLossScheduler2(LossScheduler):
    # Regime can be either:
    #   'constant': Weights are always constant
    #   'deep_to_shallow': Starts giving more weights to deeper layers and gradually goes to givin more weights to shallower ones in the end
    #   'shallow_to_deep': Starts giving more weights to shallower layers and gradually goes to givin more weights to deeper ones in the end
    def __init__(self, loss: PerceptualLoss, n_iter: int, beta: float=0.5, regime: str='constant'):
        self.original_weights = loss.layer_weights.data.clone()
        self.beta = beta

        if regime == 'deep_to_shallow':
            r = r1
        elif regime == 'constant':
            r = r2
        elif regime == 'shallow_to_deep':
            r = r3
        else:
            raise Exception(f'Regime "{regime}" not supported for perceptual loss scheduler')
        
        self.r = r
        super().__init__(loss, n_iter)
    
    def _update_loss(self):
        if self.original_weights.device != self.loss.layer_weights.device:
            self.original_weights = self.original_weights.to(self.loss.layer_weights.device)
        
        perc_idx = torch.arange(self.loss.layer_weights.shape[0], device=self.loss.layer_weights.device)
        self.loss.layer_weights.copy_(self.original_weights * self.r(self.beta, self.n_iter, self.iter, perc_idx))


class PerceptualLossScheduler(LossScheduler):
    def __init__(self, loss, n_iter, regime='falling_perceptual'):
        self.regime = regime
        self.original_weights = loss.weights.data.clone()
        
        assert self.regime in ['constant', 'falling_perceptual'], f'Invalid loss scheduler regime: "{self.regime}"'
        
        super().__init__(loss, n_iter)
    
    def _update_loss(self):
        if self.regime == 'constant':
            return
        
        if self.original_weights.device != self.loss.weights.device:
            self.original_weights = self.original_weights.to(self.loss.weights.device)
        
        perc = min(1.0, 1.3 * self.iter / self.n_iter) # 1.3 is to make it fall faster than the cosine lr
        c = math.cos((perc - 1) * math.pi) / 2 + 0.5
        r = 0.1
        fall1 = (r - 1) * c + 1  # goes from 1 to r
        fall2 = 1 - c  # goes from 1 to 0
        fall3 = 1 - c  # goes from 1 to 0
        weights = torch.tensor([1.0, fall1, 1.0, fall2, fall3], device=self.loss.weights.device)
        weights = self.original_weights * weights
        self.loss.weights.copy_(weights)
