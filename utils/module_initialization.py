import torch
import torch.nn as nn


def _init_weights(module, std=0.02):
    if isinstance(module, nn.Linear):
        nn.init.normal_(module.weight, mean=0, std=std)
        if module.bias is not None:
            nn.init.zeros_(module.bias)


def init_module_weights(module, std=0.02):
    module.apply(lambda m: _init_weights(m, std))


def init_transformer_weights(transformer, std=0.02):
    for i, b in enumerate(transformer.blocks):
        b.apply(lambda m: _init_weights(m, std / (2 * (i + 1)) ** 0.5))
