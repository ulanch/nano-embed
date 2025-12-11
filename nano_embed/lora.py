import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class LoRALayer(nn.Module):
    def __init__(self, in_dim, out_dim, rank, alpha):
        super().__init__()
        self.lora_a = nn.Parameter(torch.zeros(in_dim, rank))
        self.lora_b = nn.Parameter(torch.zeros(rank, out_dim))
        self.alpha = alpha
        nn.init.kaiming_uniform_(self.lora_a, a=math.sqrt(5))

    def forward(self, x):
        return (x @ self.lora_a @ self.lora_b) * self.alpha


class LoRALinear(nn.Module):
    def __init__(self, linear_layer, rank, alpha):
        super().__init__()
        self.linear = linear_layer
        self.lora = LoRALayer(
            in_dim=linear_layer.in_features,
            out_dim=linear_layer.out_features,
            rank=rank,
            alpha=alpha,
        )

    def forward(self, x):
        return self.linear(x) + self.lora(x)

    @property
    def weight(self):
        return self.linear.weight

    def merge_lora_weights(self):
        self.linear.weight.data += self.lora.lora_b @ self.lora.lora_a * self.lora.alpha
        return self.linear


def merge_lora_weights_in_model(model):
    for module in model.modules():
        if isinstance(module, LoRALinear):
            module.merge_lora_weights()
    return model
