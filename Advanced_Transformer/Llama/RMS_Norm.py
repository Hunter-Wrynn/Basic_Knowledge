import torch
import torch.nn as nn
import torch.nn.functional as F

class LlamaRMSNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(hidden_size))

    def forward(self, hidden_states):
        input_dtype=hidden_states.dtype
        hidden_states=hidden_states.to(torch.float32)
        var = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(var + self.eps)
        return self.weight * hidden_states.to(input_dtype)
    
