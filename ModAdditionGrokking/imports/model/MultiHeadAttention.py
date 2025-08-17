import torch
import torch.nn as nn
import einops


class MultiHeadAttention(nn.Module):
    """
    'Multi-head self-attention' block, however it attention is not required and positional encoding is not used, 
    since order doesn't matter.
    """

    def __init__(self, l_context, d_token, N_heads, d_head):
        super().__init__()
        self.sa_heads = [nn.Linear(d_token, d_head, bias=False) for _ in range(N_heads)]

        # projects from concatenated sa_head outputs to something to add to the residual stream
        self.proj = nn.Linear(N_heads*d_head, d_token) 

    def forward(self, x):
        # might still need to add in positional encoding? (doubt it though)
        x_concat = einops.reduce(
            torch.concat([sa(x) for sa in self.sa_heads], dim=2),
            "b t c -> b c", "sum"
        )
        x_att = self.proj(x_concat)
        
        return x_att