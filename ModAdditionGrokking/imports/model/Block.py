import torch.nn as nn

from imports.model.MultiHeadAttention import MultiHeadAttention


class Block(nn.Module):
    """
    Transformer block containing a multi-head self-attention layer followed by MLP.
    NOTE: no layer norm
    """
    def __init__(self, l_context, d_token, 
                 N_heads, d_head, 
                 d_MLP):
        super().__init__()
        
        self.sa_heads = MultiHeadAttention(l_context, d_token, N_heads, d_head)

        self.MLP = nn.Sequential(
            nn.Linear(d_token, d_MLP),
            nn.ReLU(),
            nn.Linear(d_MLP, d_token)
        )



    def forward(self, x):
        x_sa = self.sa_heads(x)
        W_out = x_sa + self.MLP(x_sa)

        return W_out