import torch
import torch.nn as nn
import einops

from imports.model.Block import Block


class ModAdditionModel(nn.Module):
    def __init__(self, p, d, N_heads, d_head, n):
        """
        Transformer model trained to perform modular addition.
        
        X = [a, b]  ->  c = (a + b) % p
        """

        super().__init__()

        ## Model hyperparameters
        self.d_inout = p                # prime number on the RHS of '%'
        # Residual stream
        self.d_token = d                # embedding dimension of tokens
        self.l_context = 2              # context length
        # Attention heads
        self.N_heads = N_heads          # no. of sa_heads working in parallel
        self.d_head = d_head            # dimension of each sa_head (ie. final dim of K, Q, V arrays)
        # Feed-forward layer (within block)
        self.d_MLP = n                  # hidden dimension of the MLP layer

        ## Layers
        # Embedding matrix          W_E ∈ R^{d x p}
        self.W_E = nn.Embedding(
            num_embeddings=self.d_inout, 
            embedding_dim=self.d_token
        )
        
        self.block = Block(
            self.l_context, self.d_token, 
            self.N_heads, self.d_head, 
            self.d_MLP)

        # Unembedding matrix        W_U ∈ R^{p x d}
        self.W_U = nn.Linear(self.d_token, self.d_inout)

        self.cross_entropy = nn.CrossEntropyLoss()
        
    def forward(self, X, Y_target=None):
        x_token = self.W_E(X)           # no positional encoding needed
        W_out = self.block(x_token)
        logits = self.W_U(W_out)

        # inference
        if Y_target is None:
            loss = None
            return logits, loss
        
        _, C = logits.shape
        assert C == self.d_inout

        loss = self.cross_entropy(logits, Y_target)
        return logits, loss

    def calculate(self, X):
        X = torch.tensor(X)

        if len(X.shape) == 2:
            pass
        elif len(X.shape) == 1:
            # add batch dimension
            X = einops.rearrange(X, "c -> 1 c")
        else:
            raise ValueError(f"X (shape {X.shape}) has the wrong number of dimensions")

        assert X.shape[1] == 2, "X must be of shape (N, 2)"

        logits = self.forward(X)[0]
        Y_pred = torch.argmax(logits, dim=1)
        return Y_pred