import torch
import torch.nn as nn
from torch import Tensor
import torch.nn.functional as F
from module import ws

class Attention(nn.Module):
    def __init__(
        self,
        embed_dim,
        num_heads,
        dropout_p=0.1,
        weight_standardization=False,
    ):
        super(Attention, self).__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.dropout_p = dropout_p

        if weight_standardization:
            self.q_proj = ws.Linear(embed_dim, embed_dim)
            self.k_proj = ws.Linear(embed_dim, embed_dim)
            self.v_proj = ws.Linear(embed_dim, embed_dim)
            self.out_proj = ws.Linear(embed_dim, embed_dim)
        else:
            self.q_proj = nn.Linear(embed_dim, embed_dim)
            self.k_proj = nn.Linear(embed_dim, embed_dim)
            self.v_proj = nn.Linear(embed_dim, embed_dim)
            self.out_proj = nn.Linear(embed_dim, embed_dim)
    
    def _separate_heads(self, x: Tensor, num_heads: int) -> Tensor:
        '''
        args:
            x: torch.Tensor, [batch_size, n_tokens, embed_dim]
            num_heads: int
        return:
            torch.Tensor, [batch_size, num_heads, n_tokens, embed_dim // num_heads]
        '''
        batch_size, n_tokens, embed_dim = x.shape
        x = x.reshape(batch_size, n_tokens, num_heads, embed_dim // num_heads)
        return x.transpose(1, 2) # [batch_size, num_heads, n_tokens, embed_dim // num_heads]
    
    def _recombine_heads(self, x: Tensor) -> Tensor:
        '''
        args:
            x: torch.Tensor, [batch_size, num_heads, n_tokens, embed_dim // num_heads]
        return:
            torch.Tensor, [batch_size, n_tokens, embed_dim]
        '''
        batch_size, num_heads, n_tokens, embed_dim_per_head = x.shape
        x = x.transpose(1, 2)
        return x.reshape(batch_size, n_tokens, embed_dim_per_head * num_heads)
    
    def forward(self, q: Tensor, k: Tensor, v: Tensor) -> Tensor:
        '''
        args:
            q: torch.Tensor, [batch_size, n_tokens, embed_dim]
            k: torch.Tensor, [batch_size, n_tokens, embed_dim]
            v: torch.Tensor, [batch_size, n_tokens, embed_dim]
        return:
            torch.Tensor, [batch_size, n_tokens, embed_dim]
        '''
        q = self.q_proj(q) # [batch_size, n_tokens, embed_dim]
        k = self.k_proj(k) # [batch_size, n_tokens, embed_dim]
        v = self.v_proj(v) # [batch_size, n_tokens, embed_dim]

        q = self._separate_heads(q, self.num_heads)
        k = self._separate_heads(k, self.num_heads)
        v = self._separate_heads(v, self.num_heads)

        out = F.scaled_dot_product_attention(q, k, v, dropout_p=(self.dropout_p if self.training else 0.0))
        out = self._recombine_heads(out)
        out = self.out_proj(out)
        return out