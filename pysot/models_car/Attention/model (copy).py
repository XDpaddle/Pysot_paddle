import torch
import  torch.nn as nn
from .self_attention import SelfAttention
from .mlp import Mlp
import torch.nn.functional as F

class SelfAttentionBlock(nn.Module):
    def __init__(self, dim, num_heads, act_layer=nn.GELU, dropout=0.1):
        super(SelfAttentionBlock, self).__init__()
        self.embedding_dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads

        assert (self.head_dim * num_heads == dim), "Embedding dimension needs to be divisible by the number of heads"

        self.wq = nn.Linear(dim, dim)
        self.wk = nn.Linear(dim, dim)
        self.wv = nn.Linear(dim, dim)

        self.fc = nn.Linear(dim, dim)

        self.dropout = nn.Dropout(p=dropout)
        mlp_hidden_dim = int(dim * 4)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=0)

    def forward(self, v, k, q, mask):
        '''

        Args:
            v:
            k:
            q:
            mask:

        Returns:

        '''
        batch_size = q.size(0)

        q = self.wq(q)
        k = self.wk(k)
        v = self.wv(v)

        q = self.split_heads(q, batch_size)
        k = self.split_heads(k, batch_size)
        v = self.split_heads(v, batch_size)

        scaled_attention, attention_weights = self.scaled_dot_product_attention(q, k, v, mask)

        scaled_attention = scaled_attention.transpose(1, 2).contiguous().view(batch_size, -1, self.embedding_dim)

        output = v + self.fc(scaled_attention)
        output = output + nn.Identity(self.mlp())


        return output

    def split_heads(self, x, batch_size):
        x = x.view(batch_size, -1, self.num_heads, self.head_dim)
        return x.transpose(1, 2)

    def scaled_dot_product_attention(self, q, k, v, mask):
        matmul_qk = torch.matmul(q, k.transpose(-2, -1))
        dk = q.size(-1)
        scaled_attention_logits = matmul_qk / torch.sqrt(torch.tensor(dk).float())

        if mask is not None:
            scaled_attention_logits += mask * -1e9

        attention_weights = F.softmax(scaled_attention_logits, dim=-1)

        attention = self.dropout(attention_weights)
        output = torch.matmul(attention, v)

        return output, attention_weights