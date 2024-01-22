import paddle.nn as nn
import paddle

class SelfAttention(nn.Layer):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.,
                 attn_pos_encoding_only=False):
        super().__init__()
        assert dim % num_heads == 0, f"dim {dim} should be divided by num_heads {num_heads}."

        self.dim = dim
        self.num_heads = num_heads
        head_dim = dim // num_heads         # 256//8
        self.scale = qk_scale or head_dim ** -0.5

        if attn_pos_encoding_only:
            self.qkv = nn.Linear(dim, 3 * dim, bias_attr=qkv_bias)
        else:
            self.q = nn.Linear(dim, dim, bias_attr=qkv_bias)
            self.k = nn.Linear(dim, dim, bias_attr=qkv_bias)
            self.v = nn.Linear(dim, dim, bias_attr=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        self.attn_pos_encoding_only = attn_pos_encoding_only

    def forward(self, x, attn_pos):
        '''

        Args:
            x (torch.Tensor): (B, L, C)
            q_ape (torch.Tensor | None): (1 or B, L, C), absolute positional encoding for q
            k_ape (torch.Tensor | None): (1 or B, L, C), absolute positional encoding for k
            attn_pos (torch.Tensor | None):  (1 or B, num_heads, L, L), united positional encoding

        Returns:
            torch.Tensor: (B, L, C)
        '''
        B, N, C = x.shape
        q_ape = None
        k_ape = None
        attn_pos1 = paddle.concat([attn_pos,attn_pos,attn_pos], axis=1)

        q = x + attn_pos1
        q = self.q(q).reshape([B, N, self.num_heads, C // self.num_heads]).transpose([0, 2, 1, 3])

        kv = x 
        k = kv + attn_pos1
        k = self.k(k).reshape([B, -1, self.num_heads, C // self.num_heads]).transpose([0, 2, 1, 3])
        v = self.v(kv).reshape([B, -1, self.num_heads, C // self.num_heads]).transpose([0, 2, 1, 3])

        # attn = q @ k.transpose(-2, -1)
        attn = q @ k.transpose([0,1,3,2])
        attn = attn * self.scale
#        if attn_pos is not None:
#            attn = attn + attn_pos
        attn = attn.softmax(axis=-1)
        attn = self.attn_drop(attn)

        x = attn @ v
        # x = x.transpose(1, 2).reshape([B, N, C])
        x = x.transpose([0,2,1,3]).reshape([B, N, C])
        x = self.proj(x)
        x = self.proj_drop(x)

        return x