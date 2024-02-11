import torch
from torch import nn, einsum
import torch.nn.functional as F

from einops import rearrange, repeat
from einops.layers.torch import Rearrange

# helpers

def pair(t):
    return t if isinstance(t, tuple) else (t, t)

# classes

class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn
    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)


class PreNorm_qkv(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm_q = nn.LayerNorm(dim)
        self.norm_k = nn.LayerNorm(dim)
        self.norm_v = nn.LayerNorm(dim)
        self.fn = fn

    def forward(self, q, k, v, **kwargs):
        q = self.norm_q(q)
        k = self.norm_k(k)
        v = self.norm_v(v)

        return self.fn(q, k, v)


# class CrossPreNorm(nn.Module):
#     def __init__(self, dim, fn):
#         super().__init__()
#         self.norm_source = nn.LayerNorm(dim)
#         self.norm_target = nn.LayerNorm(dim)
#         self.fn = fn
#
#     def forward(self, source_x, target_x, **kwargs):
#         source_x = self.norm_source(source_x)
#         target_x = self.norm_target(target_x)
#
#         return self.fn(source_x,target_x)


class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout = 0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )
    def forward(self, x):
        return self.net(x)


class Attention(nn.Module):
    def __init__(self, dim, heads = 8, dim_head = 64, dropout = 0.):
        super().__init__()
        inner_dim = dim_head *  heads
        project_out = not (heads == 1 and dim_head == dim)

        self.heads = heads
        self.scale = dim_head ** -0.5

        self.attend = nn.Softmax(dim = -1)
        self.to_q = nn.Linear(dim, inner_dim, bias=False)
        self.to_k = nn.Linear(dim, inner_dim, bias=False)
        self.to_v = nn.Linear(dim, inner_dim, bias=False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        ) if project_out else nn.Identity()

    def forward(self, q, k, v):
        # print("1", q.shape, k.shape, v.shape)
        b, n, _, h = *q.shape, self.heads
        # qkv = self.to_qkv(x).chunk(3, dim = -1)

        q = self.to_q(q)
        k = self.to_k(k)
        v = self.to_v(v)

        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=h), (q, k, v))
        # print("2", q.shape, k.shape, v.shape)
        dots = einsum('b h i d, b h j d -> b h i j', q, k) * self.scale

        attn = self.attend(dots)

        out = einsum('b h i j, b h j d -> b h i d', attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')

        return self.to_out(out)


class TransformerEncoder(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout = 0.):
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                PreNorm_qkv(dim, Attention(dim, heads = heads, dim_head = dim_head, dropout = dropout)),
                PreNorm(dim, FeedForward(dim, mlp_dim, dropout = dropout))
            ]))

    def forward(self, x):
        for attn, ff in self.layers:
            x = attn(x, x, x) + x
            x = ff(x) + x
        return x


class TransformerDecoder(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout = 0.):
        super().__init__()
        self.layers = nn.ModuleList([])

        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                PreNorm_qkv(dim, Attention(dim, heads = heads, dim_head = dim_head, dropout = dropout)),
                PreNorm_qkv(dim, Attention(dim, heads=heads, dim_head=dim_head, dropout=dropout)),
                PreNorm(dim, FeedForward(dim, mlp_dim, dropout = dropout))
            ]))

    def forward(self, tgt, memory):
        for attn1, attn2, ff in self.layers:
            tgt = attn1(tgt, tgt, tgt) + tgt
            tgt = attn1(tgt, memory, memory) + tgt
            tgt = ff(tgt) + tgt
        return tgt



class CrossTransformerEncoder(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout = 0.):
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                PreNorm_qkv(dim, Attention(dim, heads = heads, dim_head = dim_head, dropout = dropout)),
                PreNorm(dim, FeedForward(dim, mlp_dim, dropout = dropout))
            ]))

    def forward(self, source_x, target_x):
        for attn, ff in self.layers:
            target_x = attn(target_x, source_x, source_x) + target_x
            target_x = ff(target_x) + target_x
        return target_x



class Transformer(nn.Module):
    def __init__(self, *, num_frames, num_classes, dim, depth, heads, mlp_dim, pool = 'cls', channels = 3, dim_head = 64, dropout = 0., emb_dropout = 0.):
        super().__init__()

        self.pos_embedding = nn.Parameter(torch.randn(1, num_frames, dim))
        self.pos_embedding_decoder = nn.Parameter(torch.randn(1, num_frames, dim))

        self.dropout = nn.Dropout(emb_dropout)

        self.encoder = TransformerEncoder(dim, depth, heads, dim_head, mlp_dim, dropout)
        # self.decoder = TransformerDecoder(dim, depth, heads, dim_head, mlp_dim, dropout)

        self.pool = pool
        self.to_latent = nn.Identity()


    def forward(self, input):
        # x = self.to_patch_embedding(img)
        b, n, _ = input.shape

        tgt = input + self.pos_embedding[:, :n]
        x = self.dropout(tgt)
        x = self.encoder(x)
        # x_rec = self.decoder(torch.flip(input + self.pos_embedding_decoder[:, :n], dims=[1]), x)

        return x

class ProjetTransformer(nn.Module):
    def __init__(self, num_frames, dim, depth, heads, mlp_dim, dim_head = 64, len_invariant = 8, len_specific = 8, dropout = 0., emb_dropout = 0.):
        super().__init__()

        self.encoder = TransformerEncoder(dim, depth, heads, dim_head, mlp_dim, dropout)

        self.cls_token = nn.Parameter(torch.randn(1, len_invariant+len_specific, dim))

        self.pos_embedding = nn.Parameter(torch.randn(1, num_frames+len_invariant+len_specific, dim))

        self.dropout = nn.Dropout(emb_dropout)


    def forward(self, input):
        # x = self.to_patch_embedding(img)
        b, n, _ = input.shape
        projet_tokens = repeat(self.cls_token, '1 f d -> b f d', b = b)
        x = torch.cat((projet_tokens, input), dim=1)
        x = x + self.pos_embedding
        x = self.dropout(x)
        x = self.encoder(x)
        return x


class CrossTransformer(nn.Module):
    def __init__(self, *, source_num_frames, tgt_num_frames, dim, depth, heads, mlp_dim, pool = 'cls', dim_head = 64, dropout = 0., emb_dropout = 0.):
        super().__init__()

        self.pos_embedding_s = nn.Parameter(torch.randn(1, source_num_frames, dim))
        self.pos_embedding_t = nn.Parameter(torch.randn(1, tgt_num_frames, dim))
        self.dropout = nn.Dropout(emb_dropout)

        self.CrossTransformerEncoder = CrossTransformerEncoder(dim, depth, heads, dim_head, mlp_dim, dropout)

        self.pool = pool

    def forward(self, source_x, target_x):
        b, n_s, _ = source_x.shape
        b, n_t, _ = target_x.shape

        source_x = source_x + self.pos_embedding_s[:, : n_s]
        target_x = target_x + self.pos_embedding_t[:, : n_t]
        source_x = self.dropout(source_x)
        target_x = self.dropout(target_x)

        x_s2t = self.CrossTransformerEncoder(source_x, target_x)

        return x_s2t


if __name__ == '__main__':
    v = CrossTransformer(
        num_frames = 16,
        dim=512,
        depth=2,
        heads =4,
        mlp_dim=512,
        dropout=0.1,
        emb_dropout=0.1
    ).cuda()

    source = torch.randn(8, 8, 512).cuda()
    target = torch.randn(8, 8, 512).cuda()

    preds = v(source, target)
    # print(preds[0].shape)暂时注释