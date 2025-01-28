import torch
import torch.nn as nn
from transformers import BartForConditionalGeneration

class BARTAutoencoderLatent(BartForConditionalGeneration):
    def __init__(self, num_encoder_latents=32, num_decoder_latents=32, dim_ae=64, num_layers=3, l2_normalize_latents=False):
        self.num_encoder_latents = num_encoder_latents
        self.num_decoder_latents = num_decoder_latents
        self.dim_ae = dim_ae
        self.l2_normalize_latents = l2_normalize_latents

        self.ae = PerceiverAutoencoder(dim_lm=dim_ae, dim_ae=dim_ae, depth=num_layers, num_encoder_latents=num_encoder_latents, num_decoder_latents=num_decoder_latents, l2_normalize_latents=l2_normalize_latents)

    def get_latents(self, x, attention_mask):
        hidden_state = encoder_outputs[0]
        latent = self.ae.encode(hidden_state, attention_mask)
        return latent

    def get_output(self, latent):
        return self.ae.decode(latent)

    def encoder_output_to_decoder_input(self, encoder_outputs, attention_mask):
        latent = self.get_latents(encoder_outputs, attention_mask)  
        output = self.get_output(latent)
        return output   

class PerceiverAutoencoder(nn.Module):
    def __init__(
        self, 
        *,
        dim_lm,
        dim_ae,
        depth,
        dim_head=64,
        num_encoder_latents=32,
        num_decoder_latents=32,
        max_seq_len=64,
        ff_mult=4,
        encoder_only=False,
        transformer_decoder=False,
        l2_normalize_latents=False,
    ):
        super().__init__()
        self.perceiver_encoder = PerceiverResampler(dim=dim_lm, dim_latent=dim_ae, depth=depth, dim_head=dim_head,
                                                    num_latents=num_encoder_latents, max_seq_len=max_seq_len, ff_mult=ff_mult)
        self.perceiver_decoder = PerceiverResampler(dim=dim_ae, dim_latent=dim_lm, depth=depth, dim_head=dim_head,
                                                        num_latents=num_decoder_latents, max_seq_len=num_encoder_latents, ff_mult=ff_mult)

    def decode(self, latent):
        return self.perceiver_decoder(latent)

    def encode(self, output, attention_mask):
        return self.perceiver_encoder(output, attention_mask)

    def forward(self, output, attention_mask):
        latent = self.perceiver_encoder(output, attention_mask)
        return self.perceiver_decoder(latent)

class PerceiverResampler(nn.Module):
    def __init__(
        self,
        *,
        dim,
        dim_latent,
        depth,
        dim_head=64,
        num_latents=16,
        max_seq_len=64,
        ff_mult=4,
        legacy=False,
    ):
        super().__init__()
        self.pos_emb = AbsolutePositionalEmbedding(dim, max_seq_len)

        if legacy:
            dim_out = dim_latent
            dim_latent = dim

        print(num_latents, dim_latent)

        self.latents = nn.Parameter(torch.randn(num_latents, dim_latent))
        nn.init.normal_(self.latents, std=0.02)

        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                PerceiverAttention(
                    dim=dim, dim_latent=dim_latent, dim_head=dim_head),
                FeedForward(dim=dim_latent, mult=ff_mult)
            ]))

        self.final_norm = nn.LayerNorm(dim_latent)
        self.output_proj = nn.Linear(dim_latent, dim_out) if legacy else nn.Identity()

    def forward(self, x, mask=None):
        pos_emb = self.pos_emb(x)
        x_pos = x + pos_emb

        latents = repeat(self.latents, 'n d -> b n d', b=x.shape[0])

        for attn, ff in self.layers:
            latents = attn(x_pos, latents, mask=mask) + latents
            latents = ff(latents) + latents

        latents = self.output_proj(self.final_norm(latents))
        return latents

class PerceiverAttention(nn.Module):
    def __init__(
        self,
        *,
        dim,
        dim_latent,
        dim_head=64,
        qk_norm=True,
    ):
        super().__init__()
        self.scale = dim_head ** -0.5

        inner_dim = max(dim_latent, dim)
        self.heads = inner_dim // dim_head

        self.norm = nn.LayerNorm(dim)
        self.norm_latents = nn.LayerNorm(dim_latent)

        self.query_norm = RMSNorm(dim_head)
        self.key_norm = RMSNorm(dim_head)

        self.to_q = nn.Linear(dim_latent, inner_dim, bias=False)

        if dim_latent != dim:
            self.latent_to_kv = nn.Linear(dim_latent, inner_dim * 2, bias=False)
        else:
            self.latent_to_kv = None

        self.to_kv = nn.Linear(dim, inner_dim * 2, bias=False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim_latent),
        )

    def forward(self, x, latents, mask=None): 
        x = self.norm(x)
        latents = self.norm_latents(latents)

        b, h = x.shape[0], self.heads

        q = self.to_q(latents)

        if exists(self.latent_to_kv):
            kv_input = torch.cat([self.to_kv(x), self.latent_to_kv(latents)], dim=1)
        else:
            kv_input = torch.cat([self.to_kv(x), self.to_kv(latents)], dim=1)

        k, v = rearrange(kv_input, 'b n (split d) -> split b n d', split=2)

        q, k, v = map(lambda t: rearrange(
            t, 'b n (h d) -> b h n d', h=h), (q, k, v))

        sim = einsum('... i d, ... j d  -> ... i j',
                     self.query_norm(q) * self.scale, self.key_norm(k))

        if exists(mask):
            max_neg_value = -torch.finfo(sim.dtype).max
            mask = F.pad(mask, (0, latents.shape[-2]), value=True)
            mask = rearrange(mask, 'b j -> b 1 1 j')
            sim = sim.masked_fill(~mask, max_neg_value)

        attn = sim.softmax(dim=-1, dtype=torch.float32)
        attn = attn.to(sim.dtype)

        out = einsum('... i j, ... j d -> ... i d', attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)', h=h)
        return self.to_out(out) 

def FeedForward(dim, mult=4, dropout=0.):
    hidden_dim = int(dim * mult)
    return nn.Sequential(
        nn.LayerNorm(dim),
        nn.Linear(dim, hidden_dim),
        nn.GELU(),
        nn.Dropout(dropout),
        nn.Linear(hidden_dim, dim)
    )

class LayerNorm(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.gamma = nn.Parameter(torch.ones(dim))
        self.register_buffer("beta", torch.zeros(dim))

    def forward(self, x):
        return F.layer_norm(x, x.shape[-1:], self.gamma, self.beta)

class RMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-8):
        super().__init__()
        self.scale = dim ** -0.5
        self.eps = eps
        self.gamma = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        norm = torch.norm(x, dim=-1, keepdim=True) * self.scale
        return x / norm.clamp(min=self.eps) * self.gamma

class AbsolutePositionalEmbedding(nn.Module):
    def __init__(self, dim, max_seq_len):
        super().__init__()
        self.scale = dim ** -0.5 
        self.max_seq_len = max_seq_len
        self.embedding = nn.Embedding(max_seq_len, dim)

    def forward(self, x, pos=None):
        seq_len = x.shape[1]

        if not exists(pos):
            pos = torch.arange(seq_len, device=x.device)

        return self.embedding(pos) * self.scale