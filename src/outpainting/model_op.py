import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from util.model_util import VisionRotaryEmbeddingFast, get_2d_sincos_pos_embed, RMSNorm
torch.set_float32_matmul_precision('high')

def zero_module(module):
    """Zero out parameters of a module to ensure it acts as identity residuals."""
    if module is None: return None
    for p in module.parameters():
        p.detach().zero_()
    return module

def modulate(x, shift, scale):
    return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)

# --------------------------------------------------------
# Standard Blocks 
# --------------------------------------------------------

class BottleneckPatchEmbed(nn.Module):
    """ Image to Patch Embedding """
    def __init__(self, img_size=224, patch_size=16, in_chans=3, pca_dim=768, embed_dim=768, bias=True):
        super().__init__()
        img_size = (img_size, img_size)
        patch_size = (patch_size, patch_size)
        num_patches = (img_size[1] // patch_size[1]) * (img_size[0] // patch_size[0])
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = num_patches

        self.proj1 = nn.Conv2d(in_chans, pca_dim, kernel_size=patch_size, stride=patch_size, bias=False)
        self.proj2 = nn.Conv2d(pca_dim, embed_dim, kernel_size=1, stride=1, bias=bias)

    def forward(self, x):
        x = self.proj2(self.proj1(x)).flatten(2).transpose(1, 2)
        return x

class TimestepEmbedder(nn.Module):
    def __init__(self, hidden_size, frequency_embedding_size=256):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(frequency_embedding_size, hidden_size, bias=True),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size, bias=True),
        )
        self.frequency_embedding_size = frequency_embedding_size

    @staticmethod
    def timestep_embedding(t, dim, max_period=10000):
        half = dim // 2
        freqs = torch.exp(
            -math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32) / half
        ).to(device=t.device)
        args = t[:, None].float() * freqs[None]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if dim % 2:
            embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
        return embedding

    def forward(self, t):
        t_freq = self.timestep_embedding(t, self.frequency_embedding_size)
        t_emb = self.mlp(t_freq)
        return t_emb

class Attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=True, qk_norm=True, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads

        self.q_norm = RMSNorm(head_dim) if qk_norm else nn.Identity()
        self.k_norm = RMSNorm(head_dim) if qk_norm else nn.Identity()

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x, rope):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2] 

        q = self.q_norm(q)
        k = self.k_norm(k)

        q = rope(q)
        k = rope(k)

        x = F.scaled_dot_product_attention(
            q, k, v,
            attn_mask=None,
            dropout_p=self.attn_drop.p if self.training else 0.,
            is_causal=False
            )

        x = x.transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

class CrossAttention(nn.Module):
    def __init__(self, dim, num_heads, attn_drop=0., proj_drop=0.):
        super().__init__()
        assert dim % num_heads == 0
        self.num_heads = num_heads
        self.head_dim = dim // num_heads

        self.q_proj = nn.Linear(dim, dim)
        self.k_proj = nn.Linear(dim, dim)
        self.v_proj = nn.Linear(dim, dim)

        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = zero_module(nn.Linear(dim, dim))
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x, context, mask=None):
        B, N, D = x.shape
        _, M, _ = context.shape

        q = self.q_proj(x).reshape(B, N, self.num_heads, self.head_dim)
        k = self.k_proj(context).reshape(B, M, self.num_heads, self.head_dim)
        v = self.v_proj(context).reshape(B, M, self.num_heads, self.head_dim)

        q = q.permute(0, 2, 1, 3) 
        k = k.permute(0, 2, 1, 3)
        v = v.permute(0, 2, 1, 3)

        attn = (q @ k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        if mask is not None:
            mask = mask.to(attn.device)
            attn = attn.masked_fill(mask == 0, float("-inf"))

        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = attn @ v
        x = x.permute(0, 2, 1, 3).reshape(B, N, D)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

class SwiGLUFFN(nn.Module):
    def __init__(self, dim, hidden_dim, drop=0.0, bias=True):
        super().__init__()
        hidden_dim = int(hidden_dim * 2 / 3)
        self.w12 = nn.Linear(dim, 2 * hidden_dim, bias=bias)
        self.w3 = nn.Linear(hidden_dim, dim, bias=bias)
        self.ffn_dropout = nn.Dropout(drop)

    def forward(self, x):
        x12 = self.w12(x)
        x1, x2 = x12.chunk(2, dim=-1)
        hidden = F.silu(x1) * x2
        return self.w3(self.ffn_dropout(hidden))

class FinalLayer(nn.Module):
    def __init__(self, hidden_size, patch_size, out_channels):
        super().__init__()
        self.norm_final = RMSNorm(hidden_size)
        self.linear = nn.Linear(hidden_size, patch_size * patch_size * out_channels, bias=True)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 2 * hidden_size, bias=True)
        )

    def forward(self, x, c):
        shift, scale = self.adaLN_modulation(c).chunk(2, dim=1)
        x = modulate(self.norm_final(x), shift, scale)
        x = self.linear(x)
        return x

class JiTBlock(nn.Module):
    def __init__(self, hidden_size, num_heads, mlp_ratio=4.0, attn_drop=0.0, proj_drop=0.0, **_):
        super().__init__()
        self.norm1 = RMSNorm(hidden_size, eps=1e-6)
        self.attn = Attention(hidden_size, num_heads=num_heads, qkv_bias=True, qk_norm=True,
                              attn_drop=attn_drop, proj_drop=proj_drop)
        self.norm2 = RMSNorm(hidden_size, eps=1e-6)
        mlp_hidden_dim = int(hidden_size * mlp_ratio)
        self.mlp = SwiGLUFFN(hidden_size, mlp_hidden_dim, drop=proj_drop)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 6 * hidden_size, bias=True)
        )
        # Zero-initialized cross-attention for outpainting context (known region)
        self.op_cross_attn = CrossAttention(hidden_size, num_heads)
        # Optional semantic cross-attention (DINO tokens)
        self.parent_cross_attn = CrossAttention(hidden_size, num_heads)

    def forward(self, x, c, feat_rope=None,
                op_context=None, op_context_mask=None,
                parent_tokens=None, parent_mask=None):
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = \
            self.adaLN_modulation(c).chunk(6, dim=-1)
        h = modulate(self.norm1(x), shift_msa, scale_msa)

        x = x + gate_msa.unsqueeze(1) * self.attn(h, rope=feat_rope)
        if op_context is not None:
            x = x + gate_msa.unsqueeze(1) * self.op_cross_attn(h, op_context, mask=op_context_mask)
        if parent_tokens is not None:
            x = x + gate_msa.unsqueeze(1) * self.parent_cross_attn(h, parent_tokens, mask=parent_mask)
        x = x + gate_mlp.unsqueeze(1) * self.mlp(modulate(self.norm2(x), shift_mlp, scale_mlp))
        return x

class JiT_Outpainting(nn.Module):
    """
    Outpainting Just Image Transformer.

    Conditioning signals:
      inpaint_cond  : 7-channel input (zt ‖ mask ‖ masked_image) via context_embedder
      DINO tokens   : frozen ViT-L features from the known-region crop (optional)
      zoom_level    : target tile zoom level scalar (always active); 0 = unknown/GSD
                      unavailable. Embedded via the same sinusoidal MLP as the diffusion
                      timestep and added to the AdaLN conditioning vector c.
    """
    def __init__(
        self,
        input_size=256,
        patch_size=16,
        in_channels=3,
        hidden_size=1024,
        depth=24,
        num_heads=16,
        mlp_ratio=4.0,
        attn_drop=0.0,
        proj_drop=0.0,
        bottleneck_dim=128,
        cond_drop_prob=0.1,
        **_,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = in_channels
        self.patch_size = patch_size
        self.num_heads = num_heads
        self.hidden_size = hidden_size
        self.input_size = input_size
        self.cond_drop_prob = cond_drop_prob

        # time embed
        self.t_embedder = TimestepEmbedder(hidden_size)
        # Target-tile zoom-level embedder: same sinusoidal+MLP architecture as t_embedder.
        # zoom=0 acts as the "unknown GSD" sentinel — learned during cond_drop_prob training.
        self.zoom_embedder = TimestepEmbedder(hidden_size)

        # linear embed
        self.x_embedder = BottleneckPatchEmbed(input_size, patch_size, in_channels, bottleneck_dim, hidden_size, bias=True)

        # pos embed
        num_patches = self.x_embedder.num_patches
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches, hidden_size), requires_grad=False)
        
        # rope
        half_head_dim = hidden_size // num_heads // 2
        hw_seq_len = input_size // patch_size
        self.feat_rope = VisionRotaryEmbeddingFast(
            dim=half_head_dim,
            pt_seq_len=hw_seq_len,
            num_cls_token=0
        )

        # transformer blocks
        self.blocks = nn.ModuleList([
            JiTBlock(hidden_size, num_heads, mlp_ratio=mlp_ratio,
                     attn_drop=attn_drop if (depth // 4 * 3 > i >= depth // 4) else 0.0,
                     proj_drop=proj_drop if (depth // 4 * 3 > i >= depth // 4) else 0.0)
            for i in range(depth)
        ])

        # linear predict
        self.final_layer = FinalLayer(hidden_size, patch_size, self.out_channels)
        
        # ----------------------------------------------------------------
        # OP ADDITIONS
        # ----------------------------------------------------------------
        # 7-channel Context Embedder (x_t, mask, masked image) like SD inpaint
        # Zero-module so it starts as identity
        self.context_embedder = zero_module(
            nn.Conv2d(7, hidden_size, kernel_size=patch_size, stride=patch_size)
        )

        # Optional semantic tokens (e.g., DINO) projected to hidden size
        self.parent_proj = zero_module(nn.Linear(1024, hidden_size))

        self.initialize_weights()

    def initialize_weights(self):
        # Initialize transformer layers:
        def _basic_init(module):
            if isinstance(module, nn.Linear):
                torch.nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
        self.apply(_basic_init)

        # Initialize (and freeze) pos_embed by sin-cos embedding:
        pos_embed = get_2d_sincos_pos_embed(self.pos_embed.shape[-1], int(self.x_embedder.num_patches ** 0.5))
        self.pos_embed.data.copy_(torch.from_numpy(pos_embed).float().unsqueeze(0))

        # Initialize patch_embed
        w1 = self.x_embedder.proj1.weight.data
        nn.init.xavier_uniform_(w1.view([w1.shape[0], -1]))
        w2 = self.x_embedder.proj2.weight.data
        nn.init.xavier_uniform_(w2.view([w2.shape[0], -1]))
        nn.init.constant_(self.x_embedder.proj2.bias, 0)
        

        nn.init.normal_(self.t_embedder.mlp[0].weight, std=0.02)
        nn.init.normal_(self.t_embedder.mlp[2].weight, std=0.02)

        # Zero-out adaLN modulation layers:
        for block in self.blocks:
            nn.init.constant_(block.adaLN_modulation[-1].weight, 0)
            nn.init.constant_(block.adaLN_modulation[-1].bias, 0)

        # Zero-out output layers:
        nn.init.constant_(self.final_layer.adaLN_modulation[-1].weight, 0)
        nn.init.constant_(self.final_layer.adaLN_modulation[-1].bias, 0)

        nn.init.constant_(self.final_layer.linear.weight, 0)
        nn.init.constant_(self.final_layer.linear.bias, 0)


    def unpatchify(self, x, p):
        c = self.out_channels
        h = w = int(x.shape[1] ** 0.5)
        assert h * w == x.shape[1]
        x = x.reshape(shape=(x.shape[0], h, w, p, p, c))
        x = torch.einsum('nhwpqc->nchpwq', x)
        imgs = x.reshape(shape=(x.shape[0], c, h * p, h * p))
        return imgs

    def forward(self, x, t, conditions=None):
        """
        x          : (B, 3, H, W)    noisy image
        t          : (B,)            diffusion timestep ∈ [0, 1]
        conditions : dict with:
                       inpaint_cond  (B, 7, H, W)   — 7-ch inpainting context (required)
                       mask          (B, 1, H, W)   — 1=hole, 0=known
                       dino_feats    (B, 256, 1024)  — optional DINO semantic tokens
                       zoom_level    (B,) float      — tile zoom level; 0 = unknown/GSD N/A
        """
        B = x.shape[0]

        t_emb = self.t_embedder(t)
        c = t_emb

        op_context = None
        op_context_mask = None
        parent_tokens = None
        parent_mask = None

        if conditions is not None:
            drop_all_conds = (
                self.training
                and self.cond_drop_prob > 0
                and torch.rand((), device=x.device) < self.cond_drop_prob
            )

            if (not drop_all_conds) and conditions.get("inpaint_cond") is not None \
                    and conditions.get("mask") is not None:
                ctx_input = conditions["inpaint_cond"]
                op_context = self.context_embedder(ctx_input).flatten(2).transpose(1, 2)
                patch_mask = F.avg_pool2d(conditions["mask"], kernel_size=self.patch_size, stride=self.patch_size)
                patch_mask = patch_mask.view(B, -1)
                valid = (patch_mask < 0.5).float()
                valid_sum = valid.sum(dim=1, keepdim=True)
                valid = torch.where(valid_sum == 0, torch.ones_like(valid), valid)
                op_context_mask = valid.unsqueeze(1).unsqueeze(2)

            if (not drop_all_conds) and conditions.get("dino_feats") is not None:
                parent_tokens = self.parent_proj(conditions["dino_feats"])
                parent_mask = None

            # Target-tile zoom-level conditioning — adds to c via sinusoidal embedding
            if not drop_all_conds and conditions.get("zoom_level") is not None:
                zl = conditions["zoom_level"].float().to(x.device)  # (B,)
                c = c + self.zoom_embedder(zl)

        x = self.x_embedder(x)
        x += self.pos_embed

        for block in self.blocks:
            x = block(
                x, c,
                feat_rope=self.feat_rope,
                op_context=op_context,
                op_context_mask=op_context_mask,
                parent_tokens=parent_tokens,
                parent_mask=parent_mask,
            )

        x = self.final_layer(x, c)
        return self.unpatchify(x, self.patch_size)

def JiT_B_16(**kwargs):
    return JiT_Outpainting(depth=12, hidden_size=768, num_heads=12,
               bottleneck_dim=128, patch_size=16, **kwargs)

def JiT_B_32(**kwargs):
    return JiT_Outpainting(depth=12, hidden_size=768, num_heads=12,
               bottleneck_dim=128, patch_size=32, **kwargs)

def JiT_L_16(**kwargs):
    return JiT_Outpainting(depth=24, hidden_size=1024, num_heads=16,
               bottleneck_dim=128, patch_size=16, **kwargs)

def JiT_L_32(**kwargs):
    return JiT_Outpainting(depth=24, hidden_size=1024, num_heads=16,
               bottleneck_dim=128, patch_size=32, **kwargs)

def JiT_H_16(**kwargs):
    return JiT_Outpainting(depth=32, hidden_size=1280, num_heads=16,
               bottleneck_dim=256, patch_size=16, **kwargs)

def JiT_H_32(**kwargs):
    return JiT_Outpainting(depth=32, hidden_size=1280, num_heads=16,
               bottleneck_dim=256, patch_size=32, **kwargs)

JiT_models = {
    'JiT-B/16': JiT_B_16,
    'JiT-B/32': JiT_B_32,
    'JiT-L/16': JiT_L_16,
    'JiT-L/32': JiT_L_32,
    'JiT-H/16': JiT_H_16,
    'JiT-H/32': JiT_H_32,
}

if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    B, H, W = 2, 256, 256
    x    = torch.randn(B, 3, H, W, device=device)
    t    = torch.rand(B, device=device)
    mask = torch.randint(0, 2, (B, 1, H, W), dtype=torch.float32, device=device)
    masked_image = x * (1 - mask)
    cond = {
        "inpaint_cond": torch.cat([x, mask, masked_image], dim=1),
        "mask":         mask,
        "dino_feats":   torch.randn(B, 256, 1024, device=device),
    }
    for name, fn in [("B16", JiT_B_16), ("H16", JiT_H_16)]:
        m = fn(input_size=H).to(device)
        out = m(x, t, conditions=cond)
        assert out.shape == (B, 3, H, W)
        print(f"[{name}] {x.shape} → {out.shape} | {sum(p.numel() for p in m.parameters())/1e6:.1f}M params")
    print("✓ passed")
