import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from util.model_util import VisionRotaryEmbeddingFast, get_2d_sincos_pos_embed, RMSNorm
torch.set_float32_matmul_precision('high')


# --------------------------------------------------------
# Singleton DINOv3
# --------------------------------------------------------

_DINO_MODEL = None

def get_dino_embeddings(images, weights_path=None):
    global _DINO_MODEL
    if _DINO_MODEL is None:
        if not weights_path:
            raise ValueError(
                "weights_path must be provided to get_dino_embeddings; "
                "no default DINO checkpoint path is configured."
            )
        ckpt_path = weights_path
        if not os.path.exists(ckpt_path):
            raise FileNotFoundError(f"Missing DINOv3 weights at {ckpt_path}")
        print(f"Loading DINOv3 from {ckpt_path}...")
        model = torch.hub.load('facebookresearch/dinov3', 'dinov3_vitl16', weights=ckpt_path)
        model.eval()
        for param in model.parameters(): param.requires_grad = False
        _DINO_MODEL = model.to(images.device)
    if next(_DINO_MODEL.parameters()).device != images.device:
        _DINO_MODEL = _DINO_MODEL.to(images.device)
    with torch.no_grad():
        return _DINO_MODEL.forward_features(images.float())['x_norm_patchtokens']


def zero_module(module):
    if module is None: return None
    for p in module.parameters():
        p.detach().zero_()
    return module

def modulate(x, shift, scale):
    return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)


# --------------------------------------------------------
# Shared building blocks (ViT body — unchanged from pretrain)
# --------------------------------------------------------

class BottleneckPatchEmbed(nn.Module):
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
        return self.mlp(t_freq)


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
        x = F.scaled_dot_product_attention(q, k, v, attn_mask=None,
                                            dropout_p=self.attn_drop.p if self.training else 0.,
                                            is_causal=False)
        x = x.transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        return self.proj_drop(x)


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
        q = self.q_proj(x).reshape(B, N, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        k = self.k_proj(context).reshape(B, M, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        v = self.v_proj(context).reshape(B, M, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        attn = (q @ k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        if mask is not None:
            mask = mask.to(attn.device)
            attn = attn.masked_fill(mask == 0, float("-inf"))
        attn = self.attn_drop(attn.softmax(dim=-1))
        x = (attn @ v).permute(0, 2, 1, 3).reshape(B, N, D)
        return self.proj_drop(self.proj(x))


class SwiGLUFFN(nn.Module):
    def __init__(self, dim, hidden_dim, drop=0.0, bias=True):
        super().__init__()
        hidden_dim = int(hidden_dim * 2 / 3)
        self.w12 = nn.Linear(dim, 2 * hidden_dim, bias=bias)
        self.w3 = nn.Linear(hidden_dim, dim, bias=bias)
        self.ffn_dropout = nn.Dropout(drop)

    def forward(self, x):
        x1, x2 = self.w12(x).chunk(2, dim=-1)
        return self.w3(self.ffn_dropout(F.silu(x1) * x2))


class FinalLayer(nn.Module):
    """Projects each token to patch_size² × out_channels values for unpatchify."""
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
        return self.linear(x)


def unpatchify(x, p, out_channels=3):
    """
    x: (B, N, p*p*C)  — N must be a perfect square (h*w tokens)
    Returns (B, C, h*p, w*p)
    """
    B, N, _ = x.shape
    h = w = int(N ** 0.5)
    assert h * w == N
    x = x.reshape(B, h, w, p, p, out_channels)
    x = torch.einsum('nhwpqc->nchpwq', x)
    return x.reshape(B, out_channels, h * p, w * p)


class JiTBlock(nn.Module):
    def __init__(self, hidden_size, num_heads, mlp_ratio=4.0, attn_drop=0.0, proj_drop=0.0, **_):
        super().__init__()
        self.norm1 = RMSNorm(hidden_size, eps=1e-6)
        self.attn = Attention(hidden_size, num_heads=num_heads, qkv_bias=True, qk_norm=True,
                              attn_drop=attn_drop, proj_drop=proj_drop)
        self.norm2 = RMSNorm(hidden_size, eps=1e-6)
        self.mlp = SwiGLUFFN(hidden_size, int(hidden_size * mlp_ratio), drop=proj_drop)
        self.adaLN_modulation = nn.Sequential(nn.SiLU(), nn.Linear(hidden_size, 6 * hidden_size, bias=True))
        self.sr_cross_attn = CrossAttention(hidden_size, num_heads)

    def forward(self, x, c, feat_rope=None, parent_tokens=None, parent_mask=None):
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = \
            self.adaLN_modulation(c).chunk(6, dim=-1)
        h = modulate(self.norm1(x), shift_msa, scale_msa)
        x = x + gate_msa.unsqueeze(1) * self.attn(h, rope=feat_rope)
        if parent_tokens is not None:
            x = x + self.sr_cross_attn(h, parent_tokens, mask=parent_mask)
        x = x + gate_mlp.unsqueeze(1) * self.mlp(modulate(self.norm2(x), shift_mlp, scale_mlp))
        return x


# --------------------------------------------------------

# --------------------------------------------------------
# SR_JiT  (full-tile 4x super-resolution: 256×256 → 512×512)
# --------------------------------------------------------

class SR_JiT(nn.Module):
    """
    Full-tile Super-Resolution JiT.

    Given a 256×256 parent tile at ZL, generates the stitched 2×2 child mosaic
    at ZL+1 as a 512×512 image. Only zoom_diff=1 pairs are used.

    ViT body:  identical to the pretrained unconditional JiT — fully reusable.
    x_embedder: BottleneckPatchEmbed(512, patch_size=32) → 16×16 = 256 tokens
                (patch_size doubles vs pretrain; weights are re-initialized)

    Decoder: FinalLayer(p=32) + unpatchify — each of the 256 ViT tokens predicts
             its own 32×32 output patch via a single linear projection. ~0 extra
             parameters; t-conditioning is already provided by AdaLN in the backbone.

    Parent conditioning:
      Pixel embedder is ALWAYS active:
        parent_context_embedder(parent_img) → (B, 256, D)
        patch_size=16 on 256×256 parent → 16×16 = 256 spatially-aligned tokens.

      DINO is an OPTIONAL additional signal (use_dino=True):
        DINOv3(parent_img) → (B, 256, 1024) → parent_proj → (B, 256, D)
        Concatenated with pixel tokens → (B, 512, D) context for sr_cross_attn.

      When use_dino=False: context = pixel tokens only (B, 256, D).
      When use_dino=True:  context = cat([pixel_tokens, dino_tokens], dim=1) (B, 512, D).

    Zoom-level conditioning (always active):
      The **target** tile zoom level (= parent ZL + 1, integer typically 15–19) is embedded
      with a sinusoidal MLP (same architecture as the diffusion timestep embedder) and
      added to the AdaLN conditioning vector c = t_emb + zoom_emb. This lets the model
      learn resolution-specific hallucination patterns (e.g. urban detail at ZL 18 vs
      coarse geography at ZL 14). zoom=0 is the "unknown" sentinel — passed through the
      embedder as-is; the model learns to treat it as a null signal via cond_drop_prob.
    """
    def __init__(
        self,
        output_size=512,
        in_channels=3,
        hidden_size=1024,
        depth=24,
        num_heads=16,
        mlp_ratio=4.0,
        attn_drop=0.0,
        proj_drop=0.0,
        bottleneck_dim=128,
        cond_drop_prob=0.1,
        use_dino=True,  # whether to use DINO features on top of pixel conditioning
    ):
        super().__init__()

        self.in_channels = in_channels
        self.out_channels = in_channels
        self.num_heads = num_heads
        self.hidden_size = hidden_size
        self.output_size = output_size      # child/output image size  (512)
        self.parent_size = output_size // 2 # parent tile size          (256)
        self.cond_drop_prob = cond_drop_prob
        self.use_dino = use_dino

        # Child patch size: ensures 16×16 = 256 tokens regardless of output_size
        child_patch_size = output_size // 16
        self.child_patch_size = child_patch_size

        # ---- Standard JiT components (identical structure to pretrain) ----

        self.t_embedder = TimestepEmbedder(hidden_size)
        # Target-tile zoom-level embedder: same sinusoidal+MLP architecture as t_embedder.
        # zoom=0 acts as the "unknown GSD" sentinel — learned during cond_drop_prob training.
        self.zoom_embedder = TimestepEmbedder(hidden_size)

        # Noisy child at output_size × output_size → 256 tokens (16×16 grid)
        self.x_embedder = BottleneckPatchEmbed(
            output_size, child_patch_size, in_channels, bottleneck_dim, hidden_size, bias=True
        )

        num_patches = self.x_embedder.num_patches  # always 256
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches, hidden_size), requires_grad=False)

        # RoPE: hw_seq_len = output_size // child_patch_size = 16 (identical to pretrain at 256/16)
        half_head_dim = hidden_size // num_heads // 2
        self.feat_rope = VisionRotaryEmbeddingFast(
            dim=half_head_dim,
            pt_seq_len=output_size // child_patch_size,  # = 16
            num_cls_token=0,
        )

        self.blocks = nn.ModuleList([
            JiTBlock(
                hidden_size, num_heads, mlp_ratio=mlp_ratio,
                attn_drop=attn_drop if (depth // 4 * 3 > i >= depth // 4) else 0.0,
                proj_drop=proj_drop if (depth // 4 * 3 > i >= depth // 4) else 0.0,
            )
            for i in range(depth)
        ])

        # ---- Decoder: FinalLayer (direct) ----
        # Each of the 256 ViT tokens predicts its own 32×32 patch via a linear projection.
        self.decoder = FinalLayer(hidden_size, child_patch_size, self.out_channels)

        # Parent conditioning:
        #   Pixel embedder always present — patch_size=16 on 256×256 parent → 256 tokens
        self.parent_context_embedder = zero_module(
            nn.Conv2d(in_channels, hidden_size, kernel_size=16, stride=16)
        )
        # DINO projection — only when use_dino=True
        if use_dino:
            self.parent_proj = zero_module(nn.Linear(1024, hidden_size))
        else:
            self.parent_proj = None

        self.initialize_weights()

    def initialize_weights(self):
        def _basic_init(module):
            if isinstance(module, nn.Linear):
                torch.nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
        self.apply(_basic_init)

        # Sincos pos embed for 16×16 token grid (same grid size as pretrain)
        pos_embed = get_2d_sincos_pos_embed(self.pos_embed.shape[-1], int(self.x_embedder.num_patches ** 0.5))
        self.pos_embed.data.copy_(torch.from_numpy(pos_embed).float().unsqueeze(0))

        # x_embedder (patch_size=32, newly initialized — pretrain weights are incompatible)
        w1 = self.x_embedder.proj1.weight.data
        nn.init.xavier_uniform_(w1.view([w1.shape[0], -1]))
        w2 = self.x_embedder.proj2.weight.data
        nn.init.xavier_uniform_(w2.view([w2.shape[0], -1]))
        nn.init.constant_(self.x_embedder.proj2.bias, 0)

        nn.init.normal_(self.t_embedder.mlp[0].weight, std=0.02)
        nn.init.normal_(self.t_embedder.mlp[2].weight, std=0.02)

        # Zero-out adaLN modulations in all ViT blocks
        for block in self.blocks:
            nn.init.constant_(block.adaLN_modulation[-1].weight, 0)
            nn.init.constant_(block.adaLN_modulation[-1].bias, 0)

        # Decoder init: zero-out FinalLayer outputs for stable early training
        nn.init.constant_(self.decoder.adaLN_modulation[-1].weight, 0)
        nn.init.constant_(self.decoder.adaLN_modulation[-1].bias, 0)
        nn.init.constant_(self.decoder.linear.weight, 0)
        nn.init.constant_(self.decoder.linear.bias, 0)

    def forward(self, x, t, conditions=None):
        """
        x          : (B, 3, output_size, output_size)  noisy child mosaic
        t          : (B,)                               diffusion timestep ∈ [0, 1]
        conditions : dict with:
                       parent_img  (B, 3, 256, 256)  — always required
                       dino_feats  (B, 256, 1024)    — optional, only when use_dino=True
                       zoom_level  (B,) float        — **target** tile zoom level (= parent ZL+1);
                                                       0 = unknown
        Parent context = pixel_tokens always; + dino_tokens appended when use_dino=True.
        Zoom level is embedded via sinusoidal MLP and added to the AdaLN vector c.
        """
        B = x.shape[0]

        t_emb = self.t_embedder(t)
        c = t_emb

        parent_tokens = None

        if conditions is not None:
            drop_all_conds = (
                self.training
                and self.cond_drop_prob > 0
                and torch.rand((), device=x.device) < self.cond_drop_prob
            )
            if not drop_all_conds:
                # Pixel tokens — always computed from parent_img
                if conditions.get("parent_img") is not None:
                    pixel_tokens = self.parent_context_embedder(conditions["parent_img"]) \
                                       .flatten(2).transpose(1, 2)   # (B, 256, D)
                    parent_tokens = pixel_tokens

                # DINO tokens — appended when use_dino=True
                if self.use_dino and self.parent_proj is not None \
                        and conditions.get("dino_feats") is not None:
                    dino_tokens = self.parent_proj(conditions["dino_feats"])   # (B, 256, D)
                    if parent_tokens is not None:
                        parent_tokens = torch.cat([parent_tokens, dino_tokens], dim=1)  # (B, 512, D)
                    else:
                        parent_tokens = dino_tokens

                # Target-tile zoom-level conditioning — adds to c via sinusoidal embedding
                if conditions.get("zoom_level") is not None:
                    zl = conditions["zoom_level"].float().to(x.device)  # (B,)
                    c = c + self.zoom_embedder(zl)

        # ViT forward — parent_mask=None: all 256 parent DINO tokens attend equally (no ROI)
        x = self.x_embedder(x)
        x += self.pos_embed

        for block in self.blocks:
            x = block(
                x, c,
                feat_rope=self.feat_rope,
                parent_tokens=parent_tokens,
                parent_mask=None,
            )

        # Decoder: (B, 256, D) → (B, 3, output_size, output_size)
        x = self.decoder(x, c)
        return unpatchify(x, self.child_patch_size, self.out_channels)


# --------------------------------------------------------
# Model factory functions
# --------------------------------------------------------

def JiT_B_16(**kwargs):
    return SR_JiT(depth=12, hidden_size=768,  num_heads=12, bottleneck_dim=128, **kwargs)

def JiT_B_32(**kwargs):
    return SR_JiT(depth=12, hidden_size=768,  num_heads=12, bottleneck_dim=128, **kwargs)

def JiT_L_16(**kwargs):
    return SR_JiT(depth=24, hidden_size=1024, num_heads=16, bottleneck_dim=128, **kwargs)

def JiT_L_32(**kwargs):
    return SR_JiT(depth=24, hidden_size=1024, num_heads=16, bottleneck_dim=128, **kwargs)

def JiT_H_16(**kwargs):
    return SR_JiT(depth=32, hidden_size=1280, num_heads=16, bottleneck_dim=256, **kwargs)

def JiT_H_32(**kwargs):
    return SR_JiT(depth=32, hidden_size=1280, num_heads=16, bottleneck_dim=256, **kwargs)


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
    B = 2
    t = torch.rand(B, device=device)
    x = torch.randn(B, 3, 512, 512, device=device)
    parent = torch.randn(B, 3, 256, 256, device=device)
    for use_dino, label in ((False, "pixel only"), (True, "pixel + dino")):
        cond = {"parent_img": parent}
        if use_dino:
            cond["dino_feats"] = torch.randn(B, 256, 1024, device=device)
        m = JiT_B_16(output_size=512, use_dino=use_dino).to(device)
        out = m(x, t, conditions=cond)
        assert out.shape == (B, 3, 512, 512)
        print(f"[{label:15s}] {x.shape} → {out.shape} | "
              f"{sum(p.numel() for p in m.parameters())/1e6:.1f}M params")
    print("✓ passed")
