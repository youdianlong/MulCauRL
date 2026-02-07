import torch
import torch.nn as nn
import torch.nn.functional as F
import math

from cra_modules.m_cra.causaltransformer import EncoderLayer, SublayerConnection, \
    MultiHeadedAttention, LayerNorm, PositionwiseFeedForward, clones
class FDI(nn.Module):
    """
    Front-Door Intervention (FDI):
      ctx = LWA(mediator_tokens, feature_tokens, feature_tokens)
      out = LWA(feature_tokens, mediator_tokens, ctx)
      return out
    """
    def __init__(self, embed_dim: int):
        super().__init__()
        self.d = embed_dim

        # Only used when proj=True (to match your original AF behavior)
        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.out_proj = nn.Linear(embed_dim, embed_dim)

    def LWA(self, query, key, value, use_proj: bool = False, use_scale: bool = True):
        """
        Lightweight (Learnable) Weighted Attention (LWA)
        Args:
            query: (B, Q, D)
            key:   (B, K, D)
            value: (B, K, D)
            use_proj: whether to apply q_proj + out_proj (proj branch)
            use_scale: whether to scale dot-product by sqrt(D) (only meaningful when use_proj=True)
        Returns:
            ctx: (B, Q, D)
        """
        if use_proj:
            q = self.q_proj(query)  # (B,Q,D)
            logits = torch.matmul(q, key.transpose(-1, -2))  # (B,Q,K)
            if use_scale:
                logits = logits / math.sqrt(self.d)
            attn = logits.softmax(dim=-1)                    # (B,Q,K)
            ctx = torch.matmul(attn, value)                  # (B,Q,D)
            ctx = self.out_proj(ctx)                         # (B,Q,D)
            return ctx
        else:
            logits = torch.matmul(query, key.transpose(-1, -2))  # (B,Q,K)
            attn = logits.softmax(dim=-1)                        # (B,Q,K)
            ctx = torch.matmul(attn, value)                      # (B,Q,D)
            return ctx

    def forward(self, feat_tokens, med_tokens, use_proj: bool = False):
        """
        Args:
            feat_tokens: (B,D) or (B,N,D)
            med_tokens:  (B,D) or (B,K,D)
        Returns:
            (B,D)
        """
        x = feat_tokens
        m = med_tokens

        if x.dim() == 2:
            x = x.unsqueeze(1)  # (B,1,D)
        if m.dim() == 2:
            m = m.unsqueeze(1)  # (B,1,D)

        # Stage 1: mediator queries features -> context tokens
        ctx = self.LWA(query=m, key=x, value=x, use_proj=use_proj)     # (B,K,D)

        # Stage 2: features query mediator -> refined feature
        out = self.LWA(query=x, key=m, value=ctx, use_proj=use_proj)   # (B,1,D)

        return out.squeeze(1)  # (B,D)


class _Mamba1DEncoder(nn.Module):
    def __init__(self, d_model, d_state=16, expand=2):
        super().__init__()
        self.use_mamba = False
        try:
            from mamba_ssm.torch import Mamba
            self.block = Mamba(d_model=d_model, d_state=d_state, d_conv=4, expand=expand)
            self.use_mamba = True
        except Exception:
            self.block = nn.Sequential(
                nn.Conv1d(d_model, d_model, kernel_size=5, padding=2, groups=1),
                nn.GELU(),
                nn.Conv1d(d_model, d_model, kernel_size=3, padding=1, groups=1),
            )

    def forward(self, x):  # x: (B,L,D)
        if self.use_mamba:
            return self.block(x)  # (B,L,D)
        else:
            y = self.block(x.transpose(1, 2)).transpose(1, 2)  # (B,L,D)
            return y


class GlobalSampleMamba(nn.Module):

    def __init__(self, embed_dim, bins=49):
        super().__init__()
        self.encoder = _Mamba1DEncoder(embed_dim)
        self.bins = bins
    def forward(self, x):
        B, L, D = x.size()
        y = self.encoder(x)                     # (B,L,D)
        if L >= self.bins:
            seg = torch.linspace(0, L, steps=self.bins+1, device=x.device, dtype=torch.long)
            pools = []
            for i in range(self.bins):
                s, e = seg[i].item(), seg[i+1].item()
                pools.append(y[:, s:e, :].mean(1, keepdim=True))  # (B,1,D)
            fg = torch.cat(pools, dim=1)         # (B,bins,D)
        else:
            fg = F.adaptive_avg_pool1d(y.transpose(1, 2), self.bins).transpose(1, 2)
        return fg


class DualStreamResidual(nn.Module):
    """
    Residual wrapper for a sublayer that returns two streams:
      (main_out, comp_out)
    """
    def __init__(self, embed_dim: int, dropout: float):
        super().__init__()
        self.norm = LayerNorm(embed_dim)
        self.drop = nn.Dropout(dropout)

    def forward(self, tokens, sublayer_fn):
        main_out, comp_out = sublayer_fn(self.norm(tokens))
        return tokens + self.drop(main_out), self.drop(comp_out)


class TopKHeadSelector(nn.Module):
    """
    Compose attention matrices across layers, then select Top-K indices per head
    from the CLS-to-patch attention vector.
    """
    def forward(self, attn_stack, top_k: int = 6):
        attn_prod = attn_stack[0]
        for i in range(1, len(attn_stack)):
            attn_prod = torch.matmul(attn_stack[i], attn_prod)

        # take CLS(0) attending to others (exclude CLS itself)
        cls_to_patches = attn_prod[:, :, 0, 1:]  # (B, H, L-1)

        top_scores, top_indices = cls_to_patches.sort(dim=2, descending=True)  # (B,H,L-1)
        top_indices = top_indices[:, :, :top_k].reshape(cls_to_patches.size(0), -1)  # (B, H*top_k)
        top_scores  = top_scores[:, :, :top_k].reshape(cls_to_patches.size(0), -1)   # (B, H*top_k)
        return top_scores, top_indices


class BidirectionalAttn(nn.Module):
    """
    Two complementary attention branches:
      attn_neg = softmax(-logits)
      attn_pos = softmax(+logits)
    Returns (feat_neg, feat_pos)
    """
    def __init__(self, num_heads: int, model_dim: int, dropout: float = 0.1):
        super().__init__()
        assert model_dim % num_heads == 0
        self.num_heads = num_heads
        self.head_dim = model_dim // num_heads
        self.proj = clones(nn.Linear(model_dim, model_dim), 4)
        self.drop = nn.Dropout(p=dropout)

    def forward(self, tokens):
        q = k = v = tokens
        batch_size = q.size(0)
        q, k, v = [
            layer(inp).view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
            for layer, inp in zip(self.proj, (q, k, v))
        ]

        logits = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)  

        attn_neg = F.softmax(-logits, dim=-1)
        attn_pos = F.softmax(logits, dim=-1)
        attn_neg, attn_pos = self.drop(attn_neg), self.drop(attn_pos)

        feat_neg = torch.matmul(attn_neg, v) 
        feat_pos = torch.matmul(attn_pos, v)

        feat_neg = feat_neg.transpose(1, 2).contiguous().view(batch_size, -1, self.num_heads * self.head_dim)
        feat_pos = feat_pos.transpose(1, 2).contiguous().view(batch_size, -1, self.num_heads * self.head_dim)

        feat_neg = self.proj[-1](feat_neg)
        feat_pos = self.proj[-1](feat_pos)
        return feat_neg, feat_pos


class AggregationBlock(nn.Module):
    """
    One block: bidirectional causal attention + FFN,
    and merge complementary stream into main stream.
    """
    def __init__(self, embed_dim: int, num_heads: int, ff_dim: int, dropout: float):
        super().__init__()
        self.causal_attn = BidirectionalAttn(num_heads, embed_dim, dropout)
        self.ffn = PositionwiseFeedForward(embed_dim, ff_dim, dropout)

        self.res_attn = DualStreamResidual(embed_dim, dropout)
        self.res_ffn = SublayerConnection(embed_dim, dropout)

    def forward(self, tokens, mask=None):
        tokens, comp_tokens = self.res_attn(tokens, lambda t: self.causal_attn(t))
        tokens = self.res_ffn(tokens, self.ffn)
        tokens = tokens + self.res_ffn(comp_tokens, self.ffn)
        return tokens


class TopKSampler(nn.Module):
    """
    Select Top-K patch tokens guided by stacked attention, then refine with causal aggregation block.
    """
    def __init__(self, embed_dim: int, num_heads: int, ff_dim: int, dropout: float = 0.0):
        super().__init__()
        self.selector = TopKHeadSelector()
        self.causal_refiner = AggregationBlock(embed_dim, num_heads, ff_dim, dropout)

    def forward(self, tokens, attn_stack, top_k: int):
        _, top_indices = self.selector(attn_stack, top_k)
        top_indices = top_indices + 1 

        picked_tokens = []
        B, _ = top_indices.shape
        for b in range(B):
            picked_tokens.append(tokens[b, top_indices[b, :]])  
        local_tokens = torch.stack(picked_tokens, dim=0)  

        local_tokens = torch.cat([tokens[:, :1, :], local_tokens], dim=1)
        local_tokens = self.causal_refiner(local_tokens)[:, 1:, :]
        return local_tokens

class LGCA(nn.Module):
    def __init__(self, embed_dim, noise_std=0.1):
        super().__init__()
        self.noise_std = noise_std
        self.norm_l = LayerNorm(embed_dim)
        self.norm_g = LayerNorm(embed_dim)
        self.llf = MultiHeadedAttention(8, embed_dim)
        self.lgf = MultiHeadedAttention(8, embed_dim)
        self.proj = nn.Linear(embed_dim * 2, embed_dim)
        self.norm = LayerNorm(embed_dim)
        self.ff = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 4),
            nn.GELU(),
            nn.Linear(embed_dim * 4, embed_dim),
        )

    def forward(self, fl, fg):
        fl = self.norm_l(fl)
        fg = self.norm_g(fg)

        fll = fl + self.llf(fl, fl, fl)
        flg = fl + self.lgf(fl, fg, fg)

        fused = torch.cat([fll, flg], dim=-1)
        if self.training and self.noise_std > 0:
            fused = fused + torch.randn_like(fused) * self.noise_std

        out = self.proj(fused)
        out = self.norm(out)
        out = out + self.ff(out)
        return out


class VCP(nn.Module):

    def __init__(self, embed_dim, num_heads=8, ff_dim=None, dropout=.1, bins=49, topk=6):
        super(VCP, self).__init__()
        self.local = TopKSampler(embed_dim, num_heads, 4*embed_dim if ff_dim is None else ff_dim, dropout)
        self.global_mamba = GlobalSampleMamba(embed_dim, bins=bins)
        self.fuse = LGCA(embed_dim)
        self.out_pool = nn.AdaptiveAvgPool1d(1)
        self.topk = topk

    def forward(self, x_all, attn_list):
        fl = self.local(x_all, attn_list, self.topk)    
        fg = self.global_mamba(x_all[:, 1:, :])          
        mt = self.fuse(fl, fg)                         
        Mv = self.out_pool(mt.transpose(1, 2)).transpose(1, 2).squeeze(1) 
        return Mv, fl


class CrossLayer(EncoderLayer):
    """cross-attn + FFN"""
    def __init__(self, embed_dim, num_heads, ff_dim, dropout):
        super(CrossLayer, self).__init__(embed_dim, num_heads, ff_dim, dropout)
    def forward(self, x, y):
        x = self.sublayer_attn(x, lambda x: self.attn(x, y, y, None))
        x = self.sublayer_ff(x, self.feed_forward)
        return x


class GCP(nn.Module):

    def __init__(self, embed_dim_img=512, embed_dim_gene=128):
        super(GCP, self).__init__()
        self.embed_dim_img = embed_dim_img
        self.embed_dim_gene = embed_dim_gene

        self.map_gene = nn.Linear(embed_dim_gene, embed_dim_img)

        self.cross_v = CrossLayer(embed_dim_img, 8, 4 * embed_dim_img, 0.1)
        self.cross_l = CrossLayer(embed_dim_img, 8, 4 * embed_dim_img, 0.1)

        self.norm = LayerNorm(embed_dim_img)
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.ffn = nn.Sequential(
            nn.Linear(embed_dim_img, embed_dim_img * 4),
            nn.GELU(),
            nn.Linear(embed_dim_img * 4, embed_dim_img),
        )

    def forward(self, h_vl, Ml, g_vec):
        g_proj = self.map_gene(g_vec)
        g_proj = g_proj.unsqueeze(1)  
        gv = self.cross_v(g_proj, h_vl) + g_proj 
        if Ml.dim() == 2:
            Ml = Ml.unsqueeze(1)  
        gl = self.cross_l(gv, Ml) + gv            
        out = self.ffn(gl)
        out = self.norm(out + gl)
        Mg = self.pool(out.transpose(1, 2)).transpose(1, 2).squeeze(1)
        return Mg
