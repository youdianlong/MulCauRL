from transformers.activations import gelu
from functools import partial
import torch.nn as nn
import numpy as np
import torch
import math
import copy
import torch.nn.functional as F
from cra_modules.m_tempclip.transformer import LayerNorm
from cra_modules.m_tempclip.language_model import Bert


class LanModel(nn.Module):

    def __init__(self, tokenizer, lan, word_dim=768, out_dim=512, max_len=128):
        super(LanModel, self).__init__()
        self.bert = Bert(tokenizer, lan)            
        self.linear_text = nn.Linear(word_dim, out_dim) 
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.word_dim = word_dim

        self.word_attn = nn.Sequential(
            nn.Linear(word_dim, 1),
            nn.Softmax(dim=-2)
        )

    def _check_device(self, module, x):
        for name, param in module.named_parameters():
            if param.device != x.device:
                print(f"Parameter {name} is on device {param.device}, expected {x.device}")

    def forward(self, description):

        if isinstance(description, list):
            tokenized = self.tokenizer(
                description,
                padding=True,
                truncation=True,
                max_length=self.max_len,
                return_tensors="pt"
            )
            device = next(self.bert.parameters()).device
            tokenized = {k: v.to(device) for k, v in tokenized.items()}
        else:
            tokenized = description

        self._check_device(self.bert, tokenized['input_ids'])

        # (bs, L, word_dim)
        description_emb = self.bert(tokenized)

        # (bs, L, out_dim)
        description_emb = self.linear_text(description_emb)

        # (bs, L, 1)
        word_attn = self.word_attn(description_emb)

        # (bs, out_dim)
        description_g = (description_emb * word_attn).sum(dim=1)
        
        return description_g

    def get_vocab_matrix(self, detach: bool = True) -> torch.Tensor:

        W = None
        if hasattr(self.bert, "get_input_embeddings"):
            try:
                W = self.bert.get_input_embeddings().weight  # (V, word_dim)
            except Exception:
                W = None
        if W is None and hasattr(self.bert, "embeddings") and hasattr(self.bert.embeddings, "word_embeddings"):
            W = self.bert.embeddings.word_embeddings.weight
        if W is None and hasattr(self.bert, "bert") and hasattr(self.bert.bert, "embeddings"):
            if hasattr(self.bert.bert.embeddings, "word_embeddings"):
                W = self.bert.bert.embeddings.word_embeddings.weight

        if W is None:
            raise RuntimeError(
            )
        return W.detach() if detach else W


def clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])


class PositionwiseFeedForward(nn.Module):
    def __init__(self, d_model, d_ff, dropout=0.1):
        super(PositionwiseFeedForward, self).__init__()
        self.w_1 = nn.Linear(d_model, d_ff)
        self.w_2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, time_param=None):
        return self.w_2(self.dropout(F.relu(self.w_1(x))))


class SublayerConnection(nn.Module):
    def __init__(self, embed_dim, dropout):
        super(SublayerConnection, self).__init__()
        self.norm = LayerNorm(embed_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, sublayer):
        return x + self.dropout(sublayer(self.norm(x)))


class MultiHeadedAttention(nn.Module):
    def __init__(self, h, d_model, dropout=0.1):
        super(MultiHeadedAttention, self).__init__()
        assert d_model % h == 0
        self.d_k = d_model // h
        self.h = h
        self.linears = clones(nn.Linear(d_model, d_model), 4)
        self.attn = None
        self.dropout = nn.Dropout(p=dropout)
        self.attention = self.ori_attention
        self.selector = nn.Linear(self.d_k, 2)
        self.time_reparam = nn.Linear(2, self.d_k)

    def ori_attention(self, query, key, value, mask=None, dropout=None):
        d_k = query.size(-1)
        scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        p_attn = F.softmax(scores, dim=-1)
        if dropout is not None:
            p_attn = dropout(p_attn)
        return torch.matmul(p_attn, value), p_attn

    def forward(self, query, key, value, mask=None):
        nbatches = query.size(0)
        query, key, value = \
            [l(x).view(nbatches, -1, self.h, self.d_k).transpose(1, 2)
             for l, x in zip(self.linears, (query, key, value))]
        x, self.attn = self.attention(query, key, value, mask=mask, dropout=self.dropout)
        x = x.transpose(1, 2).contiguous().view(nbatches, -1, self.h * self.d_k)
        return self.linears[-1](x)


class DecoderLayer(nn.Module):
    def __init__(self, embed_dim, num_heads, ff_dim, dropout):
        super(DecoderLayer, self).__init__()
        self.self_attn = MultiHeadedAttention(num_heads, embed_dim, dropout)
        self.feed_forward = PositionwiseFeedForward(embed_dim, ff_dim, dropout)
        self.sublayer_self = SublayerConnection(embed_dim, dropout)
        self.sublayer_ff = SublayerConnection(embed_dim, dropout)

    def forward(self, x):
        x = self.sublayer_self(x, lambda x: self.self_attn(x, x, x))
        x = self.sublayer_ff(x, lambda x: self.feed_forward(x))
        return x


class Decoder(nn.Module):
    def __init__(self, embed_dim=768, num_layer=2, num_heads=8, ff_dim=768, dropout=0.1):
        super(Decoder, self).__init__()
        self.layers = clones(DecoderLayer(embed_dim, num_heads, ff_dim, dropout), num_layer)
        self.norm = LayerNorm(embed_dim)

        self.attention_video_feature = nn.Sequential(
            nn.Linear(embed_dim, embed_dim // 2),
            nn.Tanh(),
            nn.Linear(embed_dim // 2, 1),
            nn.Softmax(dim=-2)
        )

        self.video_proj = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(embed_dim, embed_dim)
        )

    def forward(self, x, qa, self_mask=None):
        for i in range(len(self.layers)):
            x = self.layers[i](x)
        x = self.norm(x)
        video_attn = self.attention_video_feature(x)
        x = torch.sum(x * video_attn, dim=1)
        x = self.video_proj(x)
        return x, video_attn
