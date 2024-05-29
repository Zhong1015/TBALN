# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
"""
DETR Transformer class.

Copy-paste from torch.nn.Transformer with modifications:
    * positional encodings are passed in MHattention
    * extra LN at the end of encoder is removed
    * decoder returns a stack of activations from all decoding layers
"""
import copy
from tkinter.tix import Tree
from typing import Optional, List

import torch
import torch.nn.functional as F
from torch import nn, Tensor
from torch.nn import MultiheadAttention

try:
    from .transformer_encoder import TransformerEncoder, TransformerEncoderLayer
    from .transformer_decoder import TransformerDecoder, TransformerDecoderLayer
    from .misc import _get_clones, _get_activation_fn
except ImportError:
    from .transformer_encoder import TransformerEncoder, TransformerEncoderLayer
    from .transformer_decoder import TransformerDecoder, TransformerDecoderLayer
    from .misc import _get_clones, _get_activation_fn

class Transformer(nn.Module):

    def __init__(self, d_model=512, nhead=8, num_encoder_layers=6,
                 num_decoder_layers=6, dim_feedforward=2048, dropout=0.1,
                 activation="relu", normalize_before=False,
                 return_intermediate_dec=False):
        super(Transformer, self).__init__()
        self.num_encoder_layers = num_encoder_layers
        self.num_decoder_layers = num_decoder_layers
        self.d_model = d_model
        self.nhead = nhead
        self.rm_self_attn_dec = True
        self.rm_first_self_attn = True
        
        encoder_layer = TransformerEncoderLayer(d_model, nhead, dim_feedforward,
                                                dropout, activation, normalize_before)
        encoder_norm = nn.LayerNorm(d_model) if normalize_before else None
        self.encoder = TransformerEncoder(encoder_layer, num_encoder_layers, encoder_norm)

        decoder_layer = TransformerDecoderLayer(d_model, nhead, dim_feedforward,
                                                dropout, activation, normalize_before)
        decoder_norm = nn.LayerNorm(d_model)
        self.decoder = TransformerDecoder(decoder_layer, num_decoder_layers, decoder_norm,
                                          return_intermediate=return_intermediate_dec)

        if self.rm_self_attn_dec or self.rm_first_self_attn:
            self.rm_self_attn_dec_func()

        self._reset_parameters()

    def rm_self_attn_dec_func(self):
        total_modifie_layer_num = 0
        rm_list = []
        for idx, layer in enumerate(self.decoder.layers):
            if idx == 0 and not self.rm_first_self_attn:
                continue
            if idx != 0 and not self.rm_self_attn_dec:
                continue
            
            layer.omit_selfattn = True
            del layer.self_attn
            del layer.dropout1
            del layer.norm1

            total_modifie_layer_num += 1
            rm_list.append(idx)
        # remove some self-attention layer
        print("rm {} layer: {}".format(total_modifie_layer_num, rm_list))

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, src, mask, query_embed, pos_embed):
        # flatten NxCxHxW to HWxNxC
        bs, c, h, w = src.shape
        src = src.flatten(2).permute(2, 0, 1)
        pos_embed = pos_embed.flatten(2).permute(2, 0, 1)
        query_embed = query_embed.unsqueeze(1).repeat(1, bs, 1)
        
        if mask is not None:
            mask = mask.flatten(1)# .permute(1, 0)#.permute(1, 0)
        
        if self.num_encoder_layers > 0:
            memory = self.encoder(src, src_key_padding_mask=mask, pos=pos_embed)
        else:
            memory = src # for fast　inference

        tgt = torch.zeros_like(query_embed)
        hs = self.decoder(tgt, memory, memory_key_padding_mask=mask,
                                pos=pos_embed, query_pos=query_embed)
        
        return hs.transpose(1, 2), memory.permute(1, 2, 0).reshape(bs, c, h, w)


def build_transformer(d_model, nhead, num_encoder_layers, num_decoder_layers):
    return Transformer(
        d_model=d_model, nhead=nhead, num_encoder_layers=num_encoder_layers,
        num_decoder_layers=num_decoder_layers, dim_feedforward=2048, dropout=0,
        activation="gelu", normalize_before=False,
        return_intermediate_dec=False)
    '''
    return Transformer(
        d_model=cfg.hidden_dim,
        dropout=cfg.dropout,
        nhead=cfg.nheads,
        dim_feedforward=cfg.dim_feedforward,
        num_encoder_layers=cfg.enc_layers,
        num_decoder_layers=cfg.dec_layers,
        normalize_before=cfg.pre_norm,
        return_intermediate_dec=True,
    )
    '''

# def _get_activation_fn(activation):
#     """Return an activation function given a string"""
#     if activation == "relu":
#         return F.relu
#     if activation == "gelu":
#         return F.gelu
#     if activation == "glu":
#         return F.glu
#     raise RuntimeError("activation should be relu/gelu, not {activation}.")

# def _get_clones(module, N):
#     return nn.ModuleList([copy.deepcopy(module) for i in range(N)])