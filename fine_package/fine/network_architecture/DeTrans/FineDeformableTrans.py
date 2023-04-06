# ------------------------------------------------------------------------
# 3D Deformable Transformer
# ------------------------------------------------------------------------
# Modified from Deformable DETR 
# Copyright (c) 2020 SenseTime. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]

import copy
from typing import Optional, List
import math
import numpy as np
from einops import rearrange, repeat

import torch
import torch.nn.functional as F
from torch import nn
from torch.nn.init import xavier_uniform_, constant_, normal_
from .ops.modules import FineMSDeformAttn
from .position_encoding import build_position_encoding

class FineDeformableTransformer(nn.Module):
    def __init__(self, d_model=256, nhead=8,
                 num_encoder_layers=6, dim_feedforward=1024, dropout=0.1,
                 activation="relu", num_feature_levels=4, enc_n_points=4, 
                 n_vt=8, vt_map=[3,5,5]):
        super().__init__()

        self.d_model = d_model
        self.nhead = nhead
        self.n_vt = n_vt
        self.n_levels = num_feature_levels

        encoder_layer = FineDeformableTransformerEncoderLayer(d_model, dim_feedforward,
                                                          dropout, activation,
                                                          num_feature_levels, nhead, enc_n_points)
        self.encoder = FineDeformableTransformerEncoder(encoder_layer, num_encoder_layers, num_feature_levels, n_vt)

        self.level_embed = nn.Parameter(torch.Tensor(num_feature_levels, d_model))

        # FINE
        self.all_volume_tokens = torch.nn.Parameter(torch.randn(num_feature_levels, vt_map[0]*vt_map[1]*vt_map[2], d_model))
        self.all_volume_tokens.requires_grad = True


        self._reset_parameters()

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
        for m in self.modules():
            if isinstance(m, FineMSDeformAttn):
                m._reset_parameters()
        normal_(self.level_embed)

    def get_valid_ratio(self, mask):
        _, D, H, W = mask.shape
        valid_D = torch.sum(~mask[:, :, 0, 0], 1)
        valid_H = torch.sum(~mask[:, 0, :, 0], 1)
        valid_W = torch.sum(~mask[:, 0, 0, :], 1)

        valid_ratio_d = valid_D.float() / D
        valid_ratio_h = valid_H.float() / H
        valid_ratio_w = valid_W.float() / W
        valid_ratio = torch.stack([valid_ratio_d, valid_ratio_w, valid_ratio_h], -1)
        return valid_ratio

    def forward(self, srcs, masks, pos_embeds, vt_pos=None):

        # prepare input for encoder
        src_flatten = []
        mask_flatten = []
        lvl_pos_embed_flatten = []
        spatial_shapes = []
        for lvl, (src, mask, pos_embed) in enumerate(zip(srcs, masks, pos_embeds)):
            bs, c, d, h, w = src.shape
            spatial_shape = (d, h, w)
            spatial_shapes.append(spatial_shape)
            src = src.flatten(2).transpose(1, 2)
            mask = mask.flatten(1)
            pos_embed = pos_embed.flatten(2).transpose(1, 2)
            lvl_pos_embed = pos_embed + self.level_embed[lvl].view(1, 1, -1)
            lvl_pos_embed_flatten.append(lvl_pos_embed)
            src_flatten.append(src)
            mask_flatten.append(mask)
        src_flatten = torch.cat(src_flatten, 1)
        mask_flatten = torch.cat(mask_flatten, 1)
        lvl_pos_embed_flatten = torch.cat(lvl_pos_embed_flatten, 1)
        spatial_shapes = torch.as_tensor(spatial_shapes, dtype=torch.long, device=src_flatten.device)
        level_start_index = torch.cat((spatial_shapes.new_zeros((1, )), spatial_shapes.prod(1).cumsum(0)[:-1]))
        valid_ratios = torch.stack([self.get_valid_ratio(m) for m in masks], 1)

        # Get the selected volume tokens from all volume tokens for the encoder
        vt_pos = np.random.randint(0, 3*5*5, 16)
        print('* vt_pos', vt_pos)
        B = src_flatten.shape[0]
        tmp = rearrange(self.all_volume_tokens, "l v d -> v l d")
        tmp = repeat(tmp, "v l d -> b v l d", b=B)
        tmp = rearrange(tmp, "b v l d -> (b v) l d")
        sel_vt =  tmp[vt_pos] #(batch*n_vt, n_levels, d_model)
        sel_vt = rearrange(sel_vt, "(b n) l d -> b n l d", b=B) # (batch, n_vt, n_levels, d_model)
        sel_vt = rearrange(sel_vt, "b n l d -> b l n d", b=B)   # (batch, n_levels, n_vt, d_model)
        sel_vt = rearrange(sel_vt, "b l n d -> b (l n) d", b=B) # (batch, n_levels*n_vt,  d_model)

        # encoder
        memory = self.encoder(src_flatten, spatial_shapes, level_start_index, valid_ratios, lvl_pos_embed_flatten, mask_flatten, all_vt=self.all_volume_tokens, sel_vt=sel_vt)

        

        return memory#[:, :-self.n_vt*self.n_levels, :]



class FineDeformableTransformerEncoderLayer(nn.Module):
    def __init__(self,
                 d_model=256, d_ffn=1024,
                 dropout=0.1, activation="relu",
                 n_levels=4, n_heads=8, n_points=4, 
                 n_vt=8, vt_map = [3, 5, 5]):
        super().__init__()
        self.n_levels = n_levels
        self.n_vt = n_vt
        self.d_model = d_model

        # self attention
        self.self_attn = FineMSDeformAttn(d_model, n_levels, n_heads, n_points)
        self.dropout1 = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(d_model)

        # ffn
        self.linear1 = nn.Linear(d_model, d_ffn)
        self.activation = _get_activation_fn(activation)
        self.dropout2 = nn.Dropout(dropout)
        self.linear2 = nn.Linear(d_ffn, d_model)
        self.dropout3 = nn.Dropout(dropout)
        self.norm2 = nn.LayerNorm(d_model)

        # FINE




    @staticmethod
    def with_pos_embed(tensor, pos):
        return tensor if pos is None else tensor + pos

    def forward_ffn(self, src):
        src2 = self.linear2(self.dropout2(self.activation(self.linear1(src))))
        src = src + self.dropout3(src2)
        src = self.norm2(src)
        return src

    def forward(self, src, pos, reference_points, spatial_shapes, level_start_index, padding_mask=None, all_vt=None, sel_vt=None):
        # self attention
        N, L, D = src.shape
        L_ = np.sum([np.prod(spatial_shapes[i].numpy()) for i in range(spatial_shapes.shape[0])])

        # ## Select the region linked volume tokens if needed
        # if L != L_ + self.n_levels*self.n_vt:
        #     sel_vt = torch.rand((N, self.n_levels*self.n_vt, self.d_model))
        # else:
        #     sel_vt = src[:, L_:, :]
        #     src = src[:, :L_, :]

        src2 = self.self_attn(torch.cat([self.with_pos_embed(src, pos), sel_vt], dim=1), reference_points, torch.cat([src, sel_vt], dim=1), spatial_shapes, level_start_index, padding_mask)
        src = torch.cat([src, sel_vt], dim=1) + self.dropout1(src2)
        src = self.norm1(src)

        # ffn
        src = self.forward_ffn(src)

        # FINE: G-MSA

        return src[:, :L_, :], all_vt, src[:, L_:, :]


class FineDeformableTransformerEncoder(nn.Module):
    def __init__(self, encoder_layer, num_layers, n_levels, n_vt):
        super().__init__()
        self.layers = _get_clones(encoder_layer, num_layers)
        self.num_layers = num_layers
        self.n_levels = n_levels
        self.n_vt = n_vt

        

    @staticmethod
    def get_reference_points(spatial_shapes, valid_ratios, device):
        reference_points_list = []
        for lvl, (D_, H_, W_) in enumerate(spatial_shapes):

            ref_d, ref_y, ref_x = torch.meshgrid(torch.linspace(0.5, D_ - 0.5, D_, dtype=torch.float32, device=device),
                                                 torch.linspace(0.5, H_ - 0.5, H_, dtype=torch.float32, device=device),
                                                 torch.linspace(0.5, W_ - 0.5, W_, dtype=torch.float32, device=device))

            ref_d = ref_d.reshape(-1)[None] / (valid_ratios[:, None, lvl, 0] * D_)
            ref_y = ref_y.reshape(-1)[None] / (valid_ratios[:, None, lvl, 2] * H_)
            ref_x = ref_x.reshape(-1)[None] / (valid_ratios[:, None, lvl, 1] * W_)

            ref = torch.stack((ref_d, ref_x, ref_y), -1)   # D W H
            print("xxxxxxx> ref {}".format(lvl), ref.shape)
            reference_points_list.append(ref)
        reference_points = torch.cat(reference_points_list, 1)
        print("xxxxxxx> reference_points", reference_points.shape)
        reference_points = reference_points[:, :, None] * valid_ratios[:, None]
        print("xxxxxxx> reference_points", reference_points.shape)
        print("xxxxxxx> valid_ratios", valid_ratios.shape)

        return reference_points

    def forward(self, src, spatial_shapes, level_start_index, valid_ratios, pos=None, padding_mask=None, all_vt=None, sel_vt=None):
        output = src
        reference_points = self.get_reference_points(spatial_shapes, valid_ratios, device=src.device)

        #get reference points of the sel_vt
        #(N, self.n_levels*self.n_vt, self.n_levels, 3)
        B, L, P, S = reference_points.shape
        rp_vt = torch.zeros((B, self.n_levels, self.n_vt, P, S), dtype=torch.float32, device=src.device)
        for lvl, (D_, W_, H_) in enumerate(spatial_shapes):
            tmp = [0, 
                    H_-1, 
                    H_*(W_-1) -1, 
                    H_*W_ -1,  
                    H_*W_*(D_-1),  
                    H_*W_*(D_-1) + H_-1,  
                    H_*W_*(D_-1) + H_*(W_-1) -1,  
                    H_*W_*(D_-1) + H_*W_ -1]

            for i in range(self.n_vt):
                rp_vt[:, lvl, i, :, :] = reference_points[:, tmp[i], :, :]

        rp_vt = rearrange(rp_vt, 'b l v p s -> b (l v) p s')
        reference_points = torch.cat([reference_points, rp_vt], dim=1)

        for _, layer in enumerate(self.layers):
            output, all_vt, sel_vt = layer(output, pos, reference_points, spatial_shapes, level_start_index, padding_mask, all_vt=all_vt, sel_vt=sel_vt)

        return output


def _get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])


def _get_activation_fn(activation):
    """Return an activation function given a string"""
    if activation == "relu":
        return F.relu
    if activation == "gelu":
        return F.gelu
    if activation == "glu":
        return F.glu
    raise RuntimeError(F"activation should be relu/gelu, not {activation}.")


