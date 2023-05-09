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

from fine.network_architecture.finev3 import ClassicAttention, Mlp
from timm.models.layers import DropPath

class FineDeformableTransformer(nn.Module):
    def __init__(self, d_model=256, nhead=8,
                 num_encoder_layers=6, dim_feedforward=1024, dropout=0.1,
                 activation="relu", num_feature_levels=4, enc_n_points=4, 
                 n_vt=8, n_gt=1, imsize=[64,128,128], max_imsize=[218,660,660]):
        super().__init__()

        self.d_model = d_model
        self.nhead = nhead
        self.n_vt = n_vt
        self.n_gt = n_gt
        self.n_levels = num_feature_levels
        # self.vt_map = vt_map
        self.imsize = imsize
        self.max_imsize = max_imsize

        # FINE
        self.pos_grid, self.show_grid, self.vt_map = self.filled_grid()
        self.vt_check = torch.nn.Parameter(torch.zeros(self.vt_map[0]*self.vt_map[1]*self.vt_map[2],1))
        self.vt_check.requires_grad = False

        self.all_volume_tokens = torch.nn.Parameter(torch.randn(num_feature_levels, self.vt_map[0]*self.vt_map[1]*self.vt_map[2], d_model*n_gt))
        self.all_volume_tokens.requires_grad = True


        # encoder_layer = FineDeformableTransformerEncoderLayer(d_model, dim_feedforward, dropout, activation, num_feature_levels, nhead, enc_n_points)

        self.encoder = FineDeformableTransformerEncoder(num_encoder_layers, d_model, dim_feedforward, 
                                                        dropout, activation, num_feature_levels, 
                                                        nhead, enc_n_points, num_feature_levels, 
                                                        n_vt*n_gt, self.vt_map)

        self.level_embed = nn.Parameter(torch.Tensor(num_feature_levels, d_model))

        


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

    def filled_grid(self):
        cd, ch, cw = self.imsize
        d, h, w = self.max_imsize
        grid = np.zeros(self.max_imsize, dtype = int)
        nd, nh, nw = d//cd, h//ch, w//cw
        vtm = [nd, nh, nw]
        pd, ph, pw = d%cd, h%ch, w%cw
        show = ""
        for i in range(nd):
            for j in range(nh):
                for k in range(nw):
                    show += str(nh*nw*i + nw*j + k)+" "
                    tmp_pd_0 = (pd//2)*(0**(i==0))
                    tmp_pd_1 = pd//2 + (pd//2)*(0**(i!=nd-1)) + pd%2

                    tmp_ph_0 = (ph//2)*(0**(j==0))
                    tmp_ph_1 = ph//2 + (ph//2)*(0**(j!=nh-1)) + ph%2
                    
                    tmp_pw_0 = (pw//2)*(0**(k==0))
                    tmp_pw_1 = pw//2 + (pw//2)*(0**(k!=nw-1)) + pw%2

                    grid[i*cd+tmp_pd_0:(i+1)*cd+tmp_pd_1, j*ch+tmp_ph_0:(j+1)*ch+tmp_ph_1, k*cw+tmp_pw_0:(k+1)*cw+tmp_pw_1] = nh*nw*i + nw*j + k
                    
                show += "\n"
            show += "\n"
        return grid, show, vtm

    def border_check(self, pos):
        ret = [i for i in pos]
        size = self.max_imsize
        crop_size = self.imsize
        for i in range(len(ret)):
            pad_i = (size[i]%crop_size[i])//2
            
            check_if_pos_on_frontiere = (pos[i]-pad_i)%crop_size[i] == 0
            check_if_pos_over_border_marge = (pos[i] + crop_size[i] >= size[i] -  pad_i)
            check_if_pos_under_border_marge = (pos[i] <= (size[i]%crop_size[i])//2)
            
            if check_if_pos_on_frontiere or check_if_pos_over_border_marge or check_if_pos_under_border_marge:
                if pos[i] < size[i]//2:
                    ret[i] += 1 + pad_i
                else:
                    tmp = 0
                    if pos[i]+crop_size[i] > size[i]:
                        tmp = pos[i]+crop_size[i] - size[i]
                    ret[i] -= 1 + pad_i + tmp + (size[i]%crop_size[i])%2
        return ret

    def get_tokens_idx(self, pos):
        # Myr : We put the crop in the bigger image referential
        z, x, y = [int(pos[i] + self.max_imsize[i]//2) for i in range(3)]
        pos = (z, x, y)
        pos = self.border_check(pos)
        z, x, y = pos
        cd, ch, cw = self.imsize
        tmp = self.pos_grid[z:z+cd, x:x+ch, y:y+cw]
        idx = np.unique(tmp)
        return idx

    def forward(self, srcs, masks, pos_embeds, pos=None):
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


        # Prepare vt_pos and check
        B = src_flatten.shape[0]
        vt_pos = []
        tmp_check = torch.zeros(B, self.vt_map[0]*self.vt_map[1]*self.vt_map[2],1, device=self.vt_check.device)
        for b, p in enumerate(pos):
            tmp = self.get_tokens_idx(p)
            # print("b, tmp pos", b, tmp)
            tmp_check[b,...] = self.vt_check[...]
            tmp_check[b, tmp, :] += 1
            vt_pos.append(tmp + b*np.array(self.vt_map).prod())
        vt_pos = np.array(vt_pos).flatten()



        # Get the selected volume tokens from all volume tokens for the encoder
        # vt_pos = np.random.randint(0, 3*5*5, 16)
        tmp = rearrange(self.all_volume_tokens, "l v d -> v l d")
        tmp = repeat(tmp, "v l d -> b v l d", b=B) # repeat on batch
        tmp = rearrange(tmp, "b v l d -> (b v) l d") # align batch/volume tokens
        sel_vt =  tmp[vt_pos] #(batch*n_vt, n_levels, n_gt*d_model)
        sel_vt = rearrange(sel_vt, "(b n) l d -> b n l d", b=B) # (batch, n_vt, n_levels, n_gt*d_model)
        sel_vt = rearrange(sel_vt, "b n l d -> b l n d", b=B)   # (batch, n_levels, n_vt, n_gt*d_model)
        sel_vt = rearrange(sel_vt, "b l n (g d) -> b l n g d", g=self.n_gt)   # (batch, n_levels, n_vt, n_gt, d_model)
        sel_vt = rearrange(sel_vt, "b l n g d -> b l (n g) d", g=self.n_gt)   # (batch, n_levels, n_vt*n_gt, d_model)
        sel_vt = rearrange(sel_vt, "b l n d -> b (l n) d", b=B) # (batch, n_levels*n_vt*n_gt,  d_model)

        # Get all seen volume tokens for the WG-MSA
        # check = torch.nn.Parameter(torch.rand(self.vt_map[0]*self.vt_map[1]*self.vt_map[2],1)) > 0.5
        check_pos = repeat(self.vt_check >= 1, 'n c -> (b n c)', b=B) # in fact c=1
        self.vt_check += tmp_check.sum(dim=0)

        # print("---> check", check.shape)
        # tmp = rearrange(tmp, "n l d -> n (l d)")
        seen_vts = tmp[check_pos, :, :]
        # if seen_vts.shape[0] != 0:
        seen_vts = rearrange(seen_vts, "(b n) p c -> b n p c", b=B)


        # encoder
        memory = self.encoder(src_flatten, spatial_shapes, level_start_index, valid_ratios, lvl_pos_embed_flatten, mask_flatten, seen_vts=seen_vts, sel_vt=sel_vt)

        

        return memory



class FineDeformableTransformerEncoderLayer(nn.Module):
    def __init__(self,
                 d_model=256, d_ffn=1024,
                 dropout=0.1, activation="relu",
                 n_levels=4, n_heads=8, n_points=4, 
                 n_vt=8, vt_map = [3, 5, 5], drop_path=0.2, mlp_ratio=4.):
        super().__init__()
        self.n_levels = n_levels
        self.n_vt = n_vt
        self.d_model = d_model

        # self attention
        self.self_attn = FineMSDeformAttn(d_model, n_levels, n_heads, n_points, n_vt)
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
        self.vttrans = []
        for i in range(n_levels):
            self.vttrans.append(VTTransformerLayer(d_model=d_model, n_heads=n_heads, drop_path=drop_path, mlp_ratio=mlp_ratio))
        self.vttrans = nn.ModuleList(self.vttrans)



    @staticmethod
    def with_pos_embed(tensor, pos):
        return tensor if pos is None else tensor + pos

    def forward_ffn(self, src):
        src2 = self.linear2(self.dropout2(self.activation(self.linear1(src))))
        src = src + self.dropout3(src2)
        src = self.norm2(src)
        return src

    def forward(self, src, pos, reference_points, spatial_shapes, level_start_index, padding_mask=None, seen_vts=None, sel_vt=None):
        # self attention
        N, L, D = src.shape
        L_ = np.sum([np.prod([s.item() for s in spatial_shapes[i]]) for i in range(spatial_shapes.shape[0])])

        src2 = self.self_attn(torch.cat([self.with_pos_embed(src, pos), sel_vt], dim=1), reference_points, torch.cat([src, sel_vt], dim=1), spatial_shapes, level_start_index, padding_mask)
        src = torch.cat([src, sel_vt], dim=1) + self.dropout1(src2)
        src = self.norm1(src)

        # ffn
        src = self.forward_ffn(src)

        # FINE: G-MSA
        sel_vt = rearrange(src[:, L_:, :], "b (l n) d -> b l n d", l=self.n_levels)
        for i in range(self.n_levels):
            # sel_vt[:,:,i,:], seen_vts[:,:,i,:] = self.vttrans[i](sel_vt[:,:,i,:], seen_vts[:,:,i,:])
            sel_vt[:,i,:,:], seen_vts[:,i,:,:] = self.vttrans[i](sel_vt[:,i,:,:], seen_vts[:,i,:,:])
        
        sel_vt = rearrange(sel_vt, "b l n d -> b (l n) d")

        return src[:, :L_, :], seen_vts, sel_vt

class VTTransformerLayer(nn.Module):
    """docstring for volume_tokens_transformer_layer"""
    def __init__(self, d_model=256, n_heads=8, drop_path=0.2, mlp_ratio=4.):
        super(VTTransformerLayer, self).__init__()

        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

        mlp_hidden_dim = int(d_model * mlp_ratio)
        self.mlp2 = Mlp(in_features=d_model, hidden_features=mlp_hidden_dim, act_layer=nn.GELU, drop=0.)
        self.norm3 = nn.LayerNorm(d_model)
        self.norm4 = nn.LayerNorm(d_model)
        self.vt_attn = ClassicAttention(dim=d_model, window_size=(0,0), num_heads=n_heads, 
                                        qkv_bias=True, qk_scale=None, attn_drop=0., 
                                        proj_drop=0.)

    def forward(self, sel_vt, seen_vts):
        B, N, D = sel_vt.shape
        _, N_, _ = seen_vts.shape

        if seen_vts.shape[1] != 0:
            vts = torch.cat([sel_vt, seen_vts], dim=1)
        else:
            vts = sel_vt

        skip_vts = vts
        # vts = self.norm3(vts)
        vts = self.vt_attn(vts, None, None)
        vts = self.drop_path(vts) + skip_vts
        vts = self.norm3(vts)

        # vts = vts + self.drop_path(self.mlp2(self.norm4(vts)))
        vts = self.norm4(vts + self.drop_path(self.mlp2(vts)))
        
        return vts[:, :N, :], vts[:, N:, :]
        

class FineDeformableTransformerEncoder(nn.Module):
    def __init__(self, num_layers, d_model, dim_feedforward, dropout, activation, num_feature_levels, nhead, enc_n_points, n_levels, n_vt, vt_map):
        super().__init__()
        self.dpr = [x.item() for x in torch.linspace(0, 0.2, num_layers)]

        self.layers = []
        for i in range(num_layers):
            self.layers.append(
                FineDeformableTransformerEncoderLayer(d_model, dim_feedforward, 
                                                    dropout, activation, 
                                                    num_feature_levels, nhead, 
                                                    enc_n_points,
                                                    n_vt=n_vt, vt_map=vt_map,
                                                    drop_path=self.dpr[i])
                )
        self.layers = nn.ModuleList(self.layers)
        # self.layers = _get_clones(encoder_layer, num_layers)
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
            reference_points_list.append(ref)
        reference_points = torch.cat(reference_points_list, 1)
        reference_points = reference_points[:, :, None] * valid_ratios[:, None]

        return reference_points

    def forward(self, src, spatial_shapes, level_start_index, valid_ratios, pos=None, padding_mask=None, seen_vts=None, sel_vt=None):
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
            n_gt = self.n_vt//8

            # for i in range(self.n_vt):
            for i in range(8):
                rp_vt[:, lvl, i*n_gt:(i+1)*n_gt:, :, :] = reference_points[:, tmp[i], :, :]

        rp_vt = rearrange(rp_vt, 'b l v p s -> b (l v) p s')
        reference_points = torch.cat([reference_points, rp_vt], dim=1)

        for _, layer in enumerate(self.layers):
            output, seen_vts, sel_vt = layer(output, pos, reference_points, spatial_shapes, level_start_index, padding_mask, seen_vts=seen_vts, sel_vt=sel_vt)

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


