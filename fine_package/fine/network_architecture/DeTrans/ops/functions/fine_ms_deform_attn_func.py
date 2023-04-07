# ------------------------------------------------------------------------
# 3D Deformable Self-attention
# ------------------------------------------------------------------------
# Modified from Deformable DETR
# Copyright (c) 2020 SenseTime. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------

from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

import torch
import torch.nn.functional as F
from torch.autograd import Function
from torch.autograd.function import once_differentiable
import einops

def fine_ms_deform_attn_core_pytorch_3D(value, value_spatial_shapes, sampling_locations, attention_weights, n_vt=8):
    N_, S_, M_, D_ = value.shape
    _, Lq_, M_, L_, P_, _ = sampling_locations.shape
    value_list = value[:, :-n_vt*L_, :].split([T_ * H_ * W_ for T_, H_, W_ in value_spatial_shapes], dim=1)
    # value_list = value.split([T_ * H_ * W_ for T_, H_, W_ in value_spatial_shapes], dim=1)
    sampling_grids = 2 * sampling_locations - 1
    # sampling_grids = 3 * sampling_locations - 1
    sampling_value_list = []
    tmp_vt_value_list =  []
    for lid_, (T_, H_, W_) in enumerate(value_spatial_shapes):
        value_l_ = value_list[lid_].flatten(2).transpose(1, 2).reshape(N_*M_, D_, T_, H_, W_)
        sampling_grid_l_ = sampling_grids[:, :, :, lid_].transpose(1, 2).flatten(0, 1)[:,None,:,:,:]
        sampling_value_l_ = F.grid_sample(value_l_, sampling_grid_l_.to(dtype=value_l_.dtype), mode='bilinear', padding_mode='zeros', align_corners=False)[:,:,0]
        st = S_- n_vt*L_ + n_vt*lid_
        en = S_- n_vt*L_ + n_vt*(lid_+1)
        tmp_vt_value = value[:, st:en, :].transpose(1,2).reshape(N_*M_, 1, n_vt, D_)
        tmp_vt_value_list.append(tmp_vt_value)

        sampling_value_list.append(sampling_value_l_)

    attention_weights = attention_weights.transpose(1, 2).reshape(N_*M_, 1, Lq_, L_*(P_+n_vt))

    vt_values = torch.cat(tmp_vt_value_list, dim=1)
    vt_values = einops.repeat(vt_values, "n l v d -> n l a v d", a = Lq_)
    vt_values = einops.rearrange(vt_values, "n l a v d -> n d a (l v)")

    tmp = torch.stack(sampling_value_list, dim=-2)
    tmp = tmp.flatten(-2)
    tmp = torch.cat([tmp, vt_values], dim=-1)

    output = (tmp * attention_weights).sum(-1).view(N_, M_*D_, Lq_)
    
    return output.transpose(1, 2).contiguous()