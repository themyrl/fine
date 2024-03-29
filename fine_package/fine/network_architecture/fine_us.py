from einops import rearrange
from copy import deepcopy
from nnunet.utilities.nd_softmax import softmax_helper
from torch import nn
import torch
import numpy as np
from nnunet.network_architecture.initialization import InitWeights_He
from fine.network_architecture.neural_network_ext import SegmentationNetwork
import torch.nn.functional


import torch.nn.functional as F
import torch.utils.checkpoint as checkpoint
from timm.models.layers import DropPath, to_3tuple, trunc_normal_


from einops import repeat

# V5 + multiple vts by intersec

# SYNAPSE
#MAX : 660 660 \218
#AVG : 529 529 \150
#MIN : 401 401 \93
SYNAPSE_MAX=[218,660,660]

#CROP : 128 128 \64
#MAX NCROP : 5 5 \3



# BRAIN TUMOR SEG
#MAX : 187 160 \149
#MIN : 144 122 \119
#AVG : 168 138 \137
#CROP : 128 128 \128
#MAX NCROP : 2 2 \2
BRAIN_TUMOR_MAX=[149,187,160]

class Mlp(nn.Module):
    """ Multilayer perceptron."""

    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


def window_partition(x, window_size):
    """
    Args:
        x: (B, H, W, C)
        window_size (int): window size

    Returns:
        windows: (num_windows*B, window_size, window_size, C)
    """
    B, S, H, W, C = x.shape
    x = x.view(B, S // window_size, window_size, H // window_size, window_size, W // window_size, window_size, C)
    windows = x.permute(0, 1, 3, 5, 2, 4, 6, 7).contiguous().view(-1, window_size, window_size, window_size, C)
    return windows


def window_reverse(windows, window_size, S, H, W):
    """
    Args:
        windows: (num_windows*B, window_size, window_size, C)
        window_size (int): Window size
        H (int): Height of image
        W (int): Width of image

    Returns:
        x: (B, H, W, C)
    """
    B = int(windows.shape[0] / (S * H * W / window_size / window_size / window_size))
    x = windows.view(B, S // window_size, H // window_size, W // window_size, window_size, window_size, window_size, -1)
    x = x.permute(0, 1, 4, 2, 5, 3, 6, 7).contiguous().view(B, S, H, W, -1)
    return x
class ClassicAttention(nn.Module):
    r""" Window based multi-head self attention (W-MSA) module with relative position bias.
    It supports both of shifted and non-shifted window.
    Args:
        dim (int): Number of input channels.
        window_size (tuple[int]): The height and width of the window.
        num_heads (int): Number of attention heads.
        qkv_bias (bool, optional):  If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set
        attn_drop (float, optional): Dropout ratio of attention weight. Default: 0.0
        proj_drop (float, optional): Dropout ratio of output. Default: 0.0
    """

    def __init__(self, dim, window_size, num_heads, qkv_bias=True, qk_scale=None, attn_drop=0., proj_drop=0., n_vts=4):

        super().__init__()
        self.n_vts = n_vts
        self.dim = dim
        self.window_size = window_size  # Wh, Ww
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        # self.pe = nn.Parameter(torch.zeros(window_size[0]*window_size[1], dim))
        # trunc_normal_(self.pe, std=.02)

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        self.softmax = nn.Softmax(dim=-1)


    def forward(self, x, pe, mask=None):
        """
        Args:
            x: input features with shape of (num_windows*B, N, C)
            mask: (0/-inf) mask with shape of (num_windows, Wh*Ww, Wh*Ww) or None
        """
        B_, N, C = x.shape

        if pe != None:
            N_ = N - self.n_vts
            m = pe.shape[0]
            strt = m//2-N_//2
            pe = pe[strt:strt+N_,:]
            x[:, self.n_vts:, :] = x[:, self.n_vts:, :] + pe

        # print(x.shape)
        # print(B_, N, 3, self.num_heads, C // self.num_heads)
        # print(self.qkv(x).shape)
        # exit(0)
        qkv = self.qkv(x).reshape(B_, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]  # make torchscript happy (cannot use tensor as tuple)

        q = q * self.scale
        attn = (q @ k.transpose(-2, -1))

        # exit(0)
        # attn = attn

        if mask != None:
            attn = attn + repeat(mask, "b m n -> b h m n", h=self.num_heads)
            
        attn = self.softmax(attn)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B_, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

    def extra_repr(self) -> str:
        return f'dim={self.dim}, window_size={self.window_size}, num_heads={self.num_heads}'

    def flops(self, N):
        # calculate flops for 1 window with token length of N
        flops = 0
        # qkv = self.qkv(x)
        flops += N * self.dim * 3 * self.dim
        # attn = (q @ k.transpose(-2, -1))
        flops += self.num_heads * N * (self.dim // self.num_heads) * N
        #  x = (attn @ v)
        flops += self.num_heads * N * N * (self.dim // self.num_heads)
        # x = self.proj(x)
        flops += N * self.dim * self.dim
        return flops

class WindowAttention(nn.Module):
    """ Window based multi-head self attention (W-MSA) module with relative position bias.
    It supports both of shifted and non-shifted window.

    Args:
        dim (int): Number of input channels.
        window_size (tuple[int]): The height and width of the window.
        num_heads (int): Number of attention heads.
        qkv_bias (bool, optional):  If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set
        attn_drop (float, optional): Dropout ratio of attention weight. Default: 0.0
        proj_drop (float, optional): Dropout ratio of output. Default: 0.0
    """

    def __init__(self, dim, window_size, num_heads, qkv_bias=True, qk_scale=None, attn_drop=0., proj_drop=0., gt_num=1):

        super().__init__()
        self.dim = dim
        self.window_size = window_size  
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5
        self.gt_num = gt_num

        # define a parameter table of relative position bias
        self.relative_position_bias_table = nn.Parameter(
            torch.zeros((2 * window_size[0] - 1) * (2 * window_size[1] - 1) * (2 * window_size[2] - 1),
                        num_heads))  

        # get pair-wise relative position index for each token inside the window
        coords_s = torch.arange(self.window_size[0])
        coords_h = torch.arange(self.window_size[1])
        coords_w = torch.arange(self.window_size[2])
        coords = torch.stack(torch.meshgrid([coords_s, coords_h, coords_w]))  
        coords_flatten = torch.flatten(coords, 1)  
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]  
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()  
        relative_coords[:, :, 0] += self.window_size[0] - 1  # shift to start from 0
        relative_coords[:, :, 1] += self.window_size[1] - 1
        relative_coords[:, :, 2] += self.window_size[2] - 1

        
        relative_coords[:, :, 0] *= 3 * self.window_size[1] - 1
        relative_coords[:, :, 1] *= 2 * self.window_size[1] - 1

        relative_position_index = relative_coords.sum(-1)  
        self.register_buffer("relative_position_index", relative_position_index)

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        trunc_normal_(self.relative_position_bias_table, std=.02)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x, mask=None, gt=None):
        """ Forward function.

        Args:
            x: input features with shape of (num_windows*B, N, C)
            mask: (0/-inf) mask with shape of (num_windows, Wh*Ww, Wh*Ww) or None
        """
        B_, N_, C = x.shape

        
        
        
        x = torch.cat([gt, x], dim=1) # x of shape (num_windows*B, G+N_, C)
        B_, N, C = x.shape


        qkv = self.qkv(x)

        # print("---> qkv shape", qkv.shape)
        
        qkv=qkv.reshape(B_, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]  # make torchscript happy (cannot use tensor as tuple)
        
        q = q * self.scale
        attn = (q @ k.transpose(-2, -1))
        
        relative_position_bias = self.relative_position_bias_table[self.relative_position_index.view(-1)].view(
            self.window_size[0] * self.window_size[1] * self.window_size[2],
            self.window_size[0] * self.window_size[1] * self.window_size[2], -1)  
        relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()  
        # attn = attn + relative_position_bias.unsqueeze(0)
        attn[:,:,self.gt_num:,self.gt_num:] = attn[:,:,self.gt_num:,self.gt_num:] + relative_position_bias.unsqueeze(0)
        

        if mask is not None:
            nW = mask.shape[0]
            attn_ = attn.view(B_ // nW, nW, self.num_heads, N, N)[:,:,:,self.gt_num:,self.gt_num:] + mask.unsqueeze(1).unsqueeze(0)
            attn[:,:,self.gt_num:,self.gt_num:] = attn_.view(-1, self.num_heads, N_, N_)
            attn = self.softmax(attn)
        else:
            attn = self.softmax(attn)

        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B_, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)

        gt = x[:,:-N_,:]
        x = x[:,-N_:,:] # x of size (B_, N_, C)
        return x, gt


class SwinTransformerBlock(nn.Module):
    """ Swin Transformer Block.

    Args:
        dim (int): Number of input channels.
        num_heads (int): Number of attention heads.
        window_size (int): Window size.
        shift_size (int): Shift size for SW-MSA.
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim.
        qkv_bias (bool, optional): If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set.
        drop (float, optional): Dropout rate. Default: 0.0
        attn_drop (float, optional): Attention dropout rate. Default: 0.0
        drop_path (float, optional): Stochastic depth rate. Default: 0.0
        act_layer (nn.Module, optional): Activation layer. Default: nn.GELU
        norm_layer (nn.Module, optional): Normalization layer.  Default: nn.LayerNorm
    """

    def __init__(self, dim, input_resolution, num_heads, window_size=7, shift_size=0,
                 mlp_ratio=4., qkv_bias=True, qk_scale=None, drop=0., attn_drop=0., drop_path=0.,
                 act_layer=nn.GELU, norm_layer=nn.LayerNorm, gt_num=1, n_vts=4,vt_num=1):
        super().__init__()
        self.n_vts = n_vts
        self.dim = dim
        self.input_resolution = input_resolution
        self.num_heads = num_heads
        self.window_size = window_size
        self.shift_size = shift_size
        self.mlp_ratio = mlp_ratio
        self.gt_num = gt_num
        self.vt_num=vt_num
        if min(self.input_resolution) <= self.window_size:
            # if window size is larger than input resolution, we don't partition windows
            self.shift_size = 0
            self.window_size = min(self.input_resolution)
        assert 0 <= self.shift_size < self.window_size, "shift_size must in 0-window_size"

        self.norm1 = norm_layer(dim)
        self.attn = WindowAttention(
            dim, window_size=to_3tuple(self.window_size), num_heads=num_heads,
            qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop, gt_num=gt_num)

        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

        self.gt_attn = ClassicAttention(dim=dim, window_size=(0,0), num_heads=num_heads, 
                                            qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, 
                                            proj_drop=drop, n_vts=n_vts*vt_num)
        self.vt_attn = ClassicAttention(dim=dim, window_size=(0,0), num_heads=num_heads, 
                                            qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, 
                                            proj_drop=drop)

    def forward(self, x, mask_matrix, gt, pe, vts, vt_pos, check):
        """ Forward function.

        Args:
            x: Input feature, tensor size (B, H*W, C).
            H, W: Spatial resolution of the input feature.
            mask_matrix: Attention mask for cyclic shift.
        """
        B, L, C = x.shape
        S, H, W = self.input_resolution

        # 
        
        assert L == S * H * W, "input feature has wrong size"
        
        


        shortcut = x
        
        x = self.norm1(x)
        x = x.view(B, S, H, W, C)

        # pad feature maps to multiples of window size
        pad_r = (self.window_size - W % self.window_size) % self.window_size
        pad_b = (self.window_size - H % self.window_size) % self.window_size
        pad_g = (self.window_size - S % self.window_size) % self.window_size

        x = F.pad(x, (0, 0, 0, pad_r, 0, pad_b, 0, pad_g))  
        _, Sp, Hp, Wp, _ = x.shape

        # cyclic shift
        if self.shift_size > 0:
            shifted_x = torch.roll(x, shifts=(-self.shift_size, -self.shift_size,-self.shift_size), dims=(1, 2,3))
            attn_mask = mask_matrix
        else:
            shifted_x = x
            attn_mask = None

        # partition windows
        x_windows = window_partition(shifted_x, self.window_size)  
        x_windows = x_windows.view(-1, self.window_size * self.window_size * self.window_size,
                                   C)  
        
        if len(gt.shape) != 3:
            gt = repeat(gt, "g c -> b g c", b=x_windows.shape[0])# shape of (num_windows*B, G, C)

        skip_gt = gt
        # W-MSA/SW-MSA
        attn_windows, gt = self.attn(x_windows, mask=attn_mask, gt=gt)  

        
        self.nc = vts.shape[0]//B
        if len(vts.shape) != 3:
            self.nc = vts.shape[0]
            vts = repeat(vts, "g c -> b g c", b=B)# shape of (num_windows*B, G, C)

        # vt_pos_ = [i*vts.shape[1] + vt_pos[i] for i in range(B)]
        vt_pos_ = vt_pos.copy()
        # print("len(vt_pos_)" ,len(vt_pos_))
        # print("self.n_vts", self.n_vts)
        # print("self.nc", self.nc)

        if B==2:
            vt_pos_[self.n_vts:] = [self.nc+vt_pos_[self.n_vts + i] for i in range(self.n_vts)]

        vts = rearrange(vts, "b n c -> (b n) c")
        vt = vts[vt_pos_]
        vt = rearrange(vt, "(b n) c -> b n c", b=B)
        # vts = rearrange(vts, "(b n) c -> b n c", b=B)




        # GT
        gt = skip_gt + self.drop_path(gt)
        gt = gt + self.drop_path(self.mlp(self.norm2(gt)))
        tmp, ngt, c = gt.shape
        nw = tmp//B
        gt = rearrange(gt, "(b n) g c -> b (n g) c", b=B)

        if self.vt_num != 1:
            vt = rearrange(vt, "b n (v c) -> b (n v) c", v=self.vt_num)

        gt = torch.cat([vt, gt], dim=1)
        gt = self.gt_attn(gt, pe)
        if self.vt_num != 1:
            vt = gt[:,:self.n_vts*self.vt_num,:]
            gt = gt[:,self.n_vts*self.vt_num:,:]
        else:
            vt = gt[:,:self.n_vts,:]
            gt = gt[:,self.n_vts:,:]
        gt = rearrange(gt, "b (n g) c -> (b n) g c",g=ngt, c=C)

        # New vts
        # vts_ = vts.clone().half()
        # if len(vts_.shape) != 3:
        #     vts_ = repeat(vts_, "g c -> b g c", b=B)# shape of (num_windows*B, G, C)
        # for i in range(B):
        #     vts_[i, vt_pos[i]] = vt[i]

        # Modif the vts
        z = torch.zeros(vts.shape, dtype=vt.dtype, device=vts.device)
        if self.vt_num != 1:
            vt = rearrange(vt, "b (n v) c -> b n (v c)", v=self.vt_num)
        vt = rearrange(vt, "b n c -> (b n) c")
        z[vt_pos_] = vt
        vts = vts + z
        vts = rearrange(vts, "(b n) c -> b n c", b=B)        

        check_pos = check.nonzero(as_tuple=True)[0]
        vt_mask = torch.zeros((vts.shape[1]*self.vt_num, vts.shape[1]*self.vt_num), dtype=vts.dtype, device=vts.device)-1000
        vt_mask[:, check_pos] = 0
        vt_mask = repeat(vt_mask, "n c -> b n c", b=B)
        if self.vt_num != 1:
            vts = rearrange(vts, "b n (v c) -> b (n v) c", v=self.vt_num)
        vts = self.vt_attn(vts, None,vt_mask)
        if self.vt_num != 1:
            vts = rearrange(vts, "b (n v) c -> b n (v c)", v=self.vt_num)

     
        # merge windows
        attn_windows = attn_windows.view(-1, self.window_size, self.window_size, self.window_size, C)
        shifted_x = window_reverse(attn_windows, self.window_size, Sp, Hp, Wp)  

        # reverse cyclic shift
        if self.shift_size > 0:
            x = torch.roll(shifted_x, shifts=(self.shift_size, self.shift_size, self.shift_size), dims=(1, 2, 3))
        else:
            x = shifted_x

        if pad_r > 0 or pad_b > 0 or pad_g > 0:
            x = x[:, :S, :H, :W, :].contiguous()

        x = x.view(B, S * H * W, C)

        # FFN
        x = shortcut + self.drop_path(x)
        x = x + self.drop_path(self.mlp(self.norm2(x)))


        # clamp here
        gt = torch.clamp(gt, min=-1, max=1)
        vts = torch.clamp(vts, min=-1, max=1)

        return x, gt, vts


class PatchMerging(nn.Module):
    """ Patch Merging Layer

    Args:
        dim (int): Number of input channels.
        norm_layer (nn.Module, optional): Normalization layer.  Default: nn.LayerNorm
    """

    def __init__(self, dim, norm_layer=nn.LayerNorm):
        super().__init__()
        self.dim = dim
        self.reduction = nn.Conv3d(dim,dim*2,kernel_size=2,stride=2)
        self.norm = norm_layer(dim)

    def forward(self, x, S, H, W):
        """ Forward function.

        Args:
            x: Input feature, tensor size (B, H*W, C).
            H, W: Spatial resolution of the input feature.
        """
        B, L, C = x.shape
        assert L == H * W * S, "input feature has wrong size"

        x = x.view(B, S, H, W, C)
        
        x = F.gelu(x)
        x = self.norm(x)
        x=x.permute(0,4,1,2,3)
        x=self.reduction(x)
        x=x.permute(0,2,3,4,1).view(B,-1,2*C)
        return x
    
class Patch_Expanding(nn.Module):
    def __init__(self, dim, norm_layer=nn.LayerNorm):
        super().__init__()
        self.dim = dim
        self.norm = norm_layer(dim)
        self.up=nn.ConvTranspose3d(dim,dim//2,2,2)
    def forward(self, x, S, H, W):
        """ Forward function.

        Args:
            x: Input feature, tensor size (B, H*W, C).
            H, W: Spatial resolution of the input feature.
        """
        
        B, L, C = x.shape
        assert L == H * W * S, "input feature has wrong size"

        x = x.view(B, S, H, W, C)

       
        
        x = self.norm(x)
        x=x.permute(0,4,1,2,3)
        x = self.up(x)
        x=x.permute(0,2,3,4,1).view(B,-1,C//2)
       
        return x
class BasicLayer(nn.Module):
    """ A basic Swin Transformer layer for one stage.

    Args:
        dim (int): Number of feature channels
        depth (int): Depths of this stage.
        num_heads (int): Number of attention head.
        window_size (int): Local window size. Default: 7.
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim. Default: 4.
        qkv_bias (bool, optional): If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set.
        drop (float, optional): Dropout rate. Default: 0.0
        attn_drop (float, optional): Attention dropout rate. Default: 0.0
        drop_path (float | tuple[float], optional): Stochastic depth rate. Default: 0.0
        norm_layer (nn.Module, optional): Normalization layer. Default: nn.LayerNorm
        downsample (nn.Module | None, optional): Downsample layer at the end of the layer. Default: None
        use_checkpoint (bool): Whether to use checkpointing to save memory. Default: False.
    """

    def __init__(self,
                 dim,
                 input_resolution,
                 depth,
                 num_heads,
                 window_size=7,
                 mlp_ratio=4.,
                 qkv_bias=True,
                 qk_scale=None,
                 drop=0.,
                 attn_drop=0.,
                 drop_path=0.,
                 norm_layer=nn.LayerNorm,
                 downsample=True,  
                 use_checkpoint=False, gt_num=1, id_layer=0, vt_map=(3,5,5),vt_num=1):
        super().__init__()
        self.window_size = window_size
        self.shift_size = window_size // 2
        self.depth = depth
        self.use_checkpoint = use_checkpoint
        self.vt_map = vt_map

        self.global_token = torch.nn.Parameter(torch.randn(gt_num,dim))
        self.global_token.requires_grad = True

        # self.volume_token = torch.nn.Parameter(torch.randn(vt_map[0]*vt_map[1]*vt_map[2],dim))
        self.volume_token = torch.nn.Parameter(torch.randn(vt_map[1]*vt_map[2],dim*vt_num))
        self.volume_token.requires_grad = True


        # build blocks
        self.blocks = nn.ModuleList([
            SwinTransformerBlock(
                dim=dim,
                input_resolution=input_resolution,
                num_heads=num_heads,
                window_size=window_size,
                shift_size=0 if (i % 2 == 0) else window_size // 2,
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                qk_scale=qk_scale,
                drop=drop,
                attn_drop=attn_drop,
                drop_path=drop_path[i] if isinstance(drop_path, list) else drop_path, norm_layer=norm_layer,gt_num=gt_num,vt_num=vt_num)
            for i in range(depth)])

        # if self.vt_map==(3,5,5):
        #     ws_pe = (8*gt_num//2**id_layer, 8*gt_num//2**id_layer, 8*gt_num//2**id_layer)
        # else:
        #     ws_pe = (16*gt_num//2**id_layer, 8*gt_num//2**id_layer, 8*gt_num//2**id_layer)
        ws_pe = ((32//window_size)*gt_num//2**id_layer, (32//window_size)*gt_num//2**id_layer, (32//window_size)*gt_num//2**id_layer)
        self.pe = nn.Parameter(torch.zeros(ws_pe[0]*ws_pe[1]*ws_pe[2], dim))
        trunc_normal_(self.pe, std=.02)

        # patch merging layer
        if downsample is not None:
            self.downsample = downsample(dim=dim, norm_layer=norm_layer)
        else:
            self.downsample = None

    def forward(self, x, S, H, W, vt_pos, check):
        """ Forward function.

        Args:
            x: Input feature, tensor size (B, H*W, C).
            H, W: Spatial resolution of the input feature.
        """

        # calculate attention mask for SW-MSA
        Sp = int(np.ceil(S / self.window_size)) * self.window_size
        Hp = int(np.ceil(H / self.window_size)) * self.window_size
        Wp = int(np.ceil(W / self.window_size)) * self.window_size
        img_mask = torch.zeros((1, Sp, Hp, Wp, 1), device=x.device)  
        s_slices = (slice(0, -self.window_size),
                    slice(-self.window_size, -self.shift_size),
                    slice(-self.shift_size, None))
        h_slices = (slice(0, -self.window_size),
                    slice(-self.window_size, -self.shift_size),
                    slice(-self.shift_size, None))
        w_slices = (slice(0, -self.window_size),
                    slice(-self.window_size, -self.shift_size),
                    slice(-self.shift_size, None))
        cnt = 0
        for s in s_slices:
            for h in h_slices:
                for w in w_slices:
                    img_mask[:, s, h, w, :] = cnt
                    cnt += 1

        mask_windows = window_partition(img_mask, self.window_size)  
        mask_windows = mask_windows.view(-1,
                                         self.window_size * self.window_size * self.window_size) 
        attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
        attn_mask = attn_mask.masked_fill(attn_mask != 0, float(-100.0)).masked_fill(attn_mask == 0, float(0.0))

        gt = self.global_token
        vts = self.volume_token
        # self.vt_check[vt_pos] += 1
        for blk in self.blocks:
            blk.H, blk.W = H, W
            if self.use_checkpoint:
                x = checkpoint.checkpoint(blk, x, attn_mask)
            else:
                # check = (self.vt_check.sum() >= self.vt_map[0]*self.vt_map[1]*self.vt_map[2])
                x, gt, vts = blk(x, attn_mask, gt, self.pe, vts, vt_pos, check)
        if self.downsample is not None:
            x_down = self.downsample(x, S, H, W)
            Ws, Wh, Ww = (S + 1) // 2, (H + 1) // 2, (W + 1) // 2
            return x, S, H, W, x_down, Ws, Wh, Ww
        else:
            return x, S, H, W, x, S, H, W

class BasicLayer_up(nn.Module):
    """ A basic Swin Transformer layer for one stage.

    Args:
        dim (int): Number of feature channels
        depth (int): Depths of this stage.
        num_heads (int): Number of attention head.
        window_size (int): Local window size. Default: 7.
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim. Default: 4.
        qkv_bias (bool, optional): If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set.
        drop (float, optional): Dropout rate. Default: 0.0
        attn_drop (float, optional): Attention dropout rate. Default: 0.0
        drop_path (float | tuple[float], optional): Stochastic depth rate. Default: 0.0
        norm_layer (nn.Module, optional): Normalization layer. Default: nn.LayerNorm
        downsample (nn.Module | None, optional): Downsample layer at the end of the layer. Default: None
        use_checkpoint (bool): Whether to use checkpointing to save memory. Default: False.
    """

    def __init__(self,
                 dim,
                 input_resolution,
                 depth,
                 num_heads,
                 window_size=7,
                 mlp_ratio=4.,
                 qkv_bias=True,
                 qk_scale=None,
                 drop=0.,
                 attn_drop=0.,
                 drop_path=0.,
                 norm_layer=nn.LayerNorm,
                 upsample=True, gt_num=1,id_layer=0, vt_map=(3,5,5), vt_num=1
                ):
        super().__init__()
        self.window_size = window_size
        self.shift_size = window_size // 2
        self.depth = depth
        self.vt_map = vt_map

        self.global_token = torch.nn.Parameter(torch.randn(gt_num,dim))
        self.global_token.requires_grad = True

        # self.volume_token = torch.nn.Parameter(torch.randn(vt_map[0]*vt_map[1]*vt_map[2],dim))
        self.volume_token = torch.nn.Parameter(torch.randn(vt_map[1]*vt_map[2],dim*vt_num))
        self.volume_token.requires_grad = True



        

        # build blocks
        self.blocks = nn.ModuleList([
            SwinTransformerBlock(
                dim=dim,
                input_resolution=input_resolution,
                num_heads=num_heads,
                window_size=window_size,
                shift_size=0 if (i % 2 == 0) else window_size // 2,
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                qk_scale=qk_scale,
                drop=drop,
                attn_drop=attn_drop,
                drop_path=drop_path[i] if isinstance(drop_path, list) else drop_path, norm_layer=norm_layer, gt_num=gt_num, vt_num=vt_num)
            for i in range(depth)])

        # if self.vt_map==(3,5,5):
        #     ws_pe = (8*gt_num//2**id_layer, 8*gt_num//2**id_layer, 8*gt_num//2**id_layer)
        # else:
        ws_pe = ((32//window_size)*gt_num//2**id_layer, (32//window_size)*gt_num//2**id_layer, (32//window_size)*gt_num//2**id_layer)
        self.pe = nn.Parameter(torch.zeros(ws_pe[0]*ws_pe[1]*ws_pe[2], dim))
        trunc_normal_(self.pe, std=.02)

        # patch merging layer
        
        self.Upsample = upsample(dim=2*dim, norm_layer=norm_layer)
    def forward(self, x,skip, S, H, W, vt_pos, check):
        """ Forward function.

        Args:
            x: Input feature, tensor size (B, H*W, C).
            H, W: Spatial resolution of the input feature.
        """
      
        x_up = self.Upsample(x, S, H, W)
       
        x_up+=skip
        S, H, W = S * 2, H * 2, W * 2
        # calculate attention mask for SW-MSA
        Sp = int(np.ceil(S / self.window_size)) * self.window_size
        Hp = int(np.ceil(H / self.window_size)) * self.window_size
        Wp = int(np.ceil(W / self.window_size)) * self.window_size
        img_mask = torch.zeros((1, Sp, Hp, Wp, 1), device=x.device) 
        s_slices = (slice(0, -self.window_size),
                    slice(-self.window_size, -self.shift_size),
                    slice(-self.shift_size, None))
        h_slices = (slice(0, -self.window_size),
                    slice(-self.window_size, -self.shift_size),
                    slice(-self.shift_size, None))
        w_slices = (slice(0, -self.window_size),
                    slice(-self.window_size, -self.shift_size),
                    slice(-self.shift_size, None))
        cnt = 0
        for s in s_slices:
            for h in h_slices:
                for w in w_slices:
                    img_mask[:, s, h, w, :] = cnt
                    cnt += 1

        mask_windows = window_partition(img_mask, self.window_size)  
        mask_windows = mask_windows.view(-1,
                                         self.window_size * self.window_size * self.window_size)  
        attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
        attn_mask = attn_mask.masked_fill(attn_mask != 0, float(-100.0)).masked_fill(attn_mask == 0, float(0.0))
        gt = self.global_token 
        vts = self.volume_token
        # self.vt_check[vt_pos] += 1
        for blk in self.blocks:
            # check = (self.vt_check.sum() >= self.vt_map[0]*self.vt_map[1]*self.vt_map[2])
            x_up, gt, vts = blk(x_up, attn_mask, gt, self.pe, vts, vt_pos, check)
        
        return x_up, S, H, W
        
# done
class project(nn.Module):
    def __init__(self,in_dim,out_dim,stride,padding,activate,norm,last=False):
        super().__init__()
        self.out_dim=out_dim
        self.conv1=nn.Conv3d(in_dim,out_dim,kernel_size=3,stride=stride,padding=padding)
        self.conv2=nn.Conv3d(out_dim,out_dim,kernel_size=3,stride=1,padding=1)
        self.activate=activate()
        self.norm1=norm(out_dim)
        self.last=last  
        if not last:
            self.norm2=norm(out_dim)
            
    def forward(self,x):
        x=self.conv1(x)
        x=self.activate(x)
        #norm1
        Ws, Wh, Ww = x.size(2), x.size(3), x.size(4)
        x = x.flatten(2).transpose(1, 2)
        x = self.norm1(x)
        x = x.transpose(1, 2).view(-1, self.out_dim, Ws, Wh, Ww)
        

        x=self.conv2(x)
        if not self.last:
            x=self.activate(x)
            #norm2
            Ws, Wh, Ww = x.size(2), x.size(3), x.size(4)
            x = x.flatten(2).transpose(1, 2)
            x = self.norm2(x)
            x = x.transpose(1, 2).view(-1, self.out_dim, Ws, Wh, Ww)
        return x
        
    

class PatchEmbed(nn.Module):
    """ Image to Patch Embedding

    Args:
        patch_size (int): Patch token size. Default: 4.
        in_chans (int): Number of input image channels. Default: 3.
        embed_dim (int): Number of linear projection output channels. Default: 96.
        norm_layer (nn.Module, optional): Normalization layer. Default: None
    """

    def __init__(self, patch_size=4, in_chans=4, embed_dim=96, norm_layer=None):
        super().__init__()
        patch_size = to_3tuple(patch_size)
        self.patch_size = patch_size

        self.in_chans = in_chans
        self.embed_dim = embed_dim

        if patch_size[0] == 4:
            stride1=[patch_size[0]//2,patch_size[1]//2,patch_size[2]//2]
            stride2=[patch_size[0]//2,patch_size[1]//2,patch_size[2]//2]
        else:
            stride1=[2,2,2]
            stride2=[1,2,2]

        # stride1=[patch_size[0],patch_size[1]//2,patch_size[2]//2]
        # stride2=[patch_size[0]//2,patch_size[1]//2,patch_size[2]//2]
        
        self.proj1 = project(in_chans,embed_dim//2,stride1,1,nn.GELU,nn.LayerNorm,False)
        self.proj2 = project(embed_dim//2,embed_dim,stride2,1,nn.GELU,nn.LayerNorm,True)
        
        # self.proj1 = project(in_chans,embed_dim//2,[2,2,2],1,nn.GELU,nn.LayerNorm,False)
        # self.proj2 = project(embed_dim//2,embed_dim,[1,2,2],1,nn.GELU,nn.LayerNorm,True)
        if norm_layer is not None:
            self.norm = norm_layer(embed_dim)
        else:
            self.norm = None

    def forward(self, x):
        """Forward function."""
        # padding
        _, _, S, H, W = x.size()
        if W % self.patch_size[2] != 0:
            x = F.pad(x, (0, self.patch_size[2] - W % self.patch_size[2]))
        if H % self.patch_size[1] != 0:
            x = F.pad(x, (0, 0, 0, self.patch_size[1] - H % self.patch_size[1]))
        if S % self.patch_size[0] != 0:
            x = F.pad(x, (0, 0, 0, 0, 0, self.patch_size[0] - S % self.patch_size[0]))
     
        x = self.proj1(x)  
        
        x = self.proj2(x)  
 
        if self.norm is not None:
            Ws, Wh, Ww = x.size(2), x.size(3), x.size(4)
            x = x.flatten(2).transpose(1, 2)
            x = self.norm(x)
            x = x.transpose(1, 2).view(-1, self.embed_dim, Ws, Wh, Ww)

        return x



class SwinTransformer(nn.Module):
    """ Swin Transformer backbone.
        A PyTorch impl of : `Swin Transformer: Hierarchical Vision Transformer using Shifted Windows`  -
          https://arxiv.org/pdf/2103.14030

    Args:
        pretrain_img_size (int): Input image size for training the pretrained model,
            used in absolute postion embedding. Default 224.
        patch_size (int | tuple(int)): Patch size. Default: 4.
        in_chans (int): Number of input image channels. Default: 3.
        embed_dim (int): Number of linear projection output channels. Default: 96.
        depths (tuple[int]): Depths of each Swin Transformer stage.
        num_heads (tuple[int]): Number of attention head of each stage.
        window_size (int): Window size. Default: 7.
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim. Default: 4.
        qkv_bias (bool): If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float): Override default qk scale of head_dim ** -0.5 if set.
        drop_rate (float): Dropout rate.
        attn_drop_rate (float): Attention dropout rate. Default: 0.
        drop_path_rate (float): Stochastic depth rate. Default: 0.2.
        norm_layer (nn.Module): Normalization layer. Default: nn.LayerNorm.
        ape (bool): If True, add absolute position embedding to the patch embedding. Default: False.
        patch_norm (bool): If True, add normalization after patch embedding. Default: True.
        out_indices (Sequence[int]): Output from which stages.
        frozen_stages (int): Stages to be frozen (stop grad and set eval mode).
            -1 means not freezing any parameters.
        use_checkpoint (bool): Whether to use checkpointing to save memory. Default: False.
    """

    def __init__(self,
                 pretrain_img_size=224,
                 patch_size=4,
                 in_chans=1  ,
                 embed_dim=96,
                 depths=[2, 2, 2, 2],
                 num_heads=[4, 8, 16, 32],
                 window_size=7,
                 mlp_ratio=4.,
                 qkv_bias=True,
                 qk_scale=None,
                 drop_rate=0.,
                 attn_drop_rate=0.,
                 drop_path_rate=0.2,
                 norm_layer=nn.LayerNorm,
                 ape=False,
                 patch_norm=True,
                 out_indices=(0, 1, 2, 3),
                 frozen_stages=-1,
                 use_checkpoint=False, gt_num=1, vt_map=(3,5,5),vt_num=1):
        super().__init__()

        self.pretrain_img_size = pretrain_img_size

        self.num_layers = len(depths)
        self.embed_dim = embed_dim
        self.ape = ape
        self.patch_norm = patch_norm
        self.out_indices = out_indices
        self.frozen_stages = frozen_stages

        # split image into non-overlapping patches
        self.patch_embed = PatchEmbed(
            patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dim,
            norm_layer=norm_layer if self.patch_norm else None)

        # absolute position embedding
        if self.ape:
            pretrain_img_size = to_3tuple(pretrain_img_size)
            patch_size = to_3tuple(patch_size)
            # �м���patch
            patches_resolution = [pretrain_img_size[0] // patch_size[0], pretrain_img_size[1] // patch_size[1],
                                  pretrain_img_size[2] // patch_size[2]]

            self.absolute_pos_embed = nn.Parameter(
                torch.zeros(1, embed_dim, patches_resolution[0], patches_resolution[1], patches_resolution[2]))
            trunc_normal_(self.absolute_pos_embed, std=.02)

        self.pos_drop = nn.Dropout(p=drop_rate)

        # stochastic depth
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]  # stochastic depth decay rule

        # build layers
        self.layers = nn.ModuleList()
        for i_layer in range(self.num_layers):
            layer = BasicLayer(
                dim=int(embed_dim * 2 ** i_layer),
                input_resolution=(
                    pretrain_img_size[0] // patch_size[0] // 2 ** i_layer, pretrain_img_size[1] // patch_size[1] // 2 ** i_layer,
                    pretrain_img_size[2] // patch_size[2] // 2 ** i_layer),
                depth=depths[i_layer],
                num_heads=num_heads[i_layer],
                window_size=window_size[i_layer],
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                qk_scale=qk_scale,
                drop=drop_rate,
                attn_drop=attn_drop_rate,
                drop_path=dpr[sum(
                    depths[:i_layer]):sum(depths[:i_layer + 1])],
                norm_layer=norm_layer,
                downsample=PatchMerging,
                use_checkpoint=use_checkpoint, gt_num=gt_num, id_layer=i_layer, vt_map=vt_map,vt_num=vt_num)
            self.layers.append(layer)

        num_features = [int(embed_dim * 2 ** i) for i in range(self.num_layers)]
        self.num_features = num_features


        # add a norm layer for each output
        for i_layer in out_indices:
            layer = norm_layer(num_features[i_layer])
            layer_name = f'norm{i_layer}'
            self.add_module(layer_name, layer)



        self._freeze_stages()

    def _freeze_stages(self):
        if self.frozen_stages >= 0:
            self.patch_embed.eval()
            for param in self.patch_embed.parameters():
                param.requires_grad = False

        if self.frozen_stages >= 1 and self.ape:
            self.absolute_pos_embed.requires_grad = False

        if self.frozen_stages >= 2:
            self.pos_drop.eval()
            for i in range(0, self.frozen_stages - 1):
                m = self.layers[i]
                m.eval()
                for param in m.parameters():
                    param.requires_grad = False

    

    def forward(self, x, vt_pos, check):
        """Forward function."""
        
        x = self.patch_embed(x)
        down=[]
       
        Ws, Wh, Ww = x.size(2), x.size(3), x.size(4)
        if self.ape:
            # interpolate the position embedding to the corresponding size
            absolute_pos_embed = F.interpolate(self.absolute_pos_embed, size=(Ws, Wh, Ww), align_corners=True,
                                               mode='trilinear')
            x = (x + absolute_pos_embed).flatten(2).transpose(1, 2) 
        else:
            x = x.flatten(2).transpose(1, 2)
        x = self.pos_drop(x)
        
      
        for i in range(self.num_layers):
            layer = self.layers[i]
            x_out, S, H, W, x, Ws, Wh, Ww = layer(x, Ws, Wh, Ww, vt_pos, check)
            if i in self.out_indices:
                norm_layer = getattr(self, f'norm{i}')
                x_out = norm_layer(x_out)

                out = x_out.view(-1, S, H, W, self.num_features[i]).permute(0, 4, 1, 2, 3).contiguous()
              
                down.append(out)
        return down

    def train(self, mode=True):
        """Convert the model into training mode while keep layers freezed."""
        super(SwinTransformer, self).train(mode)
        self._freeze_stages()

class encoder(nn.Module):
    def __init__(self,
                 pretrain_img_size,
                 embed_dim,
                 patch_size=4,
                 depths=[2,2,2],
                 num_heads=[24,12,6],
                 window_size=4,
                 mlp_ratio=4.,
                 qkv_bias=True,
                 qk_scale=None,
                 drop_rate=0.,
                 attn_drop_rate=0.,
                 drop_path_rate=0.2,
                 norm_layer=nn.LayerNorm, gt_num=1, vt_map=(3,5,5),vt_num=1
                 ):
        super().__init__()
        

        self.num_layers = len(depths)
        self.pos_drop = nn.Dropout(p=drop_rate)

        # stochastic depth
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]  # stochastic depth decay rule

        # build layers
        self.layers = nn.ModuleList()
        for i_layer in range(self.num_layers)[::-1]:
            
            layer = BasicLayer_up(
                dim=int(embed_dim * 2 ** (len(depths)-i_layer-1)),
                input_resolution=(
                    pretrain_img_size[0] // patch_size[0] // 2 ** (len(depths)-i_layer-1), pretrain_img_size[1] // patch_size[1] // 2 ** (len(depths)-i_layer-1),
                    pretrain_img_size[2] // patch_size[2] // 2 ** (len(depths)-i_layer-1)),
                depth=depths[i_layer],
                num_heads=num_heads[i_layer],
                window_size=window_size[i_layer],
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                qk_scale=qk_scale,
                drop=drop_rate,
                attn_drop=attn_drop_rate,
                drop_path=dpr[sum(
                    depths[:i_layer]):sum(depths[:i_layer + 1])],
                norm_layer=norm_layer,
                upsample=Patch_Expanding, gt_num=gt_num,id_layer=len(depths)-i_layer-1, vt_map=vt_map,vt_num=vt_num
                )
            self.layers.append(layer)
        self.num_features = [int(embed_dim * 2 ** i) for i in range(self.num_layers)]
    def forward(self,x,skips,vt_pos, check):
            
        outs=[]
        S, H, W = x.size(2), x.size(3), x.size(4)
        x = x.flatten(2).transpose(1, 2)
        for index,i in enumerate(skips):
             i = i.flatten(2).transpose(1, 2)
             skips[index]=i
        x = self.pos_drop(x)
            
        for i in range(self.num_layers)[::-1]:
            
            layer = self.layers[i]
            
           
            x, S, H, W,  = layer(x,skips[i], S, H, W, vt_pos, check)
            out = x.view(-1, S, H, W, self.num_features[i])
            outs.append(out)
        return outs










        
class final_patch_expanding(nn.Module):
    def __init__(self,dim,num_class,patch_size):
        super().__init__()
        self.up=nn.ConvTranspose3d(dim,num_class,patch_size,patch_size)

    def forward(self,x):
        x=x.permute(0,4,1,2,3)
        x=self.up(x)
       
        return x    




                                         
class swintransformer(SegmentationNetwork):

    def __init__(self, input_channels=1, base_num_features=64, num_classes=14, num_pool=4, num_conv_per_stage=2,
                 feat_map_mul_on_downscale=2, conv_op=nn.Conv2d,
                 norm_op=nn.BatchNorm2d, norm_op_kwargs=None,
                 dropout_op=nn.Dropout2d, dropout_op_kwargs=None,
                 nonlin=nn.LeakyReLU, nonlin_kwargs=None, deep_supervision=True, dropout_in_localization=False,
                 final_nonlin=softmax_helper, weightInitializer=InitWeights_He(1e-2), pool_op_kernel_sizes=None,
                 conv_kernel_sizes=None,
                 upscale_logits=False, convolutional_pooling=False, convolutional_upsampling=False,
                 max_num_features=None, basic_block=None,
                 seg_output_use_bias=False, gt_num=1, vt_map=(3,5,5), imsize=[64,128,128], dataset='SYNAPSE', vt_num=1, 
                 max_imsize=[218,660,660]):
    
        super(swintransformer, self).__init__()

        if dataset=="SYNAPSE":
            self.imsize=[64,128,128]
            # self.vt_map=(3,5,5)
            # self.max_imsize=SYNAPSE_MAX
            embed_dim=192
            depths=[2, 2, 2, 2]
            num_heads=[6, 12, 24, 48]
            patch_size=[2,4,4]
            window_size=[4,4,4,4]
        elif dataset=="BRAIN_TUMOR":
            self.imsize=[128,128,128]
            # self.vt_map=(2,2,2)
            # self.max_imsize=BRAIN_TUMOR_MAX
            embed_dim=96
            depths=[2, 2, 2, 2]
            num_heads=[3, 6, 12, 24]
            patch_size=[4,4,4]
            window_size=[4,4,8,4]
        
        self.max_imsize = max_imsize
        self._deep_supervision = deep_supervision
        self.do_ds = deep_supervision
        self.num_classes=num_classes
        self.conv_op=conv_op
        self.vt_map = vt_map
       
        
        self.upscale_logits_ops = []
     
        
        self.upscale_logits_ops.append(lambda x: x)
        
        # n_windows = (64x128x128) / (2x4x4)x(4x4x4) = 512
        # embed_dim=192
        # depths=[2, 2, 2, 2]
        # num_heads=[6, 12, 24, 48]
        # patch_size=[2,4,4]
        self.model_down=SwinTransformer(pretrain_img_size=self.imsize,window_size=window_size,embed_dim=embed_dim,patch_size=patch_size,
                                        depths=depths,num_heads=num_heads,in_chans=input_channels, gt_num=gt_num, vt_map=self.vt_map, vt_num=vt_num)
        self.encoder=encoder(pretrain_img_size=self.imsize,embed_dim=embed_dim,window_size=window_size[::-1][1:],patch_size=patch_size,num_heads=[24,12,6],
                            depths=[2,2,2], gt_num=gt_num, vt_map=self.vt_map, vt_num=vt_num)
   
        self.final=[]
        self.final.append(final_patch_expanding(embed_dim*2**0,num_classes,patch_size=patch_size))
        for i in range(1,len(depths)-1):
            self.final.append(final_patch_expanding(embed_dim*2**i,num_classes,patch_size=(4,4,4)))
        self.final=nn.ModuleList(self.final)

        self.vt_check = torch.nn.Parameter(torch.zeros(vt_map[0]*vt_map[1]*vt_map[2],1))
        self.vt_check.requires_grad = False

        self.pos_grid = self.filled_grid()

        self.iter = 0

    def filled_grid(self):
        #h, w = grid.shape
        cd, ch, cw = self.imsize
        d, h, w = self.max_imsize
        grid = np.zeros(self.max_imsize, dtype = int)
        nd, nh, nw = d//cd, h//ch, w//cw

        pd, ph, pw = d%cd, h%ch, w%cw

        for i in range(nd):
            for j in range(nh):
                for k in range(nw):
                    # grid[i*cd:(i+1)*cd, j*ch:(j+1)*ch, k*cw:(k+1)*cw] = nh*nw*i + nw*j + k

                    tmp_pd_0 = (pd//2)*(0**(i==0))
                    tmp_pd_1 = pd//2 + (pd//2)*(0**(i!=nd-1)) + pd%2

                    tmp_ph_0 = (ph//2)*(0**(j==0))
                    tmp_ph_1 = ph//2 + (ph//2)*(0**(j!=nh-1)) + ph%2
                    
                    tmp_pw_0 = (pw//2)*(0**(k==0))
                    tmp_pw_1 = pw//2 + (pw//2)*(0**(k!=nw-1)) + pw%2

                    # grid[i*ch+tmp_ph_0:(i+1)*ch+tmp_ph_1, j*cw+tmp_pw_0:(j+1)*cw+tmp_pw_1] = nw*i + j
                    grid[i*cd+tmp_pd_0:(i+1)*cd+tmp_pd_1, j*ch+tmp_ph_0:(j+1)*ch+tmp_ph_1, k*cw+tmp_pw_0:(k+1)*cw+tmp_pw_1] = nh*nw*i + nw*j + k
       
        return grid

    def border_check(self, pos):
        ret = [i for i in pos]
        size = self.max_imsize
        crop_size = self.imsize
        for i in range(len(ret)):
            #print(pos[i], crop_size[i])
            # if pos[i]%self.imsize[i] == 0:
            #     if pos[i] < self.max_imsize[i]//2:
            #         ret[i] += 1
            #     else:
            #         ret[i] -= 1
            pad_i = (size[i]%crop_size[i])//2
            if (pos[i]%crop_size[i] == 0) or (pos[i] > size[i] - (crop_size[i] + pad_i)) or (pos[i] < (size[i]%crop_size[i])//2):
                if pos[i] < size[i]//2:
                    ret[i] += 1 + pad_i
                else:
                    ret[i] -= 1 + pad_i
        return ret

    def get_tokens_idx(self, pos):
        pos = self.border_check(pos)
        # z, x, y = pos
        # z, x, y = int(z), int(x), int(y)

        # Myr : We put the crop in the bigger image referential
        z, x, y = [int(pos[i] + self.max_imsize[i]//2) for i in range(3)]
        # z, x, y = pos
        print('--> pos', pos)
        print('--> max size', self.max_imsize)
        print('--> z, x, y', z, x, y)


        cd, ch, cw = self.imsize
        tmp = self.pos_grid[z:z+cd, x:x+ch, y:y+cw]
        idx = np.unique(tmp)
        return idx

    def pos2vtpos(self, pos):
        # dim = [64,128,128]
        # max_dim = [218,660,660]
        dim=self.imsize
        max_dim=self.max_imsize

        # Myr : We put the crop in the bigger image referential
        rc_pos = [[p[i] + max_dim[i]//2 for i in range(3)] for p in pos]

        pad = [(max_dim[i]-dim[i]*self.vt_map[i])//2 + 0**((max_dim[i]-dim[i]*self.vt_map[i])%2 == 0) for i in range(3)]

        # get the vt pos
        # vt_pos = [[(rc[i]-pad[i])//dim[i] for i in range(3)] for rc in rc_pos]
        vt_pos = [[(rc[i]-pad[i] - dim[i]//2)//dim[i] for i in range(3)] for rc in rc_pos]


        # deal with borders
        vt_pos = [[vt[i]*(0**(vt[i]<0)) for i in range(3)] for vt in vt_pos]
        vt_pos = [[vt_pos[j][i]*(0**((rc_pos[j][i] - pad[i])>=(self.vt_map[i]*dim[i]))) for i in range(3)] for j in range(len(vt_pos))]

        # int it
        vt_pos = [[int(i) for i in j] for j in vt_pos]

        # vt_pos = [vt[0]*self.vt_map[1]*self.vt_map[2] + vt[1]*self.vt_map[2] + vt[2] for vt in vt_pos]
        # vt_pos = [vt[1]*self.vt_map[2] + vt[2] for vt in vt_pos]
        ret = []
        for vt in vt_pos:
            ##### --> not good find an other way ;)
            # deal with borders
            end_right = False
            end_botom = False
            if vt[1] == self.vt_map[1]-1:
                end_right = True
            if vt[2] == self.vt_map[2]-1:
                end_botom = True


            # 1) add nearest token pos (left )
            p1 = vt[1]*self.vt_map[2] + vt[2]
            ret.append(p1)

            # 2) up right corner
            if end_right:
                p2 = (vt[1]-1)*self.vt_map[2] + vt[2]
            else:
                p2 = (vt[1]+1)*self.vt_map[2] + vt[2]
            ret.append(p2)

            # 3) botom left
            if end_botom:
                p3 = vt[1]*self.vt_map[2] + vt[2]-1
            else:
                p3 = vt[1]*self.vt_map[2] + vt[2]+1
            ret.append(p3)


            # 4) 
            if end_right and end_botom:
                p4 = (vt[1]-1)*self.vt_map[2] + vt[2]-1
            elif end_right:
                p4 = (vt[1]-1)*self.vt_map[2] + vt[2]+1
            elif end_botom:
                p4 = (vt[1]+1)*self.vt_map[2] + vt[2]-1
            else:
                p4 = (vt[1]+1)*self.vt_map[2] + vt[2]+1
            ret.append(p4)


            # vt_pos = [vt[1]*self.vt_map[2] + vt[2] for vt in vt_pos]


        return ret
    
    def forward(self, x, pos):
        print("self.vt_map", self.vt_map)
        print("grid unique", np.unique(self.pos_grid))
        print("pos, max_imsize", pos, self.max_imsize)
        vt_pos = self.pos2vtpos(pos)
        print("pos2vtpos", vt_pos)
        vt_pos = []

        print("self.pos_grid", self.pos_grid.shape)
        for p in pos:
            vt_pos.append(self.get_tokens_idx(p))
        print("get_tokens_idx", vt_pos)
        exit(0)

        
        pr_check = ((self.vt_check >= 1).sum() >= self.vt_map[0]*self.vt_map[1]*self.vt_map[2])
        self.vt_check[vt_pos] += 1
        check = ((self.vt_check >= 1).sum() >= self.vt_map[0]*self.vt_map[1]*self.vt_map[2])

        if (pr_check == False) and check:
            torch.save(self.vt_check, "./log_chech_iter_"+str(self.iter)+".pt")
        self.iter += 1

        # print('|      Stats :')
        # print('| check', check.item())
        # print('| mean', self.vt_check.mean().item())
        # print('| min', self.vt_check.min().item())
        # print('| max', self.vt_check.max().item())
        # print('| n 0', (self.vt_check == 0).sum().item())


            # print(self.vt_check)

        
            
        seg_outputs=[]
        skips = self.model_down(x, vt_pos, self.vt_check >= 1)
        neck=skips[-1]
       
        out=self.encoder(neck,skips,vt_pos, self.vt_check >= 1)
        
        for i in range(len(out)):  
            seg_outputs.append(self.final[-(i+1)](out[i]))

        if self._deep_supervision and self.do_ds:
            return tuple([seg_outputs[-1]] + [i(j) for i, j in
                                              zip(list(self.upscale_logits_ops)[::-1], seg_outputs[:-1][::-1])])
        else:
            
            return seg_outputs[-1]
        
        
        
   

    @staticmethod
    def compute_approx_vram_consumption(patch_size, num_pool_per_axis, base_num_features, max_num_features,
                                        num_modalities, num_classes, pool_op_kernel_sizes, deep_supervision=False,
                                        conv_per_stage=2):
        """
        This only applies for num_conv_per_stage and convolutional_upsampling=True
        not real vram consumption. just a constant term to which the vram consumption will be approx proportional
        (+ offset for parameter storage)
        :param deep_supervision:
        :param patch_size:
        :param num_pool_per_axis:
        :param base_num_features:
        :param max_num_features:
        :param num_modalities:
        :param num_classes:
        :param pool_op_kernel_sizes:
        :return:
        """
        if not isinstance(num_pool_per_axis, np.ndarray):
            num_pool_per_axis = np.array(num_pool_per_axis)

        npool = len(pool_op_kernel_sizes)

        map_size = np.array(patch_size)
        tmp = np.int64((conv_per_stage * 2 + 1) * np.prod(map_size, dtype=np.int64) * base_num_features +
                       num_modalities * np.prod(map_size, dtype=np.int64) +
                       num_classes * np.prod(map_size, dtype=np.int64))

        num_feat = base_num_features

        for p in range(npool):
            for pi in range(len(num_pool_per_axis)):
                map_size[pi] /= pool_op_kernel_sizes[p][pi]
            num_feat = min(num_feat * 2, max_num_features)
            num_blocks = (conv_per_stage * 2 + 1) if p < (npool - 1) else conv_per_stage  # conv_per_stage + conv_per_stage for the convs of encode/decode and 1 for transposed conv
            tmp += num_blocks * np.prod(map_size, dtype=np.int64) * num_feat
            if deep_supervision and p < (npool - 2):
                tmp += np.prod(map_size, dtype=np.int64) * num_classes
            # print(p, map_size, num_feat, tmp)
        return tmp
