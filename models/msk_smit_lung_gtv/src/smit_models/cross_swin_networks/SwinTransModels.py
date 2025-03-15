'''
Swin-Transformer with UNet

Swin-Transformer code retrieved from:
https://github.com/SwinTransformer/Swin-Transformer-Semantic-Segmentation

Original paper:
Liu, Z., Lin, Y., Cao, Y., Hu, H., Wei, Y., Zhang, Z., ... & Guo, B. (2021).
Swin transformer: Hierarchical vision transformer using shifted windows.
arXiv preprint arXiv:2103.14030.

Modified and tested by:
Junyu Chen
jchen245@jhmi.edu
Johns Hopkins University
'''
from typing import Tuple, Union
import torch
import torch.nn as nn
import torch.utils.checkpoint as checkpoint
from timm.models.layers import DropPath, trunc_normal_, to_3tuple
from torch.distributions.normal import Normal
import torch.nn.functional as nnf
import numpy as np
import configs_sw as configs
import sys
from monai.networks.blocks import UnetrBasicBlock, UnetrPrUpBlock, UnetrUpBlock,UnetrBasicBlock_No_DownSampling#,UnetrUpOnlyBlock
class Mlp(nn.Module):
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
    B, H, W, L, C = x.shape
    x = x.view(B, H // window_size[0], window_size[0], W // window_size[1], window_size[1], L // window_size[2], window_size[2], C)

    windows = x.permute(0, 1, 3, 5, 2, 4, 6, 7).contiguous().view(-1, window_size[0], window_size[1], window_size[2], C)
    return windows


def window_reverse(windows, window_size, H, W, L):
    """
    Args:
        windows: (num_windows*B, window_size, window_size, C)
        window_size (int): Window size
        H (int): Height of image
        W (int): Width of image
    Returns:
        x: (B, H, W, C)
    """
    B = int(windows.shape[0] / (H * W * L / window_size[0] / window_size[1] / window_size[2]))
    x = windows.view(B, H // window_size[0], W // window_size[1], L // window_size[2], window_size[0], window_size[1], window_size[2], -1)
    x = x.permute(0, 1, 4, 2, 5, 3, 6, 7).contiguous().view(B, H, W, L, -1)
    return x


class RelativeSinPosEmbed(nn.Module):
    '''
    Rotary Position Embedding
    '''
    def __init__(self,):
        super(RelativeSinPosEmbed, self).__init__()

    def forward(self, attn):
        batch_sz, _, n_patches, emb_dim = attn.shape
        position_ids = torch.arange(0, n_patches).float().cuda()
        indices = torch.arange(0, emb_dim//2).float().cuda()
        indices = torch.pow(10000.0, -2 * indices / emb_dim)
        embeddings = torch.einsum('b,d->bd', position_ids, indices)
        embeddings = torch.stack([torch.sin(embeddings), torch.cos(embeddings)], dim=-1)
        embeddings = torch.reshape(embeddings.view(n_patches, emb_dim), (1, 1, n_patches, emb_dim))
        #embeddings = embeddings.permute(0, 3, 1, 2)
        return embeddings

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

    def __init__(self, dim, window_size, num_heads, qkv_bias=True, qk_scale=None, attn_drop=0., proj_drop=0., pos_embed_method='relative'):

        super().__init__()
        self.dim = dim
        self.window_size = window_size  # Wh, Ww
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        # define a parameter table of relative position bias
        self.relative_position_bias_table = nn.Parameter(
            torch.zeros((2 * window_size[0] - 1) * (2 * window_size[1] - 1) * (2 * window_size[2] - 1), num_heads))  # 2*Wh-1 * 2*Ww-1 * 2*Wt-1, nH

        # get pair-wise relative position index for each token inside the window
        coords_h = torch.arange(self.window_size[0])
        coords_w = torch.arange(self.window_size[1])
        coords_t = torch.arange(self.window_size[2])
        coords = torch.stack(torch.meshgrid([coords_h, coords_w, coords_t]))  # 3, Wh, Ww, Wt
        coords_flatten = torch.flatten(coords, 1)  # 3, Wh*Ww*Wt
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]  # 3, Wh*Ww*Wt, Wh*Ww*Wt
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()  # Wh*Ww*Wt, Wh*Ww*Wt, 3
        relative_coords[:, :, 0] += self.window_size[0] - 1  # shift to start from 0
        relative_coords[:, :, 1] += self.window_size[1] - 1
        relative_coords[:, :, 2] += self.window_size[2] - 1
        relative_coords[:, :, 0] *= (2 * self.window_size[1] - 1) * (2 * self.window_size[2] - 1)
        relative_coords[:, :, 1] *= 2 * self.window_size[2] - 1
        relative_position_index = relative_coords.sum(-1)  # Wh*Ww*Wt, Wh*Ww*Wt
        self.register_buffer("relative_position_index", relative_position_index)

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        self.pos_embed_method = pos_embed_method
        trunc_normal_(self.relative_position_bias_table, std=.02)
        self.softmax = nn.Softmax(dim=-1)
        self.sinposembed = RelativeSinPosEmbed()

    def forward(self, x, mask=None):
        """ Forward function.
        Args:
            x: input features with shape of (num_windows*B, N, C)
            mask: (0/-inf) mask with shape of (num_windows, Wh*Ww, Wh*Ww) or None
        """
        B_, N, C = x.shape
        qkv = self.qkv(x).reshape(B_, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]  # make torchscript happy (cannot use tensor as tuple)

        q = q * self.scale
        if self.pos_embed_method == 'rotary':
            pos_embed = self.sinposembed(q)
            cos_pos = pos_embed[..., 1::2].repeat(1, 1, 1, 2).cuda()
            sin_pos = pos_embed[..., ::2].repeat(1, 1, 1, 2).cuda()
            qw2 = torch.stack([-q[..., 1::2], q[..., ::2]], 4)
            qw2 = torch.reshape(qw2, q.shape)
            q = q * cos_pos + qw2 * sin_pos
            kw2 = torch.stack([-k[..., 1::2], k[..., ::2]], 4)
            kw2 = torch.reshape(kw2, k.shape)
            k = k * cos_pos + kw2 * sin_pos

        attn = (q @ k.transpose(-2, -1))
        if self.pos_embed_method == 'relative':
            relative_position_bias = self.relative_position_bias_table[self.relative_position_index.view(-1)].view(
                self.window_size[0] * self.window_size[1] * self.window_size[2],
                self.window_size[0] * self.window_size[1] * self.window_size[2], -1)  # Wh*Ww*Wt,Wh*Ww*Wt,nH
            relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()  # nH, Wh*Ww*Wt, Wh*Ww*Wt
            attn = attn + relative_position_bias.unsqueeze(0)

        if mask is not None:
            nW = mask.shape[0]
            attn = attn.view(B_ // nW, nW, self.num_heads, N, N) + mask.unsqueeze(1).unsqueeze(0)
            attn = attn.view(-1, self.num_heads, N, N)
            attn = self.softmax(attn)
        else:
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



class WindowAttention_crossModality(nn.Module):
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

    def __init__(self, dim, window_size, num_heads, qkv_bias=True, qk_scale=None, attn_drop=0., proj_drop=0., pos_embed_method='relative'):

        super().__init__()
        self.dim = dim
        self.window_size = window_size  # Wh, Ww
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        # define a parameter table of relative position bias
        self.relative_position_bias_table = nn.Parameter(
            torch.zeros((2 * window_size[0] - 1) * (2 * window_size[1] - 1) * (2 * window_size[2] - 1), num_heads))  # 2*Wh-1 * 2*Ww-1 * 2*Wt-1, nH

        # get pair-wise relative position index for each token inside the window
        coords_h = torch.arange(self.window_size[0])
        coords_w = torch.arange(self.window_size[1])
        coords_t = torch.arange(self.window_size[2])
        coords = torch.stack(torch.meshgrid([coords_h, coords_w, coords_t]))  # 3, Wh, Ww, Wt
        coords_flatten = torch.flatten(coords, 1)  # 3, Wh*Ww*Wt
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]  # 3, Wh*Ww*Wt, Wh*Ww*Wt
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()  # Wh*Ww*Wt, Wh*Ww*Wt, 3
        relative_coords[:, :, 0] += self.window_size[0] - 1  # shift to start from 0
        relative_coords[:, :, 1] += self.window_size[1] - 1
        relative_coords[:, :, 2] += self.window_size[2] - 1
        relative_coords[:, :, 0] *= (2 * self.window_size[1] - 1) * (2 * self.window_size[2] - 1)
        relative_coords[:, :, 1] *= 2 * self.window_size[2] - 1
        relative_position_index = relative_coords.sum(-1)  # Wh*Ww*Wt, Wh*Ww*Wt
        self.register_buffer("relative_position_index", relative_position_index)

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        self.pos_embed_method = pos_embed_method
        trunc_normal_(self.relative_position_bias_table, std=.02)
        self.softmax = nn.Softmax(dim=-1)
        self.sinposembed = RelativeSinPosEmbed()

    def forward(self, x, x_1, mask=None):
        """ Forward function.
        Args:
            x: input features with shape of (num_windows*B, N, C)
            mask: (0/-inf) mask with shape of (num_windows, Wh*Ww, Wh*Ww) or None
        """
        B_, N, C = x.shape
        qkv = self.qkv(x).reshape(B_, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        qkv_mod1 = self.qkv(x_1).reshape(B_, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)

        q, k, v = qkv[0], qkv_mod1[1], qkv_mod1[2]  # make torchscript happy (cannot use tensor as tuple)
        q_mod1, k_mod1, v_mod1 = qkv_mod1[0], qkv[1], qkv[2]  # make torchscript happy (cannot use tensor as tuple)

        q = q * self.scale
        q_mod1 = q_mod1 * self.scale
        if self.pos_embed_method == 'rotary':
            pos_embed = self.sinposembed(q)
            cos_pos = pos_embed[..., 1::2].repeat(1, 1, 1, 2).cuda()
            sin_pos = pos_embed[..., ::2].repeat(1, 1, 1, 2).cuda()
            qw2 = torch.stack([-q[..., 1::2], q[..., ::2]], 4)
            qw2 = torch.reshape(qw2, q.shape)
            q = q * cos_pos + qw2 * sin_pos
            kw2 = torch.stack([-k[..., 1::2], k[..., ::2]], 4)
            kw2 = torch.reshape(kw2, k.shape)
            k = k * cos_pos + kw2 * sin_pos

        attn = (q @ k.transpose(-2, -1))
        attn_mod1 = (q_mod1 @ k_mod1.transpose(-2, -1))

        if self.pos_embed_method == 'relative':
            relative_position_bias = self.relative_position_bias_table[self.relative_position_index.view(-1)].view(
                self.window_size[0] * self.window_size[1] * self.window_size[2],
                self.window_size[0] * self.window_size[1] * self.window_size[2], -1)  # Wh*Ww*Wt,Wh*Ww*Wt,nH
            relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()  # nH, Wh*Ww*Wt, Wh*Ww*Wt
            attn = attn + relative_position_bias.unsqueeze(0)
            attn_mod1 = attn_mod1 + relative_position_bias.unsqueeze(0)


        if mask is not None:
            nW = mask.shape[0]
            attn = attn.view(B_ // nW, nW, self.num_heads, N, N) + mask.unsqueeze(1).unsqueeze(0)
            attn = attn.view(-1, self.num_heads, N, N)
            attn = self.softmax(attn)
            attn_mod1 = attn_mod1.view(B_ // nW, nW, self.num_heads, N, N) + mask.unsqueeze(1).unsqueeze(0)
            attn_mod1 = attn_mod1.view(-1, self.num_heads, N, N)
            attn_mod1 = self.softmax(attn_mod1)
        else:
            attn = self.softmax(attn)
            attn_mod1 = self.softmax(attn_mod1)


        attn = self.attn_drop(attn)
        attn_mod1 = self.attn_drop(attn_mod1)


        x = (attn @ v).transpose(1, 2).reshape(B_, N, C) #mod 1 dot mod2 * mod2
        x_1 = (attn_mod1 @ v_mod1).transpose(1, 2).reshape(B_, N, C) #mod 2 dot mod 1 * mod1

        x = self.proj(x)
        x_1 = self.proj(x_1)

        x = self.proj_drop(x)
        x_1 = self.proj_drop(x_1)

        return x,x_1

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


class WindowAttention_crossModality_4attns(nn.Module):
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

    def __init__(self, dim, window_size, num_heads, qkv_bias=True, qk_scale=None, attn_drop=0., proj_drop=0., pos_embed_method='relative'):

        super().__init__()
        self.dim = dim
        self.window_size = window_size  # Wh, Ww
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        # define a parameter table of relative position bias
        self.relative_position_bias_table = nn.Parameter(
            torch.zeros((2 * window_size[0] - 1) * (2 * window_size[1] - 1) * (2 * window_size[2] - 1), num_heads))  # 2*Wh-1 * 2*Ww-1 * 2*Wt-1, nH

        # get pair-wise relative position index for each token inside the window
        coords_h = torch.arange(self.window_size[0])
        coords_w = torch.arange(self.window_size[1])
        coords_t = torch.arange(self.window_size[2])
        coords = torch.stack(torch.meshgrid([coords_h, coords_w, coords_t]))  # 3, Wh, Ww, Wt
        coords_flatten = torch.flatten(coords, 1)  # 3, Wh*Ww*Wt
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]  # 3, Wh*Ww*Wt, Wh*Ww*Wt
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()  # Wh*Ww*Wt, Wh*Ww*Wt, 3
        relative_coords[:, :, 0] += self.window_size[0] - 1  # shift to start from 0
        relative_coords[:, :, 1] += self.window_size[1] - 1
        relative_coords[:, :, 2] += self.window_size[2] - 1
        relative_coords[:, :, 0] *= (2 * self.window_size[1] - 1) * (2 * self.window_size[2] - 1)
        relative_coords[:, :, 1] *= 2 * self.window_size[2] - 1
        relative_position_index = relative_coords.sum(-1)  # Wh*Ww*Wt, Wh*Ww*Wt
        self.register_buffer("relative_position_index", relative_position_index)

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        self.pos_embed_method = pos_embed_method
        trunc_normal_(self.relative_position_bias_table, std=.02)
        self.softmax = nn.Softmax(dim=-1)
        self.sinposembed = RelativeSinPosEmbed()

    def forward(self, x, x_1, mask=None):
        """ Forward function.
        Args:
            x: input features with shape of (num_windows*B, N, C)
            mask: (0/-inf) mask with shape of (num_windows, Wh*Ww, Wh*Ww) or None
        """
        B_, N, C = x.shape
        qkv = self.qkv(x).reshape(B_, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        qkv_mod1 = self.qkv(x_1).reshape(B_, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)

        q, k, v = qkv[0], qkv_mod1[1], qkv_mod1[2]  # make torchscript happy (cannot use tensor as tuple)
        q_mod1, k_mod1, v_mod1 = qkv_mod1[0], qkv[1], qkv[2]  # make torchscript happy (cannot use tensor as tuple)

        q = q * self.scale
        q_mod1 = q_mod1 * self.scale
        if self.pos_embed_method == 'rotary':
            pos_embed = self.sinposembed(q)
            cos_pos = pos_embed[..., 1::2].repeat(1, 1, 1, 2).cuda()
            sin_pos = pos_embed[..., ::2].repeat(1, 1, 1, 2).cuda()
            qw2 = torch.stack([-q[..., 1::2], q[..., ::2]], 4)
            qw2 = torch.reshape(qw2, q.shape)
            q = q * cos_pos + qw2 * sin_pos
            kw2 = torch.stack([-k[..., 1::2], k[..., ::2]], 4)
            kw2 = torch.reshape(kw2, k.shape)
            k = k * cos_pos + kw2 * sin_pos

        attn = (q @ k.transpose(-2, -1))
        attn_mod1 = (q_mod1 @ k_mod1.transpose(-2, -1))

        #self_attn
        attn_self = (q @ k_mod1.transpose(-2, -1))
        attn_self_mod1 = (q_mod1 @ k.transpose(-2, -1))

        if self.pos_embed_method == 'relative':
            relative_position_bias = self.relative_position_bias_table[self.relative_position_index.view(-1)].view(
                self.window_size[0] * self.window_size[1] * self.window_size[2],
                self.window_size[0] * self.window_size[1] * self.window_size[2], -1)  # Wh*Ww*Wt,Wh*Ww*Wt,nH
            relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()  # nH, Wh*Ww*Wt, Wh*Ww*Wt
            attn = attn + relative_position_bias.unsqueeze(0)
            attn_mod1 = attn_mod1 + relative_position_bias.unsqueeze(0)
            # self_attn
            attn_self = attn_self + relative_position_bias.unsqueeze(0)
            attn_self_mod1 = attn_self_mod1 + relative_position_bias.unsqueeze(0)


        if mask is not None:
            nW = mask.shape[0]
            attn = attn.view(B_ // nW, nW, self.num_heads, N, N) + mask.unsqueeze(1).unsqueeze(0)
            attn = attn.view(-1, self.num_heads, N, N)
            attn = self.softmax(attn)
            attn_mod1 = attn_mod1.view(B_ // nW, nW, self.num_heads, N, N) + mask.unsqueeze(1).unsqueeze(0)
            attn_mod1 = attn_mod1.view(-1, self.num_heads, N, N)
            attn_mod1 = self.softmax(attn_mod1)
            # self_attn
            attn_self = attn_self.view(B_ // nW, nW, self.num_heads, N, N) + mask.unsqueeze(1).unsqueeze(0)
            attn_self = attn_self.view(-1, self.num_heads, N, N)
            attn_self = self.softmax(attn_self)
            attn_self_mod1 = attn_self_mod1.view(B_ // nW, nW, self.num_heads, N, N) + mask.unsqueeze(1).unsqueeze(0)
            attn_self_mod1 = attn_self_mod1.view(-1, self.num_heads, N, N)
            attn_self_mod1 = self.softmax(attn_self_mod1)


        else:
            attn = self.softmax(attn)
            attn_mod1 = self.softmax(attn_mod1)
            # self_attn
            attn_self = self.softmax(attn_self)
            attn_self_mod1 = self.softmax(attn_self_mod1)


        attn = self.attn_drop(attn)
        attn_mod1 = self.attn_drop(attn_mod1)
        # self_attn
        attn_self = self.attn_drop(attn_self)
        attn_self_mod1 = self.attn_drop(attn_self_mod1)

        x = (attn @ v).transpose(1, 2).reshape(B_, N, C)
        x_1 = (attn_mod1 @ v_mod1).transpose(1, 2).reshape(B_, N, C)
        # self_attn
        x_self = (attn_self @ v_mod1).transpose(1, 2).reshape(B_, N, C)
        x_1_self = (attn_self_mod1 @ v).transpose(1, 2).reshape(B_, N, C)

        x = self.proj(x)
        x_1 = self.proj(x_1)
        # self_attn
        x_self = self.proj(x_self)
        x_1_self = self.proj(x_1_self)

        x = self.proj_drop(x)
        x_1 = self.proj_drop(x_1)
        # self_attn
        x_self = self.proj_drop(x_self)
        x_1_self = self.proj_drop(x_1_self)

        return x+x_self,x_1+x_1_self

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


class WindowAttention_dualModality(nn.Module):
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

    def __init__(self, dim, window_size, num_heads, qkv_bias=True, qk_scale=None, attn_drop=0., proj_drop=0., pos_embed_method='relative'):

        super().__init__()
        self.dim = dim
        self.window_size = window_size  # Wh, Ww
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        # define a parameter table of relative position bias
        self.relative_position_bias_table = nn.Parameter(
            torch.zeros((2 * window_size[0] - 1) * (2 * window_size[1] - 1) * (2 * window_size[2] - 1), num_heads))  # 2*Wh-1 * 2*Ww-1 * 2*Wt-1, nH

        # get pair-wise relative position index for each token inside the window
        coords_h = torch.arange(self.window_size[0])
        coords_w = torch.arange(self.window_size[1])
        coords_t = torch.arange(self.window_size[2])
        coords = torch.stack(torch.meshgrid([coords_h, coords_w, coords_t]))  # 3, Wh, Ww, Wt
        coords_flatten = torch.flatten(coords, 1)  # 3, Wh*Ww*Wt
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]  # 3, Wh*Ww*Wt, Wh*Ww*Wt
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()  # Wh*Ww*Wt, Wh*Ww*Wt, 3
        relative_coords[:, :, 0] += self.window_size[0] - 1  # shift to start from 0
        relative_coords[:, :, 1] += self.window_size[1] - 1
        relative_coords[:, :, 2] += self.window_size[2] - 1
        relative_coords[:, :, 0] *= (2 * self.window_size[1] - 1) * (2 * self.window_size[2] - 1)
        relative_coords[:, :, 1] *= 2 * self.window_size[2] - 1
        relative_position_index = relative_coords.sum(-1)  # Wh*Ww*Wt, Wh*Ww*Wt
        self.register_buffer("relative_position_index", relative_position_index)

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        self.pos_embed_method = pos_embed_method
        trunc_normal_(self.relative_position_bias_table, std=.02)
        self.softmax = nn.Softmax(dim=-1)
        self.sinposembed = RelativeSinPosEmbed()

    def forward(self, x, x_1, mask=None):
        """ Forward function.
        Args:
            x: input features with shape of (num_windows*B, N, C)
            mask: (0/-inf) mask with shape of (num_windows, Wh*Ww, Wh*Ww) or None
        """
        B_, N, C = x.shape
        qkv = self.qkv(x).reshape(B_, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        qkv_mod1 = self.qkv(x_1).reshape(B_, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)

        q, k, v = qkv[0], qkv[1], qkv[2]  # make torchscript happy (cannot use tensor as tuple)
        q_mod1, k_mod1, v_mod1 = qkv_mod1[0], qkv_mod1[1], qkv_mod1[2]  # make torchscript happy (cannot use tensor as tuple)

        q = q * self.scale
        if self.pos_embed_method == 'rotary':
            pos_embed = self.sinposembed(q)
            cos_pos = pos_embed[..., 1::2].repeat(1, 1, 1, 2).cuda()
            sin_pos = pos_embed[..., ::2].repeat(1, 1, 1, 2).cuda()
            qw2 = torch.stack([-q[..., 1::2], q[..., ::2]], 4)
            qw2 = torch.reshape(qw2, q.shape)
            q = q * cos_pos + qw2 * sin_pos
            kw2 = torch.stack([-k[..., 1::2], k[..., ::2]], 4)
            kw2 = torch.reshape(kw2, k.shape)
            k = k * cos_pos + kw2 * sin_pos

        attn = (q @ k.transpose(-2, -1))
        if self.pos_embed_method == 'relative':
            relative_position_bias = self.relative_position_bias_table[self.relative_position_index.view(-1)].view(
                self.window_size[0] * self.window_size[1] * self.window_size[2],
                self.window_size[0] * self.window_size[1] * self.window_size[2], -1)  # Wh*Ww*Wt,Wh*Ww*Wt,nH
            relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()  # nH, Wh*Ww*Wt, Wh*Ww*Wt
            attn = attn + relative_position_bias.unsqueeze(0)

        if mask is not None:
            nW = mask.shape[0]
            attn = attn.view(B_ // nW, nW, self.num_heads, N, N) + mask.unsqueeze(1).unsqueeze(0)
            attn = attn.view(-1, self.num_heads, N, N)
            attn = self.softmax(attn)
        else:
            attn = self.softmax(attn)

        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B_, N, C)
        x_1 = (attn @ v_mod1).transpose(1, 2).reshape(B_, N, C)

        x = self.proj(x)
        x_1 = self.proj(x_1)

        x = self.proj_drop(x)
        x_1 = self.proj_drop(x_1)

        return x,x_1

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


class SwinTransformerBlock(nn.Module):
    r""" Swin Transformer Block.
    Args:
        dim (int): Number of input channels.
        input_resolution (tuple[int]): Input resulotion.
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

    def __init__(self, dim, num_heads, window_size=(7, 7, 7), shift_size=(0, 0, 0),
                 mlp_ratio=4., qkv_bias=True, qk_scale=None, drop=0., attn_drop=0., drop_path=0.,
                 act_layer=nn.GELU, norm_layer=nn.LayerNorm,  pos_embed_method='relative', concatenated_input=True):
        super().__init__()
        if concatenated_input:
            self.dim = dim *2
        else:
            self.dim = dim
        self.num_heads = num_heads
        self.window_size = window_size
        self.shift_size = shift_size
        self.mlp_ratio = mlp_ratio
        assert 0 <= min(self.shift_size) < min(self.window_size), "shift_size must in 0-window_size"

        self.norm1 = norm_layer(self.dim)
        self.attn = WindowAttention(
            self.dim, window_size=self.window_size, num_heads=num_heads,
            qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop, pos_embed_method=pos_embed_method)

        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(self.dim)
        mlp_hidden_dim = int(self.dim * mlp_ratio)
        self.mlp = Mlp(in_features=self.dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

        self.H = None
        self.W = None
        self.T = None


    def forward(self, x, mask_matrix):
        H, W, T = self.H, self.W, self.T
        B, L, C = x.shape
        #C = C * 2
        assert L == H * W * T, "input feature has wrong size"

        shortcut = x
        x = self.norm1(x)
        x = x.view(B, H, W, T, C)

        # pad feature maps to multiples of window size
        pad_l = pad_t = pad_f = 0
        pad_r = (self.window_size[0] - H % self.window_size[0]) % self.window_size[0]
        pad_b = (self.window_size[1] - W % self.window_size[1]) % self.window_size[1]
        pad_h = (self.window_size[2] - T % self.window_size[2]) % self.window_size[2]
        x = nnf.pad(x, (0, 0, pad_l, pad_r, pad_t, pad_b, pad_f, pad_h))
        _, Hp, Wp, Tp, _ = x.shape

        # cyclic shift
        if min(self.shift_size) > 0:
            shifted_x = torch.roll(x, shifts=(-self.shift_size[0], -self.shift_size[1], -self.shift_size[2]), dims=(1, 2, 3))
            attn_mask = mask_matrix
        else:
            shifted_x = x
            attn_mask = None

        # partition windows
        x_windows = window_partition(shifted_x, self.window_size)  # nW*B, window_size, window_size, C
        x_windows = x_windows.view(-1, self.window_size[0] * self.window_size[1] * self.window_size[2], C)  # nW*B, window_size*window_size, C

        # W-MSA/SW-MSA
        attn_windows = self.attn(x_windows, mask=attn_mask)  # nW*B, window_size*window_size, C

        # merge windows
        attn_windows = attn_windows.view(-1, self.window_size[0], self.window_size[1], self.window_size[2], C)
        shifted_x = window_reverse(attn_windows, self.window_size, Hp, Wp, Tp)  # B H' W' C

        # reverse cyclic shift
        if min(self.shift_size) > 0:
            x = torch.roll(shifted_x, shifts=(self.shift_size[0], self.shift_size[1], self.shift_size[2]), dims=(1, 2, 3))
        else:
            x = shifted_x

        if pad_r > 0 or pad_b > 0:
            x = x[:, :H, :W, :T, :].contiguous()

        x = x.view(B, H * W * T, C)

        # FFN
        x = shortcut + self.drop_path(x)
        x = x + self.drop_path(self.mlp(self.norm2(x)))

        return x

class SwinTransformerBlock_dualModality(nn.Module):
    r""" Swin Transformer Block.
    Args:
        dim (int): Number of input channels.
        input_resolution (tuple[int]): Input resulotion.
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

    def __init__(self, dim, num_heads, window_size=(7, 7, 7), shift_size=(0, 0, 0),
                 mlp_ratio=4., qkv_bias=True, qk_scale=None, drop=0., attn_drop=0., drop_path=0.,
                 act_layer=nn.GELU, norm_layer=nn.LayerNorm,  pos_embed_method='relative', concatenated_input=True):
        super().__init__()
        if concatenated_input:
            self.dim = dim *2
        else:
            self.dim = dim
        self.num_heads = num_heads
        self.window_size = window_size
        self.shift_size = shift_size
        self.mlp_ratio = mlp_ratio
        assert 0 <= min(self.shift_size) < min(self.window_size), "shift_size must in 0-window_size"

        self.norm1 = norm_layer(self.dim)
        self.attn = WindowAttention_dualModality(
            self.dim, window_size=self.window_size, num_heads=num_heads,
            qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop, pos_embed_method=pos_embed_method)

        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(self.dim)
        mlp_hidden_dim = int(self.dim * mlp_ratio)
        self.mlp = Mlp(in_features=self.dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

        self.H = None
        self.W = None
        self.T = None


    def forward(self, x,x_1, mask_matrix):
        H, W, T = self.H, self.W, self.T
        B, L, C = x.shape
        #C = C * 2
        assert L == H * W * T, "input feature has wrong size"

        shortcut = x
        x = self.norm1(x)
        x_1 = self.norm1(x_1)

        x = x.view(B, H, W, T, C)
        x_1 = x_1.view(B, H, W, T, C)



        # pad feature maps to multiples of window size
        pad_l = pad_t = pad_f = 0
        pad_r = (self.window_size[0] - H % self.window_size[0]) % self.window_size[0]
        pad_b = (self.window_size[1] - W % self.window_size[1]) % self.window_size[1]
        pad_h = (self.window_size[2] - T % self.window_size[2]) % self.window_size[2]
        x = nnf.pad(x, (0, 0, pad_l, pad_r, pad_t, pad_b, pad_f, pad_h))
        x_1 = nnf.pad(x_1, (0, 0, pad_l, pad_r, pad_t, pad_b, pad_f, pad_h))

        _, Hp, Wp, Tp, _ = x.shape

        # cyclic shift
        if min(self.shift_size) > 0:
            shifted_x = torch.roll(x, shifts=(-self.shift_size[0], -self.shift_size[1], -self.shift_size[2]), dims=(1, 2, 3))
            shifted_x_1 = torch.roll(x_1, shifts=(-self.shift_size[0], -self.shift_size[1], -self.shift_size[2]), dims=(1, 2, 3))
            attn_mask = mask_matrix
        else:
            shifted_x = x
            shifted_x_1 = x_1
            attn_mask = None

        # partition windows
        x_windows = window_partition(shifted_x, self.window_size)  # nW*B, window_size, window_size, C
        x_windows = x_windows.view(-1, self.window_size[0] * self.window_size[1] * self.window_size[2], C)  # nW*B, window_size*window_size, C

        x_1_windows = window_partition(shifted_x_1, self.window_size)  # nW*B, window_size, window_size, C
        x_1_windows = x_1_windows.view(-1, self.window_size[0] * self.window_size[1] * self.window_size[2],
                                   C)  # nW*B, window_size*window_size, C


        # W-MSA/SW-MSA
        attn_windows,attn_windows_x_1 = self.attn(x_windows,x_1_windows, mask=attn_mask)  # nW*B, window_size*window_size, C

        # merge windows
        attn_windows = attn_windows.view(-1, self.window_size[0], self.window_size[1], self.window_size[2], C)
        shifted_x = window_reverse(attn_windows, self.window_size, Hp, Wp, Tp)  # B H' W' C
        attn_windows_x_1 = attn_windows_x_1.view(-1, self.window_size[0], self.window_size[1], self.window_size[2], C)
        shifted_x_1 = window_reverse(attn_windows_x_1, self.window_size, Hp, Wp, Tp)  # B H' W' C
        # reverse cyclic shift
        if min(self.shift_size) > 0:
            x = torch.roll(shifted_x, shifts=(self.shift_size[0], self.shift_size[1], self.shift_size[2]), dims=(1, 2, 3))
            x_1 = torch.roll(shifted_x_1, shifts=(self.shift_size[0], self.shift_size[1], self.shift_size[2]), dims=(1, 2, 3))

        else:
            x = shifted_x
            x_1 = shifted_x_1


        if pad_r > 0 or pad_b > 0:
            x = x[:, :H, :W, :T, :].contiguous()
            x_1 = x_1[:, :H, :W, :T, :].contiguous()


        x = x.view(B, H * W * T, C)
        x_1 = x_1.view(B, H * W * T, C)


        # FFN
        x = shortcut + self.drop_path(x)
        x_1 = shortcut + self.drop_path(x_1)

        x = x + self.drop_path(self.mlp(self.norm2(x)))
        x_1 = x_1 + self.drop_path(self.mlp(self.norm2(x_1)))

        return x,x_1



class SwinTransformerBlock_crossModality(nn.Module):
    r""" Swin Transformer Block.
    Args:
        dim (int): Number of input channels.
        input_resolution (tuple[int]): Input resulotion.
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

    def __init__(self, dim, num_heads, window_size=(7, 7, 7), shift_size=(0, 0, 0),
                 mlp_ratio=4., qkv_bias=True, qk_scale=None, drop=0., attn_drop=0., drop_path=0.,
                 act_layer=nn.GELU, norm_layer=nn.LayerNorm,  pos_embed_method='relative', concatenated_input=True):
        super().__init__()
        if concatenated_input:
            self.dim = dim *2
        else:
            self.dim = dim
        self.num_heads = num_heads
        self.window_size = window_size
        self.shift_size = shift_size
        self.mlp_ratio = mlp_ratio
        assert 0 <= min(self.shift_size) < min(self.window_size), "shift_size must in 0-window_size"

        self.norm1 = norm_layer(self.dim)
        self.attn = WindowAttention_crossModality(
            self.dim, window_size=self.window_size, num_heads=num_heads,
            qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop, pos_embed_method=pos_embed_method)

        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(self.dim)
        mlp_hidden_dim = int(self.dim * mlp_ratio)
        self.mlp = Mlp(in_features=self.dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

        self.H = None
        self.W = None
        self.T = None


    def forward(self, x,x_1, mask_matrix):
        H, W, T = self.H, self.W, self.T
        B, L, C = x.shape
        #C = C * 2
        assert L == H * W * T, "input feature has wrong size"

        shortcut = x
        x = self.norm1(x)
        x_1 = self.norm1(x_1)

        x = x.view(B, H, W, T, C)
        x_1 = x_1.view(B, H, W, T, C)



        # pad feature maps to multiples of window size
        pad_l = pad_t = pad_f = 0
        pad_r = (self.window_size[0] - H % self.window_size[0]) % self.window_size[0]
        pad_b = (self.window_size[1] - W % self.window_size[1]) % self.window_size[1]
        pad_h = (self.window_size[2] - T % self.window_size[2]) % self.window_size[2]
        x = nnf.pad(x, (0, 0, pad_l, pad_r, pad_t, pad_b, pad_f, pad_h))
        x_1 = nnf.pad(x_1, (0, 0, pad_l, pad_r, pad_t, pad_b, pad_f, pad_h))

        _, Hp, Wp, Tp, _ = x.shape

        # cyclic shift
        if min(self.shift_size) > 0:
            shifted_x = torch.roll(x, shifts=(-self.shift_size[0], -self.shift_size[1], -self.shift_size[2]), dims=(1, 2, 3))
            shifted_x_1 = torch.roll(x_1, shifts=(-self.shift_size[0], -self.shift_size[1], -self.shift_size[2]), dims=(1, 2, 3))
            attn_mask = mask_matrix
        else:
            shifted_x = x
            shifted_x_1 = x_1
            attn_mask = None

        # partition windows
        x_windows = window_partition(shifted_x, self.window_size)  # nW*B, window_size, window_size, C
        x_windows = x_windows.view(-1, self.window_size[0] * self.window_size[1] * self.window_size[2], C)  # nW*B, window_size*window_size, C

        x_1_windows = window_partition(shifted_x_1, self.window_size)  # nW*B, window_size, window_size, C
        x_1_windows = x_1_windows.view(-1, self.window_size[0] * self.window_size[1] * self.window_size[2],
                                   C)  # nW*B, window_size*window_size, C


        # W-MSA/SW-MSA
        attn_windows,attn_windows_x_1 = self.attn(x_windows,x_1_windows, mask=attn_mask)  # nW*B, window_size*window_size, C

        # merge windows
        attn_windows = attn_windows.view(-1, self.window_size[0], self.window_size[1], self.window_size[2], C)
        shifted_x = window_reverse(attn_windows, self.window_size, Hp, Wp, Tp)  # B H' W' C
        attn_windows_x_1 = attn_windows_x_1.view(-1, self.window_size[0], self.window_size[1], self.window_size[2], C)
        shifted_x_1 = window_reverse(attn_windows_x_1, self.window_size, Hp, Wp, Tp)  # B H' W' C
        # reverse cyclic shift
        if min(self.shift_size) > 0:
            x = torch.roll(shifted_x, shifts=(self.shift_size[0], self.shift_size[1], self.shift_size[2]), dims=(1, 2, 3))
            x_1 = torch.roll(shifted_x_1, shifts=(self.shift_size[0], self.shift_size[1], self.shift_size[2]), dims=(1, 2, 3))

        else:
            x = shifted_x
            x_1 = shifted_x_1


        if pad_r > 0 or pad_b > 0:
            x = x[:, :H, :W, :T, :].contiguous()
            x_1 = x_1[:, :H, :W, :T, :].contiguous()


        x = x.view(B, H * W * T, C)
        x_1 = x_1.view(B, H * W * T, C)


        # FFN
        x = shortcut + self.drop_path(x)
        x_1 = shortcut + self.drop_path(x_1)

        x = x + self.drop_path(self.mlp(self.norm2(x)))
        x_1 = x_1 + self.drop_path(self.mlp(self.norm2(x_1)))

        return x,x_1

class PatchMerging(nn.Module):
    r""" Patch Merging Layer.
    Args:
        input_resolution (tuple[int]): Resolution of input feature.
        dim (int): Number of input channels.
        norm_layer (nn.Module, optional): Normalization layer.  Default: nn.LayerNorm
    """

    def __init__(self, dim, norm_layer=nn.LayerNorm, reduce_factor=2, concatenated_input=False):
        super().__init__()
        if concatenated_input:
            self.dim = dim * 2
        else:
            self.dim = dim
        self.reduction = nn.Linear(8 * self.dim, (8//reduce_factor) * self.dim, bias=False)
        self.norm = norm_layer(8 * self.dim)


    def forward(self, x, H, W, T):
        """
        x: B, H*W, C
        """
        B, L, C = x.shape
        assert L == H * W * T, "input feature has wrong size"
        assert H % 2 == 0 and W % 2 == 0 and T % 2 == 0, f"x size ({H}*{W}) are not even."

        x = x.view(B, H, W, T, C)

        # padding
        pad_input = (H % 2 == 1) or (W % 2 == 1) or (T % 2 == 1)
        if pad_input:
            x = nnf.pad(x, (0, 0, 0, T % 2, 0, W % 2, 0, H % 2))

        x0 = x[:, 0::2, 0::2, 0::2, :]  # B H/2 W/2 C
        x1 = x[:, 1::2, 0::2, 0::2, :]  # B H/2 W/2 C
        x2 = x[:, 0::2, 1::2, 0::2, :]  # B H/2 W/2 C
        x3 = x[:, 0::2, 0::2, 1::2, :]  # B H/2 W/2 C
        x4 = x[:, 1::2, 1::2, 0::2, :]  # B H/2 W/2 C
        x5 = x[:, 0::2, 1::2, 1::2, :]  # B H/2 W/2 C
        x6 = x[:, 1::2, 0::2, 1::2, :]  # B H/2 W/2 C
        x7 = x[:, 1::2, 1::2, 1::2, :]  # B H/2 W/2 C
        x = torch.cat([x0, x1, x2, x3, x4, x5, x6, x7], -1)  # B H/2 W/2 T/2 8*C
        x = x.view(B, -1, 8 * C)  # B H/2*W/2*T/2 8*C

        x = self.norm(x)
        x = self.reduction(x)

        return x

class PatchConvPool(nn.Module):
    r""" Patch Merging Layer.
    Args:
        input_resolution (tuple[int]): Resolution of input feature.
        dim (int): Number of input channels.
        norm_layer (nn.Module, optional): Normalization layer.  Default: nn.LayerNorm
    """

    def __init__(self, dim, norm_layer=nn.LayerNorm, reduce_factor=2, concatenated_input=False):
        super().__init__()
        if concatenated_input:
            self.dim = dim * 2
        else:
            self.dim = dim
        #self.reduction = nn.Linear(8 * self.dim, (8//reduce_factor) * self.dim, bias=False)
        #self.norm = norm_layer(8 * self.dim)

        self.conv_du = nn.Sequential(
            nn.Conv3d(self.dim, 2 * self.dim, 1, stride=1, padding=0),
            nn.ReLU(inplace=True),
            nn.BatchNorm3d(2 * self.dim),
            nn.Upsample(scale_factor=0.5, mode='trilinear', align_corners=False)
        )

    def forward(self, x, H, W, T):
        """
        x: B, H*W, C
        """
        B, L, C = x.shape
        assert L == H * W * T, "input feature has wrong size"
        assert H % 2 == 0 and W % 2 == 0 and T % 2 == 0, f"x size ({H}*{W}) are not even."

        x = x.view(B, C, H, W, T)

        # padding
        pad_input = (H % 2 == 1) or (W % 2 == 1) or (T % 2 == 1)
        if pad_input:
            x = nnf.pad(x, (0, 0, 0, T % 2, 0, W % 2, 0, H % 2))
        x = self.conv_du(x)
        # x0 = x[:, 0::2, 0::2, 0::2, :]  # B H/2 W/2 C
        # x1 = x[:, 1::2, 0::2, 0::2, :]  # B H/2 W/2 C
        # x2 = x[:, 0::2, 1::2, 0::2, :]  # B H/2 W/2 C
        # x3 = x[:, 0::2, 0::2, 1::2, :]  # B H/2 W/2 C
        # x4 = x[:, 1::2, 1::2, 0::2, :]  # B H/2 W/2 C
        # x5 = x[:, 0::2, 1::2, 1::2, :]  # B H/2 W/2 C
        # x6 = x[:, 1::2, 0::2, 1::2, :]  # B H/2 W/2 C
        # x7 = x[:, 1::2, 1::2, 1::2, :]  # B H/2 W/2 C
        # x = torch.cat([x0, x1, x2, x3, x4, x5, x6, x7], -1)  # B H/2 W/2 T/2 8*C
        x = x.view(B, -1, 2 * C)  # B H/2*W/2*T/2 8*C

        #x = self.norm(x)
        #x = self.reduction(x)

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
                 depth,
                 num_heads,
                 window_size=(7, 7, 7),
                 mlp_ratio=4.,
                 qkv_bias=True,
                 qk_scale=None,
                 drop=0.,
                 attn_drop=0.,
                 drop_path=0.,
                 norm_layer=nn.LayerNorm,
                 downsample=None,
                 use_checkpoint=False,
                 pat_merg_rf=2,
                 pos_embed_method='relative',
                 concatenated_input=True):
        super().__init__()
        self.window_size = window_size
        self.shift_size = (window_size[0] // 2, window_size[1] // 2, window_size[2] // 2)
        self.depth = depth
        self.use_checkpoint = use_checkpoint
        self.pat_merg_rf = pat_merg_rf
        # build blocks
        self.blocks = nn.ModuleList([
            SwinTransformerBlock(
                dim=dim,
                num_heads=num_heads,
                window_size=window_size,
                shift_size=(0, 0, 0) if (i % 2 == 0) else (window_size[0] // 2, window_size[1] // 2, window_size[2] // 2),
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                qk_scale=qk_scale,
                drop=drop,
                attn_drop=attn_drop,
                drop_path=drop_path[i] if isinstance(drop_path, list) else drop_path,
                norm_layer=norm_layer,
                pos_embed_method=pos_embed_method,
                concatenated_input=concatenated_input)
            for i in range(depth)])

        # patch merging layer
        if downsample is not None:
            self.downsample = downsample(dim=dim, norm_layer=norm_layer, reduce_factor=self.pat_merg_rf,concatenated_input=concatenated_input)
        else:
            self.downsample = None

    def forward(self, x, H, W, T):
        """ Forward function.
        Args:
            x: Input feature, tensor size (B, H*W, C).
            H, W: Spatial resolution of the input feature.
        """

        # calculate attention mask for SW-MSA
        Hp = int(np.ceil(H / self.window_size[0])) * self.window_size[0]
        Wp = int(np.ceil(W / self.window_size[1])) * self.window_size[1]
        Tp = int(np.ceil(T / self.window_size[2])) * self.window_size[2]
        img_mask = torch.zeros((1, Hp, Wp, Tp, 1), device=x.device)  # 1 Hp Wp 1
        h_slices = (slice(0, -self.window_size[0]),
                    slice(-self.window_size[0], -self.shift_size[0]),
                    slice(-self.shift_size[0], None))
        w_slices = (slice(0, -self.window_size[1]),
                    slice(-self.window_size[1], -self.shift_size[1]),
                    slice(-self.shift_size[1], None))
        t_slices = (slice(0, -self.window_size[2]),
                    slice(-self.window_size[2], -self.shift_size[2]),
                    slice(-self.shift_size[2], None))
        cnt = 0
        for h in h_slices:
            for w in w_slices:
                for t in t_slices:
                    img_mask[:, h, w, t, :] = cnt
                    cnt += 1

        mask_windows = window_partition(img_mask, self.window_size)  # nW, window_size, window_size, 1
        mask_windows = mask_windows.view(-1, self.window_size[0] * self.window_size[1] * self.window_size[2])
        attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
        attn_mask = attn_mask.masked_fill(attn_mask != 0, float(-100.0)).masked_fill(attn_mask == 0, float(0.0))

        for blk in self.blocks:
            blk.H, blk.W, blk.T = H, W, T
            if self.use_checkpoint:
                x = checkpoint.checkpoint(blk, x, attn_mask)
            else:
                x = blk(x, attn_mask)
        if self.downsample is not None:
            x_down = self.downsample(x, H, W, T)
            Wh, Ww, Wt = (H + 1) // 2, (W + 1) // 2, (T + 1) // 2
            return x, H, W, T, x_down, Wh, Ww, Wt
        else:
            return x, H, W, T, x, H, W, T



class BasicLayer_dualModality(nn.Module):
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
                 depth,
                 num_heads,
                 window_size=(7, 7, 7),
                 mlp_ratio=4.,
                 qkv_bias=True,
                 qk_scale=None,
                 drop=0.,
                 attn_drop=0.,
                 drop_path=0.,
                 norm_layer=nn.LayerNorm,
                 downsample=None,
                 use_checkpoint=False,
                 pat_merg_rf=2,
                 pos_embed_method='relative',
                 concatenated_input=True):
        super().__init__()
        self.window_size = window_size
        self.shift_size = (window_size[0] // 2, window_size[1] // 2, window_size[2] // 2)
        self.depth = depth
        self.use_checkpoint = use_checkpoint
        self.pat_merg_rf = pat_merg_rf
        # build blocks
        self.blocks = nn.ModuleList([
            SwinTransformerBlock_dualModality(
                dim=dim,
                num_heads=num_heads,
                window_size=window_size,
                shift_size=(0, 0, 0) if (i % 2 == 0) else (window_size[0] // 2, window_size[1] // 2, window_size[2] // 2),
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                qk_scale=qk_scale,
                drop=drop,
                attn_drop=attn_drop,
                drop_path=drop_path[i] if isinstance(drop_path, list) else drop_path,
                norm_layer=norm_layer,
                pos_embed_method=pos_embed_method,
                concatenated_input=concatenated_input)
            for i in range(depth)])

        # patch merging layer
        if downsample is not None:
            self.downsample = downsample(dim=dim, norm_layer=norm_layer, reduce_factor=self.pat_merg_rf,concatenated_input=concatenated_input)
        else:
            self.downsample = None

    def forward(self, x,x_1, H, W, T):
        """ Forward function.
        Args:
            x: Input feature, tensor size (B, H*W, C).
            H, W: Spatial resolution of the input feature.
        """

        # calculate attention mask for SW-MSA
        Hp = int(np.ceil(H / self.window_size[0])) * self.window_size[0]
        Wp = int(np.ceil(W / self.window_size[1])) * self.window_size[1]
        Tp = int(np.ceil(T / self.window_size[2])) * self.window_size[2]
        img_mask = torch.zeros((1, Hp, Wp, Tp, 1), device=x.device)  # 1 Hp Wp 1
        h_slices = (slice(0, -self.window_size[0]),
                    slice(-self.window_size[0], -self.shift_size[0]),
                    slice(-self.shift_size[0], None))
        w_slices = (slice(0, -self.window_size[1]),
                    slice(-self.window_size[1], -self.shift_size[1]),
                    slice(-self.shift_size[1], None))
        t_slices = (slice(0, -self.window_size[2]),
                    slice(-self.window_size[2], -self.shift_size[2]),
                    slice(-self.shift_size[2], None))
        cnt = 0
        for h in h_slices:
            for w in w_slices:
                for t in t_slices:
                    img_mask[:, h, w, t, :] = cnt
                    cnt += 1

        mask_windows = window_partition(img_mask, self.window_size)  # nW, window_size, window_size, 1
        mask_windows = mask_windows.view(-1, self.window_size[0] * self.window_size[1] * self.window_size[2])
        attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
        attn_mask = attn_mask.masked_fill(attn_mask != 0, float(-100.0)).masked_fill(attn_mask == 0, float(0.0))

        for blk in self.blocks:
            blk.H, blk.W, blk.T = H, W, T
            if self.use_checkpoint:
                x = checkpoint.checkpoint(blk, x, attn_mask)
            else:
                x,x_1 = blk(x, x_1, attn_mask)
        if self.downsample is not None:
            x_down = self.downsample(x, H, W, T)
            x_1_down = self.downsample(x_1, H, W, T)
            Wh, Ww, Wt = (H + 1) // 2, (W + 1) // 2, (T + 1) // 2
            return x,x_1, H, W, T, x_down,x_1_down, Wh, Ww, Wt
        else:
            return x, x_1, H, W, T, x,x_1, H, W, T



class BasicLayer_crossModality(nn.Module):
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
                 depth,
                 num_heads,
                 window_size=(7, 7, 7),
                 mlp_ratio=4.,
                 qkv_bias=True,
                 qk_scale=None,
                 drop=0.,
                 attn_drop=0.,
                 drop_path=0.,
                 norm_layer=nn.LayerNorm,
                 downsample=None,
                 use_checkpoint=False,
                 pat_merg_rf=2,
                 pos_embed_method='relative',
                 concatenated_input=True):
        super().__init__()
        self.window_size = window_size
        self.shift_size = (window_size[0] // 2, window_size[1] // 2, window_size[2] // 2)
        self.depth = depth
        self.use_checkpoint = use_checkpoint
        self.pat_merg_rf = pat_merg_rf
        # build blocks
        self.blocks = nn.ModuleList([
            SwinTransformerBlock_crossModality(
                dim=dim,
                num_heads=num_heads,
                window_size=window_size,
                shift_size=(0, 0, 0) if (i % 2 == 0) else (window_size[0] // 2, window_size[1] // 2, window_size[2] // 2),
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                qk_scale=qk_scale,
                drop=drop,
                attn_drop=attn_drop,
                drop_path=drop_path[i] if isinstance(drop_path, list) else drop_path,
                norm_layer=norm_layer,
                pos_embed_method=pos_embed_method,
                concatenated_input=concatenated_input)
            for i in range(depth)])

        # patch merging layer
        if downsample is not None:
            self.downsample = downsample(dim=dim, norm_layer=norm_layer, reduce_factor=self.pat_merg_rf,concatenated_input=concatenated_input)
        else:
            self.downsample = None

    def forward(self, x, x_1, H, W, T):
        """ Forward function.
        Args:
            x: Input feature, tensor size (B, H*W, C).
            H, W: Spatial resolution of the input feature.
        """

        # calculate attention mask for SW-MSA
        Hp = int(np.ceil(H / self.window_size[0])) * self.window_size[0]
        Wp = int(np.ceil(W / self.window_size[1])) * self.window_size[1]
        Tp = int(np.ceil(T / self.window_size[2])) * self.window_size[2]
        img_mask = torch.zeros((1, Hp, Wp, Tp, 1), device=x.device)  # 1 Hp Wp 1
        h_slices = (slice(0, -self.window_size[0]),
                    slice(-self.window_size[0], -self.shift_size[0]),
                    slice(-self.shift_size[0], None))
        w_slices = (slice(0, -self.window_size[1]),
                    slice(-self.window_size[1], -self.shift_size[1]),
                    slice(-self.shift_size[1], None))
        t_slices = (slice(0, -self.window_size[2]),
                    slice(-self.window_size[2], -self.shift_size[2]),
                    slice(-self.shift_size[2], None))
        cnt = 0
        for h in h_slices:
            for w in w_slices:
                for t in t_slices:
                    img_mask[:, h, w, t, :] = cnt
                    cnt += 1

        mask_windows = window_partition(img_mask, self.window_size)  # nW, window_size, window_size, 1
        mask_windows = mask_windows.view(-1, self.window_size[0] * self.window_size[1] * self.window_size[2])
        attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
        attn_mask = attn_mask.masked_fill(attn_mask != 0, float(-100.0)).masked_fill(attn_mask == 0, float(0.0))

        for blk in self.blocks:
            blk.H, blk.W, blk.T = H, W, T
            if self.use_checkpoint:
                x = checkpoint.checkpoint(blk, x, attn_mask)
            else:
                x,x_1 = blk(x, x_1, attn_mask)
        if self.downsample is not None:
            x_down = self.downsample(x, H, W, T)
            x_1_down = self.downsample(x_1, H, W, T)
            Wh, Ww, Wt = (H + 1) // 2, (W + 1) // 2, (T + 1) // 2
            return x,x_1, H, W, T, x_down,x_1_down, Wh, Ww, Wt
        else:
            return x, x_1, H, W, T, x,x_1, H, W, T


class PatchEmbed(nn.Module):
    """ Image to Patch Embedding
    Args:
        patch_size (int): Patch token size. Default: 4.
        in_chans (int): Number of input image channels. Default: 3.
        embed_dim (int): Number of linear projection output channels. Default: 96.
        norm_layer (nn.Module, optional): Normalization layer. Default: None
    """

    def __init__(self, patch_size=4, in_chans=3, embed_dim=96, norm_layer=None):
        super().__init__()
        patch_size = to_3tuple(patch_size)
        self.patch_size = patch_size

        self.in_chans = in_chans
        self.embed_dim = embed_dim

        self.proj = nn.Conv3d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)
        if norm_layer is not None:
            self.norm = norm_layer(embed_dim)
        else:
            self.norm = None

    def forward(self, x):
        """Forward function."""
        # padding
        _, _, H, W, T = x.size()
        if T % self.patch_size[2] != 0:
            x = nnf.pad(x, (0, self.patch_size[2] - T % self.patch_size[2]))
        if W % self.patch_size[1] != 0:
            x = nnf.pad(x, (0, 0, 0, self.patch_size[1] - W % self.patch_size[1]))
        if H % self.patch_size[0] != 0:
            x = nnf.pad(x, (0, 0, 0, 0, 0, self.patch_size[0] - H % self.patch_size[0]))

        x = self.proj(x)  # B C Wh Ww Wt
        if self.norm is not None:
            Wh, Ww, Wt = x.size(2), x.size(3), x.size(4)
            x = x.flatten(2).transpose(1, 2)
            x = self.norm(x)
            x = x.transpose(1, 2).view(-1, self.embed_dim, Wh, Ww, Wt)

        return x

class SinusoidalPositionEmbedding(nn.Module):
    '''
    Rotary Position Embedding
    '''
    def __init__(self,):
        super(SinusoidalPositionEmbedding, self).__init__()

    def forward(self, x):
        batch_sz, n_patches, hidden = x.shape
        position_ids = torch.arange(0, n_patches).float().cuda()
        indices = torch.arange(0, hidden//2).float().cuda()
        indices = torch.pow(10000.0, -2 * indices / hidden)
        embeddings = torch.einsum('b,d->bd', position_ids, indices)
        embeddings = torch.stack([torch.sin(embeddings), torch.cos(embeddings)], dim=-1)
        embeddings = torch.reshape(embeddings, (1, n_patches, hidden))
        return embeddings
from monai.networks.blocks.patchembedding import PatchEmbeddingBlock

class SwinTransformer(nn.Module):
    r""" Swin Transformer
        A PyTorch impl of : `Swin Transformer: Hierarchical Vision Transformer using Shifted Windows`  -
          https://arxiv.org/pdf/2103.14030
    Args:
        img_size (int | tuple(int)): Input image size. Default 224
        patch_size (int | tuple(int)): Patch size. Default: 4
        in_chans (int): Number of input image channels. Default: 3
        num_classes (int): Number of classes for classification head. Default: 1000
        embed_dim (int): Patch embedding dimension. Default: 96
        depths (tuple(int)): Depth of each Swin Transformer layer.
        num_heads (tuple(int)): Number of attention heads in different layers.
        window_size (tuple): Window size. Default: 7
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim. Default: 4
        qkv_bias (bool): If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float): Override default qk scale of head_dim ** -0.5 if set. Default: None
        drop_rate (float): Dropout rate. Default: 0
        attn_drop_rate (float): Attention dropout rate. Default: 0
        drop_path_rate (float): Stochastic depth rate. Default: 0.1
        norm_layer (nn.Module): Normalization layer. Default: nn.LayerNorm.
        ape (bool): If True, add absolute position embedding to the patch embedding. Default: False
        patch_norm (bool): If True, add normalization after patch embedding. Default: True
        use_checkpoint (bool): Whether to use checkpointing to save memory. Default: False
    """

    def __init__(self, pretrain_img_size=96,
                 patch_size=4,
                 in_chans=3,
                 embed_dim=96,
                 depths=[2, 2, 6, 2],
                 num_heads=[3, 6, 12, 24],
                 window_size=(7, 7, 7),
                 mlp_ratio=4.,
                 qkv_bias=True,
                 qk_scale=None,
                 drop_rate=0.,
                 attn_drop_rate=0.,
                 drop_path_rate=0.2,
                 norm_layer=nn.LayerNorm,
                 ape=False,
                 spe=False,
                 patch_norm=True,
                 out_indices=(0, 1, 2, 3, 4),
                 frozen_stages=-1,
                 use_checkpoint=False,
                 pat_merg_rf=2,
                 pos_embed_method='relative',
                 concatenated_input=True):
        super().__init__()
        self.pretrain_img_size = pretrain_img_size
        self.num_layers = len(depths)
        self.embed_dim = embed_dim
        self.ape = ape
        self.spe = spe
        self.patch_norm = patch_norm
        self.out_indices = out_indices
        self.frozen_stages = frozen_stages

        # split image into non-overlapping patches
        self.patch_embed = PatchEmbed(
            patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dim,
            norm_layer=norm_layer if self.patch_norm else None)
        # self.patch_embedding = PatchEmbeddingBlock(
        #     in_channels=in_chans,
        #     img_size=pretrain_img_size,
        #     patch_size=patch_size,
        #     hidden_size=embed_dim,
        #     num_heads=4,
        #     pos_embed='perceptron',
        #     dropout_rate=drop_path_rate,
        #     spatial_dims=3,
        # )

        # absolute position embedding
        if self.ape:
            pretrain_img_size = to_3tuple(self.pretrain_img_size)
            patch_size = to_3tuple(patch_size)
            patches_resolution = [pretrain_img_size[0] // patch_size[0], pretrain_img_size[1] // patch_size[1], pretrain_img_size[2] // patch_size[2]]

            self.absolute_pos_embed = nn.Parameter(
                torch.zeros(1, embed_dim, patches_resolution[0], patches_resolution[1], patches_resolution[2]))
            trunc_normal_(self.absolute_pos_embed, std=.02)
        elif self.spe:
            self.pos_embd = SinusoidalPositionEmbedding().cuda()
        self.pos_drop = nn.Dropout(p=drop_rate)

        # stochastic depth
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]  # stochastic depth decay rule

        # build layers
        self.layers = nn.ModuleList()
        for i_layer in range(self.num_layers):
            layer = BasicLayer(dim=int(embed_dim * 2 ** i_layer),
                                depth=depths[i_layer],
                                num_heads=num_heads[i_layer],
                                window_size=window_size,
                                mlp_ratio=mlp_ratio,
                                qkv_bias=qkv_bias,
                                qk_scale=qk_scale,
                                drop=drop_rate,
                                attn_drop=attn_drop_rate,
                                drop_path=dpr[sum(depths[:i_layer]):sum(depths[:i_layer + 1])],
                                norm_layer=norm_layer,
                                downsample=PatchMerging if (i_layer < self.num_layers) else None,
                                use_checkpoint=use_checkpoint,
                                pat_merg_rf=pat_merg_rf,
                                pos_embed_method=pos_embed_method,
                                concatenated_input=concatenated_input)
            self.layers.append(layer)

        num_features = [int(embed_dim * 2 ** i) for i in range(1,self.num_layers+1)]
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

    def init_weights(self, pretrained=None):
        """Initialize the weights in backbone.
        Args:
            pretrained (str, optional): Path to pre-trained weights.
                Defaults to None.
        """

        def _init_weights(m):
            if isinstance(m, nn.Linear):
                trunc_normal_(m.weight, std=.02)
                if isinstance(m, nn.Linear) and m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.LayerNorm):
                nn.init.constant_(m.bias, 0)
                nn.init.constant_(m.weight, 1.0)

        if isinstance(pretrained, str):
            self.apply(_init_weights)
        elif pretrained is None:
            self.apply(_init_weights)
        else:
            raise TypeError('pretrained must be a str or None')

    def forward(self, x):
        """Forward function."""
        outs = []
        #x = self.patch_embedding(x).transpose(1, 2)
        x = self.patch_embed(x)
        #x = self.norm(x)
        #x = x.transpose(1, 2).view(-1, self.embed_dim, 48, 48, 48)
        outs.append(x)

        Wh, Ww, Wt = x.size(2), x.size(3), x.size(4)
        if self.ape:
            # interpolate the position embedding to the corresponding size
            absolute_pos_embed = nnf.interpolate(self.absolute_pos_embed, size=(Wh, Ww, Wt), mode='trilinear')
            x = (x + absolute_pos_embed).flatten(2).transpose(1, 2)  # B Wh*Ww*Wt C
        elif self.spe:
            print(self.spe)
            x = x.flatten(2).transpose(1, 2)
            x += self.pos_embd(x)
        else:
            x = x.flatten(2).transpose(1, 2)

        x = self.pos_drop(x)


        for i in range(self.num_layers):
            layer = self.layers[i]
            x_out, H, W, T, x, Wh, Ww, Wt = layer(x, Wh, Ww, Wt)
            if i in self.out_indices:
                norm_layer = getattr(self, f'norm{i}')
                x = norm_layer(x)

                out = x.view(-1, Wh, Ww, Wt, self.num_features[i]).permute(0, 4, 1, 2, 3).contiguous()
                #print(out.shape)
                outs.append(out)

        return outs

    def train(self, mode=True):
        """Convert the model into training mode while keep layers freezed."""
        super(SwinTransformer, self).train(mode)
        self._freeze_stages()


class SwinTransformer_dense(nn.Module):
    r""" Swin Transformer
        A PyTorch impl of : `Swin Transformer: Hierarchical Vision Transformer using Shifted Windows`  -
          https://arxiv.org/pdf/2103.14030
    Args:
        img_size (int | tuple(int)): Input image size. Default 224
        patch_size (int | tuple(int)): Patch size. Default: 4
        in_chans (int): Number of input image channels. Default: 3
        num_classes (int): Number of classes for classification head. Default: 1000
        embed_dim (int): Patch embedding dimension. Default: 96
        depths (tuple(int)): Depth of each Swin Transformer layer.
        num_heads (tuple(int)): Number of attention heads in different layers.
        window_size (tuple): Window size. Default: 7
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim. Default: 4
        qkv_bias (bool): If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float): Override default qk scale of head_dim ** -0.5 if set. Default: None
        drop_rate (float): Dropout rate. Default: 0
        attn_drop_rate (float): Attention dropout rate. Default: 0
        drop_path_rate (float): Stochastic depth rate. Default: 0.1
        norm_layer (nn.Module): Normalization layer. Default: nn.LayerNorm.
        ape (bool): If True, add absolute position embedding to the patch embedding. Default: False
        patch_norm (bool): If True, add normalization after patch embedding. Default: True
        use_checkpoint (bool): Whether to use checkpointing to save memory. Default: False
    """

    def __init__(self, pretrain_img_size=96,
                 patch_size=4,
                 in_chans=3,
                 embed_dim=96,
                 depths=[2, 2, 6, 2],
                 num_heads=[3, 6, 12, 24],
                 window_size=(7, 7, 7),
                 mlp_ratio=4.,
                 qkv_bias=True,
                 qk_scale=None,
                 drop_rate=0.,
                 attn_drop_rate=0.,
                 drop_path_rate=0.2,
                 norm_layer=nn.LayerNorm,
                 ape=False,
                 spe=False,
                 patch_norm=True,
                 out_indices=(0, 1, 2, 3, 4),
                 frozen_stages=-1,
                 use_checkpoint=False,
                 pat_merg_rf=2,
                 pos_embed_method='relative',
                 concatenated_input=True):
        super().__init__()
        self.pretrain_img_size = pretrain_img_size
        self.num_layers = len(depths)
        self.embed_dim = embed_dim
        self.ape = ape
        self.spe = spe
        self.patch_norm = patch_norm
        self.out_indices = out_indices
        self.frozen_stages = frozen_stages

        # split image into non-overlapping patches
        self.patch_embed = PatchEmbed(
            patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dim,
            norm_layer=norm_layer if self.patch_norm else None)
        # self.patch_embedding = PatchEmbeddingBlock(
        #     in_channels=in_chans,
        #     img_size=pretrain_img_size,
        #     patch_size=patch_size,
        #     hidden_size=embed_dim,
        #     num_heads=4,
        #     pos_embed='perceptron',
        #     dropout_rate=drop_path_rate,
        #     spatial_dims=3,
        # )

        # absolute position embedding
        if self.ape:
            pretrain_img_size = to_3tuple(self.pretrain_img_size)
            patch_size = to_3tuple(patch_size)
            patches_resolution = [pretrain_img_size[0] // patch_size[0], pretrain_img_size[1] // patch_size[1], pretrain_img_size[2] // patch_size[2]]

            self.absolute_pos_embed = nn.Parameter(
                torch.zeros(1, embed_dim, patches_resolution[0], patches_resolution[1], patches_resolution[2]))
            trunc_normal_(self.absolute_pos_embed, std=.02)
        elif self.spe:
            self.pos_embd = SinusoidalPositionEmbedding().cuda()
        self.pos_drop = nn.Dropout(p=drop_rate)

        # stochastic depth
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]  # stochastic depth decay rule

        # build layers
        self.layers = nn.ModuleList()
        self.patch_merging_layers = nn.ModuleList()
        for i_layer in range(self.num_layers):
            layer = BasicLayer(dim=int(embed_dim * 2 ** i_layer),
                                depth=depths[i_layer],
                                num_heads=num_heads[i_layer],
                                window_size=window_size,
                                mlp_ratio=mlp_ratio,
                                qkv_bias=qkv_bias,
                                qk_scale=qk_scale,
                                drop=drop_rate,
                                attn_drop=attn_drop_rate,
                                drop_path=dpr[sum(depths[:i_layer]):sum(depths[:i_layer + 1])],
                                norm_layer=norm_layer,
                                downsample=PatchMerging if (i_layer < self.num_layers) else None,
                                use_checkpoint=use_checkpoint,
                                pat_merg_rf=pat_merg_rf,
                                pos_embed_method=pos_embed_method,
                                concatenated_input=concatenated_input)
            self.layers.append(layer)
        patch_merging_layer = PatchConvPool(int(embed_dim * 2 ** i_layer), concatenated_input=False)
        self.patch_merging_layers.append(patch_merging_layer)
        num_features = [int(embed_dim * 2 ** i) for i in range(1,self.num_layers+1)]
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

    def init_weights(self, pretrained=None):
        """Initialize the weights in backbone.
        Args:
            pretrained (str, optional): Path to pre-trained weights.
                Defaults to None.
        """

        def _init_weights(m):
            if isinstance(m, nn.Linear):
                trunc_normal_(m.weight, std=.02)
                if isinstance(m, nn.Linear) and m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.LayerNorm):
                nn.init.constant_(m.bias, 0)
                nn.init.constant_(m.weight, 1.0)

        if isinstance(pretrained, str):
            self.apply(_init_weights)
        elif pretrained is None:
            self.apply(_init_weights)
        else:
            raise TypeError('pretrained must be a str or None')

    def forward(self, x):
        """Forward function."""
        outs = []
        #x = self.patch_embedding(x).transpose(1, 2)
        x = self.patch_embed(x)
        #x = self.norm(x)
        #x = x.transpose(1, 2).view(-1, self.embed_dim, 48, 48, 48)
        outs.append(x)

        Wh, Ww, Wt = x.size(2), x.size(3), x.size(4)
        if self.ape:
            # interpolate the position embedding to the corresponding size
            absolute_pos_embed = nnf.interpolate(self.absolute_pos_embed, size=(Wh, Ww, Wt), mode='trilinear')
            x = (x + absolute_pos_embed).flatten(2).transpose(1, 2)  # B Wh*Ww*Wt C
        elif self.spe:
            print(self.spe)
            x = x.flatten(2).transpose(1, 2)
            x += self.pos_embd(x)
        else:
            x = x.flatten(2).transpose(1, 2)

        x = self.pos_drop(x)

        for i in range(self.num_layers):
            layer = self.layers[i]
            #x_pre = x
            x_pre_down = self.patch_merging_layers[i](x, Wh, Ww, Wt)
            x_out, H, W, T, x, Wh, Ww, Wt = layer(x, Wh, Ww, Wt)
            x = x_pre_down + x
            if i in self.out_indices:
                norm_layer = getattr(self, f'norm{i}')
                x = norm_layer(x)
                out = x.view(-1, Wh, Ww, Wt, self.num_features[i]).permute(0, 4, 1, 2, 3).contiguous()
                #print(out.shape)
                outs.append(out)

        return outs

    def train(self, mode=True):
        """Convert the model into training mode while keep layers freezed."""
        super(SwinTransformer_dense, self).train(mode)
        self._freeze_stages()

class SwinTransformer_wFeatureTalk(nn.Module):
    r""" Swin Transformer modified to process images from two modalities; feature talks between two images are introduced in encoder
        A PyTorch impl of : `Swin Transformer: Hierarchical Vision Transformer using Shifted Windows`  -
          https://arxiv.org/pdf/2103.14030
    Args:
        img_size (int | tuple(int)): Input image size. Default 224
        patch_size (int | tuple(int)): Patch size. Default: 4
        in_chans (int): Number of input image channels. Default: 3
        num_classes (int): Number of classes for classification head. Default: 1000
        embed_dim (int): Patch embedding dimension. Default: 96
        depths (tuple(int)): Depth of each Swin Transformer layer.
        num_heads (tuple(int)): Number of attention heads in different layers.
        window_size (tuple): Window size. Default: 7
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim. Default: 4
        qkv_bias (bool): If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float): Override default qk scale of head_dim ** -0.5 if set. Default: None
        drop_rate (float): Dropout rate. Default: 0
        attn_drop_rate (float): Attention dropout rate. Default: 0
        drop_path_rate (float): Stochastic depth rate. Default: 0.1
        norm_layer (nn.Module): Normalization layer. Default: nn.LayerNorm.
        ape (bool): If True, add absolute position embedding to the patch embedding. Default: False
        patch_norm (bool): If True, add normalization after patch embedding. Default: True
        use_checkpoint (bool): Whether to use checkpointing to save memory. Default: False
    """

    def __init__(self, pretrain_img_size=160,
                 patch_size=4,
                 in_chans=3,
                 embed_dim=96,
                 depths=[2, 2, 6, 2],
                 num_heads=[3, 6, 12, 24],
                 window_size=(7, 7, 7),
                 mlp_ratio=4.,
                 qkv_bias=True,
                 qk_scale=None,
                 drop_rate=0.,
                 attn_drop_rate=0.,
                 drop_path_rate=0.2,
                 norm_layer=nn.LayerNorm,
                 ape=False,
                 spe=False,
                 patch_norm=True,
                 out_indices=(0, 1, 2, 3),
                 frozen_stages=-1,
                 use_checkpoint=False,
                 pat_merg_rf=2,
                 pos_embed_method='relative'):
        super().__init__()
        self.pretrain_img_size = pretrain_img_size
        self.num_layers = len(depths)
        self.embed_dim = embed_dim
        self.ape = ape
        self.spe = spe
        self.patch_norm = patch_norm
        self.out_indices = out_indices
        self.frozen_stages = frozen_stages

        # split image into non-overlapping patches
        self.patch_embed = PatchEmbed(
            patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dim,
            norm_layer=norm_layer if self.patch_norm else None)

        # absolute position embedding
        if self.ape:
            pretrain_img_size = to_3tuple(self.pretrain_img_size)
            patch_size = to_3tuple(patch_size)
            patches_resolution = [pretrain_img_size[0] // patch_size[0], pretrain_img_size[1] // patch_size[1], pretrain_img_size[2] // patch_size[2]]

            self.absolute_pos_embed = nn.Parameter(
                torch.zeros(1, embed_dim, patches_resolution[0], patches_resolution[1], patches_resolution[2]))
            trunc_normal_(self.absolute_pos_embed, std=.02)
        elif self.spe:
            self.pos_embd = SinusoidalPositionEmbedding().cuda()
        self.pos_drop = nn.Dropout(p=drop_rate)

        # stochastic depth
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]  # stochastic depth decay rule

        # build layers
        self.layers = nn.ModuleList()
        self.patch_merging_layers = nn.ModuleList()

        for i_layer in range(self.num_layers):
            layer = BasicLayer(dim=int(embed_dim * 2 ** i_layer),
                                depth=depths[i_layer],
                                num_heads=num_heads[i_layer],
                                window_size=window_size,
                                mlp_ratio=mlp_ratio,
                                qkv_bias=qkv_bias,
                                qk_scale=qk_scale,
                                drop=drop_rate,
                                attn_drop=attn_drop_rate,
                                drop_path=dpr[sum(depths[:i_layer]):sum(depths[:i_layer + 1])],
                                norm_layer=norm_layer,
                                downsample=PatchMerging if (i_layer < self.num_layers - 1) else None,
                                use_checkpoint=use_checkpoint,
                                pat_merg_rf=pat_merg_rf,
                                pos_embed_method=pos_embed_method)
            self.layers.append(layer)

            patch_merging_layer = PatchMerging(int(embed_dim * 2 ** i_layer), reduce_factor=4, concatenated_input=False)
            self.patch_merging_layers.append(patch_merging_layer)

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

    def init_weights(self, pretrained=None):
        """Initialize the weights in backbone.
        Args:
            pretrained (str, optional): Path to pre-trained weights.
                Defaults to None.
        """

        def _init_weights(m):
            if isinstance(m, nn.Linear):
                trunc_normal_(m.weight, std=.02)
                if isinstance(m, nn.Linear) and m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.LayerNorm):
                nn.init.constant_(m.bias, 0)
                nn.init.constant_(m.weight, 1.0)

        if isinstance(pretrained, str):
            self.apply(_init_weights)
        elif pretrained is None:
            self.apply(_init_weights)
        else:
            raise TypeError('pretrained must be a str or None')

    def forward(self, x_0,x_1):
        """Forward function."""
        #PET image
        x_0 = self.patch_embed(x_0)
        Wh_x0, Ww_x0, Wt_x0 = x_0.size(2), x_0.size(3), x_0.size(4)
        if self.ape:
            # interpolate the position embedding to the corresponding size
            absolute_pos_embed_x0 = nnf.interpolate(self.absolute_pos_embed, size=(Wh_x0, Ww_x0, Wt_x0), mode='trilinear')
            x_0 = (x_0 + absolute_pos_embed_x0).flatten(2).transpose(1, 2)  # B Wh*Ww*Wt C
        elif self.spe:
            print(self.spe)
            x_0 = x_0.flatten(2).transpose(1, 2)
            x_0 += self.pos_embd(x_0)
        else:
            x_0 = x_0.flatten(2).transpose(1, 2)
        x_0 = self.pos_drop(x_0)

        #CT image
        x_1 = self.patch_embed(x_1)
        Wh_x1, Ww_x1, Wt_x1 = x_1.size(2), x_1.size(3), x_1.size(4)
        if self.ape:
            # interpolate the position embedding to the corresponding size
            absolute_pos_embed_x1 = nnf.interpolate(self.absolute_pos_embed, size=(Wh_x1, Ww_x1, Wt_x1),
                                                    mode='trilinear')
            x_1 = (x_1 + absolute_pos_embed_x1).flatten(2).transpose(1, 2)  # B Wh*Ww*Wt C
        elif self.spe:
            print(self.spe)
            x_1 = x_1.flatten(2).transpose(1, 2)
            x_1 += self.pos_embd(x_1)
        else:
            x_1 = x_1.flatten(2).transpose(1, 2)
        x_1 = self.pos_drop(x_1)


        outs = []

        #############layer0################
        layer = self.layers[0]
        # concatenate x_0 and x_1 in dimension 1
        x_0_1 = torch.cat((x_0,x_1),dim=2) #concatenate in the channel dimension so that the image dimension is still Wh_x0, Ww_x0, Wt_x0 which is needed to be compatible for SWINTR
        # send the concatenated feature vector to transformer
        x_out_x0_1_l0, H_x0_1, W_x0_1, T_x0_1, x_0_1, Wh_x0_1_l0, Ww_x0_1_l0, Wt_x0_1_l0 = layer(x_0_1, Wh_x0, Ww_x0, Wt_x0)
        # split the resulting feature vector in dimension 1 to get processed x_1_processed, x_0_processed
        x_out_x0_l0 = x_out_x0_1_l0[:,:,0:self.embed_dim]
        x_out_x1_l0 = x_out_x0_1_l0[:,:,self.embed_dim:]
        # add x_1_process and x_0_processed to x_1 and x_0 and start layer 1
        x_0_out = x_out_x0_l0 + x_0 # updated x_0
        x_1_out = x_out_x1_l0 + x_1 # updated x_1
        out_x0_x1_l0 = x_0_out + x_1_out
        norm_layer = getattr(self, f'norm{0}')
        x_out_l0 = norm_layer(out_x0_x1_l0)
        out = x_out_l0.view(-1, Wh_x0, Ww_x0, Wt_x0, self.embed_dim).permute(0, 4, 1, 2, 3).contiguous()
        outs.append(out) #layer 0 output

        #construct the input for the next layer
        x_0_down = self.patch_merging_layers[0](x_0, Wh_x0, Ww_x0, Wt_x0)
        x_1_down = self.patch_merging_layers[0](x_1, Wh_x1, Ww_x1, Wt_x1)
        x_0 = x_0_1[:,:,0:self.embed_dim*2] + x_0_down
        x_1 = x_0_1[:,:,self.embed_dim*2:] + x_1_down

        #############layer1################
        layer = self.layers[1]
        # concatenate x_0 and x_1 in dimension 1
        x_0_1 = torch.cat((x_0, x_1),
                          dim=2)  # concatenate in the channel dimension so that the image dimension is still Wh_x0, Ww_x0, Wt_x0 which is needed to be compatible for SWINTR
        # send the concatenated feature vector to transformer
        x_out_x0_1_l1, H_x0_1, W_x0_1, T_x0_1, x_0_1, Wh_x0_1_l1, Ww_x0_1_l1, Wt_x0_1_l1 = layer(x_0_1, Wh_x0_1_l0, Ww_x0_1_l0, Wt_x0_1_l0)
        # split the resulting feature vector in dimension 1 to get processed x_1_processed, x_0_processed
        x_out_x0_l1 = x_out_x0_1_l1[:, :, 0:self.embed_dim*2]
        x_out_x1_l1 = x_out_x0_1_l1[:, :, self.embed_dim*2:]
        x_0_out = x_out_x0_l1 + x_0  # updated x_0
        x_1_out = x_out_x1_l1 + x_1  # updated x_1
        out_x0_x1_l1 = x_0_out + x_1_out #should I use the sum or concat for decoder?
        norm_layer = getattr(self, f'norm{1}')
        x_out_l1 = norm_layer(out_x0_x1_l1)
        out = x_out_l1.view(-1, Wh_x0_1_l0, Ww_x0_1_l0, Wt_x0_1_l0, self.embed_dim*2).permute(0, 4, 1, 2, 3).contiguous()
        outs.append(out)  # layer 1 output

        #construct the input for the next layer
        x_0_down = self.patch_merging_layers[1](x_0, Wh_x0_1_l0, Ww_x0_1_l0, Wt_x0_1_l0)
        x_1_down = self.patch_merging_layers[1](x_1, Wh_x0_1_l0, Ww_x0_1_l0, Wt_x0_1_l0)
        x_0 = x_0_1[:, :, 0:self.embed_dim * 4] + x_0_down
        x_1 = x_0_1[:, :, self.embed_dim * 4:] + x_1_down

        #############layer2################
        layer = self.layers[2]
        # concatenate x_0 and x_1 in dimension 1
        x_0_1 = torch.cat((x_0, x_1),
                          dim=2)  # concatenate in the channel dimension so that the image dimension is still Wh_x0, Ww_x0, Wt_x0 which is needed to be compatible for SWINTR
        # send the concatenated feature vector to transformer
        x_out_x0_1_l2, H_x0_1, W_x0_1, T_x0_1, x_0_1, Wh_x0_1_l2, Ww_x0_1_l2, Wt_x0_1_l2 = layer(x_0_1, Wh_x0_1_l1,
                                                                                                 Ww_x0_1_l1, Wt_x0_1_l1)
        # split the resulting feature vector in dimension 1 to get processed x_1_processed, x_0_processed
        x_out_x0_l2 = x_out_x0_1_l2[:, :, 0:self.embed_dim * 4]
        x_out_x1_l2 = x_out_x0_1_l2[:, :, self.embed_dim * 4:]
        x_0_out = x_out_x0_l2 + x_0  # updated x_0
        x_1_out = x_out_x1_l2 + x_1  # updated x_1
        out_x0_x1_l2 = x_0_out + x_1_out
        norm_layer = getattr(self, f'norm{2}')
        x_out_l2 = norm_layer(out_x0_x1_l2)
        out = x_out_l2.view(-1, Wh_x0_1_l1, Ww_x0_1_l1, Wt_x0_1_l1, self.embed_dim*4).permute(0, 4, 1, 2, 3).contiguous()
        outs.append(out)  # layer 2 output

        # construct the input for the next layer
        x_0_down = self.patch_merging_layers[2](x_0, Wh_x0_1_l1, Ww_x0_1_l1, Wt_x0_1_l1)
        x_1_down = self.patch_merging_layers[2](x_1, Wh_x0_1_l1, Ww_x0_1_l1, Wt_x0_1_l1)
        x_0 = x_0_1[:, :, 0:self.embed_dim * 8] + x_0_down
        x_1 = x_0_1[:, :, self.embed_dim * 8:] + x_1_down

        #############layer3################
        layer = self.layers[3]
        # concatenate x_0 and x_1 in dimension 1
        x_0_1 = torch.cat((x_0, x_1),
                          dim=2)  # concatenate in the channel dimension so that the image dimension is still Wh_x0, Ww_x0, Wt_x0 which is needed to be compatible for SWINTR
        # send the concatenated feature vector to transformer
        x_out_x0_1_l3, H_x0_1, W_x0_1, T_x0_1, x_0_1, Wh_x0_1_l3, Ww_x0_1_l3, Wt_x0_1_l3 = layer(x_0_1, Wh_x0_1_l2,
                                                                                                 Ww_x0_1_l2, Wt_x0_1_l2)
        # split the resulting feature vector in dimension 1 to get processed x_1_processed, x_0_processed
        x_out_x0_l3 = x_out_x0_1_l3[:, :, 0:self.embed_dim * 8]
        x_out_x1_l3 = x_out_x0_1_l3[:, :, self.embed_dim * 8:]
        x_0_out = x_out_x0_l3 + x_0  # updated x_0
        x_1_out = x_out_x1_l3 + x_1  # updated x_1
        out_x0_x1_l3 = x_0_out + x_1_out
        norm_layer = getattr(self, f'norm{3}')
        x_out_l3 = norm_layer(out_x0_x1_l3)
        out = x_out_l3.view(-1, Wh_x0_1_l2, Ww_x0_1_l2, Wt_x0_1_l2, self.embed_dim*8).permute(0, 4, 1, 2, 3).contiguous()
        outs.append(out)  # layer 2 output

        return outs

    def train(self, mode=True):
        """Convert the model into training mode while keep layers freezed."""
        super(SwinTransformer_wFeatureTalk, self).train(mode)
        self._freeze_stages()

class SwinTransformer_wFeatureTalk_concat(nn.Module):
    r""" Swin Transformer modified to process images from two modalities; feature talks between two images are introduced in encoder
        A PyTorch impl of : `Swin Transformer: Hierarchical Vision Transformer using Shifted Windows`  -
          https://arxiv.org/pdf/2103.14030
    Args:
        img_size (int | tuple(int)): Input image size. Default 224
        patch_size (int | tuple(int)): Patch size. Default: 4
        in_chans (int): Number of input image channels. Default: 3
        num_classes (int): Number of classes for classification head. Default: 1000
        embed_dim (int): Patch embedding dimension. Default: 96
        depths (tuple(int)): Depth of each Swin Transformer layer.
        num_heads (tuple(int)): Number of attention heads in different layers.
        window_size (tuple): Window size. Default: 7
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim. Default: 4
        qkv_bias (bool): If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float): Override default qk scale of head_dim ** -0.5 if set. Default: None
        drop_rate (float): Dropout rate. Default: 0
        attn_drop_rate (float): Attention dropout rate. Default: 0
        drop_path_rate (float): Stochastic depth rate. Default: 0.1
        norm_layer (nn.Module): Normalization layer. Default: nn.LayerNorm.
        ape (bool): If True, add absolute position embedding to the patch embedding. Default: False
        patch_norm (bool): If True, add normalization after patch embedding. Default: True
        use_checkpoint (bool): Whether to use checkpointing to save memory. Default: False
    """

    def __init__(self, pretrain_img_size=160,
                 patch_size=4,
                 in_chans=3,
                 embed_dim=96,
                 depths=[2, 2, 6, 2],
                 num_heads=[3, 6, 12, 24],
                 window_size=(7, 7, 7),
                 mlp_ratio=4.,
                 qkv_bias=True,
                 qk_scale=None,
                 drop_rate=0.,
                 attn_drop_rate=0.,
                 drop_path_rate=0.2,
                 norm_layer=nn.LayerNorm,
                 ape=False,
                 spe=False,
                 patch_norm=True,
                 out_indices=(0, 1, 2, 3),
                 frozen_stages=-1,
                 use_checkpoint=False,
                 pat_merg_rf=2,
                 pos_embed_method='relative'):
        super().__init__()
        self.pretrain_img_size = pretrain_img_size
        self.num_layers = len(depths)
        self.embed_dim = embed_dim
        self.ape = ape
        self.spe = spe
        self.patch_norm = patch_norm
        self.out_indices = out_indices
        self.frozen_stages = frozen_stages

        # split image into non-overlapping patches
        self.patch_embed = PatchEmbed(
            patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dim,
            norm_layer=norm_layer if self.patch_norm else None)

        # absolute position embedding
        if self.ape:
            pretrain_img_size = to_3tuple(self.pretrain_img_size)
            patch_size = to_3tuple(patch_size)
            patches_resolution = [pretrain_img_size[0] // patch_size[0], pretrain_img_size[1] // patch_size[1], pretrain_img_size[2] // patch_size[2]]

            self.absolute_pos_embed = nn.Parameter(
                torch.zeros(1, embed_dim, patches_resolution[0], patches_resolution[1], patches_resolution[2]))
            trunc_normal_(self.absolute_pos_embed, std=.02)
        elif self.spe:
            self.pos_embd = SinusoidalPositionEmbedding().cuda()
        self.pos_drop = nn.Dropout(p=drop_rate)

        # stochastic depth
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]  # stochastic depth decay rule

        # build layers
        self.layers = nn.ModuleList()
        self.patch_merging_layers = nn.ModuleList()

        for i_layer in range(self.num_layers):
            layer = BasicLayer(dim=int(embed_dim * 2 ** i_layer),
                                depth=depths[i_layer],
                                num_heads=num_heads[i_layer],
                                window_size=window_size,
                                mlp_ratio=mlp_ratio,
                                qkv_bias=qkv_bias,
                                qk_scale=qk_scale,
                                drop=drop_rate,
                                attn_drop=attn_drop_rate,
                                drop_path=dpr[sum(depths[:i_layer]):sum(depths[:i_layer + 1])],
                                norm_layer=norm_layer,
                                downsample=PatchMerging if (i_layer < self.num_layers - 1) else None,
                                use_checkpoint=use_checkpoint,
                                pat_merg_rf=pat_merg_rf,
                                pos_embed_method=pos_embed_method)
            self.layers.append(layer)

            patch_merging_layer = PatchMerging(int(embed_dim * 2 ** i_layer), reduce_factor=4, concatenated_input=False)
            self.patch_merging_layers.append(patch_merging_layer)

        num_features = [int(embed_dim * 2 ** i) for i in range(self.num_layers)]
        self.num_features = num_features



        # add a norm layer for each output
        for i_layer in out_indices:
            layer = norm_layer(num_features[i_layer]*2)
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

    def init_weights(self, pretrained=None):
        """Initialize the weights in backbone.
        Args:
            pretrained (str, optional): Path to pre-trained weights.
                Defaults to None.
        """

        def _init_weights(m):
            if isinstance(m, nn.Linear):
                trunc_normal_(m.weight, std=.02)
                if isinstance(m, nn.Linear) and m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.LayerNorm):
                nn.init.constant_(m.bias, 0)
                nn.init.constant_(m.weight, 1.0)

        if isinstance(pretrained, str):
            self.apply(_init_weights)
        elif pretrained is None:
            self.apply(_init_weights)
        else:
            raise TypeError('pretrained must be a str or None')

    def forward(self, x_0,x_1):
        """Forward function."""
        #PET image
        x_0 = self.patch_embed(x_0)
        Wh_x0, Ww_x0, Wt_x0 = x_0.size(2), x_0.size(3), x_0.size(4)
        if self.ape:
            # interpolate the position embedding to the corresponding size
            absolute_pos_embed_x0 = nnf.interpolate(self.absolute_pos_embed, size=(Wh_x0, Ww_x0, Wt_x0), mode='trilinear')
            x_0 = (x_0 + absolute_pos_embed_x0).flatten(2).transpose(1, 2)  # B Wh*Ww*Wt C
        elif self.spe:
            print(self.spe)
            x_0 = x_0.flatten(2).transpose(1, 2)
            x_0 += self.pos_embd(x_0)
        else:
            x_0 = x_0.flatten(2).transpose(1, 2)
        x_0 = self.pos_drop(x_0)

        #CT image
        x_1 = self.patch_embed(x_1)
        Wh_x1, Ww_x1, Wt_x1 = x_1.size(2), x_1.size(3), x_1.size(4)
        if self.ape:
            # interpolate the position embedding to the corresponding size
            absolute_pos_embed_x1 = nnf.interpolate(self.absolute_pos_embed, size=(Wh_x1, Ww_x1, Wt_x1),
                                                    mode='trilinear')
            x_1 = (x_1 + absolute_pos_embed_x1).flatten(2).transpose(1, 2)  # B Wh*Ww*Wt C
        elif self.spe:
            print(self.spe)
            x_1 = x_1.flatten(2).transpose(1, 2)
            x_1 += self.pos_embd(x_1)
        else:
            x_1 = x_1.flatten(2).transpose(1, 2)
        x_1 = self.pos_drop(x_1)


        outs = []

        #############layer0################
        layer = self.layers[0]
        # concatenate x_0 and x_1 in dimension 1
        x_0_1 = torch.cat((x_0,x_1),dim=2) #concatenate in the channel dimension so that the image dimension is still Wh_x0, Ww_x0, Wt_x0 which is needed to be compatible for SWINTR
        # send the concatenated feature vector to transformer
        x_out_x0_1_l0, H_x0_1, W_x0_1, T_x0_1, x_0_1, Wh_x0_1_l0, Ww_x0_1_l0, Wt_x0_1_l0 = layer(x_0_1, Wh_x0, Ww_x0, Wt_x0)
        # split the resulting feature vector in dimension 1 to get processed x_1_processed, x_0_processed
        x_out_x0_l0 = x_out_x0_1_l0[:,:,0:self.embed_dim]
        x_out_x1_l0 = x_out_x0_1_l0[:,:,self.embed_dim:]
        # add x_1_process and x_0_processed to x_1 and x_0 and start layer 1
        x_0_out = x_out_x0_l0 + x_0 # updated x_0
        x_1_out = x_out_x1_l0 + x_1 # updated x_1
        #out_x0_x1_l0 = x_0_out + x_1_out
        out_x0_x1_l0 = torch.concat((x_0_out , x_1_out),dim=2)

        norm_layer = getattr(self, f'norm{0}')
        x_out_l0 = norm_layer(out_x0_x1_l0)
        out = x_out_l0.view(-1, Wh_x0, Ww_x0, Wt_x0, self.embed_dim*2).permute(0, 4, 1, 2, 3).contiguous()
        outs.append(out) #layer 0 output

        #construct the input for the next layer
        x_0_down = self.patch_merging_layers[0](x_0, Wh_x0, Ww_x0, Wt_x0)
        x_1_down = self.patch_merging_layers[0](x_1, Wh_x1, Ww_x1, Wt_x1)
        x_0 = x_0_1[:,:,0:self.embed_dim*2] + x_0_down
        x_1 = x_0_1[:,:,self.embed_dim*2:] + x_1_down

        #############layer1################
        layer = self.layers[1]
        # concatenate x_0 and x_1 in dimension 1
        x_0_1 = torch.cat((x_0, x_1),
                          dim=2)  # concatenate in the channel dimension so that the image dimension is still Wh_x0, Ww_x0, Wt_x0 which is needed to be compatible for SWINTR
        # send the concatenated feature vector to transformer
        x_out_x0_1_l1, H_x0_1, W_x0_1, T_x0_1, x_0_1, Wh_x0_1_l1, Ww_x0_1_l1, Wt_x0_1_l1 = layer(x_0_1, Wh_x0_1_l0, Ww_x0_1_l0, Wt_x0_1_l0)
        # split the resulting feature vector in dimension 1 to get processed x_1_processed, x_0_processed
        x_out_x0_l1 = x_out_x0_1_l1[:, :, 0:self.embed_dim*2]
        x_out_x1_l1 = x_out_x0_1_l1[:, :, self.embed_dim*2:]
        x_0_out = x_out_x0_l1 + x_0  # updated x_0
        x_1_out = x_out_x1_l1 + x_1  # updated x_1
        #out_x0_x1_l1 = x_0_out + x_1_out #should I use the sum or concat for decoder?
        out_x0_x1_l1 = torch.concat((x_0_out , x_1_out),dim=2)

        norm_layer = getattr(self, f'norm{1}')
        x_out_l1 = norm_layer(out_x0_x1_l1)
        out = x_out_l1.view(-1, Wh_x0_1_l0, Ww_x0_1_l0, Wt_x0_1_l0, self.embed_dim*4).permute(0, 4, 1, 2, 3).contiguous()
        outs.append(out)  # layer 1 output

        #construct the input for the next layer
        x_0_down = self.patch_merging_layers[1](x_0, Wh_x0_1_l0, Ww_x0_1_l0, Wt_x0_1_l0)
        x_1_down = self.patch_merging_layers[1](x_1, Wh_x0_1_l0, Ww_x0_1_l0, Wt_x0_1_l0)
        x_0 = x_0_1[:, :, 0:self.embed_dim * 4] + x_0_down
        x_1 = x_0_1[:, :, self.embed_dim * 4:] + x_1_down

        #############layer2################
        layer = self.layers[2]
        # concatenate x_0 and x_1 in dimension 1
        x_0_1 = torch.cat((x_0, x_1),
                          dim=2)  # concatenate in the channel dimension so that the image dimension is still Wh_x0, Ww_x0, Wt_x0 which is needed to be compatible for SWINTR
        # send the concatenated feature vector to transformer
        x_out_x0_1_l2, H_x0_1, W_x0_1, T_x0_1, x_0_1, Wh_x0_1_l2, Ww_x0_1_l2, Wt_x0_1_l2 = layer(x_0_1, Wh_x0_1_l1,
                                                                                                 Ww_x0_1_l1, Wt_x0_1_l1)
        # split the resulting feature vector in dimension 1 to get processed x_1_processed, x_0_processed
        x_out_x0_l2 = x_out_x0_1_l2[:, :, 0:self.embed_dim * 4]
        x_out_x1_l2 = x_out_x0_1_l2[:, :, self.embed_dim * 4:]
        x_0_out = x_out_x0_l2 + x_0  # updated x_0
        x_1_out = x_out_x1_l2 + x_1  # updated x_1
        #out_x0_x1_l2 = x_0_out + x_1_out
        out_x0_x1_l2 = torch.concat((x_0_out , x_1_out),dim=2)

        norm_layer = getattr(self, f'norm{2}')
        x_out_l2 = norm_layer(out_x0_x1_l2)
        out = x_out_l2.view(-1, Wh_x0_1_l1, Ww_x0_1_l1, Wt_x0_1_l1, self.embed_dim*8).permute(0, 4, 1, 2, 3).contiguous()
        outs.append(out)  # layer 2 output

        # construct the input for the next layer
        x_0_down = self.patch_merging_layers[2](x_0, Wh_x0_1_l1, Ww_x0_1_l1, Wt_x0_1_l1)
        x_1_down = self.patch_merging_layers[2](x_1, Wh_x0_1_l1, Ww_x0_1_l1, Wt_x0_1_l1)
        x_0 = x_0_1[:, :, 0:self.embed_dim * 8] + x_0_down
        x_1 = x_0_1[:, :, self.embed_dim * 8:] + x_1_down

        #############layer3################
        layer = self.layers[3]
        # concatenate x_0 and x_1 in dimension 1
        x_0_1 = torch.cat((x_0, x_1),
                          dim=2)  # concatenate in the channel dimension so that the image dimension is still Wh_x0, Ww_x0, Wt_x0 which is needed to be compatible for SWINTR
        # send the concatenated feature vector to transformer
        x_out_x0_1_l3, H_x0_1, W_x0_1, T_x0_1, x_0_1, Wh_x0_1_l3, Ww_x0_1_l3, Wt_x0_1_l3 = layer(x_0_1, Wh_x0_1_l2,
                                                                                                 Ww_x0_1_l2, Wt_x0_1_l2)
        # split the resulting feature vector in dimension 1 to get processed x_1_processed, x_0_processed
        x_out_x0_l3 = x_out_x0_1_l3[:, :, 0:self.embed_dim * 8]
        x_out_x1_l3 = x_out_x0_1_l3[:, :, self.embed_dim * 8:]
        x_0_out = x_out_x0_l3 + x_0  # updated x_0
        x_1_out = x_out_x1_l3 + x_1  # updated x_1
        #out_x0_x1_l3 = x_0_out + x_1_out
        out_x0_x1_l3 = torch.concat((x_0_out , x_1_out),dim=2)

        norm_layer = getattr(self, f'norm{3}')
        x_out_l3 = norm_layer(out_x0_x1_l3)
        out = x_out_l3.view(-1, Wh_x0_1_l2, Ww_x0_1_l2, Wt_x0_1_l2, self.embed_dim*16).permute(0, 4, 1, 2, 3).contiguous()
        outs.append(out)  # layer 2 output

        return outs

    def train(self, mode=True):
        """Convert the model into training mode while keep layers freezed."""
        super(SwinTransformer_wFeatureTalk_concat, self).train(mode)
        self._freeze_stages()

class SwinTransformer_wFeatureTalk_concat_noCrossModalUpdating(nn.Module):
    r""" Swin Transformer modified to process images from two modalities; feature talks between two images are introduced in encoder
        A PyTorch impl of : `Swin Transformer: Hierarchical Vision Transformer using Shifted Windows`  -
          https://arxiv.org/pdf/2103.14030
    Args:
        img_size (int | tuple(int)): Input image size. Default 224
        patch_size (int | tuple(int)): Patch size. Default: 4
        in_chans (int): Number of input image channels. Default: 3
        num_classes (int): Number of classes for classification head. Default: 1000
        embed_dim (int): Patch embedding dimension. Default: 96
        depths (tuple(int)): Depth of each Swin Transformer layer.
        num_heads (tuple(int)): Number of attention heads in different layers.
        window_size (tuple): Window size. Default: 7
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim. Default: 4
        qkv_bias (bool): If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float): Override default qk scale of head_dim ** -0.5 if set. Default: None
        drop_rate (float): Dropout rate. Default: 0
        attn_drop_rate (float): Attention dropout rate. Default: 0
        drop_path_rate (float): Stochastic depth rate. Default: 0.1
        norm_layer (nn.Module): Normalization layer. Default: nn.LayerNorm.
        ape (bool): If True, add absolute position embedding to the patch embedding. Default: False
        patch_norm (bool): If True, add normalization after patch embedding. Default: True
        use_checkpoint (bool): Whether to use checkpointing to save memory. Default: False
    """

    def __init__(self, pretrain_img_size=160,
                 patch_size=4,
                 in_chans=3,
                 embed_dim=96,
                 depths=[2, 2, 6, 2],
                 num_heads=[3, 6, 12, 24],
                 window_size=(7, 7, 7),
                 mlp_ratio=4.,
                 qkv_bias=True,
                 qk_scale=None,
                 drop_rate=0.,
                 attn_drop_rate=0.,
                 drop_path_rate=0.2,
                 norm_layer=nn.LayerNorm,
                 ape=False,
                 spe=False,
                 patch_norm=True,
                 out_indices=(0, 1, 2, 3),
                 frozen_stages=-1,
                 use_checkpoint=False,
                 pat_merg_rf=2,
                 pos_embed_method='relative'):
        super().__init__()
        self.pretrain_img_size = pretrain_img_size
        self.num_layers = len(depths)
        self.embed_dim = embed_dim
        self.ape = ape
        self.spe = spe
        self.patch_norm = patch_norm
        self.out_indices = out_indices
        self.frozen_stages = frozen_stages

        # split image into non-overlapping patches
        self.patch_embed = PatchEmbed(
            patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dim,
            norm_layer=norm_layer if self.patch_norm else None)

        # absolute position embedding
        if self.ape:
            pretrain_img_size = to_3tuple(self.pretrain_img_size)
            patch_size = to_3tuple(patch_size)
            patches_resolution = [pretrain_img_size[0] // patch_size[0], pretrain_img_size[1] // patch_size[1], pretrain_img_size[2] // patch_size[2]]

            self.absolute_pos_embed = nn.Parameter(
                torch.zeros(1, embed_dim, patches_resolution[0], patches_resolution[1], patches_resolution[2]))
            trunc_normal_(self.absolute_pos_embed, std=.02)
        elif self.spe:
            self.pos_embd = SinusoidalPositionEmbedding().cuda()
        self.pos_drop = nn.Dropout(p=drop_rate)

        # stochastic depth
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]  # stochastic depth decay rule

        # build layers
        self.layers = nn.ModuleList()
        self.patch_merging_layers = nn.ModuleList()

        for i_layer in range(self.num_layers):
            layer = BasicLayer(dim=int(embed_dim * 2 ** i_layer),
                                depth=depths[i_layer],
                                num_heads=num_heads[i_layer],
                                window_size=window_size,
                                mlp_ratio=mlp_ratio,
                                qkv_bias=qkv_bias,
                                qk_scale=qk_scale,
                                drop=drop_rate,
                                attn_drop=attn_drop_rate,
                                drop_path=dpr[sum(depths[:i_layer]):sum(depths[:i_layer + 1])],
                                norm_layer=norm_layer,
                                downsample=PatchMerging if (i_layer < self.num_layers - 1) else None,
                                use_checkpoint=use_checkpoint,
                                pat_merg_rf=pat_merg_rf,
                                pos_embed_method=pos_embed_method)
            self.layers.append(layer)

            patch_merging_layer = PatchMerging(int(embed_dim * 2 ** i_layer), reduce_factor=4, concatenated_input=False)
            self.patch_merging_layers.append(patch_merging_layer)

        num_features = [int(embed_dim * 2 ** i) for i in range(self.num_layers)]
        self.num_features = num_features



        # add a norm layer for each output
        for i_layer in out_indices:
            layer = norm_layer(num_features[i_layer]*2)
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

    def init_weights(self, pretrained=None):
        """Initialize the weights in backbone.
        Args:
            pretrained (str, optional): Path to pre-trained weights.
                Defaults to None.
        """

        def _init_weights(m):
            if isinstance(m, nn.Linear):
                trunc_normal_(m.weight, std=.02)
                if isinstance(m, nn.Linear) and m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.LayerNorm):
                nn.init.constant_(m.bias, 0)
                nn.init.constant_(m.weight, 1.0)

        if isinstance(pretrained, str):
            self.apply(_init_weights)
        elif pretrained is None:
            self.apply(_init_weights)
        else:
            raise TypeError('pretrained must be a str or None')

    def forward(self, x_0,x_1):
        """Forward function."""
        #PET image
        x_0 = self.patch_embed(x_0)
        Wh_x0, Ww_x0, Wt_x0 = x_0.size(2), x_0.size(3), x_0.size(4)
        if self.ape:
            # interpolate the position embedding to the corresponding size
            absolute_pos_embed_x0 = nnf.interpolate(self.absolute_pos_embed, size=(Wh_x0, Ww_x0, Wt_x0), mode='trilinear')
            x_0 = (x_0 + absolute_pos_embed_x0).flatten(2).transpose(1, 2)  # B Wh*Ww*Wt C
        elif self.spe:
            print(self.spe)
            x_0 = x_0.flatten(2).transpose(1, 2)
            x_0 += self.pos_embd(x_0)
        else:
            x_0 = x_0.flatten(2).transpose(1, 2)
        x_0 = self.pos_drop(x_0)

        #CT image
        x_1 = self.patch_embed(x_1)
        Wh_x1, Ww_x1, Wt_x1 = x_1.size(2), x_1.size(3), x_1.size(4)
        if self.ape:
            # interpolate the position embedding to the corresponding size
            absolute_pos_embed_x1 = nnf.interpolate(self.absolute_pos_embed, size=(Wh_x1, Ww_x1, Wt_x1),
                                                    mode='trilinear')
            x_1 = (x_1 + absolute_pos_embed_x1).flatten(2).transpose(1, 2)  # B Wh*Ww*Wt C
        elif self.spe:
            print(self.spe)
            x_1 = x_1.flatten(2).transpose(1, 2)
            x_1 += self.pos_embd(x_1)
        else:
            x_1 = x_1.flatten(2).transpose(1, 2)
        x_1 = self.pos_drop(x_1)


        outs = []

        #############layer0################
        layer = self.layers[0]
        # concatenate x_0 and x_1 in dimension 1
        x_0_1 = torch.cat((x_0,x_1),dim=2) #concatenate in the channel dimension so that the image dimension is still Wh_x0, Ww_x0, Wt_x0 which is needed to be compatible for SWINTR
        # send the concatenated feature vector to transformer
        x_out_x0_1_l0, H_x0_1, W_x0_1, T_x0_1, x_0_1, Wh_x0_1_l0, Ww_x0_1_l0, Wt_x0_1_l0 = layer(x_0_1, Wh_x0, Ww_x0, Wt_x0)
        # split the resulting feature vector in dimension 1 to get processed x_1_processed, x_0_processed
        x_out_x0_l0 = x_out_x0_1_l0[:,:,0:self.embed_dim]
        x_out_x1_l0 = x_out_x0_1_l0[:,:,self.embed_dim:]
        # add x_1_process and x_0_processed to x_1 and x_0 and start layer 1
        x_0_out = x_out_x0_l0 + x_0 # updated x_0
        x_1_out = x_out_x1_l0 + x_1 # updated x_1
        #out_x0_x1_l0 = x_0_out + x_1_out
        out_x0_x1_l0 = torch.concat((x_0_out , x_1_out),dim=2)

        norm_layer = getattr(self, f'norm{0}')
        x_out_l0 = norm_layer(out_x0_x1_l0)
        out = x_out_l0.view(-1, Wh_x0, Ww_x0, Wt_x0, self.embed_dim*2).permute(0, 4, 1, 2, 3).contiguous()
        outs.append(out) #layer 0 output

        #construct the input for the next layer
        x_0_down = self.patch_merging_layers[0](x_0, Wh_x0, Ww_x0, Wt_x0)
        x_1_down = self.patch_merging_layers[0](x_1, Wh_x1, Ww_x1, Wt_x1)
        x_0 = x_0_down
        x_1 = x_1_down

        #############layer1################
        layer = self.layers[1]
        # concatenate x_0 and x_1 in dimension 1
        x_0_1 = torch.cat((x_0, x_1),
                          dim=2)  # concatenate in the channel dimension so that the image dimension is still Wh_x0, Ww_x0, Wt_x0 which is needed to be compatible for SWINTR
        # send the concatenated feature vector to transformer
        x_out_x0_1_l1, H_x0_1, W_x0_1, T_x0_1, x_0_1, Wh_x0_1_l1, Ww_x0_1_l1, Wt_x0_1_l1 = layer(x_0_1, Wh_x0_1_l0, Ww_x0_1_l0, Wt_x0_1_l0)
        # split the resulting feature vector in dimension 1 to get processed x_1_processed, x_0_processed
        x_out_x0_l1 = x_out_x0_1_l1[:, :, 0:self.embed_dim*2]
        x_out_x1_l1 = x_out_x0_1_l1[:, :, self.embed_dim*2:]
        x_0_out = x_out_x0_l1 + x_0  # updated x_0
        x_1_out = x_out_x1_l1 + x_1  # updated x_1
        #out_x0_x1_l1 = x_0_out + x_1_out #should I use the sum or concat for decoder?
        out_x0_x1_l1 = torch.concat((x_0_out , x_1_out),dim=2)

        norm_layer = getattr(self, f'norm{1}')
        x_out_l1 = norm_layer(out_x0_x1_l1)
        out = x_out_l1.view(-1, Wh_x0_1_l0, Ww_x0_1_l0, Wt_x0_1_l0, self.embed_dim*4).permute(0, 4, 1, 2, 3).contiguous()
        outs.append(out)  # layer 1 output

        #construct the input for the next layer
        x_0_down = self.patch_merging_layers[1](x_0, Wh_x0_1_l0, Ww_x0_1_l0, Wt_x0_1_l0)
        x_1_down = self.patch_merging_layers[1](x_1, Wh_x0_1_l0, Ww_x0_1_l0, Wt_x0_1_l0)
        x_0 = x_0_down
        x_1 =  x_1_down

        #############layer2################
        layer = self.layers[2]
        # concatenate x_0 and x_1 in dimension 1
        x_0_1 = torch.cat((x_0, x_1),
                          dim=2)  # concatenate in the channel dimension so that the image dimension is still Wh_x0, Ww_x0, Wt_x0 which is needed to be compatible for SWINTR
        # send the concatenated feature vector to transformer
        x_out_x0_1_l2, H_x0_1, W_x0_1, T_x0_1, x_0_1, Wh_x0_1_l2, Ww_x0_1_l2, Wt_x0_1_l2 = layer(x_0_1, Wh_x0_1_l1,
                                                                                                 Ww_x0_1_l1, Wt_x0_1_l1)
        # split the resulting feature vector in dimension 1 to get processed x_1_processed, x_0_processed
        x_out_x0_l2 = x_out_x0_1_l2[:, :, 0:self.embed_dim * 4]
        x_out_x1_l2 = x_out_x0_1_l2[:, :, self.embed_dim * 4:]
        x_0_out = x_out_x0_l2 + x_0  # updated x_0
        x_1_out = x_out_x1_l2 + x_1  # updated x_1
        #out_x0_x1_l2 = x_0_out + x_1_out
        out_x0_x1_l2 = torch.concat((x_0_out , x_1_out),dim=2)

        norm_layer = getattr(self, f'norm{2}')
        x_out_l2 = norm_layer(out_x0_x1_l2)
        out = x_out_l2.view(-1, Wh_x0_1_l1, Ww_x0_1_l1, Wt_x0_1_l1, self.embed_dim*8).permute(0, 4, 1, 2, 3).contiguous()
        outs.append(out)  # layer 2 output

        # construct the input for the next layer
        x_0_down = self.patch_merging_layers[2](x_0, Wh_x0_1_l1, Ww_x0_1_l1, Wt_x0_1_l1)
        x_1_down = self.patch_merging_layers[2](x_1, Wh_x0_1_l1, Ww_x0_1_l1, Wt_x0_1_l1)
        x_0 = x_0_down
        x_1 = x_1_down

        #############layer3################
        layer = self.layers[3]
        # concatenate x_0 and x_1 in dimension 1
        x_0_1 = torch.cat((x_0, x_1),
                          dim=2)  # concatenate in the channel dimension so that the image dimension is still Wh_x0, Ww_x0, Wt_x0 which is needed to be compatible for SWINTR
        # send the concatenated feature vector to transformer
        x_out_x0_1_l3, H_x0_1, W_x0_1, T_x0_1, x_0_1, Wh_x0_1_l3, Ww_x0_1_l3, Wt_x0_1_l3 = layer(x_0_1, Wh_x0_1_l2,
                                                                                                 Ww_x0_1_l2, Wt_x0_1_l2)
        # split the resulting feature vector in dimension 1 to get processed x_1_processed, x_0_processed
        x_out_x0_l3 = x_out_x0_1_l3[:, :, 0:self.embed_dim * 8]
        x_out_x1_l3 = x_out_x0_1_l3[:, :, self.embed_dim * 8:]
        x_0_out = x_out_x0_l3 + x_0  # updated x_0
        x_1_out = x_out_x1_l3 + x_1  # updated x_1
        #out_x0_x1_l3 = x_0_out + x_1_out
        out_x0_x1_l3 = torch.concat((x_0_out , x_1_out),dim=2)

        norm_layer = getattr(self, f'norm{3}')
        x_out_l3 = norm_layer(out_x0_x1_l3)
        out = x_out_l3.view(-1, Wh_x0_1_l2, Ww_x0_1_l2, Wt_x0_1_l2, self.embed_dim*16).permute(0, 4, 1, 2, 3).contiguous()
        outs.append(out)  # layer 2 output

        return outs

    def train(self, mode=True):
        """Convert the model into training mode while keep layers freezed."""
        super(SwinTransformer_wFeatureTalk_concat_noCrossModalUpdating, self).train(mode)
        self._freeze_stages()





class SwinTransformer_wFeatureTalk_concat_PETUpdatingOnly_5stageOuts(nn.Module):
    r""" Swin Transformer modified to process images from two modalities; feature talks between two images are introduced in encoder
        A PyTorch impl of : `Swin Transformer: Hierarchical Vision Transformer using Shifted Windows`  -
          https://arxiv.org/pdf/2103.14030
    Args:
        img_size (int | tuple(int)): Input image size. Default 224
        patch_size (int | tuple(int)): Patch size. Default: 4
        in_chans (int): Number of input image channels. Default: 3
        num_classes (int): Number of classes for classification head. Default: 1000
        embed_dim (int): Patch embedding dimension. Default: 96
        depths (tuple(int)): Depth of each Swin Transformer layer.
        num_heads (tuple(int)): Number of attention heads in different layers.
        window_size (tuple): Window size. Default: 7
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim. Default: 4
        qkv_bias (bool): If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float): Override default qk scale of head_dim ** -0.5 if set. Default: None
        drop_rate (float): Dropout rate. Default: 0
        attn_drop_rate (float): Attention dropout rate. Default: 0
        drop_path_rate (float): Stochastic depth rate. Default: 0.1
        norm_layer (nn.Module): Normalization layer. Default: nn.LayerNorm.
        ape (bool): If True, add absolute position embedding to the patch embedding. Default: False
        patch_norm (bool): If True, add normalization after patch embedding. Default: True
        use_checkpoint (bool): Whether to use checkpointing to save memory. Default: False
    """

    def __init__(self, pretrain_img_size=160,
                 patch_size=4,
                 in_chans=3,
                 embed_dim=96,
                 depths=[2, 2, 6, 2],
                 num_heads=[3, 6, 12, 24],
                 window_size=(7, 7, 7),
                 mlp_ratio=4.,
                 qkv_bias=True,
                 qk_scale=None,
                 drop_rate=0.,
                 attn_drop_rate=0.,
                 drop_path_rate=0.2,
                 norm_layer=nn.LayerNorm,
                 ape=False,
                 spe=False,
                 patch_norm=True,
                 out_indices=(0, 1, 2, 3),
                 frozen_stages=-1,
                 use_checkpoint=False,
                 pat_merg_rf=2,
                 pos_embed_method='relative'):
        super().__init__()
        self.pretrain_img_size = pretrain_img_size
        self.num_layers = len(depths)
        self.embed_dim = embed_dim
        self.ape = ape
        self.spe = spe
        self.patch_norm = patch_norm
        self.out_indices = out_indices
        self.frozen_stages = frozen_stages

        # split image into non-overlapping patches
        self.patch_embed_mod0 = PatchEmbed(
            patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dim,
            norm_layer=norm_layer if self.patch_norm else None)

        self.patch_embed_mod1 = PatchEmbed(
            patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dim,
            norm_layer=norm_layer if self.patch_norm else None)

        # absolute position embedding
        if self.ape:
            pretrain_img_size = to_3tuple(self.pretrain_img_size)
            patch_size = to_3tuple(patch_size)
            patches_resolution = [pretrain_img_size[0] // patch_size[0], pretrain_img_size[1] // patch_size[1],
                                  pretrain_img_size[2] // patch_size[2]]

            self.absolute_pos_embed = nn.Parameter(
                torch.zeros(1, embed_dim, patches_resolution[0], patches_resolution[1], patches_resolution[2]))
            trunc_normal_(self.absolute_pos_embed, std=.02)
        elif self.spe:
            self.pos_embd = SinusoidalPositionEmbedding().cuda()
        self.pos_drop = nn.Dropout(p=drop_rate)

        # stochastic depth
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]  # stochastic depth decay rule

        # build layers
        self.layers = nn.ModuleList()
        self.patch_merging_layers = nn.ModuleList()

        for i_layer in range(self.num_layers):
            layer = BasicLayer(dim=int((embed_dim*2) * 2 ** i_layer),
                               depth=depths[i_layer],
                               num_heads=num_heads[i_layer],
                               window_size=window_size,
                               mlp_ratio=mlp_ratio,
                               qkv_bias=qkv_bias,
                               qk_scale=qk_scale,
                               drop=drop_rate,
                               attn_drop=attn_drop_rate,
                               drop_path=dpr[sum(depths[:i_layer]):sum(depths[:i_layer + 1])],
                               norm_layer=norm_layer,
                               downsample=PatchMerging if (i_layer < self.num_layers -1 ) else None,
                               use_checkpoint=use_checkpoint,
                               pat_merg_rf=pat_merg_rf,
                               pos_embed_method=pos_embed_method)
            self.layers.append(layer)

            # patch_merging_layer = PatchMerging(int(embed_dim * 2 ** i_layer), reduce_factor=4, concatenated_input=False)
            # self.patch_merging_layers.append(patch_merging_layer)
            patch_merging_layer = PatchConvPool(int(embed_dim * 2 ** i_layer), concatenated_input=False)
            self.patch_merging_layers.append(patch_merging_layer)

        num_features = [int((embed_dim*2) * 2 ** i) for i in range(self.num_layers)]
        self.num_features = num_features

        # add a norm layer for each output
        for i_layer in out_indices:
            layer = norm_layer(num_features[i_layer] * 1)
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

    def init_weights(self, pretrained=None):
        """Initialize the weights in backbone.
        Args:
            pretrained (str, optional): Path to pre-trained weights.
                Defaults to None.
        """

        def _init_weights(m):
            if isinstance(m, nn.Linear):
                trunc_normal_(m.weight, std=.02)
                if isinstance(m, nn.Linear) and m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.LayerNorm):
                nn.init.constant_(m.bias, 0)
                nn.init.constant_(m.weight, 1.0)

        if isinstance(pretrained, str):
            self.apply(_init_weights)
        elif pretrained is None:
            self.apply(_init_weights)
        else:
            raise TypeError('pretrained must be a str or None')

    def forward(self, x):
        """Forward function."""
        outs = []
        x_0 = torch.unsqueeze(x[:, 0, :, :, :], 1)
        x_1 = torch.unsqueeze(x[:, 1, :, :, :], 1)
        # PET image
        x_0 = self.patch_embed_mod0(x_0)
        Wh_x0, Ww_x0, Wt_x0 = x_0.size(2), x_0.size(3), x_0.size(4)
        if self.ape:
            # interpolate the position embedding to the corresponding size
            absolute_pos_embed_x0 = nnf.interpolate(self.absolute_pos_embed, size=(Wh_x0, Ww_x0, Wt_x0),
                                                    mode='trilinear')
            x_0 = (x_0 + absolute_pos_embed_x0).flatten(2).transpose(1, 2)  # B Wh*Ww*Wt C
        elif self.spe:
            print(self.spe)
            x_0 = x_0.flatten(2).transpose(1, 2)
            x_0 += self.pos_embd(x_0)
        else:
            x_0 = x_0.flatten(2).transpose(1, 2)
        x_0 = self.pos_drop(x_0)

        # CT image
        x_1 = self.patch_embed_mod1(x_1)
        Wh_x1, Ww_x1, Wt_x1 = x_1.size(2), x_1.size(3), x_1.size(4)
        if self.ape:
            # interpolate the position embedding to the corresponding size
            absolute_pos_embed_x1 = nnf.interpolate(self.absolute_pos_embed, size=(Wh_x1, Ww_x1, Wt_x1),
                                                    mode='trilinear')
            x_1 = (x_1 + absolute_pos_embed_x1).flatten(2).transpose(1, 2)  # B Wh*Ww*Wt C
        elif self.spe:
            print(self.spe)
            x_1 = x_1.flatten(2).transpose(1, 2)
            x_1 += self.pos_embd(x_1)
        else:
            x_1 = x_1.flatten(2).transpose(1, 2)
        x_1 = self.pos_drop(x_1)
        #print(x_0.size())
        outs.append((x_0.view(-1, Wh_x0, Ww_x0, Wt_x0, self.embed_dim).permute(0, 4, 1, 2, 3).contiguous()))

        x_0 = self.patch_merging_layers[0](x_0, Wh_x0, Ww_x0, Wt_x0)
        x_1 = self.patch_merging_layers[0](x_1, Wh_x1, Ww_x1, Wt_x1)

        #############layer0################
        layer = self.layers[0]
        # concatenate x_0 and x_1 in dimension 1
        x_0_1 = torch.cat((x_0, x_1),
                          dim=2)  # concatenate in the spatial dimension so that the SWINTR is looking at the correlations spatially
        # send the concatenated feature vector to transformer
        x_out_x0_1_l0, H_x0_1, W_x0_1, T_x0_1, x_0_1, Wh_x0_1_l0, Ww_x0_1_l0, Wt_x0_1_l0 = layer(x_0_1, int(Wh_x0/2), int(Ww_x0/2),
                                                                                                 int(Wt_x0/2))
        # split the resulting feature vector in dimension 1 to get processed x_1_processed, x_0_processed
        #x_out_x0_1_l0 = x_0_1
        x_out_x0_l0 = x_out_x0_1_l0[:, :, 0:self.embed_dim *2]
        x_out_x1_l0 = x_out_x0_1_l0[:, :, self.embed_dim*2:]
        # add x_1_process and x_0_processed to x_1 and x_0 and start layer 1
        x_0_out = x_out_x0_l0    # do not update x_0
        x_1_out = x_out_x1_l0   # do not update x_1
        out_x0_x1_l0 = x_0_out + x_0 + x_1 + x_1_out
        #out_x0_x1_l0 = torch.concat((x_0_out, x_0), dim=2)

        norm_layer = getattr(self, f'norm{0}')
        x_out_l0 = norm_layer(out_x0_x1_l0)
        out = x_out_l0.view(-1, H_x0_1, W_x0_1, T_x0_1, self.embed_dim * 2).permute(0, 4, 1, 2, 3).contiguous()
        outs.append(out)  # layer 0 output
        # add transformer encoded back to modality-specific branch
        x_0 = x_0_out + x_0
        x_1 = x_1_out + x_1
        # construct the input for the next layer
        x_0 = self.patch_merging_layers[1](x_0, int(Wh_x0/2), int(Ww_x0/2), int(Wt_x0/2))
        x_1 = self.patch_merging_layers[1](x_1, int(Wh_x1/2), int(Ww_x1/2), int(Wt_x1/2))


        #############layer1################
        layer = self.layers[1]
        # concatenate x_0 and x_1 in dimension 1
        x_0_1 = torch.cat((x_0, x_1),
                          dim=2)  # concatenate in the channel dimension so that the image dimension is still Wh_x0, Ww_x0, Wt_x0 which is needed to be compatible for SWINTR
        # send the concatenated feature vector to transformer
        x_out_x0_1_l1, H_x0_1, W_x0_1, T_x0_1, x_0_1, Wh_x0_1_l1, Ww_x0_1_l1, Wt_x0_1_l1 = layer(x_0_1, Wh_x0_1_l0,
                                                                                                 Ww_x0_1_l0, Wt_x0_1_l0)
        # split the resulting feature vector in dimension 1 to get processed x_1_processed, x_0_processed
        #x_out_x0_1_l1 = x_0_1
        x_out_x0_l1 = x_out_x0_1_l1[:, :, 0:self.embed_dim * 4]
        x_out_x1_l1 = x_out_x0_1_l1[:, :, self.embed_dim * 4:]
        x_0_out = x_out_x0_l1   # updated x_0
        x_1_out = x_out_x1_l1  # updated x_1
        out_x0_x1_l1 = x_0_out + x_0 + x_1 + x_1_out#should I use the sum or concat for decoder?
        #out_x0_x1_l1 = torch.concat((x_0_out, x_1_out), dim=2)

        norm_layer = getattr(self, f'norm{1}')
        x_out_l1 = norm_layer(out_x0_x1_l1)
        out = x_out_l1.view(-1, H_x0_1, W_x0_1, T_x0_1, self.embed_dim * 4).permute(0, 4, 1, 2,
                                                                                                3).contiguous()
        outs.append(out)  # layer 1 output
        # add transformer encoded back to modality-specific branch
        x_0 = x_0_out + x_0
        x_1 = x_1_out + x_1
        # construct the input for the next layer
        x_0 = self.patch_merging_layers[2](x_0, Wh_x0_1_l0, Ww_x0_1_l0, Wt_x0_1_l0)
        x_1 = self.patch_merging_layers[2](x_1, Wh_x0_1_l0, Ww_x0_1_l0, Wt_x0_1_l0)


        #############layer2################
        layer = self.layers[2]
        # concatenate x_0 and x_1 in dimension 1
        x_0_1 = torch.cat((x_0, x_1),
                          dim=2)  # concatenate in the channel dimension so that the image dimension is still Wh_x0, Ww_x0, Wt_x0 which is needed to be compatible for SWINTR
        # send the concatenated feature vector to transformer
        x_out_x0_1_l2, H_x0_1, W_x0_1, T_x0_1, x_0_1, Wh_x0_1_l2, Ww_x0_1_l2, Wt_x0_1_l2 = layer(x_0_1, Wh_x0_1_l1,
                                                                                                 Ww_x0_1_l1, Wt_x0_1_l1)
        # split the resulting feature vector in dimension 1 to get processed x_1_processed, x_0_processed
        #x_out_x0_1_l2 = x_0_1
        x_out_x0_l2 = x_out_x0_1_l2[:, :, 0:self.embed_dim * 8]
        x_out_x1_l2 = x_out_x0_1_l2[:, :, self.embed_dim * 8:]
        x_0_out = x_out_x0_l2  # updated x_0
        x_1_out = x_out_x1_l2   # updated x_1
        out_x0_x1_l2 = x_0_out + x_0 + x_1 + x_1_out
        #out_x0_x1_l2 = torch.concat((x_0_out, x_1_out), dim=2)

        norm_layer = getattr(self, f'norm{2}')
        x_out_l2 = norm_layer(out_x0_x1_l2)
        out = x_out_l2.view(-1, H_x0_1, W_x0_1, T_x0_1, self.embed_dim * 8).permute(0, 4, 1, 2,
                                                                                                3).contiguous()
        outs.append(out)  # layer 2 output
        # add transformer encoded back to modality-specific branch
        x_0 = x_0_out + x_0
        x_1 = x_1_out + x_1
        # construct the input for the next layer
        x_0 = self.patch_merging_layers[3](x_0, Wh_x0_1_l1, Ww_x0_1_l1, Wt_x0_1_l1)
        x_1 = self.patch_merging_layers[3](x_1, Wh_x0_1_l1, Ww_x0_1_l1, Wt_x0_1_l1)


        #############layer3################
        layer = self.layers[3]
        # concatenate x_0 and x_1 in dimension 1
        x_0_1 = torch.cat((x_0, x_1),
                          dim=2)  # concatenate in the channel dimension so that the image dimension is still Wh_x0, Ww_x0, Wt_x0 which is needed to be compatible for SWINTR
        # send the concatenated feature vector to transformer
        x_out_x0_1_l3, H_x0_1, W_x0_1, T_x0_1, x_0_1, Wh_x0_1_l3, Ww_x0_1_l3, Wt_x0_1_l3 = layer(x_0_1, Wh_x0_1_l2,
                                                                                                 Ww_x0_1_l2, Wt_x0_1_l2)
        # split the resulting feature vector in dimension 1 to get processed x_1_processed, x_0_processed
        #x_out_x0_1_l3 = x_0_1
        x_out_x0_l3 = x_out_x0_1_l3[:, :, 0:self.embed_dim * 16]
        x_out_x1_l3 = x_out_x0_1_l3[:, :, self.embed_dim * 16:]
        x_0_out = x_out_x0_l3   # updated x_0
        x_1_out = x_out_x1_l3   # updated x_1
        out_x0_x1_l3 = x_0_out + x_0 + x_1 + x_1_out
        #out_x0_x1_l3 = torch.concat((x_0_out, x_1_out), dim=2)

        norm_layer = getattr(self, f'norm{3}')
        x_out_l3 = norm_layer(out_x0_x1_l3)
        out = x_out_l3.view(-1, H_x0_1, W_x0_1, T_x0_1, self.embed_dim * 16).permute(0, 4, 1, 2,
                                                                                                 3).contiguous()
        outs.append(out)  # layer 2 output

        return outs

    def train(self, mode=True):
        """Convert the model into training mode while keep layers freezed."""
        super(SwinTransformer_wFeatureTalk_concat_PETUpdatingOnly_5stageOuts, self).train(mode)
        self._freeze_stages()





class SwinTransformer_wDualModalityFeatureTalk_OutConcat_5stageOuts(nn.Module):
    r""" Swin Transformer modified to process images from two modalities; feature talks between two images are introduced in encoder
        A PyTorch impl of : `Swin Transformer: Hierarchical Vision Transformer using Shifted Windows`  -
          https://arxiv.org/pdf/2103.14030
    Args:
        img_size (int | tuple(int)): Input image size. Default 224
        patch_size (int | tuple(int)): Patch size. Default: 4
        in_chans (int): Number of input image channels. Default: 3
        num_classes (int): Number of classes for classification head. Default: 1000
        embed_dim (int): Patch embedding dimension. Default: 96
        depths (tuple(int)): Depth of each Swin Transformer layer.
        num_heads (tuple(int)): Number of attention heads in different layers.
        window_size (tuple): Window size. Default: 7
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim. Default: 4
        qkv_bias (bool): If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float): Override default qk scale of head_dim ** -0.5 if set. Default: None
        drop_rate (float): Dropout rate. Default: 0
        attn_drop_rate (float): Attention dropout rate. Default: 0
        drop_path_rate (float): Stochastic depth rate. Default: 0.1
        norm_layer (nn.Module): Normalization layer. Default: nn.LayerNorm.
        ape (bool): If True, add absolute position embedding to the patch embedding. Default: False
        patch_norm (bool): If True, add normalization after patch embedding. Default: True
        use_checkpoint (bool): Whether to use checkpointing to save memory. Default: False
    """

    def __init__(self, pretrain_img_size=160,
                 patch_size=4,
                 in_chans=3,
                 embed_dim=96,
                 depths=[2, 2, 6, 2],
                 num_heads=[3, 6, 12, 24],
                 window_size=(7, 7, 7),
                 mlp_ratio=4.,
                 qkv_bias=True,
                 qk_scale=None,
                 drop_rate=0.,
                 attn_drop_rate=0.,
                 drop_path_rate=0.2,
                 norm_layer=nn.LayerNorm,
                 ape=False,
                 spe=False,
                 patch_norm=True,
                 out_indices=(0, 1, 2, 3),
                 frozen_stages=-1,
                 use_checkpoint=False,
                 pat_merg_rf=2,
                 pos_embed_method='relative'):
        super().__init__()
        self.pretrain_img_size = pretrain_img_size
        self.num_layers = len(depths)
        self.embed_dim = embed_dim
        self.ape = ape
        self.spe = spe
        self.patch_norm = patch_norm
        self.out_indices = out_indices
        self.frozen_stages = frozen_stages

        # split image into non-overlapping patches
        self.patch_embed_mod0 = PatchEmbed(
            patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dim,
            norm_layer=norm_layer if self.patch_norm else None)

        self.patch_embed_mod1 = PatchEmbed(
            patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dim,
            norm_layer=norm_layer if self.patch_norm else None)

        # absolute position embedding
        if self.ape:
            pretrain_img_size = to_3tuple(self.pretrain_img_size)
            patch_size = to_3tuple(patch_size)
            patches_resolution = [pretrain_img_size[0] // patch_size[0], pretrain_img_size[1] // patch_size[1],
                                  pretrain_img_size[2] // patch_size[2]]

            self.absolute_pos_embed = nn.Parameter(
                torch.zeros(1, embed_dim, patches_resolution[0], patches_resolution[1], patches_resolution[2]))
            trunc_normal_(self.absolute_pos_embed, std=.02)
        elif self.spe:
            self.pos_embd = SinusoidalPositionEmbedding().cuda()
        self.pos_drop = nn.Dropout(p=drop_rate)

        # stochastic depth
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]  # stochastic depth decay rule

        # build layers
        self.layers = nn.ModuleList()
        self.patch_merging_layers = nn.ModuleList()

        for i_layer in range(self.num_layers):
            layer = BasicLayer_dualModality(dim=int((embed_dim) * 2 ** i_layer),
                               depth=depths[i_layer],
                               num_heads=num_heads[i_layer],
                               window_size=window_size,
                               mlp_ratio=mlp_ratio,
                               qkv_bias=qkv_bias,
                               qk_scale=qk_scale,
                               drop=drop_rate,
                               attn_drop=attn_drop_rate,
                               drop_path=dpr[sum(depths[:i_layer]):sum(depths[:i_layer + 1])],
                               norm_layer=norm_layer,
                               downsample=PatchMerging if (i_layer < self.num_layers -1 ) else None,
                               use_checkpoint=use_checkpoint,
                               pat_merg_rf=pat_merg_rf,
                               pos_embed_method=pos_embed_method)
            self.layers.append(layer)

            patch_merging_layer = PatchMerging(int(embed_dim * 2 ** i_layer), reduce_factor=4, concatenated_input=False)
            # self.patch_merging_layers.append(patch_merging_layer)
            #patch_merging_layer = PatchConvPool(int(embed_dim * 2 ** i_layer), concatenated_input=False)
            self.patch_merging_layers.append(patch_merging_layer)

        num_features = [int((embed_dim*4) * 2 ** i) for i in range(self.num_layers)]
        self.num_features = num_features

        # add a norm layer for each output
        for i_layer in out_indices:
            layer = norm_layer(num_features[i_layer] * 1)
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

    def init_weights(self, pretrained=None):
        """Initialize the weights in backbone.
        Args:
            pretrained (str, optional): Path to pre-trained weights.
                Defaults to None.
        """

        def _init_weights(m):
            if isinstance(m, nn.Linear):
                trunc_normal_(m.weight, std=.02)
                if isinstance(m, nn.Linear) and m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.LayerNorm):
                nn.init.constant_(m.bias, 0)
                nn.init.constant_(m.weight, 1.0)

        if isinstance(pretrained, str):
            self.apply(_init_weights)
        elif pretrained is None:
            self.apply(_init_weights)
        else:
            raise TypeError('pretrained must be a str or None')

    def forward(self, x):
        """Forward function."""
        outs = []
        x_0 = torch.unsqueeze(x[:, 0, :, :, :], 1)
        x_1 = torch.unsqueeze(x[:, 1, :, :, :], 1)
        # PET image
        x_0 = self.patch_embed_mod0(x_0)
        Wh_x0, Ww_x0, Wt_x0 = x_0.size(2), x_0.size(3), x_0.size(4)
        if self.ape:
            # interpolate the position embedding to the corresponding size
            absolute_pos_embed_x0 = nnf.interpolate(self.absolute_pos_embed, size=(Wh_x0, Ww_x0, Wt_x0),
                                                    mode='trilinear')
            x_0 = (x_0 + absolute_pos_embed_x0).flatten(2).transpose(1, 2)  # B Wh*Ww*Wt C
        elif self.spe:
            print(self.spe)
            x_0 = x_0.flatten(2).transpose(1, 2)
            x_0 += self.pos_embd(x_0)
        else:
            x_0 = x_0.flatten(2).transpose(1, 2)
        x_0 = self.pos_drop(x_0)

        # CT image
        x_1 = self.patch_embed_mod1(x_1) # B C, W, H ,D
        Wh_x1, Ww_x1, Wt_x1 = x_1.size(2), x_1.size(3), x_1.size(4)
        if self.ape:
            # interpolate the position embedding to the corresponding size
            absolute_pos_embed_x1 = nnf.interpolate(self.absolute_pos_embed, size=(Wh_x1, Ww_x1, Wt_x1),
                                                    mode='trilinear')
            x_1 = (x_1 + absolute_pos_embed_x1).flatten(2).transpose(1, 2)  # B Wh*Ww*Wt C
        elif self.spe:
            print(self.spe)
            x_1 = x_1.flatten(2).transpose(1, 2)
            x_1 += self.pos_embd(x_1)
        else:
            x_1 = x_1.flatten(2).transpose(1, 2)
        x_1 = self.pos_drop(x_1)
        #print(x_0.size())

        out = torch.cat((x_0,x_1),dim=2)
        #out = x_0+x_1

        outs.append((out.view(-1, Wh_x0, Ww_x0, Wt_x0, self.embed_dim*2).permute(0, 4, 1, 2, 3).contiguous()))

        x_0 = self.patch_merging_layers[0](x_0, Wh_x0, Ww_x0, Wt_x0)
        x_1 = self.patch_merging_layers[0](x_1, Wh_x1, Ww_x1, Wt_x1)

        #############layer0################
        layer = self.layers[0]
        # concatenate x_0 and x_1 in dimension 1
        # x_0_1 = torch.cat((x_0, x_1),
        #                   dim=1)  # concatenate in the spatial dimension so that the SWINTR is looking at the correlations spatially
        # send the concatenated feature vector to transformer
        x_out_x0_l0, x_out_x1_l0, H_x0_1, W_x0_1, T_x0_1, x_0_small, x_1_small, Wh_x0_1_l0, Ww_x0_1_l0, Wt_x0_1_l0 = layer(x_0,x_1,int(Wh_x0/2), int(Ww_x0/2),
                                                                                                 int(Wt_x0/2))
        # split the resulting feature vector in dimension 1 to get processed x_1_processed, x_0_processed
        #x_out_x0_1_l0 = x_0_1
        x_0_out = x_out_x0_l0
        x_1_out = x_out_x1_l0
        # add x_1_process and x_0_processed to x_1 and x_0 and start layer 1
        #x_0_out = x_out_x0_l0    # do not update x_0
        #x_1_out = x_out_x1_l0   # do not update x_1
        #out_x0_x1_l0 = x_0_out + x_1_out
        out_x0_x1_l0 = torch.concat((x_0_out, x_1_out), dim=2)

        norm_layer = getattr(self, f'norm{0}')
        x_out_l0 = norm_layer(out_x0_x1_l0)
        out = x_out_l0.view(-1, int(H_x0_1), W_x0_1, T_x0_1, self.embed_dim * 4).permute(0, 4, 1, 2, 3).contiguous()
        outs.append(out)  # layer 0 output
        # add transformer encoded back to modality-specific branch
        x_0 = x_0_out + x_0
        x_1 = x_1_out + x_1
        # construct the input for the next layer
        x_0 = self.patch_merging_layers[1](x_0, int(Wh_x0/2), int(Ww_x0/2), int(Wt_x0/2))
        x_1 = self.patch_merging_layers[1](x_1, int(Wh_x1/2), int(Ww_x1/2), int(Wt_x1/2))


        #############layer1################
        layer = self.layers[1]
        # concatenate x_0 and x_1 in dimension 1
        #x_0_1 = torch.cat((x_0, x_1),
        #                  dim=1)  # concatenate in the channel dimension so that the image dimension is still Wh_x0, Ww_x0, Wt_x0 which is needed to be compatible for SWINTR
        # send the concatenated feature vector to transformer
        x_out_x0_l1,x_out_x1_l1, H_x0_1, W_x0_1, T_x0_1, x_0_small,x_1_small, Wh_x0_1_l1, Ww_x0_1_l1, Wt_x0_1_l1 = layer(x_0,x_1, int(Wh_x0_1_l0),
                                                                                                 int(Ww_x0_1_l0), int(Wt_x0_1_l0))
        # split the resulting feature vector in dimension 1 to get processed x_1_processed, x_0_processed
        #x_out_x0_1_l1 = x_0_1
        x_0_out = x_out_x0_l1
        x_1_out = x_out_x1_l1
        #x_out_x0_l1 = x_out_x0_1_l1[:, :, 0:self.embed_dim * 4]
        #x_out_x1_l1 = x_out_x0_1_l1[:, :, self.embed_dim * 4:]
        #x_0_out = x_out_x0_l1   # updated x_0
        #x_1_out = x_out_x1_l1  # updated x_1
        #out_x0_x1_l1 = x_0_out + x_1_out #should I use the sum or concat for decoder?
        out_x0_x1_l1 = torch.concat((x_0_out, x_1_out), dim=2)

        norm_layer = getattr(self, f'norm{1}')
        x_out_l1 = norm_layer(out_x0_x1_l1)
        out = x_out_l1.view(-1, int(H_x0_1), W_x0_1, T_x0_1, self.embed_dim * 8).permute(0, 4, 1, 2,
                                                                                                3).contiguous()
        outs.append(out)  # layer 1 output
        # add transformer encoded back to modality-specific branch
        x_0 = x_0_out + x_0
        x_1 = x_1_out + x_1
        # construct the input for the next layer
        x_0 = self.patch_merging_layers[2](x_0, int(Wh_x0_1_l0), int(Ww_x0_1_l0), int(Wt_x0_1_l0))
        x_1 = self.patch_merging_layers[2](x_1, int(Wh_x0_1_l0), int(Ww_x0_1_l0), int(Wt_x0_1_l0))


        #############layer2################
        layer = self.layers[2]
        # concatenate x_0 and x_1 in dimension 1
        #x_0_1 = torch.cat((x_0, x_1),
        #                  dim=1)  # concatenate in the channel dimension so that the image dimension is still Wh_x0, Ww_x0, Wt_x0 which is needed to be compatible for SWINTR
        # send the concatenated feature vector to transformer
        x_out_x0_l2, x_out_x1_l2,H_x0_1, W_x0_1, T_x0_1, x_0_small,x_1_small, Wh_x0_1_l2, Ww_x0_1_l2, Wt_x0_1_l2 = layer(x_0,x_1, int(Wh_x0_1_l1),
                                                                                                 int(Ww_x0_1_l1), int(Wt_x0_1_l1))
        # split the resulting feature vector in dimension 1 to get processed x_1_processed, x_0_processed
        #x_out_x0_1_l2 = x_0_1
        x_0_out = x_out_x0_l2
        x_1_out = x_out_x1_l2
        #x_out_x0_l2 = x_out_x0_1_l2[:, :, 0:self.embed_dim * 8]
        #x_out_x1_l2 = x_out_x0_1_l2[:, :, self.embed_dim * 8:]
        #x_0_out = x_out_x0_l2  # updated x_0
        #x_1_out = x_out_x1_l2   # updated x_1
        #out_x0_x1_l2 = x_0_out  + x_1_out
        out_x0_x1_l2 = torch.concat((x_0_out, x_1_out), dim=2)

        norm_layer = getattr(self, f'norm{2}')
        x_out_l2 = norm_layer(out_x0_x1_l2)
        out = x_out_l2.view(-1, int(H_x0_1), W_x0_1, T_x0_1, self.embed_dim * 16).permute(0, 4, 1, 2,
                                                                                                3).contiguous()
        outs.append(out)  # layer 2 output
        # add transformer encoded back to modality-specific branch
        x_0 = x_0_out + x_0
        x_1 = x_1_out + x_1
        # construct the input for the next layer
        x_0 = self.patch_merging_layers[3](x_0, int(Wh_x0_1_l1), int(Ww_x0_1_l1), int(Wt_x0_1_l1))
        x_1 = self.patch_merging_layers[3](x_1, int(Wh_x0_1_l1), int(Ww_x0_1_l1), int(Wt_x0_1_l1))


        #############layer3################
        layer = self.layers[3]
        # concatenate x_0 and x_1 in dimension 1
        #x_0_1 = torch.cat((x_0, x_1),
        #                  dim=1)  # concatenate in the channel dimension so that the image dimension is still Wh_x0, Ww_x0, Wt_x0 which is needed to be compatible for SWINTR
        # send the concatenated feature vector to transformer
        x_out_x0_l3, x_out_x1_l3,H_x0_1, W_x0_1, T_x0_1, x_0_small,x_1_small, Wh_x0_1_l3, Ww_x0_1_l3, Wt_x0_1_l3 = layer(x_0,x_1, int(Wh_x0_1_l2),
                                                                                                 int(Ww_x0_1_l2), int(Wt_x0_1_l2))
        # split the resulting feature vector in dimension 1 to get processed x_1_processed, x_0_processed
        #x_out_x0_1_l3 = x_0_1
        x_0_out = x_out_x0_l3
        x_1_out = x_out_x1_l3
        #x_out_x0_l3 = x_out_x0_1_l3[:, :, 0:self.embed_dim * 16]
        #x_out_x1_l3 = x_out_x0_1_l3[:, :, self.embed_dim * 16:]
        #x_0_out = x_out_x0_l3   # updated x_0
        #x_1_out = x_out_x1_l3   # updated x_1
        #out_x0_x1_l3 = x_0_out + x_1_out
        out_x0_x1_l3 = torch.concat((x_0_out, x_1_out), dim=2)

        norm_layer = getattr(self, f'norm{3}')
        x_out_l3 = norm_layer(out_x0_x1_l3)
        out = x_out_l3.view(-1, int(H_x0_1), W_x0_1, T_x0_1, self.embed_dim * 32).permute(0, 4, 1, 2,
                                                                                                 3).contiguous()
        outs.append(out)  # layer 2 output

        return outs

    def train(self, mode=True):
        """Convert the model into training mode while keep layers freezed."""
        super(SwinTransformer_wDualModalityFeatureTalk_OutConcat_5stageOuts, self).train(mode)
        self._freeze_stages()



class SwinTransformer_wDualModalityFeatureTalk_OutSum_5stageOuts(nn.Module):
    r""" Swin Transformer modified to process images from two modalities; feature talks between two images are introduced in encoder
        A PyTorch impl of : `Swin Transformer: Hierarchical Vision Transformer using Shifted Windows`  -
          https://arxiv.org/pdf/2103.14030
    Args:
        img_size (int | tuple(int)): Input image size. Default 224
        patch_size (int | tuple(int)): Patch size. Default: 4
        in_chans (int): Number of input image channels. Default: 3
        num_classes (int): Number of classes for classification head. Default: 1000
        embed_dim (int): Patch embedding dimension. Default: 96
        depths (tuple(int)): Depth of each Swin Transformer layer.
        num_heads (tuple(int)): Number of attention heads in different layers.
        window_size (tuple): Window size. Default: 7
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim. Default: 4
        qkv_bias (bool): If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float): Override default qk scale of head_dim ** -0.5 if set. Default: None
        drop_rate (float): Dropout rate. Default: 0
        attn_drop_rate (float): Attention dropout rate. Default: 0
        drop_path_rate (float): Stochastic depth rate. Default: 0.1
        norm_layer (nn.Module): Normalization layer. Default: nn.LayerNorm.
        ape (bool): If True, add absolute position embedding to the patch embedding. Default: False
        patch_norm (bool): If True, add normalization after patch embedding. Default: True
        use_checkpoint (bool): Whether to use checkpointing to save memory. Default: False
    """

    def __init__(self, pretrain_img_size=160,
                 patch_size=4,
                 in_chans=3,
                 embed_dim=96,
                 depths=[2, 2, 6, 2],
                 num_heads=[3, 6, 12, 24],
                 window_size=(7, 7, 7),
                 mlp_ratio=4.,
                 qkv_bias=True,
                 qk_scale=None,
                 drop_rate=0.,
                 attn_drop_rate=0.,
                 drop_path_rate=0.2,
                 norm_layer=nn.LayerNorm,
                 ape=False,
                 spe=False,
                 patch_norm=True,
                 out_indices=(0, 1, 2, 3),
                 frozen_stages=-1,
                 use_checkpoint=False,
                 pat_merg_rf=2,
                 pos_embed_method='relative'):
        super().__init__()
        self.pretrain_img_size = pretrain_img_size
        self.num_layers = len(depths)
        self.embed_dim = embed_dim
        self.ape = ape
        self.spe = spe
        self.patch_norm = patch_norm
        self.out_indices = out_indices
        self.frozen_stages = frozen_stages

        # split image into non-overlapping patches
        self.patch_embed_mod0 = PatchEmbed(
            patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dim,
            norm_layer=norm_layer if self.patch_norm else None)

        self.patch_embed_mod1 = PatchEmbed(
            patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dim,
            norm_layer=norm_layer if self.patch_norm else None)

        # absolute position embedding
        if self.ape:
            pretrain_img_size = to_3tuple(self.pretrain_img_size)
            patch_size = to_3tuple(patch_size)
            patches_resolution = [pretrain_img_size[0] // patch_size[0], pretrain_img_size[1] // patch_size[1],
                                  pretrain_img_size[2] // patch_size[2]]

            self.absolute_pos_embed = nn.Parameter(
                torch.zeros(1, embed_dim, patches_resolution[0], patches_resolution[1], patches_resolution[2]))
            trunc_normal_(self.absolute_pos_embed, std=.02)
        elif self.spe:
            self.pos_embd = SinusoidalPositionEmbedding().cuda()
        self.pos_drop = nn.Dropout(p=drop_rate)

        # stochastic depth
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]  # stochastic depth decay rule

        # build layers
        self.layers = nn.ModuleList()
        self.patch_merging_layers = nn.ModuleList()

        for i_layer in range(self.num_layers):
            layer = BasicLayer_dualModality(dim=int((embed_dim) * 2 ** i_layer),
                               depth=depths[i_layer],
                               num_heads=num_heads[i_layer],
                               window_size=window_size,
                               mlp_ratio=mlp_ratio,
                               qkv_bias=qkv_bias,
                               qk_scale=qk_scale,
                               drop=drop_rate,
                               attn_drop=attn_drop_rate,
                               drop_path=dpr[sum(depths[:i_layer]):sum(depths[:i_layer + 1])],
                               norm_layer=norm_layer,
                               downsample=PatchMerging if (i_layer < self.num_layers -1 ) else None,
                               use_checkpoint=use_checkpoint,
                               pat_merg_rf=pat_merg_rf,
                               pos_embed_method=pos_embed_method)
            self.layers.append(layer)

            patch_merging_layer = PatchMerging(int(embed_dim * 2 ** i_layer), reduce_factor=4, concatenated_input=False)
            # self.patch_merging_layers.append(patch_merging_layer)
            #patch_merging_layer = PatchConvPool(int(embed_dim * 2 ** i_layer), concatenated_input=False)
            self.patch_merging_layers.append(patch_merging_layer)

        num_features = [int((embed_dim*2) * 2 ** i) for i in range(self.num_layers)]
        self.num_features = num_features

        # add a norm layer for each output
        for i_layer in out_indices:
            layer = norm_layer(num_features[i_layer] * 1)
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

    def init_weights(self, pretrained=None):
        """Initialize the weights in backbone.
        Args:
            pretrained (str, optional): Path to pre-trained weights.
                Defaults to None.
        """

        def _init_weights(m):
            if isinstance(m, nn.Linear):
                trunc_normal_(m.weight, std=.02)
                if isinstance(m, nn.Linear) and m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.LayerNorm):
                nn.init.constant_(m.bias, 0)
                nn.init.constant_(m.weight, 1.0)

        if isinstance(pretrained, str):
            self.apply(_init_weights)
        elif pretrained is None:
            self.apply(_init_weights)
        else:
            raise TypeError('pretrained must be a str or None')

    def forward(self, x):
        """Forward function."""
        outs = []
        x_0 = torch.unsqueeze(x[:, 0, :, :, :], 1)
        x_1 = torch.unsqueeze(x[:, 1, :, :, :], 1)
        # PET image
        x_0 = self.patch_embed_mod0(x_0)
        Wh_x0, Ww_x0, Wt_x0 = x_0.size(2), x_0.size(3), x_0.size(4)
        if self.ape:
            # interpolate the position embedding to the corresponding size
            absolute_pos_embed_x0 = nnf.interpolate(self.absolute_pos_embed, size=(Wh_x0, Ww_x0, Wt_x0),
                                                    mode='trilinear')
            x_0 = (x_0 + absolute_pos_embed_x0).flatten(2).transpose(1, 2)  # B Wh*Ww*Wt C
        elif self.spe:
            print(self.spe)
            x_0 = x_0.flatten(2).transpose(1, 2)
            x_0 += self.pos_embd(x_0)
        else:
            x_0 = x_0.flatten(2).transpose(1, 2)
        x_0 = self.pos_drop(x_0)

        # CT image
        x_1 = self.patch_embed_mod1(x_1) # B C, W, H ,D
        Wh_x1, Ww_x1, Wt_x1 = x_1.size(2), x_1.size(3), x_1.size(4)
        if self.ape:
            # interpolate the position embedding to the corresponding size
            absolute_pos_embed_x1 = nnf.interpolate(self.absolute_pos_embed, size=(Wh_x1, Ww_x1, Wt_x1),
                                                    mode='trilinear')
            x_1 = (x_1 + absolute_pos_embed_x1).flatten(2).transpose(1, 2)  # B Wh*Ww*Wt C
        elif self.spe:
            print(self.spe)
            x_1 = x_1.flatten(2).transpose(1, 2)
            x_1 += self.pos_embd(x_1)
        else:
            x_1 = x_1.flatten(2).transpose(1, 2)
        x_1 = self.pos_drop(x_1)
        #print(x_0.size())

        #out = torch.cat((x_0,x_1),dim=2)
        out = x_0 + x_1

        outs.append((out.view(-1, Wh_x0, Ww_x0, Wt_x0, self.embed_dim).permute(0, 4, 1, 2, 3).contiguous()))

        x_0 = self.patch_merging_layers[0](x_0, Wh_x0, Ww_x0, Wt_x0)
        x_1 = self.patch_merging_layers[0](x_1, Wh_x1, Ww_x1, Wt_x1)

        #############layer0################
        layer = self.layers[0]
        # concatenate x_0 and x_1 in dimension 1
        # x_0_1 = torch.cat((x_0, x_1),
        #                   dim=1)  # concatenate in the spatial dimension so that the SWINTR is looking at the correlations spatially
        # send the concatenated feature vector to transformer
        x_out_x0_l0, x_out_x1_l0, H_x0_1, W_x0_1, T_x0_1, x_0_small, x_1_small, Wh_x0_1_l0, Ww_x0_1_l0, Wt_x0_1_l0 = layer(x_0,x_1,int(Wh_x0/2), int(Ww_x0/2),
                                                                                                 int(Wt_x0/2))
        # split the resulting feature vector in dimension 1 to get processed x_1_processed, x_0_processed
        #x_out_x0_1_l0 = x_0_1
        x_0_out = x_out_x0_l0
        x_1_out = x_out_x1_l0
        # add x_1_process and x_0_processed to x_1 and x_0 and start layer 1
        #x_0_out = x_out_x0_l0    # do not update x_0
        #x_1_out = x_out_x1_l0   # do not update x_1
        out_x0_x1_l0 = x_0_out + x_1_out
        #out_x0_x1_l0 = torch.concat((x_0_out, x_1_out), dim=2)

        norm_layer = getattr(self, f'norm{0}')
        x_out_l0 = norm_layer(out_x0_x1_l0)
        out = x_out_l0.view(-1, int(H_x0_1), W_x0_1, T_x0_1, self.embed_dim * 2).permute(0, 4, 1, 2, 3).contiguous()
        outs.append(out)  # layer 0 output
        # add transformer encoded back to modality-specific branch
        x_0 = x_0_out + x_0
        x_1 = x_1_out + x_1
        # construct the input for the next layer
        x_0 = self.patch_merging_layers[1](x_0, int(Wh_x0/2), int(Ww_x0/2), int(Wt_x0/2))
        x_1 = self.patch_merging_layers[1](x_1, int(Wh_x1/2), int(Ww_x1/2), int(Wt_x1/2))


        #############layer1################
        layer = self.layers[1]
        # concatenate x_0 and x_1 in dimension 1
        #x_0_1 = torch.cat((x_0, x_1),
        #                  dim=1)  # concatenate in the channel dimension so that the image dimension is still Wh_x0, Ww_x0, Wt_x0 which is needed to be compatible for SWINTR
        # send the concatenated feature vector to transformer
        x_out_x0_l1,x_out_x1_l1, H_x0_1, W_x0_1, T_x0_1, x_0_small,x_1_small, Wh_x0_1_l1, Ww_x0_1_l1, Wt_x0_1_l1 = layer(x_0,x_1, int(Wh_x0_1_l0),
                                                                                                 int(Ww_x0_1_l0), int(Wt_x0_1_l0))
        # split the resulting feature vector in dimension 1 to get processed x_1_processed, x_0_processed
        #x_out_x0_1_l1 = x_0_1
        x_0_out = x_out_x0_l1
        x_1_out = x_out_x1_l1
        #x_out_x0_l1 = x_out_x0_1_l1[:, :, 0:self.embed_dim * 4]
        #x_out_x1_l1 = x_out_x0_1_l1[:, :, self.embed_dim * 4:]
        #x_0_out = x_out_x0_l1   # updated x_0
        #x_1_out = x_out_x1_l1  # updated x_1
        out_x0_x1_l1 = x_0_out + x_1_out #should I use the sum or concat for decoder?
        #out_x0_x1_l1 = torch.concat((x_0_out, x_1_out), dim=2)

        norm_layer = getattr(self, f'norm{1}')
        x_out_l1 = norm_layer(out_x0_x1_l1)
        out = x_out_l1.view(-1, int(H_x0_1), W_x0_1, T_x0_1, self.embed_dim * 4).permute(0, 4, 1, 2,
                                                                                                3).contiguous()
        outs.append(out)  # layer 1 output
        # add transformer encoded back to modality-specific branch
        x_0 = x_0_out + x_0
        x_1 = x_1_out + x_1
        # construct the input for the next layer
        x_0 = self.patch_merging_layers[2](x_0, int(Wh_x0_1_l0), int(Ww_x0_1_l0), int(Wt_x0_1_l0))
        x_1 = self.patch_merging_layers[2](x_1, int(Wh_x0_1_l0), int(Ww_x0_1_l0), int(Wt_x0_1_l0))


        #############layer2################
        layer = self.layers[2]
        # concatenate x_0 and x_1 in dimension 1
        #x_0_1 = torch.cat((x_0, x_1),
        #                  dim=1)  # concatenate in the channel dimension so that the image dimension is still Wh_x0, Ww_x0, Wt_x0 which is needed to be compatible for SWINTR
        # send the concatenated feature vector to transformer
        x_out_x0_l2, x_out_x1_l2,H_x0_1, W_x0_1, T_x0_1, x_0_small,x_1_small, Wh_x0_1_l2, Ww_x0_1_l2, Wt_x0_1_l2 = layer(x_0,x_1, int(Wh_x0_1_l1),
                                                                                                 int(Ww_x0_1_l1), int(Wt_x0_1_l1))
        # split the resulting feature vector in dimension 1 to get processed x_1_processed, x_0_processed
        #x_out_x0_1_l2 = x_0_1
        x_0_out = x_out_x0_l2
        x_1_out = x_out_x1_l2
        #x_out_x0_l2 = x_out_x0_1_l2[:, :, 0:self.embed_dim * 8]
        #x_out_x1_l2 = x_out_x0_1_l2[:, :, self.embed_dim * 8:]
        #x_0_out = x_out_x0_l2  # updated x_0
        #x_1_out = x_out_x1_l2   # updated x_1
        out_x0_x1_l2 = x_0_out  + x_1_out
        #out_x0_x1_l2 = torch.concat((x_0_out, x_1_out), dim=2)

        norm_layer = getattr(self, f'norm{2}')
        x_out_l2 = norm_layer(out_x0_x1_l2)
        out = x_out_l2.view(-1, int(H_x0_1), W_x0_1, T_x0_1, self.embed_dim * 8).permute(0, 4, 1, 2,
                                                                                                3).contiguous()
        outs.append(out)  # layer 2 output
        # add transformer encoded back to modality-specific branch
        x_0 = x_0_out + x_0
        x_1 = x_1_out + x_1
        # construct the input for the next layer
        x_0 = self.patch_merging_layers[3](x_0, int(Wh_x0_1_l1), int(Ww_x0_1_l1), int(Wt_x0_1_l1))
        x_1 = self.patch_merging_layers[3](x_1, int(Wh_x0_1_l1), int(Ww_x0_1_l1), int(Wt_x0_1_l1))


        #############layer3################
        layer = self.layers[3]
        # concatenate x_0 and x_1 in dimension 1
        #x_0_1 = torch.cat((x_0, x_1),
        #                  dim=1)  # concatenate in the channel dimension so that the image dimension is still Wh_x0, Ww_x0, Wt_x0 which is needed to be compatible for SWINTR
        # send the concatenated feature vector to transformer
        x_out_x0_l3, x_out_x1_l3,H_x0_1, W_x0_1, T_x0_1, x_0_small,x_1_small, Wh_x0_1_l3, Ww_x0_1_l3, Wt_x0_1_l3 = layer(x_0,x_1, int(Wh_x0_1_l2),
                                                                                                 int(Ww_x0_1_l2), int(Wt_x0_1_l2))
        # split the resulting feature vector in dimension 1 to get processed x_1_processed, x_0_processed
        #x_out_x0_1_l3 = x_0_1
        x_0_out = x_out_x0_l3
        x_1_out = x_out_x1_l3
        #x_out_x0_l3 = x_out_x0_1_l3[:, :, 0:self.embed_dim * 16]
        #x_out_x1_l3 = x_out_x0_1_l3[:, :, self.embed_dim * 16:]
        #x_0_out = x_out_x0_l3   # updated x_0
        #x_1_out = x_out_x1_l3   # updated x_1
        out_x0_x1_l3 = x_0_out + x_1_out
        #out_x0_x1_l3 = torch.concat((x_0_out, x_1_out), dim=2)

        norm_layer = getattr(self, f'norm{3}')
        x_out_l3 = norm_layer(out_x0_x1_l3)
        out = x_out_l3.view(-1, int(H_x0_1), W_x0_1, T_x0_1, self.embed_dim * 16).permute(0, 4, 1, 2,
                                                                                                 3).contiguous()
        outs.append(out)  # layer 2 output

        return outs

    def train(self, mode=True):
        """Convert the model into training mode while keep layers freezed."""
        super(SwinTransformer_wDualModalityFeatureTalk_OutSum_5stageOuts, self).train(mode)
        self._freeze_stages()

from monai.utils import optional_import
rearrange, _ = optional_import("einops", name="rearrange")
import torch.nn.functional as F

class SwinTransformer_wCrossModalityFeatureTalk_OutSum_5stageOuts(nn.Module):
    r""" Swin Transformer modified to process images from two modalities; feature talks between two images are introduced in encoder
        A PyTorch impl of : `Swin Transformer: Hierarchical Vision Transformer using Shifted Windows`  -
          https://arxiv.org/pdf/2103.14030
    Args:
        img_size (int | tuple(int)): Input image size. Default 224
        patch_size (int | tuple(int)): Patch size. Default: 4
        in_chans (int): Number of input image channels. Default: 3
        num_classes (int): Number of classes for classification head. Default: 1000
        embed_dim (int): Patch embedding dimension. Default: 96
        depths (tuple(int)): Depth of each Swin Transformer layer.
        num_heads (tuple(int)): Number of attention heads in different layers.
        window_size (tuple): Window size. Default: 7
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim. Default: 4
        qkv_bias (bool): If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float): Override default qk scale of head_dim ** -0.5 if set. Default: None
        drop_rate (float): Dropout rate. Default: 0
        attn_drop_rate (float): Attention dropout rate. Default: 0
        drop_path_rate (float): Stochastic depth rate. Default: 0.1
        norm_layer (nn.Module): Normalization layer. Default: nn.LayerNorm.
        ape (bool): If True, add absolute position embedding to the patch embedding. Default: False
        patch_norm (bool): If True, add normalization after patch embedding. Default: True
        use_checkpoint (bool): Whether to use checkpointing to save memory. Default: False
    """

    def __init__(self, pretrain_img_size=160,
                 patch_size=4,
                 in_chans=3,
                 embed_dim=96,
                 depths=[2, 2, 6, 2],
                 num_heads=[3, 6, 12, 24],
                 window_size=(7, 7, 7),
                 mlp_ratio=4.,
                 qkv_bias=True,
                 qk_scale=None,
                 drop_rate=0.,
                 attn_drop_rate=0.,
                 drop_path_rate=0.2,
                 norm_layer=nn.LayerNorm,
                 ape=False,
                 spe=False,
                 patch_norm=True,
                 out_indices=(0, 1, 2, 3),
                 frozen_stages=-1,
                 use_checkpoint=False,
                 pat_merg_rf=2,
                 pos_embed_method='relative'):
        super().__init__()
        self.pretrain_img_size = pretrain_img_size
        self.num_layers = len(depths)
        self.embed_dim = embed_dim
        self.ape = ape
        self.spe = spe
        self.patch_norm = patch_norm
        self.out_indices = out_indices
        self.frozen_stages = frozen_stages

        # split image into non-overlapping patches
        self.patch_embed_mod0 = PatchEmbed(
            patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dim,
            norm_layer=norm_layer if self.patch_norm else None)

        self.patch_embed_mod1 = PatchEmbed(
            patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dim,
            norm_layer=norm_layer if self.patch_norm else None)

        # absolute position embedding
        if self.ape:
            pretrain_img_size = to_3tuple(self.pretrain_img_size)
            patch_size = to_3tuple(patch_size)
            patches_resolution = [pretrain_img_size[0] // patch_size[0], pretrain_img_size[1] // patch_size[1],
                                  pretrain_img_size[2] // patch_size[2]]

            self.absolute_pos_embed = nn.Parameter(
                torch.zeros(1, embed_dim, patches_resolution[0], patches_resolution[1], patches_resolution[2]))
            trunc_normal_(self.absolute_pos_embed, std=.02)
        elif self.spe:
            self.pos_embd = SinusoidalPositionEmbedding().cuda()
        self.pos_drop = nn.Dropout(p=drop_rate)

        # stochastic depth
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]  # stochastic depth decay rule

        # build layers
        self.layers = nn.ModuleList()
        self.patch_merging_layers = nn.ModuleList()

        for i_layer in range(self.num_layers):
            layer = BasicLayer_crossModality(dim=int((embed_dim) * 2 ** i_layer),
                               depth=depths[i_layer],
                               num_heads=num_heads[i_layer],
                               window_size=window_size,
                               mlp_ratio=mlp_ratio,
                               qkv_bias=qkv_bias,
                               qk_scale=qk_scale,
                               drop=drop_rate,
                               attn_drop=attn_drop_rate,
                               drop_path=dpr[sum(depths[:i_layer]):sum(depths[:i_layer + 1])],
                               norm_layer=norm_layer,
                               downsample=PatchMerging if (i_layer < self.num_layers -1 ) else None,
                               use_checkpoint=use_checkpoint,
                               pat_merg_rf=pat_merg_rf,
                               pos_embed_method=pos_embed_method)
            self.layers.append(layer)

            patch_merging_layer = PatchMerging(int(embed_dim * 2 ** i_layer), reduce_factor=4, concatenated_input=False)
            # self.patch_merging_layers.append(patch_merging_layer)
            #patch_merging_layer = PatchConvPool(int(embed_dim * 2 ** i_layer), concatenated_input=False)
            self.patch_merging_layers.append(patch_merging_layer)

        num_features = [int((embed_dim*2) * 2 ** i) for i in range(self.num_layers)]
        self.num_features = num_features

        # add a norm layer for each output
        for i_layer in out_indices:
            layer = norm_layer(num_features[i_layer] * 1)
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

    def init_weights(self, pretrained=None):
        """Initialize the weights in backbone.
        Args:
            pretrained (str, optional): Path to pre-trained weights.
                Defaults to None.
        """

        def _init_weights(m):
            if isinstance(m, nn.Linear):
                trunc_normal_(m.weight, std=.02)
                if isinstance(m, nn.Linear) and m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.LayerNorm):
                nn.init.constant_(m.bias, 0)
                nn.init.constant_(m.weight, 1.0)

        if isinstance(pretrained, str):
            self.apply(_init_weights)
        elif pretrained is None:
            self.apply(_init_weights)
        else:
            raise TypeError('pretrained must be a str or None')

    def proj_out(self, x, normalize=False):
        if normalize:
            x_shape = x.size()
            if len(x_shape) == 5:
                n, ch, d, h, w = x_shape
                x = rearrange(x, "n c d h w -> n d h w c")
                x = F.layer_norm(x, [ch])
                x = rearrange(x, "n d h w c -> n c d h w")
            elif len(x_shape) == 4:
                n, ch, h, w = x_shape
                x = rearrange(x, "n c h w -> n h w c")
                x = F.layer_norm(x, [ch])
                x = rearrange(x, "n h w c -> n c h w")
        return x

    def forward(self, x,normalize=True):
        """Forward function."""
        outs = []

        #print ('info,',x.shape)
        x_0 = torch.unsqueeze(x[:, 0, :, :, :], 1)
        x_1 = torch.unsqueeze(x[:, 1, :, :, :], 1)
        # PET image
        x_0 = self.patch_embed_mod0(x_0)
        Wh_x0, Ww_x0, Wt_x0 = x_0.size(2), x_0.size(3), x_0.size(4)
        if self.ape:
            # interpolate the position embedding to the corresponding size
            absolute_pos_embed_x0 = nnf.interpolate(self.absolute_pos_embed, size=(Wh_x0, Ww_x0, Wt_x0),
                                                    mode='trilinear')
            x_0 = (x_0 + absolute_pos_embed_x0).flatten(2).transpose(1, 2)  # B Wh*Ww*Wt C
        elif self.spe:
            print(self.spe)
            x_0 = x_0.flatten(2).transpose(1, 2)
            x_0 += self.pos_embd(x_0)
        else:
            x_0 = x_0.flatten(2).transpose(1, 2)
        x_0 = self.pos_drop(x_0)
        #x_0 = self.proj_out(x_0, normalize)

        # CT image
        x_1 = self.patch_embed_mod1(x_1) # B C, W, H ,D
        Wh_x1, Ww_x1, Wt_x1 = x_1.size(2), x_1.size(3), x_1.size(4)
        if self.ape:
            # interpolate the position embedding to the corresponding size
            absolute_pos_embed_x1 = nnf.interpolate(self.absolute_pos_embed, size=(Wh_x1, Ww_x1, Wt_x1),
                                                    mode='trilinear')
            x_1 = (x_1 + absolute_pos_embed_x1).flatten(2).transpose(1, 2)  # B Wh*Ww*Wt C
        elif self.spe:
            print(self.spe)
            x_1 = x_1.flatten(2).transpose(1, 2)
            x_1 += self.pos_embd(x_1)
        else:
            x_1 = x_1.flatten(2).transpose(1, 2)
        x_1 = self.pos_drop(x_1)
        #x_1 = self.proj_out(x_1, normalize)

        #print(x_0.size())

        #out = torch.cat((x_0,x_1),dim=2)
        out = x_0 + x_1
        out = self.proj_out(out, normalize)

        outs.append((out.view(-1, Wh_x0, Ww_x0, Wt_x0, self.embed_dim).permute(0, 4, 1, 2, 3).contiguous()))

        x_0 = self.patch_merging_layers[0](x_0, Wh_x0, Ww_x0, Wt_x0)
        x_1 = self.patch_merging_layers[0](x_1, Wh_x1, Ww_x1, Wt_x1)

        #############layer0################
        layer = self.layers[0]
        # concatenate x_0 and x_1 in dimension 1
        # x_0_1 = torch.cat((x_0, x_1),
        #                   dim=1)  # concatenate in the spatial dimension so that the SWINTR is looking at the correlations spatially
        # send the concatenated feature vector to transformer
        x_out_x0_l0, x_out_x1_l0, H_x0_1, W_x0_1, T_x0_1, x_0_small, x_1_small, Wh_x0_1_l0, Ww_x0_1_l0, Wt_x0_1_l0 = layer(x_0,x_1,int(Wh_x0/2), int(Ww_x0/2),
                                                                                                 int(Wt_x0/2))
        # split the resulting feature vector in dimension 1 to get processed x_1_processed, x_0_processed
        #x_out_x0_1_l0 = x_0_1
        x_0_out = x_out_x0_l0
        x_1_out = x_out_x1_l0
        # add x_1_process and x_0_processed to x_1 and x_0 and start layer 1
        #x_0_out = x_out_x0_l0    # do not update x_0
        #x_1_out = x_out_x1_l0   # do not update x_1
        #x_0_out = self.proj_out(x_0_out, normalize)
        #x_1_out = self.proj_out(x_1_out, normalize)
        out_x0_x1_l0 = x_0_out + x_1_out
        x_out_l0 = self.proj_out(out_x0_x1_l0, normalize)

        #out_x0_x1_l0 = torch.concat((x_0_out, x_1_out), dim=2)

        #norm_layer = getattr(self, f'norm{0}')
        #x_out_l0 = norm_layer(out_x0_x1_l0)
        out = x_out_l0.view(-1, int(H_x0_1), W_x0_1, T_x0_1, self.embed_dim * 2).permute(0, 4, 1, 2, 3).contiguous()
        outs.append(out)  # layer 0 output
        # add transformer encoded back to modality-specific branch
        x_0 = x_0_out + x_0
        x_1 = x_1_out + x_1
        # construct the input for the next layer
        x_0 = self.patch_merging_layers[1](x_0, int(Wh_x0/2), int(Ww_x0/2), int(Wt_x0/2))
        x_1 = self.patch_merging_layers[1](x_1, int(Wh_x1/2), int(Ww_x1/2), int(Wt_x1/2))


        #############layer1################
        layer = self.layers[1]
        # concatenate x_0 and x_1 in dimension 1
        #x_0_1 = torch.cat((x_0, x_1),
        #                  dim=1)  # concatenate in the channel dimension so that the image dimension is still Wh_x0, Ww_x0, Wt_x0 which is needed to be compatible for SWINTR
        # send the concatenated feature vector to transformer
        x_out_x0_l1,x_out_x1_l1, H_x0_1, W_x0_1, T_x0_1, x_0_small,x_1_small, Wh_x0_1_l1, Ww_x0_1_l1, Wt_x0_1_l1 = layer(x_0,x_1, int(Wh_x0_1_l0),
                                                                                                 int(Ww_x0_1_l0), int(Wt_x0_1_l0))
        # split the resulting feature vector in dimension 1 to get processed x_1_processed, x_0_processed
        #x_out_x0_1_l1 = x_0_1
        x_0_out = x_out_x0_l1
        x_1_out = x_out_x1_l1
        #x_0_out = self.proj_out(x_0_out, normalize)
        #x_1_out = self.proj_out(x_1_out, normalize)
        #x_out_x0_l1 = x_out_x0_1_l1[:, :, 0:self.embed_dim * 4]
        #x_out_x1_l1 = x_out_x0_1_l1[:, :, self.embed_dim * 4:]
        #x_0_out = x_out_x0_l1   # updated x_0
        #x_1_out = x_out_x1_l1  # updated x_1
        out_x0_x1_l1 = x_0_out + x_1_out #should I use the sum or concat for decoder?
        x_out_l1 = self.proj_out(out_x0_x1_l1, normalize)

        #out_x0_x1_l1 = torch.concat((x_0_out, x_1_out), dim=2)

        #norm_layer = getattr(self, f'norm{1}')
        #x_out_l1 = norm_layer(out_x0_x1_l1)
        out = x_out_l1.view(-1, int(H_x0_1), W_x0_1, T_x0_1, self.embed_dim * 4).permute(0, 4, 1, 2,
                                                                                                3).contiguous()
        outs.append(out)  # layer 1 output
        # add transformer encoded back to modality-specific branch
        x_0 = x_0_out + x_0
        x_1 = x_1_out + x_1
        # construct the input for the next layer
        x_0 = self.patch_merging_layers[2](x_0, int(Wh_x0_1_l0), int(Ww_x0_1_l0), int(Wt_x0_1_l0))
        x_1 = self.patch_merging_layers[2](x_1, int(Wh_x0_1_l0), int(Ww_x0_1_l0), int(Wt_x0_1_l0))


        #############layer2################
        layer = self.layers[2]
        # concatenate x_0 and x_1 in dimension 1
        #x_0_1 = torch.cat((x_0, x_1),
        #                  dim=1)  # concatenate in the channel dimension so that the image dimension is still Wh_x0, Ww_x0, Wt_x0 which is needed to be compatible for SWINTR
        # send the concatenated feature vector to transformer
        x_out_x0_l2, x_out_x1_l2,H_x0_1, W_x0_1, T_x0_1, x_0_small,x_1_small, Wh_x0_1_l2, Ww_x0_1_l2, Wt_x0_1_l2 = layer(x_0,x_1, int(Wh_x0_1_l1),
                                                                                                 int(Ww_x0_1_l1), int(Wt_x0_1_l1))
        # split the resulting feature vector in dimension 1 to get processed x_1_processed, x_0_processed
        #x_out_x0_1_l2 = x_0_1
        x_0_out = x_out_x0_l2
        x_1_out = x_out_x1_l2
        #x_0_out = self.proj_out(x_0_out, normalize)
        #x_1_out = self.proj_out(x_1_out, normalize)
        #x_out_x0_l2 = x_out_x0_1_l2[:, :, 0:self.embed_dim * 8]
        #x_out_x1_l2 = x_out_x0_1_l2[:, :, self.embed_dim * 8:]
        #x_0_out = x_out_x0_l2  # updated x_0
        #x_1_out = x_out_x1_l2   # updated x_1
        out_x0_x1_l2 = x_0_out  + x_1_out
        x_out_l2 = self.proj_out(out_x0_x1_l2, normalize)

        #out_x0_x1_l2 = torch.concat((x_0_out, x_1_out), dim=2)

        #norm_layer = getattr(self, f'norm{2}')
        #x_out_l2 = norm_layer(out_x0_x1_l2)
        out = x_out_l2.view(-1, int(H_x0_1), W_x0_1, T_x0_1, self.embed_dim * 8).permute(0, 4, 1, 2,
                                                                                                3).contiguous()
        outs.append(out)  # layer 2 output
        # add transformer encoded back to modality-specific branch
        x_0 = x_0_out + x_0
        x_1 = x_1_out + x_1
        # construct the input for the next layer
        x_0 = self.patch_merging_layers[3](x_0, int(Wh_x0_1_l1), int(Ww_x0_1_l1), int(Wt_x0_1_l1))
        x_1 = self.patch_merging_layers[3](x_1, int(Wh_x0_1_l1), int(Ww_x0_1_l1), int(Wt_x0_1_l1))


        #############layer3################
        layer = self.layers[3]
        # concatenate x_0 and x_1 in dimension 1
        #x_0_1 = torch.cat((x_0, x_1),
        #                  dim=1)  # concatenate in the channel dimension so that the image dimension is still Wh_x0, Ww_x0, Wt_x0 which is needed to be compatible for SWINTR
        # send the concatenated feature vector to transformer
        x_out_x0_l3, x_out_x1_l3,H_x0_1, W_x0_1, T_x0_1, x_0_small,x_1_small, Wh_x0_1_l3, Ww_x0_1_l3, Wt_x0_1_l3 = layer(x_0,x_1, int(Wh_x0_1_l2),
                                                                                                 int(Ww_x0_1_l2), int(Wt_x0_1_l2))
        # split the resulting feature vector in dimension 1 to get processed x_1_processed, x_0_processed
        #x_out_x0_1_l3 = x_0_1
        x_0_out = x_out_x0_l3
        x_1_out = x_out_x1_l3
        #x_0_out = self.proj_out(x_0_out, normalize)
        #x_1_out = self.proj_out(x_1_out, normalize)
        #x_out_x0_l3 = x_out_x0_1_l3[:, :, 0:self.embed_dim * 16]
        #x_out_x1_l3 = x_out_x0_1_l3[:, :, self.embed_dim * 16:]
        #x_0_out = x_out_x0_l3   # updated x_0
        #x_1_out = x_out_x1_l3   # updated x_1
        out_x0_x1_l3 = x_0_out + x_1_out
        out_x0_x1_l3 = self.proj_out(out_x0_x1_l3, normalize)
        #out_x0_x1_l3 = torch.concat((x_0_out, x_1_out), dim=2)

        norm_layer = getattr(self, f'norm{3}')
        x_out_l3 = norm_layer(out_x0_x1_l3)
        out = x_out_l3.view(-1, int(H_x0_1), W_x0_1, T_x0_1, self.embed_dim * 16).permute(0, 4, 1, 2,
                                                                                                 3).contiguous()
        outs.append(out)  # layer 2 output

        return outs

    def train(self, mode=True):
        """Convert the model into training mode while keep layers freezed."""
        super(SwinTransformer_wCrossModalityFeatureTalk_OutSum_5stageOuts, self).train(mode)
        self._freeze_stages()



class SwinTransformer_wCrossModalityFeatureTalk_wInputFusion_OutSum_5stageOuts(nn.Module):
    r""" Swin Transformer modified to process images from two modalities; feature talks between two images are introduced in encoder
        A PyTorch impl of : `Swin Transformer: Hierarchical Vision Transformer using Shifted Windows`  -
          https://arxiv.org/pdf/2103.14030
    Args:
        img_size (int | tuple(int)): Input image size. Default 224
        patch_size (int | tuple(int)): Patch size. Default: 4
        in_chans (int): Number of input image channels. Default: 3
        num_classes (int): Number of classes for classification head. Default: 1000
        embed_dim (int): Patch embedding dimension. Default: 96
        depths (tuple(int)): Depth of each Swin Transformer layer.
        num_heads (tuple(int)): Number of attention heads in different layers.
        window_size (tuple): Window size. Default: 7
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim. Default: 4
        qkv_bias (bool): If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float): Override default qk scale of head_dim ** -0.5 if set. Default: None
        drop_rate (float): Dropout rate. Default: 0
        attn_drop_rate (float): Attention dropout rate. Default: 0
        drop_path_rate (float): Stochastic depth rate. Default: 0.1
        norm_layer (nn.Module): Normalization layer. Default: nn.LayerNorm.
        ape (bool): If True, add absolute position embedding to the patch embedding. Default: False
        patch_norm (bool): If True, add normalization after patch embedding. Default: True
        use_checkpoint (bool): Whether to use checkpointing to save memory. Default: False
    """

    def __init__(self, pretrain_img_size=160,
                 patch_size=4,
                 in_chans=3,
                 embed_dim=96,
                 depths=[2, 2, 6, 2],
                 num_heads=[3, 6, 12, 24],
                 window_size=(7, 7, 7),
                 mlp_ratio=4.,
                 qkv_bias=True,
                 qk_scale=None,
                 drop_rate=0.,
                 attn_drop_rate=0.,
                 drop_path_rate=0.2,
                 norm_layer=nn.LayerNorm,
                 ape=False,
                 spe=False,
                 patch_norm=True,
                 out_indices=(0, 1, 2, 3),
                 frozen_stages=-1,
                 use_checkpoint=False,
                 pat_merg_rf=2,
                 pos_embed_method='relative'):
        super().__init__()
        self.pretrain_img_size = pretrain_img_size
        self.num_layers = len(depths)
        self.embed_dim = embed_dim
        self.ape = ape
        self.spe = spe
        self.patch_norm = patch_norm
        self.out_indices = out_indices
        self.frozen_stages = frozen_stages

        self.res_fusionBlock = depthwise_separable_conv3d(
            nin=2,
            kernels_per_layer=48,
            nout=48,
        )
        # split image into non-overlapping patches

        self.patch_embed_mod0 = PatchEmbed(
            patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dim,
            norm_layer=norm_layer if self.patch_norm else None)

        self.patch_embed_mod1 = PatchEmbed(
            patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dim,
            norm_layer=norm_layer if self.patch_norm else None)

        # absolute position embedding
        if self.ape:
            pretrain_img_size = to_3tuple(self.pretrain_img_size)
            patch_size = to_3tuple(patch_size)
            patches_resolution = [pretrain_img_size[0] // patch_size[0], pretrain_img_size[1] // patch_size[1],
                                  pretrain_img_size[2] // patch_size[2]]

            self.absolute_pos_embed = nn.Parameter(
                torch.zeros(1, embed_dim, patches_resolution[0], patches_resolution[1], patches_resolution[2]))
            trunc_normal_(self.absolute_pos_embed, std=.02)
        elif self.spe:
            self.pos_embd = SinusoidalPositionEmbedding().cuda()
        self.pos_drop = nn.Dropout(p=drop_rate)

        # stochastic depth
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]  # stochastic depth decay rule

        # build layers
        self.layers = nn.ModuleList()
        self.patch_merging_layers = nn.ModuleList()

        for i_layer in range(self.num_layers):
            layer = BasicLayer_crossModality(dim=int((embed_dim) * 2 ** i_layer),
                                             depth=depths[i_layer],
                                             num_heads=num_heads[i_layer],
                                             window_size=window_size,
                                             mlp_ratio=mlp_ratio,
                                             qkv_bias=qkv_bias,
                                             qk_scale=qk_scale,
                                             drop=drop_rate,
                                             attn_drop=attn_drop_rate,
                                             drop_path=dpr[sum(depths[:i_layer]):sum(depths[:i_layer + 1])],
                                             norm_layer=norm_layer,
                                             downsample=PatchMerging if (i_layer < self.num_layers - 1) else None,
                                             use_checkpoint=use_checkpoint,
                                             pat_merg_rf=pat_merg_rf,
                                             pos_embed_method=pos_embed_method)
            self.layers.append(layer)

            patch_merging_layer = PatchMerging(int(embed_dim * 2 ** i_layer), reduce_factor=4, concatenated_input=False)
            # self.patch_merging_layers.append(patch_merging_layer)
            # patch_merging_layer = PatchConvPool(int(embed_dim * 2 ** i_layer), concatenated_input=False)
            self.patch_merging_layers.append(patch_merging_layer)

        num_features = [int((embed_dim * 2) * 2 ** i) for i in range(self.num_layers)]
        self.num_features = num_features

        # add a norm layer for each output
        for i_layer in out_indices:
            layer = norm_layer(num_features[i_layer] * 1)
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

    def init_weights(self, pretrained=None):
        """Initialize the weights in backbone.
        Args:
            pretrained (str, optional): Path to pre-trained weights.
                Defaults to None.
        """

        def _init_weights(m):
            if isinstance(m, nn.Linear):
                trunc_normal_(m.weight, std=.02)
                if isinstance(m, nn.Linear) and m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.LayerNorm):
                nn.init.constant_(m.bias, 0)
                nn.init.constant_(m.weight, 1.0)

        if isinstance(pretrained, str):
            self.apply(_init_weights)
        elif pretrained is None:
            self.apply(_init_weights)
        else:
            raise TypeError('pretrained must be a str or None')

    def forward(self, x):
        """Forward function."""
        outs = []
        x_0 = torch.unsqueeze(x[:, 0, :, :, :], 1)
        x_1 = torch.unsqueeze(x[:, 1, :, :, :], 1)
        # PET image
        x_0 = self.patch_embed_mod0(x_0)
        Wh_x0, Ww_x0, Wt_x0 = x_0.size(2), x_0.size(3), x_0.size(4)
        if self.ape:
            # interpolate the position embedding to the corresponding size
            absolute_pos_embed_x0 = nnf.interpolate(self.absolute_pos_embed, size=(Wh_x0, Ww_x0, Wt_x0),
                                                    mode='trilinear')
            x_0 = (x_0 + absolute_pos_embed_x0).flatten(2).transpose(1, 2)  # B Wh*Ww*Wt C
        elif self.spe:
            print(self.spe)
            x_0 = x_0.flatten(2).transpose(1, 2)
            x_0 += self.pos_embd(x_0)
        else:
            x_0 = x_0.flatten(2).transpose(1, 2)
        x_0 = self.pos_drop(x_0)

        # CT image
        x_1 = self.patch_embed_mod1(x_1)  # B C, W, H ,D
        Wh_x1, Ww_x1, Wt_x1 = x_1.size(2), x_1.size(3), x_1.size(4)
        if self.ape:
            # interpolate the position embedding to the corresponding size
            absolute_pos_embed_x1 = nnf.interpolate(self.absolute_pos_embed, size=(Wh_x1, Ww_x1, Wt_x1),
                                                    mode='trilinear')
            x_1 = (x_1 + absolute_pos_embed_x1).flatten(2).transpose(1, 2)  # B Wh*Ww*Wt C
        elif self.spe:
            print(self.spe)
            x_1 = x_1.flatten(2).transpose(1, 2)
            x_1 += self.pos_embd(x_1)
        else:
            x_1 = x_1.flatten(2).transpose(1, 2)
        x_1 = self.pos_drop(x_1)
        # print(x_0.size())

        # out = torch.cat((x_0,x_1),dim=2)
        #out = x_0 + x_1
        out = self.res_fusionBlock(x)
        outs.append(out)
        #outs.append((out.view(-1, Wh_x0, Ww_x0, Wt_x0, self.embed_dim).permute(0, 4, 1, 2, 3).contiguous()))

        x_0 = self.patch_merging_layers[0](x_0, Wh_x0, Ww_x0, Wt_x0)
        x_1 = self.patch_merging_layers[0](x_1, Wh_x1, Ww_x1, Wt_x1)

        #############layer0################
        layer = self.layers[0]
        # concatenate x_0 and x_1 in dimension 1
        # x_0_1 = torch.cat((x_0, x_1),
        #                   dim=1)  # concatenate in the spatial dimension so that the SWINTR is looking at the correlations spatially
        # send the concatenated feature vector to transformer
        x_out_x0_l0, x_out_x1_l0, H_x0_1, W_x0_1, T_x0_1, x_0_small, x_1_small, Wh_x0_1_l0, Ww_x0_1_l0, Wt_x0_1_l0 = layer(
            x_0, x_1, int(Wh_x0 / 2), int(Ww_x0 / 2),
            int(Wt_x0 / 2))
        # split the resulting feature vector in dimension 1 to get processed x_1_processed, x_0_processed
        # x_out_x0_1_l0 = x_0_1
        x_0_out = x_out_x0_l0
        x_1_out = x_out_x1_l0
        # add x_1_process and x_0_processed to x_1 and x_0 and start layer 1
        # x_0_out = x_out_x0_l0    # do not update x_0
        # x_1_out = x_out_x1_l0   # do not update x_1
        out_x0_x1_l0 = x_0_out + x_1_out
        # out_x0_x1_l0 = torch.concat((x_0_out, x_1_out), dim=2)

        norm_layer = getattr(self, f'norm{0}')
        x_out_l0 = norm_layer(out_x0_x1_l0)
        out = x_out_l0.view(-1, int(H_x0_1), W_x0_1, T_x0_1, self.embed_dim * 2).permute(0, 4, 1, 2, 3).contiguous()
        outs.append(out)  # layer 0 output
        # add transformer encoded back to modality-specific branch
        x_0 = x_0_out + x_0
        x_1 = x_1_out + x_1
        # construct the input for the next layer
        x_0 = self.patch_merging_layers[1](x_0, int(Wh_x0 / 2), int(Ww_x0 / 2), int(Wt_x0 / 2))
        x_1 = self.patch_merging_layers[1](x_1, int(Wh_x1 / 2), int(Ww_x1 / 2), int(Wt_x1 / 2))

        #############layer1################
        layer = self.layers[1]
        # concatenate x_0 and x_1 in dimension 1
        # x_0_1 = torch.cat((x_0, x_1),
        #                  dim=1)  # concatenate in the channel dimension so that the image dimension is still Wh_x0, Ww_x0, Wt_x0 which is needed to be compatible for SWINTR
        # send the concatenated feature vector to transformer
        x_out_x0_l1, x_out_x1_l1, H_x0_1, W_x0_1, T_x0_1, x_0_small, x_1_small, Wh_x0_1_l1, Ww_x0_1_l1, Wt_x0_1_l1 = layer(
            x_0, x_1, int(Wh_x0_1_l0),
            int(Ww_x0_1_l0), int(Wt_x0_1_l0))
        # split the resulting feature vector in dimension 1 to get processed x_1_processed, x_0_processed
        # x_out_x0_1_l1 = x_0_1
        x_0_out = x_out_x0_l1
        x_1_out = x_out_x1_l1
        # x_out_x0_l1 = x_out_x0_1_l1[:, :, 0:self.embed_dim * 4]
        # x_out_x1_l1 = x_out_x0_1_l1[:, :, self.embed_dim * 4:]
        # x_0_out = x_out_x0_l1   # updated x_0
        # x_1_out = x_out_x1_l1  # updated x_1
        out_x0_x1_l1 = x_0_out + x_1_out  # should I use the sum or concat for decoder?
        # out_x0_x1_l1 = torch.concat((x_0_out, x_1_out), dim=2)

        norm_layer = getattr(self, f'norm{1}')
        x_out_l1 = norm_layer(out_x0_x1_l1)
        out = x_out_l1.view(-1, int(H_x0_1), W_x0_1, T_x0_1, self.embed_dim * 4).permute(0, 4, 1, 2,
                                                                                         3).contiguous()
        outs.append(out)  # layer 1 output
        # add transformer encoded back to modality-specific branch
        x_0 = x_0_out + x_0
        x_1 = x_1_out + x_1
        # construct the input for the next layer
        x_0 = self.patch_merging_layers[2](x_0, int(Wh_x0_1_l0), int(Ww_x0_1_l0), int(Wt_x0_1_l0))
        x_1 = self.patch_merging_layers[2](x_1, int(Wh_x0_1_l0), int(Ww_x0_1_l0), int(Wt_x0_1_l0))

        #############layer2################
        layer = self.layers[2]
        # concatenate x_0 and x_1 in dimension 1
        # x_0_1 = torch.cat((x_0, x_1),
        #                  dim=1)  # concatenate in the channel dimension so that the image dimension is still Wh_x0, Ww_x0, Wt_x0 which is needed to be compatible for SWINTR
        # send the concatenated feature vector to transformer
        x_out_x0_l2, x_out_x1_l2, H_x0_1, W_x0_1, T_x0_1, x_0_small, x_1_small, Wh_x0_1_l2, Ww_x0_1_l2, Wt_x0_1_l2 = layer(
            x_0, x_1, int(Wh_x0_1_l1),
            int(Ww_x0_1_l1), int(Wt_x0_1_l1))
        # split the resulting feature vector in dimension 1 to get processed x_1_processed, x_0_processed
        # x_out_x0_1_l2 = x_0_1
        x_0_out = x_out_x0_l2
        x_1_out = x_out_x1_l2
        # x_out_x0_l2 = x_out_x0_1_l2[:, :, 0:self.embed_dim * 8]
        # x_out_x1_l2 = x_out_x0_1_l2[:, :, self.embed_dim * 8:]
        # x_0_out = x_out_x0_l2  # updated x_0
        # x_1_out = x_out_x1_l2   # updated x_1
        out_x0_x1_l2 = x_0_out + x_1_out
        # out_x0_x1_l2 = torch.concat((x_0_out, x_1_out), dim=2)

        norm_layer = getattr(self, f'norm{2}')
        x_out_l2 = norm_layer(out_x0_x1_l2)
        out = x_out_l2.view(-1, int(H_x0_1), W_x0_1, T_x0_1, self.embed_dim * 8).permute(0, 4, 1, 2,
                                                                                         3).contiguous()
        outs.append(out)  # layer 2 output
        # add transformer encoded back to modality-specific branch
        x_0 = x_0_out + x_0
        x_1 = x_1_out + x_1
        # construct the input for the next layer
        x_0 = self.patch_merging_layers[3](x_0, int(Wh_x0_1_l1), int(Ww_x0_1_l1), int(Wt_x0_1_l1))
        x_1 = self.patch_merging_layers[3](x_1, int(Wh_x0_1_l1), int(Ww_x0_1_l1), int(Wt_x0_1_l1))

        #############layer3################
        layer = self.layers[3]
        # concatenate x_0 and x_1 in dimension 1
        # x_0_1 = torch.cat((x_0, x_1),
        #                  dim=1)  # concatenate in the channel dimension so that the image dimension is still Wh_x0, Ww_x0, Wt_x0 which is needed to be compatible for SWINTR
        # send the concatenated feature vector to transformer
        x_out_x0_l3, x_out_x1_l3, H_x0_1, W_x0_1, T_x0_1, x_0_small, x_1_small, Wh_x0_1_l3, Ww_x0_1_l3, Wt_x0_1_l3 = layer(
            x_0, x_1, int(Wh_x0_1_l2),
            int(Ww_x0_1_l2), int(Wt_x0_1_l2))
        # split the resulting feature vector in dimension 1 to get processed x_1_processed, x_0_processed
        # x_out_x0_1_l3 = x_0_1
        x_0_out = x_out_x0_l3
        x_1_out = x_out_x1_l3
        # x_out_x0_l3 = x_out_x0_1_l3[:, :, 0:self.embed_dim * 16]
        # x_out_x1_l3 = x_out_x0_1_l3[:, :, self.embed_dim * 16:]
        # x_0_out = x_out_x0_l3   # updated x_0
        # x_1_out = x_out_x1_l3   # updated x_1
        out_x0_x1_l3 = x_0_out + x_1_out
        # out_x0_x1_l3 = torch.concat((x_0_out, x_1_out), dim=2)

        norm_layer = getattr(self, f'norm{3}')
        x_out_l3 = norm_layer(out_x0_x1_l3)
        out = x_out_l3.view(-1, int(H_x0_1), W_x0_1, T_x0_1, self.embed_dim * 16).permute(0, 4, 1, 2,
                                                                                          3).contiguous()
        outs.append(out)  # layer 2 output

        return outs

    def train(self, mode=True):
        """Convert the model into training mode while keep layers freezed."""
        super(SwinTransformer_wCrossModalityFeatureTalk_wInputFusion_OutSum_5stageOuts, self).train(mode)
        self._freeze_stages()


class SwinTransformer_wRandomSpatialFeatureTalk_wCrossModalUpdating_5stageOuts(nn.Module):
    r""" Swin Transformer modified to process images from two modalities; feature talks between two images are introduced in encoder
        A PyTorch impl of : `Swin Transformer: Hierarchical Vision Transformer using Shifted Windows`  -
          https://arxiv.org/pdf/2103.14030
    Args:
        img_size (int | tuple(int)): Input image size. Default 224
        patch_size (int | tuple(int)): Patch size. Default: 4
        in_chans (int): Number of input image channels. Default: 3
        num_classes (int): Number of classes for classification head. Default: 1000
        embed_dim (int): Patch embedding dimension. Default: 96
        depths (tuple(int)): Depth of each Swin Transformer layer.
        num_heads (tuple(int)): Number of attention heads in different layers.
        window_size (tuple): Window size. Default: 7
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim. Default: 4
        qkv_bias (bool): If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float): Override default qk scale of head_dim ** -0.5 if set. Default: None
        drop_rate (float): Dropout rate. Default: 0
        attn_drop_rate (float): Attention dropout rate. Default: 0
        drop_path_rate (float): Stochastic depth rate. Default: 0.1
        norm_layer (nn.Module): Normalization layer. Default: nn.LayerNorm.
        ape (bool): If True, add absolute position embedding to the patch embedding. Default: False
        patch_norm (bool): If True, add normalization after patch embedding. Default: True
        use_checkpoint (bool): Whether to use checkpointing to save memory. Default: False
    """

    def __init__(self, pretrain_img_size=160,
                 patch_size=4,
                 in_chans=3,
                 embed_dim=96,
                 depths=[2, 2, 6, 2],
                 num_heads=[3, 6, 12, 24],
                 window_size=(7, 7, 7),
                 mlp_ratio=4.,
                 qkv_bias=True,
                 qk_scale=None,
                 drop_rate=0.,
                 attn_drop_rate=0.,
                 drop_path_rate=0.2,
                 norm_layer=nn.LayerNorm,
                 ape=False,
                 spe=False,
                 patch_norm=True,
                 out_indices=(0, 1, 2, 3),
                 frozen_stages=-1,
                 use_checkpoint=False,
                 pat_merg_rf=2,
                 pos_embed_method='relative'):
        super().__init__()
        self.pretrain_img_size = pretrain_img_size
        self.num_layers = len(depths)
        self.embed_dim = embed_dim
        self.ape = ape
        self.spe = spe
        self.patch_norm = patch_norm
        self.out_indices = out_indices
        self.frozen_stages = frozen_stages

        # split image into non-overlapping patches
        self.patch_embed_mod0 = PatchEmbed(
            patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dim,
            norm_layer=norm_layer if self.patch_norm else None)

        self.patch_embed_mod1 = PatchEmbed(
            patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dim,
            norm_layer=norm_layer if self.patch_norm else None)

        # absolute position embedding
        if self.ape:
            pretrain_img_size = to_3tuple(self.pretrain_img_size)
            patch_size = to_3tuple(patch_size)
            patches_resolution = [pretrain_img_size[0] // patch_size[0], pretrain_img_size[1] // patch_size[1],
                                  pretrain_img_size[2] // patch_size[2]]

            self.absolute_pos_embed = nn.Parameter(
                torch.zeros(1, embed_dim, patches_resolution[0], patches_resolution[1], patches_resolution[2]))
            trunc_normal_(self.absolute_pos_embed, std=.02)
        elif self.spe:
            self.pos_embd = SinusoidalPositionEmbedding().cuda()
        self.pos_drop = nn.Dropout(p=drop_rate)

        # stochastic depth
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]  # stochastic depth decay rule

        # build layers
        self.layers = nn.ModuleList()
        self.patch_merging_layers = nn.ModuleList()

        for i_layer in range(self.num_layers):
            layer = BasicLayer(dim=int((embed_dim) * 2 ** i_layer),
                               depth=depths[i_layer],
                               num_heads=num_heads[i_layer],
                               window_size=window_size,
                               mlp_ratio=mlp_ratio,
                               qkv_bias=qkv_bias,
                               qk_scale=qk_scale,
                               drop=drop_rate,
                               attn_drop=attn_drop_rate,
                               drop_path=dpr[sum(depths[:i_layer]):sum(depths[:i_layer + 1])],
                               norm_layer=norm_layer,
                               downsample=PatchMerging if (i_layer < self.num_layers -1 ) else None,
                               use_checkpoint=use_checkpoint,
                               pat_merg_rf=pat_merg_rf,
                               pos_embed_method=pos_embed_method)
            self.layers.append(layer)

            patch_merging_layer = PatchMerging(int(embed_dim * 2 ** i_layer), reduce_factor=4, concatenated_input=False)
            # self.patch_merging_layers.append(patch_merging_layer)
            #patch_merging_layer = PatchConvPool(int(embed_dim * 2 ** i_layer), concatenated_input=False)
            self.patch_merging_layers.append(patch_merging_layer)

        num_features = [int((embed_dim*4) * 2 ** i) for i in range(self.num_layers)]
        self.num_features = num_features

        # add a norm layer for each output
        for i_layer in out_indices:
            layer = norm_layer(num_features[i_layer] * 1)
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

    def init_weights(self, pretrained=None):
        """Initialize the weights in backbone.
        Args:
            pretrained (str, optional): Path to pre-trained weights.
                Defaults to None.
        """

        def _init_weights(m):
            if isinstance(m, nn.Linear):
                trunc_normal_(m.weight, std=.02)
                if isinstance(m, nn.Linear) and m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.LayerNorm):
                nn.init.constant_(m.bias, 0)
                nn.init.constant_(m.weight, 1.0)

        if isinstance(pretrained, str):
            self.apply(_init_weights)
        elif pretrained is None:
            self.apply(_init_weights)
        else:
            raise TypeError('pretrained must be a str or None')

    def complement_idx(idx, dim):
        """
        Compute the complement: set(range(dim)) - set(idx).
        idx is a multi-dimensional tensor, find the complement for its trailing dimension,
        all other dimension is considered batched.
        Args:
            idx: input index, shape: [N, *, K]
            dim: the max index for complement
        """
        a = torch.arange(dim, device=idx.device)
        ndim = idx.ndim
        dims = idx.shape
        n_idx = dims[-1]
        dims = dims[:-1] + (-1,)
        for i in range(1, ndim):
            a = a.unsqueeze(0)
        a = a.expand(*dims)
        masked = torch.scatter(a, -1, idx, 0)
        compl, _ = torch.sort(masked, dim=-1, descending=False)
        compl = compl.permute(-1, *tuple(range(ndim - 1)))
        compl = compl[n_idx:].permute(*(tuple(range(1, ndim)) + (0,)))
        return compl

    def forward(self, x):
        """Forward function."""
        outs = []
        x_0 = torch.unsqueeze(x[:, 0, :, :, :], 1)
        x_1 = torch.unsqueeze(x[:, 1, :, :, :], 1)
        # PET image
        x_0 = self.patch_embed_mod0(x_0)
        Wh_x0, Ww_x0, Wt_x0 = x_0.size(2), x_0.size(3), x_0.size(4)
        if self.ape:
            # interpolate the position embedding to the corresponding size
            absolute_pos_embed_x0 = nnf.interpolate(self.absolute_pos_embed, size=(Wh_x0, Ww_x0, Wt_x0),
                                                    mode='trilinear')
            x_0 = (x_0 + absolute_pos_embed_x0).flatten(2).transpose(1, 2)  # B Wh*Ww*Wt C
        elif self.spe:
            print(self.spe)
            x_0 = x_0.flatten(2).transpose(1, 2)
            x_0 += self.pos_embd(x_0)
        else:
            x_0 = x_0.flatten(2).transpose(1, 2)
        x_0 = self.pos_drop(x_0)

        # CT image
        x_1 = self.patch_embed_mod1(x_1)
        Wh_x1, Ww_x1, Wt_x1 = x_1.size(2), x_1.size(3), x_1.size(4)
        if self.ape:
            # interpolate the position embedding to the corresponding size
            absolute_pos_embed_x1 = nnf.interpolate(self.absolute_pos_embed, size=(Wh_x1, Ww_x1, Wt_x1),
                                                    mode='trilinear')
            x_1 = (x_1 + absolute_pos_embed_x1).flatten(2).transpose(1, 2)  # B Wh*Ww*Wt C
        elif self.spe:
            print(self.spe)
            x_1 = x_1.flatten(2).transpose(1, 2)
            x_1 += self.pos_embd(x_1)
        else:
            x_1 = x_1.flatten(2).transpose(1, 2)
        x_1 = self.pos_drop(x_1)
        #print(x_0.size())

        out = torch.cat((x_0,x_1),dim=2)
        outs.append((out.view(-1, Wh_x0, Ww_x0, Wt_x0, self.embed_dim*2).permute(0, 4, 1, 2, 3).contiguous()))

        x_0 = self.patch_merging_layers[0](x_0, Wh_x0, Ww_x0, Wt_x0)
        x_1 = self.patch_merging_layers[0](x_1, Wh_x1, Ww_x1, Wt_x1)

        #############layer0################
        layer = self.layers[0]
        # concatenate x_0 and x_1 in dimension 1

        x_0_top50,x_0_top50_idx = torch.topk(x_0,int(x_0.size(dim=1)/2),dim=1)
        x_1_top50 = torch.gather(x_1, 1, x_0_top50_idx)

        x_0_1_top50 = torch.cat((x_0_top50, x_1_top50),
                          dim=1)


        x_0_1 = torch.cat((x_0, x_1),
                          dim=1)  # concatenate in the spatial dimension so that the SWINTR is looking at the correlations spatially
        # send the concatenated feature vector to transformer
        x_out_x0_1_l0, H_x0_1, W_x0_1, T_x0_1, x_0_1, Wh_x0_1_l0, Ww_x0_1_l0, Wt_x0_1_l0 = layer(x_0_1, int(Wh_x0), int(Ww_x0/2),
                                                                                                 int(Wt_x0/2))
        # split the resulting feature vector in dimension 1 to get processed x_1_processed, x_0_processed
        #x_out_x0_1_l0 = x_0_1
        x_0_out = x_out_x0_1_l0[:, :int(Wh_x0*Ww_x0/2*Wt_x0/2/2), :]
        x_1_out = x_out_x0_1_l0[:, int(Wh_x0*Ww_x0/2*Wt_x0/2/2):, :]
        # add x_1_process and x_0_processed to x_1 and x_0 and start layer 1
        #x_0_out = x_out_x0_l0    # do not update x_0
        #x_1_out = x_out_x1_l0   # do not update x_1
        #out_x0_x1_l0 = x_0_out + x_1_out
        out_x0_x1_l0 = torch.concat((x_0_out, x_1_out), dim=2)

        norm_layer = getattr(self, f'norm{0}')
        x_out_l0 = norm_layer(out_x0_x1_l0)
        out = x_out_l0.view(-1, int(H_x0_1/2), W_x0_1, T_x0_1, self.embed_dim * 4).permute(0, 4, 1, 2, 3).contiguous()
        outs.append(out)  # layer 0 output
        # add transformer encoded back to modality-specific branch
        x_0 = x_0_out + x_0
        x_1 = x_1_out + x_1
        # construct the input for the next layer
        x_0 = self.patch_merging_layers[1](x_0, int(Wh_x0/2), int(Ww_x0/2), int(Wt_x0/2))
        x_1 = self.patch_merging_layers[1](x_1, int(Wh_x1/2), int(Ww_x1/2), int(Wt_x1/2))


        #############layer1################
        layer = self.layers[1]
        # concatenate x_0 and x_1 in dimension 1
        x_0_1 = torch.cat((x_0, x_1),
                          dim=1)  # concatenate in the channel dimension so that the image dimension is still Wh_x0, Ww_x0, Wt_x0 which is needed to be compatible for SWINTR
        # send the concatenated feature vector to transformer
        x_out_x0_1_l1, H_x0_1, W_x0_1, T_x0_1, x_0_1, Wh_x0_1_l1, Ww_x0_1_l1, Wt_x0_1_l1 = layer(x_0_1, int(Wh_x0_1_l0),
                                                                                                 int(Ww_x0_1_l0), int(Wt_x0_1_l0))
        # split the resulting feature vector in dimension 1 to get processed x_1_processed, x_0_processed
        #x_out_x0_1_l1 = x_0_1
        x_0_out = x_out_x0_1_l1[:, :int(Wh_x0_1_l0 * Ww_x0_1_l0 * Wt_x0_1_l0  / 2), :]
        x_1_out = x_out_x0_1_l1[:, int(Wh_x0_1_l0 * Ww_x0_1_l0  * Wt_x0_1_l0  / 2):, :]
        #x_out_x0_l1 = x_out_x0_1_l1[:, :, 0:self.embed_dim * 4]
        #x_out_x1_l1 = x_out_x0_1_l1[:, :, self.embed_dim * 4:]
        #x_0_out = x_out_x0_l1   # updated x_0
        #x_1_out = x_out_x1_l1  # updated x_1
        #out_x0_x1_l1 = x_0_out + x_1_out #should I use the sum or concat for decoder?
        out_x0_x1_l1 = torch.concat((x_0_out, x_1_out), dim=2)

        norm_layer = getattr(self, f'norm{1}')
        x_out_l1 = norm_layer(out_x0_x1_l1)
        out = x_out_l1.view(-1, int(H_x0_1/2), W_x0_1, T_x0_1, self.embed_dim * 8).permute(0, 4, 1, 2,
                                                                                                3).contiguous()
        outs.append(out)  # layer 1 output
        # add transformer encoded back to modality-specific branch
        x_0 = x_0_out + x_0
        x_1 = x_1_out + x_1
        # construct the input for the next layer
        x_0 = self.patch_merging_layers[2](x_0, int(Wh_x0_1_l0/2), int(Ww_x0_1_l0), int(Wt_x0_1_l0))
        x_1 = self.patch_merging_layers[2](x_1, int(Wh_x0_1_l0/2), int(Ww_x0_1_l0), int(Wt_x0_1_l0))


        #############layer2################
        layer = self.layers[2]
        # concatenate x_0 and x_1 in dimension 1
        x_0_1 = torch.cat((x_0, x_1),
                          dim=1)  # concatenate in the channel dimension so that the image dimension is still Wh_x0, Ww_x0, Wt_x0 which is needed to be compatible for SWINTR
        # send the concatenated feature vector to transformer
        x_out_x0_1_l2, H_x0_1, W_x0_1, T_x0_1, x_0_1, Wh_x0_1_l2, Ww_x0_1_l2, Wt_x0_1_l2 = layer(x_0_1, int(Wh_x0_1_l1),
                                                                                                 int(Ww_x0_1_l1), int(Wt_x0_1_l1))
        # split the resulting feature vector in dimension 1 to get processed x_1_processed, x_0_processed
        #x_out_x0_1_l2 = x_0_1
        x_0_out = x_out_x0_1_l2[:, :int(Wh_x0_1_l1 * Ww_x0_1_l1 * Wt_x0_1_l1  / 2), :]
        x_1_out = x_out_x0_1_l2[:, int(Wh_x0_1_l1 * Ww_x0_1_l1 * Wt_x0_1_l1  / 2):, :]
        #x_out_x0_l2 = x_out_x0_1_l2[:, :, 0:self.embed_dim * 8]
        #x_out_x1_l2 = x_out_x0_1_l2[:, :, self.embed_dim * 8:]
        #x_0_out = x_out_x0_l2  # updated x_0
        #x_1_out = x_out_x1_l2   # updated x_1
        #out_x0_x1_l2 = x_0_out  + x_1_out
        out_x0_x1_l2 = torch.concat((x_0_out, x_1_out), dim=2)

        norm_layer = getattr(self, f'norm{2}')
        x_out_l2 = norm_layer(out_x0_x1_l2)
        out = x_out_l2.view(-1, int(H_x0_1/2), W_x0_1, T_x0_1, self.embed_dim * 16).permute(0, 4, 1, 2,
                                                                                                3).contiguous()
        outs.append(out)  # layer 2 output
        # add transformer encoded back to modality-specific branch
        x_0 = x_0_out + x_0
        x_1 = x_1_out + x_1
        # construct the input for the next layer
        x_0 = self.patch_merging_layers[3](x_0, int(Wh_x0_1_l1/2), int(Ww_x0_1_l1), int(Wt_x0_1_l1))
        x_1 = self.patch_merging_layers[3](x_1, int(Wh_x0_1_l1/2), int(Ww_x0_1_l1), int(Wt_x0_1_l1))


        #############layer3################
        layer = self.layers[3]
        # concatenate x_0 and x_1 in dimension 1
        x_0_1 = torch.cat((x_0, x_1),
                          dim=1)  # concatenate in the channel dimension so that the image dimension is still Wh_x0, Ww_x0, Wt_x0 which is needed to be compatible for SWINTR
        # send the concatenated feature vector to transformer
        x_out_x0_1_l3, H_x0_1, W_x0_1, T_x0_1, x_0_1, Wh_x0_1_l3, Ww_x0_1_l3, Wt_x0_1_l3 = layer(x_0_1, int(Wh_x0_1_l2),
                                                                                                 int(Ww_x0_1_l2), int(Wt_x0_1_l2))
        # split the resulting feature vector in dimension 1 to get processed x_1_processed, x_0_processed
        #x_out_x0_1_l3 = x_0_1
        x_0_out = x_out_x0_1_l3[:, :int(Wh_x0_1_l2 * Ww_x0_1_l2  * Wt_x0_1_l2 / 2), :]
        x_1_out = x_out_x0_1_l3[:, int(Wh_x0_1_l2 * Ww_x0_1_l2 * Wt_x0_1_l2 / 2):, :]
        #x_out_x0_l3 = x_out_x0_1_l3[:, :, 0:self.embed_dim * 16]
        #x_out_x1_l3 = x_out_x0_1_l3[:, :, self.embed_dim * 16:]
        #x_0_out = x_out_x0_l3   # updated x_0
        #x_1_out = x_out_x1_l3   # updated x_1
        #out_x0_x1_l3 = x_0_out + x_1_out
        out_x0_x1_l3 = torch.concat((x_0_out, x_1_out), dim=2)

        norm_layer = getattr(self, f'norm{3}')
        x_out_l3 = norm_layer(out_x0_x1_l3)
        out = x_out_l3.view(-1, int(H_x0_1/2), W_x0_1, T_x0_1, self.embed_dim * 32).permute(0, 4, 1, 2,
                                                                                                 3).contiguous()
        outs.append(out)  # layer 2 output

        return outs

    def train(self, mode=True):
        """Convert the model into training mode while keep layers freezed."""
        super(SwinTransformer_wRandomSpatialFeatureTalk_wCrossModalUpdating_5stageOuts, self).train(mode)
        self._freeze_stages()


class SwinTransformer_wFeatureTalk_concat_noCrossModalUpdating_ConvPoolDownsampling(nn.Module):
    r""" Swin Transformer modified to process images from two modalities; feature talks between two images are introduced in encoder
        A PyTorch impl of : `Swin Transformer: Hierarchical Vision Transformer using Shifted Windows`  -
          https://arxiv.org/pdf/2103.14030
    Args:
        img_size (int | tuple(int)): Input image size. Default 224
        patch_size (int | tuple(int)): Patch size. Default: 4
        in_chans (int): Number of input image channels. Default: 3
        num_classes (int): Number of classes for classification head. Default: 1000
        embed_dim (int): Patch embedding dimension. Default: 96
        depths (tuple(int)): Depth of each Swin Transformer layer.
        num_heads (tuple(int)): Number of attention heads in different layers.
        window_size (tuple): Window size. Default: 7
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim. Default: 4
        qkv_bias (bool): If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float): Override default qk scale of head_dim ** -0.5 if set. Default: None
        drop_rate (float): Dropout rate. Default: 0
        attn_drop_rate (float): Attention dropout rate. Default: 0
        drop_path_rate (float): Stochastic depth rate. Default: 0.1
        norm_layer (nn.Module): Normalization layer. Default: nn.LayerNorm.
        ape (bool): If True, add absolute position embedding to the patch embedding. Default: False
        patch_norm (bool): If True, add normalization after patch embedding. Default: True
        use_checkpoint (bool): Whether to use checkpointing to save memory. Default: False
    """

    def __init__(self, pretrain_img_size=160,
                 patch_size=4,
                 in_chans=3,
                 embed_dim=96,
                 depths=[2, 2, 6, 2],
                 num_heads=[3, 6, 12, 24],
                 window_size=(7, 7, 7),
                 mlp_ratio=4.,
                 qkv_bias=True,
                 qk_scale=None,
                 drop_rate=0.,
                 attn_drop_rate=0.,
                 drop_path_rate=0.2,
                 norm_layer=nn.LayerNorm,
                 ape=False,
                 spe=False,
                 patch_norm=True,
                 out_indices=(0, 1, 2, 3),
                 frozen_stages=-1,
                 use_checkpoint=False,
                 pat_merg_rf=2,
                 pos_embed_method='relative'):
        super().__init__()
        self.pretrain_img_size = pretrain_img_size
        self.num_layers = len(depths)
        self.embed_dim = embed_dim
        self.ape = ape
        self.spe = spe
        self.patch_norm = patch_norm
        self.out_indices = out_indices
        self.frozen_stages = frozen_stages

        # split image into non-overlapping patches
        self.patch_embed = PatchEmbed(
            patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dim,
            norm_layer=norm_layer if self.patch_norm else None)

        # absolute position embedding
        if self.ape:
            pretrain_img_size = to_3tuple(self.pretrain_img_size)
            patch_size = to_3tuple(patch_size)
            patches_resolution = [pretrain_img_size[0] // patch_size[0], pretrain_img_size[1] // patch_size[1], pretrain_img_size[2] // patch_size[2]]

            self.absolute_pos_embed = nn.Parameter(
                torch.zeros(1, embed_dim, patches_resolution[0], patches_resolution[1], patches_resolution[2]))
            trunc_normal_(self.absolute_pos_embed, std=.02)
        elif self.spe:
            self.pos_embd = SinusoidalPositionEmbedding().cuda()
        self.pos_drop = nn.Dropout(p=drop_rate)

        # stochastic depth
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]  # stochastic depth decay rule

        # build layers
        self.layers = nn.ModuleList()
        self.patch_downsampling_layers = nn.ModuleList()

        for i_layer in range(self.num_layers):
            layer = BasicLayer(dim=int(embed_dim * 2 ** i_layer),
                                depth=depths[i_layer],
                                num_heads=num_heads[i_layer],
                                window_size=window_size,
                                mlp_ratio=mlp_ratio,
                                qkv_bias=qkv_bias,
                                qk_scale=qk_scale,
                                drop=drop_rate,
                                attn_drop=attn_drop_rate,
                                drop_path=dpr[sum(depths[:i_layer]):sum(depths[:i_layer + 1])],
                                norm_layer=norm_layer,
                                downsample=PatchMerging if (i_layer < self.num_layers - 1) else None,
                                use_checkpoint=use_checkpoint,
                                pat_merg_rf=pat_merg_rf,
                                pos_embed_method=pos_embed_method)
            self.layers.append(layer)

            patch_downsampling_layer = PatchConvPool(int(embed_dim * 2 ** i_layer),  concatenated_input=False)
            self.patch_downsampling_layers.append(patch_downsampling_layer)

        num_features = [int(embed_dim * 2 ** i) for i in range(self.num_layers)]
        self.num_features = num_features



        # add a norm layer for each output
        for i_layer in out_indices:
            layer = norm_layer(num_features[i_layer]*2)
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

    def init_weights(self, pretrained=None):
        """Initialize the weights in backbone.
        Args:
            pretrained (str, optional): Path to pre-trained weights.
                Defaults to None.
        """

        def _init_weights(m):
            if isinstance(m, nn.Linear):
                trunc_normal_(m.weight, std=.02)
                if isinstance(m, nn.Linear) and m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.LayerNorm):
                nn.init.constant_(m.bias, 0)
                nn.init.constant_(m.weight, 1.0)

        if isinstance(pretrained, str):
            self.apply(_init_weights)
        elif pretrained is None:
            self.apply(_init_weights)
        else:
            raise TypeError('pretrained must be a str or None')

    def forward(self, x_0,x_1):
        """Forward function."""
        #PET image
        x_0 = self.patch_embed(x_0)
        Wh_x0, Ww_x0, Wt_x0 = x_0.size(2), x_0.size(3), x_0.size(4)
        if self.ape:
            # interpolate the position embedding to the corresponding size
            absolute_pos_embed_x0 = nnf.interpolate(self.absolute_pos_embed, size=(Wh_x0, Ww_x0, Wt_x0), mode='trilinear')
            x_0 = (x_0 + absolute_pos_embed_x0).flatten(2).transpose(1, 2)  # B Wh*Ww*Wt C
        elif self.spe:
            print(self.spe)
            x_0 = x_0.flatten(2).transpose(1, 2)
            x_0 += self.pos_embd(x_0)
        else:
            x_0 = x_0.flatten(2).transpose(1, 2)
        x_0 = self.pos_drop(x_0)

        #CT image
        x_1 = self.patch_embed(x_1)
        Wh_x1, Ww_x1, Wt_x1 = x_1.size(2), x_1.size(3), x_1.size(4)
        if self.ape:
            # interpolate the position embedding to the corresponding size
            absolute_pos_embed_x1 = nnf.interpolate(self.absolute_pos_embed, size=(Wh_x1, Ww_x1, Wt_x1),
                                                    mode='trilinear')
            x_1 = (x_1 + absolute_pos_embed_x1).flatten(2).transpose(1, 2)  # B Wh*Ww*Wt C
        elif self.spe:
            print(self.spe)
            x_1 = x_1.flatten(2).transpose(1, 2)
            x_1 += self.pos_embd(x_1)
        else:
            x_1 = x_1.flatten(2).transpose(1, 2)
        x_1 = self.pos_drop(x_1)


        outs = []

        #############layer0################
        layer = self.layers[0]
        # concatenate x_0 and x_1 in dimension 1
        x_0_1 = torch.cat((x_0,x_1),dim=2) #concatenate in the channel dimension so that the image dimension is still Wh_x0, Ww_x0, Wt_x0 which is needed to be compatible for SWINTR
        # send the concatenated feature vector to transformer
        x_out_x0_1_l0, H_x0_1, W_x0_1, T_x0_1, x_0_1, Wh_x0_1_l0, Ww_x0_1_l0, Wt_x0_1_l0 = layer(x_0_1, Wh_x0, Ww_x0, Wt_x0)
        # split the resulting feature vector in dimension 1 to get processed x_1_processed, x_0_processed
        x_out_x0_l0 = x_out_x0_1_l0[:,:,0:self.embed_dim]
        x_out_x1_l0 = x_out_x0_1_l0[:,:,self.embed_dim:]
        # add x_1_process and x_0_processed to x_1 and x_0 and start layer 1
        x_0_out = x_out_x0_l0 + x_0 # updated x_0
        x_1_out = x_out_x1_l0 + x_1 # updated x_1
        #out_x0_x1_l0 = x_0_out + x_1_out
        out_x0_x1_l0 = torch.concat((x_0_out , x_1_out),dim=2)

        norm_layer = getattr(self, f'norm{0}')
        x_out_l0 = norm_layer(out_x0_x1_l0)
        out = x_out_l0.view(-1, Wh_x0, Ww_x0, Wt_x0, self.embed_dim*2).permute(0, 4, 1, 2, 3).contiguous()
        outs.append(out) #layer 0 output

        #construct the input for the next layer; use conv and pool and view, to downsample input of size of (1,64000,128) to (1,8000,256)
        x_0_down = self.patch_downsampling_layers[0](x_0, Wh_x0, Ww_x0, Wt_x0)
        x_1_down = self.patch_downsampling_layers[0](x_1, Wh_x1, Ww_x1, Wt_x1)
        x_0 = x_0_down
        x_1 = x_1_down

        #############layer1################
        layer = self.layers[1]
        # concatenate x_0 and x_1 in dimension 1
        x_0_1 = torch.cat((x_0, x_1),
                          dim=2)  # concatenate in the channel dimension so that the image dimension is still Wh_x0, Ww_x0, Wt_x0 which is needed to be compatible for SWINTR
        # send the concatenated feature vector to transformer
        x_out_x0_1_l1, H_x0_1, W_x0_1, T_x0_1, x_0_1, Wh_x0_1_l1, Ww_x0_1_l1, Wt_x0_1_l1 = layer(x_0_1, Wh_x0_1_l0, Ww_x0_1_l0, Wt_x0_1_l0)
        # split the resulting feature vector in dimension 1 to get processed x_1_processed, x_0_processed
        x_out_x0_l1 = x_out_x0_1_l1[:, :, 0:self.embed_dim*2]
        x_out_x1_l1 = x_out_x0_1_l1[:, :, self.embed_dim*2:]
        x_0_out = x_out_x0_l1 + x_0  # updated x_0
        x_1_out = x_out_x1_l1 + x_1  # updated x_1
        #out_x0_x1_l1 = x_0_out + x_1_out #should I use the sum or concat for decoder?
        out_x0_x1_l1 = torch.concat((x_0_out , x_1_out),dim=2)

        norm_layer = getattr(self, f'norm{1}')
        x_out_l1 = norm_layer(out_x0_x1_l1)
        out = x_out_l1.view(-1, Wh_x0_1_l0, Ww_x0_1_l0, Wt_x0_1_l0, self.embed_dim*4).permute(0, 4, 1, 2, 3).contiguous()
        outs.append(out)  # layer 1 output

        #construct the input for the next layer
        x_0_down = self.patch_downsampling_layers[1](x_0, Wh_x0_1_l0, Ww_x0_1_l0, Wt_x0_1_l0)
        x_1_down = self.patch_downsampling_layers[1](x_1, Wh_x0_1_l0, Ww_x0_1_l0, Wt_x0_1_l0)
        x_0 = x_0_down
        x_1 = x_1_down

        #############layer2################
        layer = self.layers[2]
        # concatenate x_0 and x_1 in dimension 1
        x_0_1 = torch.cat((x_0, x_1),
                          dim=2)  # concatenate in the channel dimension so that the image dimension is still Wh_x0, Ww_x0, Wt_x0 which is needed to be compatible for SWINTR
        # send the concatenated feature vector to transformer
        x_out_x0_1_l2, H_x0_1, W_x0_1, T_x0_1, x_0_1, Wh_x0_1_l2, Ww_x0_1_l2, Wt_x0_1_l2 = layer(x_0_1, Wh_x0_1_l1,
                                                                                                 Ww_x0_1_l1, Wt_x0_1_l1)
        # split the resulting feature vector in dimension 1 to get processed x_1_processed, x_0_processed
        x_out_x0_l2 = x_out_x0_1_l2[:, :, 0:self.embed_dim * 4]
        x_out_x1_l2 = x_out_x0_1_l2[:, :, self.embed_dim * 4:]
        x_0_out = x_out_x0_l2 + x_0  # updated x_0
        x_1_out = x_out_x1_l2 + x_1  # updated x_1
        #out_x0_x1_l2 = x_0_out + x_1_out
        out_x0_x1_l2 = torch.concat((x_0_out , x_1_out),dim=2)

        norm_layer = getattr(self, f'norm{2}')
        x_out_l2 = norm_layer(out_x0_x1_l2)
        out = x_out_l2.view(-1, Wh_x0_1_l1, Ww_x0_1_l1, Wt_x0_1_l1, self.embed_dim*8).permute(0, 4, 1, 2, 3).contiguous()
        outs.append(out)  # layer 2 output

        # construct the input for the next layer
        x_0_down = self.patch_downsampling_layers[2](x_0, Wh_x0_1_l1, Ww_x0_1_l1, Wt_x0_1_l1)
        x_1_down = self.patch_downsampling_layers[2](x_1, Wh_x0_1_l1, Ww_x0_1_l1, Wt_x0_1_l1)
        x_0 = x_0_down
        x_1 = x_1_down

        #############layer3################
        layer = self.layers[3]
        # concatenate x_0 and x_1 in dimension 1
        x_0_1 = torch.cat((x_0, x_1),
                          dim=2)  # concatenate in the channel dimension so that the image dimension is still Wh_x0, Ww_x0, Wt_x0 which is needed to be compatible for SWINTR
        # send the concatenated feature vector to transformer
        x_out_x0_1_l3, H_x0_1, W_x0_1, T_x0_1, x_0_1, Wh_x0_1_l3, Ww_x0_1_l3, Wt_x0_1_l3 = layer(x_0_1, Wh_x0_1_l2,
                                                                                                 Ww_x0_1_l2, Wt_x0_1_l2)
        # split the resulting feature vector in dimension 1 to get processed x_1_processed, x_0_processed
        x_out_x0_l3 = x_out_x0_1_l3[:, :, 0:self.embed_dim * 8]
        x_out_x1_l3 = x_out_x0_1_l3[:, :, self.embed_dim * 8:]
        x_0_out = x_out_x0_l3 + x_0  # updated x_0
        x_1_out = x_out_x1_l3 + x_1  # updated x_1
        #out_x0_x1_l3 = x_0_out + x_1_out
        out_x0_x1_l3 = torch.concat((x_0_out , x_1_out),dim=2)

        norm_layer = getattr(self, f'norm{3}')
        x_out_l3 = norm_layer(out_x0_x1_l3)
        out = x_out_l3.view(-1, Wh_x0_1_l2, Ww_x0_1_l2, Wt_x0_1_l2, self.embed_dim*16).permute(0, 4, 1, 2, 3).contiguous()
        outs.append(out)  # layer 2 output

        return outs

    def train(self, mode=True):
        """Convert the model into training mode while keep layers freezed."""
        super(SwinTransformer_wFeatureTalk_concat_noCrossModalUpdating_ConvPoolDownsampling, self).train(mode)
        self._freeze_stages()




# feature 96        
class TransMorph_Unetr_CT_Lung_Tumor_Batch_Norm_Correction_Official_No_Unused_Parameters_Cross_Attention(nn.Module):
    def __init__(
        self,
        config,
        out_channels: int = 2,
        feature_size: int = 48,
        hidden_size: int = 768,
        mlp_dim: int = 3072,
        img_size: int = 128,
        num_heads: int = 12,
        pos_embed: str = "perceptron",
        norm_name: Union[Tuple, str] = "batch",
        conv_block: bool = False,
        res_block: bool = True,
        spatial_dims: int = 3,
        in_channels: int=1,
        #out_channels: int,
    ) -> None:
        '''
        TransMorph Model
        '''
        
        #super(TransMorph_Unetr, self).__init__()
        super().__init__()
        self.hidden_size = hidden_size
        self.feat_size=(img_size//32,img_size//32,img_size//32)
        
        embed_dim = 96#config.embed_dim

        #SwinTransformer_wCrossModalityFeatureTalk_OutSum_5stageOuts
        #SwinTransformer_wDualModalityFeatureTalk_OutSum_5stageOuts
        self.transformer = SwinTransformer_wDualModalityFeatureTalk_OutSum_5stageOuts(patch_size=config.patch_size,
                                           pretrain_img_size=config.img_size[0],
                                           in_chans=int(config.in_chans/2),
                                           embed_dim=config.embed_dim,
                                           depths=config.depths,
                                           num_heads=config.num_heads,
                                           window_size=config.window_size,
                                           mlp_ratio=config.mlp_ratio,
                                           qkv_bias=config.qkv_bias,
                                           drop_rate=config.drop_rate,
                                           drop_path_rate=config.drop_path_rate,
                                           ape=config.ape,
                                           spe=config.spe,
                                           patch_norm=config.patch_norm,
                                           use_checkpoint=config.use_checkpoint,
                                           out_indices=config.out_indices,
                                           pat_merg_rf=config.pat_merg_rf,
                                           )
        #below is the decoder from UnetR

        self.encoder1 = UnetrBasicBlock(
            spatial_dims=spatial_dims,
            in_channels=config.in_chans,
            out_channels=feature_size,
            kernel_size=3,
            stride=1,
            norm_name=norm_name,
            res_block=True,
        )

        self.encoder2 = UnetrBasicBlock_No_DownSampling(
            spatial_dims=spatial_dims,
            in_channels=feature_size,
            out_channels=feature_size,
            kernel_size=3,
            stride=1,
            norm_name=norm_name,
            res_block=True,
        )

        self.encoder3 = UnetrBasicBlock_No_DownSampling(
            spatial_dims=spatial_dims,
            in_channels=2 * feature_size,
            out_channels=2 * feature_size,
            kernel_size=3,
            stride=1,
            norm_name=norm_name,
            res_block=True,
        )

        self.encoder4 = UnetrBasicBlock_No_DownSampling(
            spatial_dims=spatial_dims,
            in_channels=4 * feature_size,
            out_channels=4 * feature_size,
            kernel_size=3,
            stride=1,
            norm_name=norm_name,
            res_block=True,
        )

        self.encoder10 = UnetrBasicBlock_No_DownSampling(
            spatial_dims=spatial_dims,
            in_channels=16 * feature_size,
            out_channels=16 * feature_size,
            kernel_size=3,
            stride=1,
            norm_name=norm_name,
            res_block=True,
        )

        self.decoder5 = UnetrUpBlock(
            spatial_dims=spatial_dims,
            in_channels=16 * feature_size,
            out_channels=8 * feature_size,
            kernel_size=3,
            upsample_kernel_size=2,
            norm_name=norm_name,
            res_block=True,
        )

        self.decoder4 = UnetrUpBlock(
            spatial_dims=spatial_dims,
            in_channels=feature_size * 8,
            out_channels=feature_size * 4,
            kernel_size=3,
            upsample_kernel_size=2,
            norm_name=norm_name,
            res_block=True,
        )

        self.decoder3 = UnetrUpBlock(
            spatial_dims=spatial_dims,
            in_channels=feature_size * 4,
            out_channels=feature_size * 2,
            kernel_size=3,
            upsample_kernel_size=2,
            norm_name=norm_name,
            res_block=True,
        )
        self.decoder2 = UnetrUpBlock(
            spatial_dims=spatial_dims,
            in_channels=feature_size * 2,
            out_channels=feature_size,
            kernel_size=3,
            upsample_kernel_size=2,
            norm_name=norm_name,
            res_block=True,
        )

        self.decoder1 = UnetrUpBlock(
            spatial_dims=spatial_dims,
            in_channels=feature_size,
            out_channels=feature_size,
            kernel_size=3,
            upsample_kernel_size=2,
            norm_name=norm_name,
            res_block=True,
        )

        self.out = UnetOutBlock(
            spatial_dims=spatial_dims, in_channels=feature_size, out_channels=out_channels
        )  # type: ignore

    def proj_feat(self, x, hidden_size, feat_size):
        x = x.view(x.size(0), feat_size[0], feat_size[1], feat_size[2], hidden_size)
        x = x.permute(0, 4, 1, 2, 3).contiguous()
        return x

    def forward(self, x_in):

        #x, out_feats = self.transformer(x_in)

        out_feats = self.transformer(x_in)
        
        #for item in out_feats:
        #    print ('info: size is ',item.shape)

        #info: size is  torch.Size([6, 48, 64, 64, 64])
        #info: size is  torch.Size([6, 96, 32, 32, 32])
        #info: size is  torch.Size([6, 192, 16, 16, 16])
        #info: size is  torch.Size([6, 384, 8, 8, 8])
        #info: size is  torch.Size([6, 768, 4, 4, 4])

        enc44 = out_feats[3]   # torch.Size([4, 384, 8, 8, 8])  
        enc33 = out_feats[2]   # torch.Size([4, 192, 16, 16, 16])
        enc22 = out_feats[1]   # torch.Size([4, 96, 32, 32, 32])   
        enc11 = out_feats[0]   # torch.Size([4, 48, 64, 64, 64])    
        #x=self.proj_feat(x, self.hidden_size, self.feat_size) # torch.Size([4, 768, 4, 4, 4])  
        x=out_feats[4]

        #print ('encoder x after projection size is ',x.size())

        #print ('input enc0 size ',x_in.size())
        enc0 = self.encoder1(x_in)
        #print ('out enc0 size ',enc0.size())
        enc1 = self.encoder2(enc11) #input size torch.Size([4, 96, 64, 64, 64])
        #print ('enc1 size ',enc1.size())
        enc2 = self.encoder3(enc22) #input size torch.Size([4, 192, 32, 32, 32])
        #print ('enc2 size ',enc2.size())
        enc3 = self.encoder4(enc33) #torch.Size([4, 384, 16, 16, 16])
        #print ('enc3 size ',enc3.size())

        dec4 = self.encoder10(x)

        dec3 = self.decoder5(dec4, enc44)
        dec2 = self.decoder4(dec3, enc3)
        dec1 = self.decoder3(dec2, enc2)
        dec0 = self.decoder2(dec1, enc1)
        out = self.decoder1(dec0, enc0)
        logits = self.out(out)

        

        return logits
    
class Conv3dReLU(nn.Sequential):
    def __init__(
            self,
            in_channels,
            out_channels,
            kernel_size,
            padding=0,
            stride=1,
            use_batchnorm=True,
    ):
        conv = nn.Conv3d(
            in_channels,
            out_channels,
            kernel_size,
            stride=stride,
            padding=padding,
            bias=False,
        )
        relu = nn.LeakyReLU(inplace=True)
        if not use_batchnorm:
            nm = nn.InstanceNorm3d(out_channels)
        else:
            nm = nn.BatchNorm3d(out_channels)

        super(Conv3dReLU, self).__init__(conv, nm, relu)


# Residual block
class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, downsample=None):
        super(ResidualBlock, self).__init__()
        self.conv_block = nn.Sequential(
        nn.Conv3d(in_channels, out_channels, stride),
        nn.BatchNorm3d(out_channels),
        nn.ReLU(inplace=True),
        nn.Conv3d(out_channels, out_channels, stride),
        nn.BatchNorm3d(out_channels)
        )
        self.conv_skip = nn.Sequential(
            nn.Conv3d(in_channels, out_channels, stride),
            nn.BatchNorm3d(out_channels),
        )

    def forward(self, x):
        # residual = self.conv_skip(x)
        # out = self.conv1(x)
        # out = self.bn1(out)
        # out = self.relu(out)
        # out = self.conv2(out)
        # out = self.bn2(out)
        #
        # out += residual
        # out = self.relu(out)
        return self.conv_block(x) + self.conv_skip(x)


class DecoderBlock(nn.Module):
    def __init__(
            self,
            in_channels,
            out_channels,
            skip_channels=0,
            use_batchnorm=True,
    ):
        super().__init__()
        # self.conv1 = Conv3dReLU(
        #     out_channels+skip_channels,
        #     out_channels,
        #     kernel_size=3,
        #     padding=1,
        #     use_batchnorm=use_batchnorm,
        # )
        # self.conv2 = Conv3dReLU(
        #     out_channels,
        #     out_channels,
        #     kernel_size=3,
        #     padding=1,
        #     use_batchnorm=use_batchnorm,
        # )
        self.up = nn.ConvTranspose3d(in_channels,out_channels,kernel_size=2,stride=2)

    def forward(self, x, skip=None):
        x = self.up(x)
        #if skip is not None:
        #    x = torch.cat([x, skip], dim=1)
        #x = self.conv1(x)
        #x = self.conv2(x)
        return x

class RegistrationHead(nn.Sequential):
    def __init__(self, in_channels, out_channels, kernel_size=3, upsampling=1):
        conv3d = nn.Conv3d(in_channels, out_channels, kernel_size=kernel_size, padding=kernel_size // 2)
        conv3d.weight = nn.Parameter(Normal(0, 1e-5).sample(conv3d.weight.shape))
        conv3d.bias = nn.Parameter(torch.zeros(conv3d.bias.shape))
        super().__init__(conv3d)


class SegmentationHead(nn.Sequential):
    def __init__(self, in_channels, num_classes, image_size=(128,128,48), kernel_size=3, upsampling=1):
        #conv3d = nn.Conv3d(in_channels, num_classes, kernel_size=1)
        conv3d = nn.Conv3d(in_channels, num_classes, 1,1,0,1,1,False)
        softmax = nn.Softmax(dim=1)
        #Reshape = torch.reshape([np.prod(image_size),num_classes])
        #softmax = torch.nn.functional.softmax()
        #conv3d.weight = nn.Parameter(Normal(0, 1e-5).sample(conv3d.weight.shape))
        #conv3d.bias = nn.Parameter(torch.zeros(conv3d.bias.shape))
        super(SegmentationHead, self).__init__(conv3d,softmax)

class SegmentationHead_new(nn.Sequential):
    def __init__(self, in_channels, num_classes, kernel_size=1, upsampling=1):
        #conv3d = nn.Conv3d(in_channels, num_classes, kernel_size=1)
        conv3d = nn.Conv3d(in_channels, num_classes, 1,1,0,1,1, False)
        sigmoid = nn.Sigmoid()
        #conv3d.weight = nn.Parameter(Normal(0, 1e-5).sample(conv3d.weight.shape))
        #conv3d.bias = nn.Parameter(torch.zeros(conv3d.bias.shape))
        super(SegmentationHead_new, self).__init__(conv3d,sigmoid)

class SpatialTransformer(nn.Module):
    """
    N-D Spatial Transformer

    Obtained from https://github.com/voxelmorph/voxelmorph
    """

    def __init__(self, size, mode='bilinear'):
        super().__init__()

        self.mode = mode

        # create sampling grid
        vectors = [torch.arange(0, s) for s in size]
        grids = torch.meshgrid(vectors)
        grid = torch.stack(grids)
        grid = torch.unsqueeze(grid, 0)
        grid = grid.type(torch.FloatTensor)

        # registering the grid as a buffer cleanly moves it to the GPU, but it also
        # adds it to the state dict. this is annoying since everything in the state dict
        # is included when saving weights to disk, so the model files are way bigger
        # than they need to be. so far, there does not appear to be an elegant solution.
        # see: https://discuss.pytorch.org/t/how-to-register-buffer-without-polluting-state-dict
        self.register_buffer('grid', grid)

    def forward(self, src, flow):
        # new locations
        new_locs = self.grid + flow
        shape = flow.shape[2:]

        # need to normalize grid values to [-1, 1] for resampler
        for i in range(len(shape)):
            new_locs[:, i, ...] = 2 * (new_locs[:, i, ...] / (shape[i] - 1) - 0.5)

        # move channels dim to last position
        # also not sure why, but the channels need to be reversed
        if len(shape) == 2:
            new_locs = new_locs.permute(0, 2, 3, 1)
            new_locs = new_locs[..., [1, 0]]
        elif len(shape) == 3:
            new_locs = new_locs.permute(0, 2, 3, 4, 1)
            new_locs = new_locs[..., [2, 1, 0]]

        return nnf.grid_sample(src, new_locs, align_corners=True, mode=self.mode)

class SwinVNetSkip(nn.Module):
    def __init__(self, config):
        super(SwinVNetSkip, self).__init__()
        if_convskip = config.if_convskip
        self.if_convskip = if_convskip
        if_transskip = config.if_transskip
        self.if_transskip = if_transskip
        embed_dim = config.embed_dim
        self.transformer = SwinTransformer(patch_size=config.patch_size,
                                           in_chans=config.in_chans,
                                           embed_dim=config.embed_dim,
                                           depths=config.depths,
                                           num_heads=config.num_heads,
                                           window_size=config.window_size,
                                           mlp_ratio=config.mlp_ratio,
                                           qkv_bias=config.qkv_bias,
                                           drop_rate=config.drop_rate,
                                           drop_path_rate=config.drop_path_rate,
                                           ape=config.ape,
                                           spe=config.spe,
                                           patch_norm=config.patch_norm,
                                           use_checkpoint=config.use_checkpoint,
                                           out_indices=config.out_indices,
                                           pat_merg_rf=config.pat_merg_rf,
                                           pos_embed_method=config.pos_embed_method,
                                           concatenated_input=False)
        self.up0 = DecoderBlock(embed_dim*8, embed_dim*4, skip_channels=embed_dim*4 if if_transskip else 0, use_batchnorm=False)
        self.up1 = DecoderBlock(embed_dim*4, embed_dim*2, skip_channels=embed_dim*2 if if_transskip else 0, use_batchnorm=False)  # 384, 20, 20, 64
        self.up2 = DecoderBlock(embed_dim*2, embed_dim, skip_channels=embed_dim if if_transskip else 0, use_batchnorm=False)  # 384, 40, 40, 64
        self.up3 = DecoderBlock(embed_dim, embed_dim//2, skip_channels=embed_dim//2 if if_convskip else 0, use_batchnorm=False)  # 384, 80, 80, 128
        self.up4 = DecoderBlock(embed_dim//2, config.seg_head_chan, skip_channels=config.seg_head_chan if if_convskip else 0, use_batchnorm=False)  # 384, 160, 160, 256
        self.c1 = Conv3dReLU(2, embed_dim//2, 3, 1, use_batchnorm=False)
        self.c2 = Conv3dReLU(2, config.seg_head_chan, 3, 1, use_batchnorm=False)
        self.seg_head = SegmentationHead_new(
            in_channels=config.seg_head_chan,
            num_classes=2,
            kernel_size=3,
        )
        self.spatial_trans = SpatialTransformer(config.img_size)
        self.avg_pool = nn.AvgPool3d(3, stride=2, padding=1)

    def forward(self, x):
        #source = x[:, 0:1, :, :]
        if self.if_convskip:
            x_s0 = x.clone()
            x_s1 = self.avg_pool(x)
            f4 = self.c1(x_s1)
            f5 = self.c2(x_s0)
        else:
            f4 = None
            f5 = None

        out = self.transformer(x)  # (B, n_patch, hidden)

        if self.if_transskip:
            f1 = out[-2]
            f2 = out[-3]
            f3 = out[-4]
        else:
            f1 = None
            f2 = None
            f3 = None
        x = self.up0(out[-1], f1)
        x = self.up1(x, f2)
        x = self.up2(x, f3)
        x = self.up3(x, f4)
        x = self.up4(x, f5)
        out = self.seg_head(x)
        #out = self.spatial_trans(source, flow)
        return out

from monai.networks.blocks import UnetrBasicBlock,UnetResBlock,UnetrUpBlock,UnetrPrUpBlock
from monai.networks.blocks.dynunet_block import UnetOutBlock, get_conv_layer, UnetBasicBlock

from typing import Sequence, Tuple, Union

class SWINUnetrUpBlock(nn.Module):
    """
    An upsampling module that can be used for UNETR: "Hatamizadeh et al.,
    UNETR: Transformers for 3D Medical Image Segmentation <https://arxiv.org/abs/2103.10504>"
    """

    def __init__(
        self,
        spatial_dims: int,
        in_channels: int,
        out_channels: int,
        kernel_size: Union[Sequence[int], int],
        upsample_kernel_size: Union[Sequence[int], int],
        norm_name: Union[Tuple, str],
        res_block: bool = False,
    ) -> None:
        """
        Args:
            spatial_dims: number of spatial dimensions.
            in_channels: number of input channels.
            out_channels: number of output channels.
            kernel_size: convolution kernel size.
            upsample_kernel_size: convolution kernel size for transposed convolution layers.
            norm_name: feature normalization type and arguments.
            res_block: bool argument to determine if residual block is used.

        """

        super().__init__()
        upsample_stride = upsample_kernel_size
        self.transp_conv = get_conv_layer(
            spatial_dims,
            in_channels,
            out_channels,
            kernel_size=upsample_kernel_size,
            stride=upsample_stride,
            conv_only=True,
            is_transposed=True,
        )

        if res_block:
            self.conv_block = UnetResBlock(
                spatial_dims,
                in_channels + in_channels,
                in_channels,
                kernel_size=kernel_size,
                stride=1,
                norm_name=norm_name,
            )
        else:
            self.conv_block = UnetBasicBlock(  # type: ignore
                spatial_dims,
                in_channels + in_channels,
                in_channels,
                kernel_size=kernel_size,
                stride=1,
                norm_name=norm_name,
            )

    def forward(self, inp, skip):
        # number of channels for skip should equals to out_channels
        out = torch.cat((inp, skip), dim=1)
        out = self.conv_block(out)
        out = self.transp_conv(out)

        return out


class SWINUnetrBlock(nn.Module):
    """
    An upsampling module that can be used for UNETR: "Hatamizadeh et al.,
    UNETR: Transformers for 3D Medical Image Segmentation <https://arxiv.org/abs/2103.10504>"
    """

    def __init__(
        self,
        spatial_dims: int,
        in_channels: int,
        out_channels: int,
        kernel_size: Union[Sequence[int], int],
        upsample_kernel_size: Union[Sequence[int], int],
        norm_name: Union[Tuple, str],
        res_block: bool = False,
    ) -> None:
        """
        Args:
            spatial_dims: number of spatial dimensions.
            in_channels: number of input channels.
            out_channels: number of output channels.
            kernel_size: convolution kernel size.
            upsample_kernel_size: convolution kernel size for transposed convolution layers.
            norm_name: feature normalization type and arguments.
            res_block: bool argument to determine if residual block is used.

        """

        super().__init__()
        upsample_stride = upsample_kernel_size
        self.transp_conv = get_conv_layer(
            spatial_dims,
            in_channels,
            out_channels,
            kernel_size=upsample_kernel_size,
            stride=upsample_stride,
            conv_only=True,
            is_transposed=True,
        )

        if res_block:
            self.conv_block = UnetResBlock(
                spatial_dims,
                in_channels + in_channels,
                in_channels,
                kernel_size=kernel_size,
                stride=1,
                norm_name=norm_name,
            )
        else:
            self.conv_block = UnetBasicBlock(  # type: ignore
                spatial_dims,
                in_channels + in_channels,
                in_channels,
                kernel_size=kernel_size,
                stride=1,
                norm_name=norm_name,
            )

    def forward(self, inp, skip):
        # number of channels for skip should equals to out_channels
        out = torch.cat((inp, skip), dim=1)
        out = self.conv_block(out)
        #out = self.transp_conv(out)

        return out

class SwinUNETR_self(nn.Module):
    def __init__(self, config):
        super(SwinUNETR_self, self).__init__()
        if_convskip = config.if_convskip
        self.if_convskip = if_convskip
        if_transskip = config.if_transskip
        self.if_transskip = if_transskip
        embed_dim = config.embed_dim
        self.transformer = SwinTransformer(patch_size=config.patch_size,
                                           in_chans=config.in_chans,
                                           embed_dim=config.embed_dim,
                                           depths=config.depths,
                                           num_heads=config.num_heads,
                                           window_size=config.window_size,
                                           mlp_ratio=config.mlp_ratio,
                                           qkv_bias=config.qkv_bias,
                                           drop_rate=config.drop_rate,
                                           drop_path_rate=config.drop_path_rate,
                                           ape=config.ape,
                                           spe=config.spe,
                                           patch_norm=config.patch_norm,
                                           use_checkpoint=config.use_checkpoint,
                                           out_indices=config.out_indices,
                                           pat_merg_rf=config.pat_merg_rf,
                                           pos_embed_method=config.pos_embed_method,
                                           concatenated_input=False)

        self.encoder0 = UnetrBasicBlock(
            spatial_dims=3,
            in_channels=config.in_chans,
            out_channels=embed_dim,
            kernel_size=3,
            stride=1,
            norm_name='instance',
            res_block=True,
        )

        self.encoder1 = UnetrBasicBlock(
            spatial_dims=3,
            in_channels=embed_dim,
            out_channels=embed_dim,
            kernel_size=3,
            stride=1,
            norm_name='instance',
            res_block=True,
        )

        self.encoder2 = UnetResBlock(
            spatial_dims=3,
            in_channels=embed_dim * 2,
            out_channels=embed_dim * 2,
            kernel_size=3,
            stride=1,
            norm_name='instance',
        )

        self.encoder3 = UnetResBlock(
            spatial_dims=3,
            in_channels=embed_dim * 4,
            out_channels=embed_dim * 4,
            kernel_size=3,
            stride=1,
            norm_name='instance',
        )

        self.encoder4 = UnetResBlock(
            spatial_dims=3,
            in_channels=embed_dim * 8,
            out_channels=embed_dim * 8,
            kernel_size=3,
            stride=1,
            norm_name='instance',
        )

        self.res_botneck = UnetResBlock(
                spatial_dims=3,
                in_channels=embed_dim*16,
                out_channels=embed_dim*16,
                kernel_size=3,
                stride=1,
                norm_name='instance',
        )

        self.decoder5 = UnetrPrUpBlock(
            spatial_dims=3,
            in_channels=embed_dim*16,
            out_channels=embed_dim*8,
            num_layer=0,
            kernel_size=3,
            stride=1,
            upsample_kernel_size=2,
            norm_name='instance',
            conv_block=True,
            res_block=True,
        )

        self.decoder4 = SWINUnetrUpBlock(
            spatial_dims=3,
            in_channels=embed_dim*8,
            out_channels=embed_dim*4,
            kernel_size=3,
            upsample_kernel_size=2,
            norm_name='instance',
            res_block=True,
        )

        self.decoder3 = SWINUnetrUpBlock(
            spatial_dims=3,
            in_channels=embed_dim * 4,
            out_channels=embed_dim * 2,
            kernel_size=3,
            upsample_kernel_size=2,
            norm_name='instance',
            res_block=True,
        )
        self.decoder2 = SWINUnetrUpBlock(
            spatial_dims=3,
            in_channels=embed_dim * 2,
            out_channels=embed_dim * 1,
            kernel_size=3,
            upsample_kernel_size=2,
            norm_name='instance',
            res_block=True,
        )

        self.decoder1 = SWINUnetrUpBlock(
            spatial_dims=3,
            in_channels=embed_dim * 1,
            out_channels=embed_dim * 1,
            kernel_size=3,
            upsample_kernel_size=2,
            norm_name='instance',
            res_block=True,
        )

        self.decoder0 = SWINUnetrUpBlock(
            spatial_dims=3,
            in_channels=embed_dim * 1,
            out_channels=embed_dim * 1,
            kernel_size=3,
            upsample_kernel_size=1,
            norm_name='instance',
            res_block=True,
        )

        self.out = UnetOutBlock(spatial_dims=3, in_channels=embed_dim, out_channels=2)  # type: ignore



    def forward(self, x):

        out = self.transformer(x)  # (B, n_patch, hidden)
        #print(out[-1].size())

        #stage 4 features
        enc5 = self.res_botneck(out[-1]) # B, 5,5,5,2048
        dec4 = self.decoder5(enc5) #B, 10,10,10,1024
        enc4 = self.encoder4(out[-2]) #skip features should be twice the di

        #stage 3 features
        dec3 = self.decoder4(dec4,enc4)
        enc3 = self.encoder3(out[-3]) #skip features

        #stage 2 features
        dec2 = self.decoder3(dec3,enc3)
        enc2 = self.encoder2(out[-4]) #skip features

        # stage 1 features
        dec1 = self.decoder2(dec2, enc2)
        enc1 = self.encoder1(out[-5])  # skip features

        dec0 = self.decoder1(dec1, enc1)
        enc0 = self.encoder0(x)

        head = self.decoder0(dec0, enc0)

        logits = self.out(head)

        return logits


class SwinUNETR_inputsFusion(nn.Module):
    def __init__(self, config):
        super(SwinUNETR_inputsFusion, self).__init__()
        if_convskip = config.if_convskip
        self.if_convskip = if_convskip
        if_transskip = config.if_transskip
        self.if_transskip = if_transskip
        embed_dim = config.embed_dim
        self.transformer = SwinTransformer(patch_size=config.patch_size,
                                           in_chans=1,
                                           embed_dim=config.embed_dim,
                                           depths=config.depths,
                                           num_heads=config.num_heads,
                                           window_size=config.window_size,
                                           mlp_ratio=config.mlp_ratio,
                                           qkv_bias=config.qkv_bias,
                                           drop_rate=config.drop_rate,
                                           drop_path_rate=config.drop_path_rate,
                                           ape=config.ape,
                                           spe=config.spe,
                                           patch_norm=config.patch_norm,
                                           use_checkpoint=config.use_checkpoint,
                                           out_indices=config.out_indices,
                                           pat_merg_rf=config.pat_merg_rf,
                                           pos_embed_method=config.pos_embed_method,
                                           concatenated_input=False)

        # self.res_fusionBlock = UnetResBlock(
        #     spatial_dims=3,
        #     in_channels=config.in_chans,
        #     out_channels=1,
        #     kernel_size=3,
        #     stride=1,
        #     norm_name='instance',
        # )

        self.res_fusionBlock = depthwise_separable_conv3d(
            nin=config.in_chans,
            kernels_per_layer=48,
            nout=1,
        )
        self.encoder0 = depthwise_separable_conv3d(
            nin=1,
            kernels_per_layer=48,
            nout=embed_dim,
        )


        # UnetrBasicBlock(
        #     spatial_dims=3,
        #     in_channels=1,
        #     out_channels=embed_dim,
        #     kernel_size=3,
        #     stride=1,
        #     norm_name='instance',
        #     res_block=True,
        # )

        self.encoder1 = UnetrBasicBlock(
            spatial_dims=3,
            in_channels=embed_dim,
            out_channels=embed_dim,
            kernel_size=3,
            stride=1,
            norm_name='instance',
            res_block=True,
        )

        self.encoder2 = UnetResBlock(
            spatial_dims=3,
            in_channels=embed_dim * 2,
            out_channels=embed_dim * 2,
            kernel_size=3,
            stride=1,
            norm_name='instance',
        )

        self.encoder3 = UnetResBlock(
            spatial_dims=3,
            in_channels=embed_dim * 4,
            out_channels=embed_dim * 4,
            kernel_size=3,
            stride=1,
            norm_name='instance',
        )

        self.encoder4 = UnetResBlock(
            spatial_dims=3,
            in_channels=embed_dim * 8,
            out_channels=embed_dim * 8,
            kernel_size=3,
            stride=1,
            norm_name='instance',
        )

        self.res_botneck = UnetResBlock(
            spatial_dims=3,
            in_channels=embed_dim * 16,
            out_channels=embed_dim * 16,
            kernel_size=3,
            stride=1,
            norm_name='instance',
        )

        self.decoder5 = UnetrPrUpBlock(
            spatial_dims=3,
            in_channels=embed_dim * 16,
            out_channels=embed_dim * 8,
            num_layer=0,
            kernel_size=3,
            stride=1,
            upsample_kernel_size=2,
            norm_name='instance',
            conv_block=True,
            res_block=True,
        )

        self.decoder4 = SWINUnetrUpBlock(
            spatial_dims=3,
            in_channels=embed_dim * 8,
            out_channels=embed_dim * 4,
            kernel_size=3,
            upsample_kernel_size=2,
            norm_name='instance',
            res_block=True,
        )

        self.decoder3 = SWINUnetrUpBlock(
            spatial_dims=3,
            in_channels=embed_dim * 4,
            out_channels=embed_dim * 2,
            kernel_size=3,
            upsample_kernel_size=2,
            norm_name='instance',
            res_block=True,
        )
        self.decoder2 = SWINUnetrUpBlock(
            spatial_dims=3,
            in_channels=embed_dim * 2,
            out_channels=embed_dim * 1,
            kernel_size=3,
            upsample_kernel_size=2,
            norm_name='instance',
            res_block=True,
        )

        self.decoder1 = SWINUnetrUpBlock(
            spatial_dims=3,
            in_channels=embed_dim * 1,
            out_channels=embed_dim * 1,
            kernel_size=3,
            upsample_kernel_size=2,
            norm_name='instance',
            res_block=True,
        )

        self.decoder0 = SWINUnetrUpBlock(
            spatial_dims=3,
            in_channels=embed_dim * 1,
            out_channels=embed_dim * 1,
            kernel_size=3,
            upsample_kernel_size=1,
            norm_name='instance',
            res_block=True,
        )

        self.out = UnetOutBlock(spatial_dims=3, in_channels=embed_dim, out_channels=2)  # type: ignore

    def forward(self, x):
        x = self.res_fusionBlock(x)

        out = self.transformer(x)  # (B, n_patch, hidden)
        # print(out[-1].size())

        # stage 4 features
        enc5 = self.res_botneck(out[-1])  # B, 5,5,5,2048
        dec4 = self.decoder5(enc5)  # B, 10,10,10,1024
        enc4 = self.encoder4(out[-2])  # skip features should be twice the di

        # stage 3 features
        dec3 = self.decoder4(dec4, enc4)
        enc3 = self.encoder3(out[-3])  # skip features

        # stage 2 features
        dec2 = self.decoder3(dec3, enc3)
        enc2 = self.encoder2(out[-4])  # skip features

        # stage 1 features
        dec1 = self.decoder2(dec2, enc2)
        enc1 = self.encoder1(out[-5])  # skip features

        dec0 = self.decoder1(dec1, enc1)
        enc0 = self.encoder0(x)

        head = self.decoder0(dec0, enc0)

        logits = self.out(head)

        return logits


class depthwise_separable_conv3d(nn.Module):
    def __init__(self, nin, kernels_per_layer, nout):
        super(depthwise_separable_conv3d, self).__init__()
        self.depthwise = nn.Conv3d(nin, nin * kernels_per_layer, kernel_size=3, padding=1, stride=1, groups=nin)
        self.pointwise = nn.Conv3d(nin * kernels_per_layer, nout, kernel_size=1)

    def forward(self, x):
        out = self.depthwise(x)
        out = self.pointwise(out)
        return out


class SwinUNETR_dense(nn.Module):
    def __init__(self, config):
        super(SwinUNETR_dense, self).__init__()
        if_convskip = config.if_convskip
        self.if_convskip = if_convskip
        if_transskip = config.if_transskip
        self.if_transskip = if_transskip
        embed_dim = config.embed_dim
        self.transformer = SwinTransformer_dense(patch_size=config.patch_size,
                                           in_chans=1,
                                           embed_dim=config.embed_dim,
                                           depths=config.depths,
                                           num_heads=config.num_heads,
                                           window_size=config.window_size,
                                           mlp_ratio=config.mlp_ratio,
                                           qkv_bias=config.qkv_bias,
                                           drop_rate=config.drop_rate,
                                           drop_path_rate=config.drop_path_rate,
                                           ape=config.ape,
                                           spe=config.spe,
                                           patch_norm=config.patch_norm,
                                           use_checkpoint=config.use_checkpoint,
                                           out_indices=config.out_indices,
                                           pat_merg_rf=config.pat_merg_rf,
                                           pos_embed_method=config.pos_embed_method,
                                           concatenated_input=False)

        self.res_fusionBlock = UnetResBlock(
            spatial_dims=3,
            in_channels=config.in_chans,
            out_channels=1,
            kernel_size=3,
            stride=1,
            norm_name='instance',
        )

        # self.res_fusionBlock = depthwise_separable_conv3d(
        #     nin=config.in_chans,
        #     kernels_per_layer=48,
        #     nout=1,
        # )

        self.encoder0 = UnetrBasicBlock(
            spatial_dims=3,
            in_channels=1,
            out_channels=embed_dim,
            kernel_size=3,
            stride=1,
            norm_name='instance',
            res_block=True,
        )

        self.encoder1 = UnetrBasicBlock(
            spatial_dims=3,
            in_channels=embed_dim,
            out_channels=embed_dim,
            kernel_size=3,
            stride=1,
            norm_name='instance',
            res_block=True,
        )

        self.encoder2 = UnetResBlock(
            spatial_dims=3,
            in_channels=embed_dim * 2,
            out_channels=embed_dim * 2,
            kernel_size=3,
            stride=1,
            norm_name='instance',
        )

        self.encoder3 = UnetResBlock(
            spatial_dims=3,
            in_channels=embed_dim * 4,
            out_channels=embed_dim * 4,
            kernel_size=3,
            stride=1,
            norm_name='instance',
        )

        self.encoder4 = UnetResBlock(
            spatial_dims=3,
            in_channels=embed_dim * 8,
            out_channels=embed_dim * 8,
            kernel_size=3,
            stride=1,
            norm_name='instance',
        )

        self.res_botneck = UnetResBlock(
            spatial_dims=3,
            in_channels=embed_dim * 16,
            out_channels=embed_dim * 16,
            kernel_size=3,
            stride=1,
            norm_name='instance',
        )

        self.decoder5 = UnetrPrUpBlock(
            spatial_dims=3,
            in_channels=embed_dim * 16,
            out_channels=embed_dim * 8,
            num_layer=0,
            kernel_size=3,
            stride=1,
            upsample_kernel_size=2,
            norm_name='instance',
            conv_block=True,
            res_block=True,
        )

        self.decoder4 = SWINUnetrUpBlock(
            spatial_dims=3,
            in_channels=embed_dim * 8,
            out_channels=embed_dim * 4,
            kernel_size=3,
            upsample_kernel_size=2,
            norm_name='instance',
            res_block=True,
        )

        self.decoder3 = SWINUnetrUpBlock(
            spatial_dims=3,
            in_channels=embed_dim * 4,
            out_channels=embed_dim * 2,
            kernel_size=3,
            upsample_kernel_size=2,
            norm_name='instance',
            res_block=True,
        )
        self.decoder2 = SWINUnetrUpBlock(
            spatial_dims=3,
            in_channels=embed_dim * 2,
            out_channels=embed_dim * 1,
            kernel_size=3,
            upsample_kernel_size=2,
            norm_name='instance',
            res_block=True,
        )

        self.decoder1 = SWINUnetrUpBlock(
            spatial_dims=3,
            in_channels=embed_dim * 1,
            out_channels=embed_dim * 1,
            kernel_size=3,
            upsample_kernel_size=2,
            norm_name='instance',
            res_block=True,
        )

        self.decoder0 = SWINUnetrUpBlock(
            spatial_dims=3,
            in_channels=embed_dim * 1,
            out_channels=embed_dim * 1,
            kernel_size=3,
            upsample_kernel_size=1,
            norm_name='instance',
            res_block=True,
        )

        self.out = UnetOutBlock(spatial_dims=3, in_channels=embed_dim, out_channels=2)  # type: ignore

    def forward(self, x):
        x = self.res_fusionBlock(x)
        print(x.size())
        out = self.transformer(x)  # (B, n_patch, hidden)
        # print(out[-1].size())

        # stage 4 features
        enc5 = self.res_botneck(out[-1])  # B, 5,5,5,2048
        dec4 = self.decoder5(enc5)  # B, 10,10,10,1024
        enc4 = self.encoder4(out[-2])  # skip features should be twice the di

        # stage 3 features
        dec3 = self.decoder4(dec4, enc4)
        enc3 = self.encoder3(out[-3])  # skip features

        # stage 2 features
        dec2 = self.decoder3(dec3, enc3)
        enc2 = self.encoder2(out[-4])  # skip features

        # stage 1 features
        dec1 = self.decoder2(dec2, enc2)
        enc1 = self.encoder1(out[-5])  # skip features

        dec0 = self.decoder1(dec1, enc1)
        enc0 = self.encoder0(x)

        head = self.decoder0(dec0, enc0)

        logits = self.out(head)

        return logits

class SwinVNetSkip_transfuser(nn.Module):
    def __init__(self, config):
        super(SwinVNetSkip_transfuser, self).__init__()
        if_convskip = config.if_convskip
        self.if_convskip = if_convskip
        if_transskip = config.if_transskip
        self.if_transskip = if_transskip
        embed_dim = config.embed_dim
        self.swinTransfuser = SwinTransformer_wFeatureTalk(patch_size=config.patch_size,
                                           in_chans=int(config.in_chans/2), #
                                           embed_dim=config.embed_dim,
                                           depths=config.depths,
                                           num_heads=config.num_heads,
                                           window_size=config.window_size,
                                           mlp_ratio=config.mlp_ratio,
                                           qkv_bias=config.qkv_bias,
                                           drop_rate=config.drop_rate,
                                           drop_path_rate=config.drop_path_rate,
                                           ape=config.ape,
                                           spe=config.spe,
                                           patch_norm=config.patch_norm,
                                           use_checkpoint=config.use_checkpoint,
                                           out_indices=config.out_indices,
                                           pat_merg_rf=config.pat_merg_rf,
                                           pos_embed_method=config.pos_embed_method)
        self.up0 = DecoderBlock(embed_dim*8, embed_dim*4, skip_channels=embed_dim*4 if if_transskip else 0, use_batchnorm=False)
        self.up1 = DecoderBlock(embed_dim*4, embed_dim*2, skip_channels=embed_dim*2 if if_transskip else 0, use_batchnorm=False)  # 384, 20, 20, 64
        self.up2 = DecoderBlock(embed_dim*2, embed_dim, skip_channels=embed_dim if if_transskip else 0, use_batchnorm=False)  # 384, 40, 40, 64
        self.up3 = DecoderBlock(embed_dim, embed_dim//2, skip_channels=embed_dim//2 if if_convskip else 0, use_batchnorm=False)  # 384, 80, 80, 128
        self.up4 = DecoderBlock(embed_dim//2, config.seg_head_chan, skip_channels=config.seg_head_chan if if_convskip else 0, use_batchnorm=False)  # 384, 160, 160, 256
        self.c1 = Conv3dReLU(2, embed_dim//2, 3, 1, use_batchnorm=False)
        self.c2 = Conv3dReLU(2, config.seg_head_chan, 3, 1, use_batchnorm=False)
        self.seg_head = SegmentationHead(
            in_channels=config.seg_head_chan,
            num_classes=2,
            kernel_size=3,
        )
        self.spatial_trans = SpatialTransformer(config.img_size)
        self.avg_pool = nn.AvgPool3d(3, stride=2, padding=1)

    def forward(self, x):
        #source = x[:, 0:1, :, :]
        x_0 = torch.unsqueeze(x[:, 0, :, :, :],1)
        x_1 = torch.unsqueeze(x[:, 1, :, :, :],1)
        if self.if_convskip:
            x_s0 = x.clone()
            x_s1 = self.avg_pool(x)
            f4 = self.c1(x_s1)
            f5 = self.c2(x_s0)
        else:
            f4 = None
            f5 = None

        out = self.swinTransfuser(x_0,x_1)  # (B, n_patch, hidden)

        if self.if_transskip:
            f1 = out[-2]
            f2 = out[-3]
            f3 = out[-4]
        else:
            f1 = None
            f2 = None
            f3 = None
        x = self.up0(out[-1], f1)
        x = self.up1(x, f2)
        x = self.up2(x, f3)
        x = self.up3(x, f4)
        x = self.up4(x, f5)
        out = self.seg_head(x)
        #out = self.spatial_trans(source, flow)
        return out



class SwinVNetSkip_transfuser_concat(nn.Module):
    def __init__(self, config):
        super(SwinVNetSkip_transfuser_concat, self).__init__()
        if_convskip = config.if_convskip
        self.if_convskip = if_convskip
        if_transskip = config.if_transskip
        self.if_transskip = if_transskip
        embed_dim = config.embed_dim
        self.swinTransfuser = SwinTransformer_wFeatureTalk_concat(patch_size=config.patch_size,
                                           in_chans=int(config.in_chans/2), #
                                           embed_dim=config.embed_dim,
                                           depths=config.depths,
                                           num_heads=config.num_heads,
                                           window_size=config.window_size,
                                           mlp_ratio=config.mlp_ratio,
                                           qkv_bias=config.qkv_bias,
                                           drop_rate=config.drop_rate,
                                           drop_path_rate=config.drop_path_rate,
                                           ape=config.ape,
                                           spe=config.spe,
                                           patch_norm=config.patch_norm,
                                           use_checkpoint=config.use_checkpoint,
                                           out_indices=config.out_indices,
                                           pat_merg_rf=config.pat_merg_rf,
                                           pos_embed_method=config.pos_embed_method)
        self.up0 = DecoderBlock(embed_dim*16, embed_dim*8, skip_channels=embed_dim*8 if if_transskip else 0, use_batchnorm=False)
        self.up1 = DecoderBlock(embed_dim*8, embed_dim*4, skip_channels=embed_dim*4 if if_transskip else 0, use_batchnorm=False)  # 384, 20, 20, 64
        self.up2 = DecoderBlock(embed_dim*4, embed_dim*2, skip_channels=embed_dim*2 if if_transskip else 0, use_batchnorm=False)  # 384, 40, 40, 64
        self.up3 = DecoderBlock(embed_dim*2, embed_dim, skip_channels=embed_dim if if_convskip else 0, use_batchnorm=False)  # 384, 80, 80, 128
        self.up4 = DecoderBlock(embed_dim, config.seg_head_chan, skip_channels=config.seg_head_chan if if_convskip else 0, use_batchnorm=False)  # 384, 160, 160, 256
        self.c1 = Conv3dReLU(2, embed_dim, 3, 1, use_batchnorm=False)
        self.c2 = Conv3dReLU(2, config.seg_head_chan, 3, 1, use_batchnorm=False)
        self.seg_head = SegmentationHead(
            in_channels=config.seg_head_chan,
            num_classes=2,
            kernel_size=3,
        )
        self.spatial_trans = SpatialTransformer(config.img_size)
        self.avg_pool = nn.AvgPool3d(3, stride=2, padding=1)

    def forward(self, x):
        #source = x[:, 0:1, :, :]
        x_0 = torch.unsqueeze(x[:, 0, :, :, :],1)
        x_1 = torch.unsqueeze(x[:, 1, :, :, :],1)
        if self.if_convskip:
            x_s0 = x.clone()
            x_s1 = self.avg_pool(x)
            f4 = self.c1(x_s1)
            f5 = self.c2(x_s0)
        else:
            f4 = None
            f5 = None

        out = self.swinTransfuser(x_0,x_1)  # (B, n_patch, hidden)

        if self.if_transskip:
            f1 = out[-2]
            f2 = out[-3]
            f3 = out[-4]
        else:
            f1 = None
            f2 = None
            f3 = None
        x = self.up0(out[-1], f1)
        x = self.up1(x, f2)
        x = self.up2(x, f3)
        x = self.up3(x, f4)
        x = self.up4(x, f5)
        out = self.seg_head(x)
        #out = self.spatial_trans(source, flow)
        return out



class SwinVNetSkip_transfuser_concat_noCrossModalUpdating(nn.Module):
    def __init__(self, config):
        super(SwinVNetSkip_transfuser_concat_noCrossModalUpdating, self).__init__()
        if_convskip = config.if_convskip
        self.if_convskip = if_convskip
        if_transskip = config.if_transskip
        self.if_transskip = if_transskip
        embed_dim = config.embed_dim
        self.swinTransfuser = SwinTransformer_wFeatureTalk_concat_noCrossModalUpdating(patch_size=config.patch_size,
                                           in_chans=int(config.in_chans/2), #
                                           embed_dim=config.embed_dim,
                                           depths=config.depths,
                                           num_heads=config.num_heads,
                                           window_size=config.window_size,
                                           mlp_ratio=config.mlp_ratio,
                                           qkv_bias=config.qkv_bias,
                                           drop_rate=config.drop_rate,
                                           drop_path_rate=config.drop_path_rate,
                                           ape=config.ape,
                                           spe=config.spe,
                                           patch_norm=config.patch_norm,
                                           use_checkpoint=config.use_checkpoint,
                                           out_indices=config.out_indices,
                                           pat_merg_rf=config.pat_merg_rf,
                                           pos_embed_method=config.pos_embed_method)
        self.up0 = DecoderBlock(embed_dim*16, embed_dim*8, skip_channels=embed_dim*8 if if_transskip else 0, use_batchnorm=False)
        self.up1 = DecoderBlock(embed_dim*8, embed_dim*4, skip_channels=embed_dim*4 if if_transskip else 0, use_batchnorm=False)  # 384, 20, 20, 64
        self.up2 = DecoderBlock(embed_dim*4, embed_dim*2, skip_channels=embed_dim*2 if if_transskip else 0, use_batchnorm=False)  # 384, 40, 40, 64
        self.up3 = DecoderBlock(embed_dim*2, embed_dim, skip_channels=embed_dim if if_convskip else 0, use_batchnorm=False)  # 384, 80, 80, 128
        self.up4 = DecoderBlock(embed_dim, config.seg_head_chan, skip_channels=config.seg_head_chan if if_convskip else 0, use_batchnorm=False)  # 384, 160, 160, 256
        self.c1 = Conv3dReLU(2, embed_dim, 3, 1, use_batchnorm=False)
        self.c2 = Conv3dReLU(2, config.seg_head_chan, 3, 1, use_batchnorm=False)
        self.seg_head = SegmentationHead(
            in_channels=config.seg_head_chan,
            num_classes=2,
            kernel_size=3,
        )
        self.spatial_trans = SpatialTransformer(config.img_size)
        self.avg_pool = nn.AvgPool3d(3, stride=2, padding=1)

    def forward(self, x):
        #source = x[:, 0:1, :, :]
        x_0 = torch.unsqueeze(x[:, 0, :, :, :],1)
        x_1 = torch.unsqueeze(x[:, 1, :, :, :],1)
        if self.if_convskip:
            x_s0 = x.clone()
            x_s1 = self.avg_pool(x)
            f4 = self.c1(x_s1)
            f5 = self.c2(x_s0)
        else:
            f4 = None
            f5 = None

        out = self.swinTransfuser(x_0,x_1)  # (B, n_patch, hidden)

        if self.if_transskip:
            f1 = out[-2]
            f2 = out[-3]
            f3 = out[-4]
        else:
            f1 = None
            f2 = None
            f3 = None
        x = self.up0(out[-1], f1)
        x = self.up1(x, f2)
        x = self.up2(x, f3)
        x = self.up3(x, f4)
        x = self.up4(x, f5)
        out = self.seg_head(x)
        #out = self.spatial_trans(source, flow)
        return out


class SwinVNetSkip_transfuser_concat_noCrossModalUpdating_ConvPoolDownsampling(nn.Module):
    def __init__(self, config):
        super(SwinVNetSkip_transfuser_concat_noCrossModalUpdating_ConvPoolDownsampling, self).__init__()
        if_convskip = config.if_convskip
        self.if_convskip = if_convskip
        if_transskip = config.if_transskip
        self.if_transskip = if_transskip
        embed_dim = config.embed_dim
        self.swinTransfuser = SwinTransformer_wFeatureTalk_concat_noCrossModalUpdating_ConvPoolDownsampling(patch_size=config.patch_size,
                                           in_chans=int(config.in_chans/2), #
                                           embed_dim=config.embed_dim,
                                           depths=config.depths,
                                           num_heads=config.num_heads,
                                           window_size=config.window_size,
                                           mlp_ratio=config.mlp_ratio,
                                           qkv_bias=config.qkv_bias,
                                           drop_rate=config.drop_rate,
                                           drop_path_rate=config.drop_path_rate,
                                           ape=config.ape,
                                           spe=config.spe,
                                           patch_norm=config.patch_norm,
                                           use_checkpoint=config.use_checkpoint,
                                           out_indices=config.out_indices,
                                           pat_merg_rf=config.pat_merg_rf,
                                           pos_embed_method=config.pos_embed_method)
        self.up0 = DecoderBlock(embed_dim*16, embed_dim*8, skip_channels=embed_dim*8 if if_transskip else 0, use_batchnorm=False)
        self.up1 = DecoderBlock(embed_dim*8, embed_dim*4, skip_channels=embed_dim*4 if if_transskip else 0, use_batchnorm=False)  # 384, 20, 20, 64
        self.up2 = DecoderBlock(embed_dim*4, embed_dim*2, skip_channels=embed_dim*2 if if_transskip else 0, use_batchnorm=False)  # 384, 40, 40, 64
        self.up3 = DecoderBlock(embed_dim*2, embed_dim, skip_channels=embed_dim if if_convskip else 0, use_batchnorm=False)  # 384, 80, 80, 128
        self.up4 = DecoderBlock(embed_dim, config.seg_head_chan, skip_channels=config.seg_head_chan if if_convskip else 0, use_batchnorm=False)  # 384, 160, 160, 256
        self.c1 = Conv3dReLU(2, embed_dim, 3, 1, use_batchnorm=False)
        self.c2 = Conv3dReLU(2, config.seg_head_chan, 3, 1, use_batchnorm=False)
        self.seg_head = SegmentationHead_new(
            in_channels=config.seg_head_chan,
            num_classes=2,
            kernel_size=3,
        )
        self.spatial_trans = SpatialTransformer(config.img_size)
        self.avg_pool = nn.AvgPool3d(3, stride=2, padding=1)

    def forward(self, x):
        #source = x[:, 0:1, :, :]
        x_0 = torch.unsqueeze(x[:, 0, :, :, :],1)
        x_1 = torch.unsqueeze(x[:, 1, :, :, :],1)
        if self.if_convskip:
            x_s0 = x.clone()
            x_s1 = self.avg_pool(x)
            f4 = self.c1(x_s1)
            f5 = self.c2(x_s0)
        else:
            f4 = None
            f5 = None

        out = self.swinTransfuser(x_0,x_1)  # (B, n_patch, hidden)

        if self.if_transskip:
            f1 = out[-2]
            f2 = out[-3]
            f3 = out[-4]
        else:
            f1 = None
            f2 = None
            f3 = None
        x = self.up0(out[-1], f1)
        x = self.up1(x, f2)
        x = self.up2(x, f3)
        x = self.up3(x, f4)
        x = self.up4(x, f5)
        out = self.seg_head(x)
        #out = self.spatial_trans(source, flow)
        return out


class SwinUNETR_fusion(nn.Module):
    def __init__(self, config):
        super(SwinUNETR_fusion, self).__init__()
        if_convskip = config.if_convskip
        self.if_convskip = if_convskip
        if_transskip = config.if_transskip
        self.if_transskip = if_transskip
        embed_dim = config.embed_dim
        self.transformer = SwinTransformer_wFeatureTalk_concat_PETUpdatingOnly_5stageOuts(patch_size=config.patch_size,
                                           in_chans=int(config.in_chans/2),
                                           embed_dim=config.embed_dim,
                                           depths=config.depths,
                                           num_heads=config.num_heads,
                                           window_size=config.window_size,
                                           mlp_ratio=config.mlp_ratio,
                                           qkv_bias=config.qkv_bias,
                                           drop_rate=config.drop_rate,
                                           drop_path_rate=config.drop_path_rate,
                                           ape=config.ape,
                                           spe=config.spe,
                                           patch_norm=config.patch_norm,
                                           use_checkpoint=config.use_checkpoint,
                                           out_indices=config.out_indices,
                                           pat_merg_rf=config.pat_merg_rf,
                                           pos_embed_method=config.pos_embed_method,
                                           )

        self.encoder0 = UnetrBasicBlock(
            spatial_dims=3,
            in_channels=config.in_chans,
            out_channels=embed_dim*1,
            kernel_size=3,
            stride=1,
            norm_name='instance',
            res_block=True,
        )

        self.encoder1 = UnetrBasicBlock(
            spatial_dims=3,
            in_channels=embed_dim *1,
            out_channels=embed_dim *1,
            kernel_size=3,
            stride=1,
            norm_name='instance',
            res_block=True,
        )

        self.encoder2 = UnetResBlock(
            spatial_dims=3,
            in_channels=embed_dim * 2,
            out_channels=embed_dim * 2,
            kernel_size=3,
            stride=1,
            norm_name='instance',
        )

        self.encoder3 = UnetResBlock(
            spatial_dims=3,
            in_channels=embed_dim * 4,
            out_channels=embed_dim * 4,
            kernel_size=3,
            stride=1,
            norm_name='instance',
        )

        self.encoder4 = UnetResBlock(
            spatial_dims=3,
            in_channels=embed_dim * 8,
            out_channels=embed_dim * 8,
            kernel_size=3,
            stride=1,
            norm_name='instance',
        )

        self.res_botneck = UnetResBlock(
                spatial_dims=3,
                in_channels=embed_dim*16,
                out_channels=embed_dim*16,
                kernel_size=3,
                stride=1,
                norm_name='instance',
        )

        self.decoder5 = UnetrPrUpBlock(
            spatial_dims=3,
            in_channels=embed_dim*16,
            out_channels=embed_dim*8,
            num_layer=0,
            kernel_size=3,
            stride=1,
            upsample_kernel_size=2,
            norm_name='instance',
            conv_block=True,
            res_block=True,
        )

        self.decoder4 = SWINUnetrUpBlock(
            spatial_dims=3,
            in_channels=embed_dim*8,
            out_channels=embed_dim*4,
            kernel_size=3,
            upsample_kernel_size=2,
            norm_name='instance',
            res_block=True,
        )

        self.decoder3 = SWINUnetrUpBlock(
            spatial_dims=3,
            in_channels=embed_dim * 4,
            out_channels=embed_dim * 2,
            kernel_size=3,
            upsample_kernel_size=2,
            norm_name='instance',
            res_block=True,
        )
        self.decoder2 = SWINUnetrUpBlock(
            spatial_dims=3,
            in_channels=embed_dim * 2,
            out_channels=embed_dim * 1,
            kernel_size=3,
            upsample_kernel_size=2,
            norm_name='instance',
            res_block=True,
        )

        self.decoder1 = SWINUnetrUpBlock(
            spatial_dims=3,
            in_channels=embed_dim * 1,
            out_channels=embed_dim *1 ,
            kernel_size=3,
            upsample_kernel_size=2,
            norm_name='instance',
            res_block=True,
        )

        self.decoder0 = SWINUnetrUpBlock(
            spatial_dims=3,
            in_channels=embed_dim * 1,
            out_channels=embed_dim * 1,
            kernel_size=3,
            upsample_kernel_size=1,
            norm_name='instance',
            res_block=True,
        )

        self.out = UnetOutBlock(spatial_dims=3, in_channels=embed_dim, out_channels=2)  # type: ignore


    def forward(self, x):



        out = self.transformer(x)  # (B, n_patch, hidden)

        #stage 4 features
        enc5 = self.res_botneck(out[-1]) # B, 5,5,5,2048
        dec4 = self.decoder5(enc5) #B, 10,10,10,1024
        enc4 = self.encoder4(out[-2]) #skip features should be twice the di


        #stage 3 features
        dec3 = self.decoder4(dec4,enc4)
        enc3 = self.encoder3(out[-3]) #skip features

        #stage 2 features
        dec2 = self.decoder3(dec3,enc3)
        enc2 = self.encoder2(out[-4]) #skip features

        # stage 1 features
        dec1 = self.decoder2(dec2, enc2)
        enc1 = self.encoder1(out[-5])  # skip features

        dec0 = self.decoder1(dec1, enc1)
        enc0 = self.encoder0(x)

        head = self.decoder0(dec0, enc0)

        logits = self.out(head)

        return logits




class SwinUNETR_dualModalityFusion_OutConcat(nn.Module):
    def __init__(self, config):
        super(SwinUNETR_dualModalityFusion_OutConcat, self).__init__()
        if_convskip = config.if_convskip
        self.if_convskip = if_convskip
        if_transskip = config.if_transskip
        self.if_transskip = if_transskip
        embed_dim = config.embed_dim
        self.transformer = SwinTransformer_wDualModalityFeatureTalk_OutConcat_5stageOuts(patch_size=config.patch_size,
                                           in_chans=int(config.in_chans/2),
                                           embed_dim=config.embed_dim,
                                           depths=config.depths,
                                           num_heads=config.num_heads,
                                           window_size=config.window_size,
                                           mlp_ratio=config.mlp_ratio,
                                           qkv_bias=config.qkv_bias,
                                           drop_rate=config.drop_rate,
                                           drop_path_rate=config.drop_path_rate,
                                           ape=config.ape,
                                           spe=config.spe,
                                           patch_norm=config.patch_norm,
                                           use_checkpoint=config.use_checkpoint,
                                           out_indices=config.out_indices,
                                           pat_merg_rf=config.pat_merg_rf,
                                           pos_embed_method=config.pos_embed_method,
                                           )

        self.encoder0 = UnetrBasicBlock(
            spatial_dims=3,
            in_channels=config.in_chans,
            out_channels=embed_dim*2,
            kernel_size=3,
            stride=1,
            norm_name='instance',
            res_block=True,
        )

        self.encoder1 = UnetrBasicBlock(
            spatial_dims=3,
            in_channels=embed_dim *2,
            out_channels=embed_dim *2,
            kernel_size=3,
            stride=1,
            norm_name='instance',
            res_block=True,
        )

        self.encoder2 = UnetResBlock(
            spatial_dims=3,
            in_channels=embed_dim * 4,
            out_channels=embed_dim * 4,
            kernel_size=3,
            stride=1,
            norm_name='instance',
        )

        self.encoder3 = UnetResBlock(
            spatial_dims=3,
            in_channels=embed_dim * 8,
            out_channels=embed_dim * 8,
            kernel_size=3,
            stride=1,
            norm_name='instance',
        )

        self.encoder4 = UnetResBlock(
            spatial_dims=3,
            in_channels=embed_dim * 16,
            out_channels=embed_dim * 16,
            kernel_size=3,
            stride=1,
            norm_name='instance',
        )

        self.res_botneck = UnetResBlock(
                spatial_dims=3,
                in_channels=embed_dim*32,
                out_channels=embed_dim*32,
                kernel_size=3,
                stride=1,
                norm_name='instance',
        )

        self.decoder5 = UnetrPrUpBlock(
            spatial_dims=3,
            in_channels=embed_dim*32,
            out_channels=embed_dim*16,
            num_layer=0,
            kernel_size=3,
            stride=1,
            upsample_kernel_size=2,
            norm_name='instance',
            conv_block=True,
            res_block=True,
        )

        self.decoder4 = SWINUnetrUpBlock(
            spatial_dims=3,
            in_channels=embed_dim*16,
            out_channels=embed_dim*8,
            kernel_size=3,
            upsample_kernel_size=2,
            norm_name='instance',
            res_block=True,
        )

        self.decoder3 = SWINUnetrUpBlock(
            spatial_dims=3,
            in_channels=embed_dim * 8,
            out_channels=embed_dim * 4,
            kernel_size=3,
            upsample_kernel_size=2,
            norm_name='instance',
            res_block=True,
        )
        self.decoder2 = SWINUnetrUpBlock(
            spatial_dims=3,
            in_channels=embed_dim * 4,
            out_channels=embed_dim * 2,
            kernel_size=3,
            upsample_kernel_size=2,
            norm_name='instance',
            res_block=True,
        )

        self.decoder1 = SWINUnetrUpBlock(
            spatial_dims=3,
            in_channels=embed_dim * 2,
            out_channels=embed_dim *2 ,
            kernel_size=3,
            upsample_kernel_size=2,
            norm_name='instance',
            res_block=True,
        )

        self.decoder0 = SWINUnetrUpBlock(
            spatial_dims=3,
            in_channels=embed_dim * 2,
            out_channels=embed_dim * 1,
            kernel_size=3,
            upsample_kernel_size=1,
            norm_name='instance',
            res_block=True,
        )

        self.out = UnetOutBlock(spatial_dims=3, in_channels=embed_dim, out_channels=2)  # type: ignore


    def forward(self, x):



        out = self.transformer(x)  # (B, n_patch, hidden)

        #stage 4 features
        enc5 = self.res_botneck(out[-1]) # B, 5,5,5,2048
        dec4 = self.decoder5(enc5) #B, 10,10,10,1024
        enc4 = self.encoder4(out[-2]) #skip features should be twice the di


        #stage 3 features
        dec3 = self.decoder4(dec4,enc4)
        enc3 = self.encoder3(out[-3]) #skip features

        #stage 2 features
        dec2 = self.decoder3(dec3,enc3)
        enc2 = self.encoder2(out[-4]) #skip features

        # stage 1 features
        dec1 = self.decoder2(dec2, enc2)
        enc1 = self.encoder1(out[-5])  # skip features

        dec0 = self.decoder1(dec1, enc1)
        enc0 = self.encoder0(x)

        head = self.decoder0(dec0, enc0)

        logits = self.out(head)

        return logits


class SwinUNETR_CrossModalityFusion_inputFusion_OutSum(nn.Module):
    def __init__(self, config):
        super(SwinUNETR_CrossModalityFusion_inputFusion_OutSum, self).__init__()
        if_convskip = config.if_convskip
        self.if_convskip = if_convskip
        if_transskip = config.if_transskip
        self.if_transskip = if_transskip
        embed_dim = config.embed_dim
        self.transformer = SwinTransformer_wCrossModalityFeatureTalk_wInputFusion_OutSum_5stageOuts(patch_size=config.patch_size,
                                           in_chans=int(config.in_chans/2),
                                           embed_dim=config.embed_dim,
                                           depths=config.depths,
                                           num_heads=config.num_heads,
                                           window_size=config.window_size,
                                           mlp_ratio=config.mlp_ratio,
                                           qkv_bias=config.qkv_bias,
                                           drop_rate=config.drop_rate,
                                           drop_path_rate=config.drop_path_rate,
                                           ape=config.ape,
                                           spe=config.spe,
                                           patch_norm=config.patch_norm,
                                           use_checkpoint=config.use_checkpoint,
                                           out_indices=config.out_indices,
                                           pat_merg_rf=config.pat_merg_rf,
                                           pos_embed_method=config.pos_embed_method,
                                           )

        self.encoder0 = UnetrBasicBlock(
            spatial_dims=3,
            in_channels=config.in_chans,
            out_channels=embed_dim*1,
            kernel_size=3,
            stride=1,
            norm_name='instance',
            res_block=True,
        )

        self.encoder1 = UnetrBasicBlock(
            spatial_dims=3,
            in_channels=embed_dim *1,
            out_channels=embed_dim *1,
            kernel_size=3,
            stride=1,
            norm_name='instance',
            res_block=True,
        )

        self.encoder2 = UnetResBlock(
            spatial_dims=3,
            in_channels=embed_dim * 2,
            out_channels=embed_dim * 2,
            kernel_size=3,
            stride=1,
            norm_name='instance',
        )

        self.encoder3 = UnetResBlock(
            spatial_dims=3,
            in_channels=embed_dim * 4,
            out_channels=embed_dim * 4,
            kernel_size=3,
            stride=1,
            norm_name='instance',
        )

        self.encoder4 = UnetResBlock(
            spatial_dims=3,
            in_channels=embed_dim * 8,
            out_channels=embed_dim * 8,
            kernel_size=3,
            stride=1,
            norm_name='instance',
        )

        self.res_botneck = UnetResBlock(
                spatial_dims=3,
                in_channels=embed_dim*16,
                out_channels=embed_dim*16,
                kernel_size=3,
                stride=1,
                norm_name='instance',
        )

        self.decoder5 = UnetrPrUpBlock(
            spatial_dims=3,
            in_channels=embed_dim*16,
            out_channels=embed_dim*8,
            num_layer=0,
            kernel_size=3,
            stride=1,
            upsample_kernel_size=2,
            norm_name='instance',
            conv_block=True,
            res_block=True,
        )

        self.decoder4 = SWINUnetrUpBlock(
            spatial_dims=3,
            in_channels=embed_dim*8,
            out_channels=embed_dim*4,
            kernel_size=3,
            upsample_kernel_size=2,
            norm_name='instance',
            res_block=True,
        )

        self.decoder3 = SWINUnetrUpBlock(
            spatial_dims=3,
            in_channels=embed_dim * 4,
            out_channels=embed_dim * 2,
            kernel_size=3,
            upsample_kernel_size=2,
            norm_name='instance',
            res_block=True,
        )
        self.decoder2 = SWINUnetrUpBlock(
            spatial_dims=3,
            in_channels=embed_dim * 2,
            out_channels=embed_dim * 1,
            kernel_size=3,
            upsample_kernel_size=2,
            norm_name='instance',
            res_block=True,
        )

        self.decoder1 = SWINUnetrUpBlock(
            spatial_dims=3,
            in_channels=embed_dim * 1,
            out_channels=embed_dim *1 ,
            kernel_size=3,
            upsample_kernel_size=2,
            norm_name='instance',
            res_block=True,
        )

        self.decoder0 = SWINUnetrUpBlock(
            spatial_dims=3,
            in_channels=embed_dim * 1,
            out_channels=embed_dim * 1,
            kernel_size=3,
            upsample_kernel_size=1,
            norm_name='instance',
            res_block=True,
        )

        self.out = UnetOutBlock(spatial_dims=3, in_channels=embed_dim, out_channels=2)  # type: ignore
        self.res_fusionBlock = depthwise_separable_conv3d(
            nin=config.in_chans,
            kernels_per_layer=48,
            nout=1,
        )

    def forward(self, x):


        out = self.transformer(x)  # (B, n_patch, hidden)

        #stage 4 features
        enc5 = self.res_botneck(out[-1]) # B, 5,5,5,2048
        dec4 = self.decoder5(enc5) #B, 10,10,10,1024
        enc4 = self.encoder4(out[-2]) #skip features should be twice the di

        #stage 3 features
        dec3 = self.decoder4(dec4,enc4)
        enc3 = self.encoder3(out[-3]) #skip features

        #stage 2 features
        dec2 = self.decoder3(dec3,enc3)
        enc2 = self.encoder2(out[-4]) #skip features

        # stage 1 features
        dec1 = self.decoder2(dec2, enc2)
        enc1 = self.encoder1(out[-5])  # skip features

        dec0 = self.decoder1(dec1, enc1)
        enc0 = self.encoder0(x)

        head = self.decoder0(dec0, enc0)

        logits = self.out(head)

        return logits


##########################################################################
def conv(in_channels, out_channels, kernel_size, bias=False, stride = 1):
    return nn.Conv3d(
        in_channels, out_channels, kernel_size,
        padding=(kernel_size//2), bias=bias, stride = stride)

## Channel Attention Block (CAB)

class CAB(nn.Module):
    def __init__(self, n_feat,  kernel_size, reduction=4, bias=False, act = nn.PReLU()):
        super(CAB, self).__init__()
        modules_body = []
        modules_body.append(conv(n_feat, n_feat, kernel_size, bias=bias))
        modules_body.append(act)
        modules_body.append(conv(n_feat, n_feat, kernel_size, bias=bias))

        self.CA = CALayer(n_feat, reduction, bias=bias) #n_feat = channel, noiseLevel_dim
        self.body = nn.Sequential(*modules_body)

    def forward(self, x):
        res = self.body(x) #x.shape=[4,80,32,32,32] and res.shape=[4,80,32,32,32]
        res = self.CA(res)
        res += x
        return res
## Channel Attention Layer
class CALayer(nn.Module):
    def __init__(self, channel, reduction=16, bias=False):
        super(CALayer, self).__init__()
        # global average pooling: feature --> point
        self.avg_pool = nn.AdaptiveAvgPool3d(1)
        # feature channel downscale and upscale --> channel weight
        self.conv_du = nn.Sequential(
                nn.Conv3d(channel, channel // reduction, 1, padding=0, bias=bias),
                nn.ReLU(inplace=True),
                nn.Conv3d(channel // reduction, channel, 1, padding=0, bias=bias),
                nn.Sigmoid()
        )

    def forward(self, x):
        y = self.avg_pool(x)
        y = self.conv_du(y)
        return x * y

class SwinUNETR_CrossModalityFusion_OutSum(nn.Module):
    def __init__(self, config):
        super(SwinUNETR_CrossModalityFusion_OutSum, self).__init__()
        if_convskip = config.if_convskip
        self.if_convskip = if_convskip
        if_transskip = config.if_transskip
        self.if_transskip = if_transskip
        embed_dim = config.embed_dim
        self.transformer = SwinTransformer_wCrossModalityFeatureTalk_OutSum_5stageOuts(patch_size=config.patch_size,
                                           in_chans=int(config.in_chans/2),
                                           embed_dim=config.embed_dim,
                                           depths=config.depths,
                                           num_heads=config.num_heads,
                                           window_size=config.window_size,
                                           mlp_ratio=config.mlp_ratio,
                                           qkv_bias=config.qkv_bias,
                                           drop_rate=config.drop_rate,
                                           drop_path_rate=config.drop_path_rate,
                                           ape=config.ape,
                                           spe=config.spe,
                                           patch_norm=config.patch_norm,
                                           use_checkpoint=config.use_checkpoint,
                                           out_indices=config.out_indices,
                                           pat_merg_rf=config.pat_merg_rf,
                                           pos_embed_method=config.pos_embed_method,
                                           )

        self.encoder0 = UnetrBasicBlock(
            spatial_dims=3,
            in_channels=config.in_chans,
            out_channels=embed_dim*1,
            kernel_size=3,
            stride=1,
            norm_name='instance',
            res_block=True,
        )

        self.encoder1 = UnetrBasicBlock(
            spatial_dims=3,
            in_channels=embed_dim *1,
            out_channels=embed_dim *1,
            kernel_size=3,
            stride=1,
            norm_name='instance',
            res_block=True,
        )

        self.encoder2 = UnetResBlock(
            spatial_dims=3,
            in_channels=embed_dim * 2,
            out_channels=embed_dim * 2,
            kernel_size=3,
            stride=1,
            norm_name='instance',
        )

        self.encoder3 = UnetResBlock(
            spatial_dims=3,
            in_channels=embed_dim * 4,
            out_channels=embed_dim * 4,
            kernel_size=3,
            stride=1,
            norm_name='instance',
        )


        self.encoder4 = UnetResBlock(
            spatial_dims=3,
            in_channels=embed_dim * 8,
            out_channels=embed_dim * 8,
            kernel_size=3,
            stride=1,
            norm_name='instance',
        )

        self.res_botneck = UnetResBlock(
                spatial_dims=3,
                in_channels=embed_dim*16,
                out_channels=embed_dim*16,
                kernel_size=3,
                stride=1,
                norm_name='instance',
        )

        self.decoder5 = UnetrPrUpBlock(
            spatial_dims=3,
            in_channels=embed_dim*16,
            out_channels=embed_dim*8,
            num_layer=0,
            kernel_size=3,
            stride=1,
            upsample_kernel_size=2,
            norm_name='instance',
            conv_block=True,
            res_block=True,
        )

        self.decoder4 = SWINUnetrUpBlock(
            spatial_dims=3,
            in_channels=embed_dim*8,
            out_channels=embed_dim*4,
            kernel_size=3,
            upsample_kernel_size=2,
            norm_name='instance',
            res_block=True,
        )

        self.decoder3 = SWINUnetrUpBlock(
            spatial_dims=3,
            in_channels=embed_dim * 4,
            out_channels=embed_dim * 2,
            kernel_size=3,
            upsample_kernel_size=2,
            norm_name='instance',
            res_block=True,
        )
        self.decoder2 = SWINUnetrUpBlock(
            spatial_dims=3,
            in_channels=embed_dim * 2,
            out_channels=embed_dim * 1,
            kernel_size=3,
            upsample_kernel_size=2,
            norm_name='instance',
            res_block=True,
        )

        self.decoder1 = SWINUnetrUpBlock(
            spatial_dims=3,
            in_channels=embed_dim * 1,
            out_channels=embed_dim *1 ,
            kernel_size=3,
            upsample_kernel_size=2,
            norm_name='instance',
            res_block=True,
        )

        self.decoder0 = SWINUnetrUpBlock(
            spatial_dims=3,
            in_channels=embed_dim * 1,
            out_channels=embed_dim * 1,
            kernel_size=3,
            upsample_kernel_size=1,
            norm_name='instance',
            res_block=True,
        )

        self.out = UnetOutBlock(spatial_dims=3, in_channels=embed_dim, out_channels=2)  # type: ignore

    def forward(self, x):

        out = self.transformer(x)  # (B, n_patch, hidden)

        #stage 4 features
        enc5 = self.res_botneck(out[-1]) # B, 5,5,5,2048
        dec4 = self.decoder5(enc5) #B, 10,10,10,1024
        enc4 = self.encoder4(out[-2]) #skip features should be twice the di

        #stage 3 features
        dec3 = self.decoder4(dec4,enc4)
        enc3 = self.encoder3(out[-3]) #skip features

        #stage 2 features
        dec2 = self.decoder3(dec3,enc3)
        enc2 = self.encoder2(out[-4]) #skip features

        # stage 1 features
        dec1 = self.decoder2(dec2, enc2)
        enc1 = self.encoder1(out[-5])  # skip features

        dec0 = self.decoder1(dec1, enc1)
        enc0 = self.encoder0(x)

        head = self.decoder0(dec0, enc0)

        logits = self.out(head)

        return logits


class SwinUNETR_CrossModalityFusion_OutSum_6stageOuts(nn.Module):
    def __init__(self, config):
        super(SwinUNETR_CrossModalityFusion_OutSum_6stageOuts, self).__init__()
        if_convskip = config.if_convskip
        self.if_convskip = if_convskip
        if_transskip = config.if_transskip
        self.if_transskip = if_transskip
        embed_dim = config.embed_dim
        self.transformer = SwinTransformer_wCrossModalityFeatureTalk_OutSum_5stageOuts(patch_size=config.patch_size,
                                           in_chans=int(config.in_chans/2),
                                           embed_dim=config.embed_dim,
                                           depths=config.depths,
                                           num_heads=config.num_heads,
                                           window_size=config.window_size,
                                           mlp_ratio=config.mlp_ratio,
                                           qkv_bias=config.qkv_bias,
                                           drop_rate=config.drop_rate,
                                           drop_path_rate=config.drop_path_rate,
                                           ape=config.ape,
                                           spe=config.spe,
                                           patch_norm=config.patch_norm,
                                           use_checkpoint=config.use_checkpoint,
                                           out_indices=config.out_indices,
                                           pat_merg_rf=config.pat_merg_rf,
                                           pos_embed_method=config.pos_embed_method,
                                           )

        self.encoder0 = UnetrBasicBlock(
            spatial_dims=3,
            in_channels=config.in_chans,
            out_channels=embed_dim*1,
            kernel_size=3,
            stride=1,
            norm_name='instance',
            res_block=True,
        )

        # self.encoder0 = depthwise_separable_conv3d(
        #     nin=2,
        #     kernels_per_layer=96,
        #     nout=embed_dim,
        # )

        self.encoder1 = UnetrBasicBlock(
            spatial_dims=3,
            in_channels=embed_dim *1,
            out_channels=embed_dim *1,
            kernel_size=3,
            stride=1,
            norm_name='instance',
            res_block=True,
        )

        self.encoder2 = UnetResBlock(
            spatial_dims=3,
            in_channels=embed_dim * 2,
            out_channels=embed_dim * 2,
            kernel_size=3,
            stride=1,
            norm_name='instance',
        )

        self.encoder3 = UnetResBlock(
            spatial_dims=3,
            in_channels=embed_dim * 4,
            out_channels=embed_dim * 4,
            kernel_size=3,
            stride=1,
            norm_name='instance',
        )


        self.encoder4 = UnetResBlock(
            spatial_dims=3,
            in_channels=embed_dim * 8,
            out_channels=embed_dim * 8,
            kernel_size=3,
            stride=1,
            norm_name='instance',
        )

        self.res_botneck = UnetResBlock(
                spatial_dims=3,
                in_channels=embed_dim*16,
                out_channels=embed_dim*16,
                kernel_size=3,
                stride=1,
                norm_name='instance',
        )

        self.decoder5 = UnetrPrUpBlock(
            spatial_dims=3,
            in_channels=embed_dim*16,
            out_channels=embed_dim*8,
            num_layer=0,
            kernel_size=3,
            stride=1,
            upsample_kernel_size=2,
            norm_name='instance',
            conv_block=True,
            res_block=True,
        )

        self.decoder4 = SWINUnetrUpBlock(
            spatial_dims=3,
            in_channels=embed_dim*8,
            out_channels=embed_dim*4,
            kernel_size=3,
            upsample_kernel_size=2,
            norm_name='instance',
            res_block=True,
        )

        self.decoder3 = SWINUnetrUpBlock(
            spatial_dims=3,
            in_channels=embed_dim * 4,
            out_channels=embed_dim * 2,
            kernel_size=3,
            upsample_kernel_size=2,
            norm_name='instance',
            res_block=True,
        )
        self.decoder2 = SWINUnetrUpBlock(
            spatial_dims=3,
            in_channels=embed_dim * 2,
            out_channels=embed_dim * 1,
            kernel_size=3,
            upsample_kernel_size=2,
            norm_name='instance',
            res_block=True,
        )

        self.decoder1 = SWINUnetrUpBlock(
            spatial_dims=3,
            in_channels=embed_dim * 1,
            out_channels=embed_dim *1 ,
            kernel_size=3,
            upsample_kernel_size=2,
            norm_name='instance',
            res_block=True,
        )

        self.decoder0 = SWINUnetrBlock(
            spatial_dims=3,
            in_channels=embed_dim * 1,
            out_channels=embed_dim * 1,
            kernel_size=3,
            upsample_kernel_size=1,
            norm_name='instance',
            res_block=True,
        )

        self.out = UnetOutBlock(spatial_dims=3, in_channels=embed_dim, out_channels=2)  # type: ignore

    def forward(self, x):

        out = self.transformer(x)  # (B, n_patch, hidden)
        #print(1)
        #stage 4 features
        enc5 = self.res_botneck(out[-1]) # B, 5,5,5,2048
        dec4 = self.decoder5(enc5) #B, 10,10,10,1024
        enc4 = self.encoder4(out[-2]) #skip features should be twice the di

        #stage 3 features
        dec3 = self.decoder4(dec4,enc4)
        enc3 = self.encoder3(out[-3]) #skip features

        #stage 2 features
        dec2 = self.decoder3(dec3,enc3)
        enc2 = self.encoder2(out[-4]) #skip features

        # stage 1 features
        dec1 = self.decoder2(dec2, enc2)
        enc1 = self.encoder1(out[-5])  # skip features

        dec0 = self.decoder1(dec1, enc1)
        enc0 = self.encoder0(x)

        head = self.decoder0(dec0, enc0)

        logits = self.out(head)

        return logits


class SwinUNETR_CrossModalityFusion_OutSum_wChAttn(nn.Module):
    def __init__(self, config):
        super(SwinUNETR_CrossModalityFusion_OutSum_wChAttn, self).__init__()
        if_convskip = config.if_convskip
        self.if_convskip = if_convskip
        if_transskip = config.if_transskip
        self.if_transskip = if_transskip
        embed_dim = config.embed_dim
        self.transformer = SwinTransformer_wCrossModalityFeatureTalk_OutSum_5stageOuts(patch_size=config.patch_size,
                                                                                       in_chans=int(
                                                                                           config.in_chans / 2),
                                                                                       embed_dim=config.embed_dim,
                                                                                       depths=config.depths,
                                                                                       num_heads=config.num_heads,
                                                                                       window_size=config.window_size,
                                                                                       mlp_ratio=config.mlp_ratio,
                                                                                       qkv_bias=config.qkv_bias,
                                                                                       drop_rate=config.drop_rate,
                                                                                       drop_path_rate=config.drop_path_rate,
                                                                                       ape=config.ape,
                                                                                       spe=config.spe,
                                                                                       patch_norm=config.patch_norm,
                                                                                       use_checkpoint=config.use_checkpoint,
                                                                                       out_indices=config.out_indices,
                                                                                       pat_merg_rf=config.pat_merg_rf,
                                                                                       pos_embed_method=config.pos_embed_method,
                                                                                       )

        self.encoder0 = UnetrBasicBlock(
            spatial_dims=3,
            in_channels=config.in_chans,
            out_channels=embed_dim * 1,
            kernel_size=3,
            stride=1,
            norm_name='instance',
            res_block=True,
        )

        self.CAB1 = CAB(
            n_feat=embed_dim * 1,
            kernel_size=3,
        )

        self.encoder1 = UnetrBasicBlock(
            spatial_dims=3,
            in_channels=embed_dim * 1,
            out_channels=embed_dim * 1,
            kernel_size=3,
            stride=1,
            norm_name='instance',
            res_block=True,
        )

        self.CAB2 = CAB(
            n_feat=embed_dim * 2,
            kernel_size=3,
        )

        self.encoder2 = UnetResBlock(
            spatial_dims=3,
            in_channels=embed_dim * 2,
            out_channels=embed_dim * 2,
            kernel_size=3,
            stride=1,
            norm_name='instance',
        )

        self.CAB3 = CAB(
            n_feat=embed_dim * 4,
            kernel_size=3,
        )

        self.encoder3 = UnetResBlock(
            spatial_dims=3,
            in_channels=embed_dim * 4,
            out_channels=embed_dim * 4,
            kernel_size=3,
            stride=1,
            norm_name='instance',
        )

        self.CAB4 = CAB(
            n_feat=embed_dim * 8,
            kernel_size=3,
        )

        self.encoder4 = UnetResBlock(
            spatial_dims=3,
            in_channels=embed_dim * 8,
            out_channels=embed_dim * 8,
            kernel_size=3,
            stride=1,
            norm_name='instance',
        )

        self.CAB5 = CAB(
            n_feat=embed_dim * 16,
            kernel_size=3,
        )

        self.res_botneck = UnetResBlock(
            spatial_dims=3,
            in_channels=embed_dim * 16,
            out_channels=embed_dim * 16,
            kernel_size=3,
            stride=1,
            norm_name='instance',
        )

        self.decoder5 = UnetrPrUpBlock(
            spatial_dims=3,
            in_channels=embed_dim * 16,
            out_channels=embed_dim * 8,
            num_layer=0,
            kernel_size=3,
            stride=1,
            upsample_kernel_size=2,
            norm_name='instance',
            conv_block=True,
            res_block=True,
        )

        self.decoder4 = SWINUnetrUpBlock(
            spatial_dims=3,
            in_channels=embed_dim * 8,
            out_channels=embed_dim * 4,
            kernel_size=3,
            upsample_kernel_size=2,
            norm_name='instance',
            res_block=True,
        )

        self.decoder3 = SWINUnetrUpBlock(
            spatial_dims=3,
            in_channels=embed_dim * 4,
            out_channels=embed_dim * 2,
            kernel_size=3,
            upsample_kernel_size=2,
            norm_name='instance',
            res_block=True,
        )
        self.decoder2 = SWINUnetrUpBlock(
            spatial_dims=3,
            in_channels=embed_dim * 2,
            out_channels=embed_dim * 1,
            kernel_size=3,
            upsample_kernel_size=2,
            norm_name='instance',
            res_block=True,
        )

        self.decoder1 = SWINUnetrUpBlock(
            spatial_dims=3,
            in_channels=embed_dim * 1,
            out_channels=embed_dim * 1,
            kernel_size=3,
            upsample_kernel_size=2,
            norm_name='instance',
            res_block=True,
        )

        self.decoder0 = SWINUnetrUpBlock(
            spatial_dims=3,
            in_channels=embed_dim * 1,
            out_channels=embed_dim * 1,
            kernel_size=3,
            upsample_kernel_size=1,
            norm_name='instance',
            res_block=True,
        )

        self.out = UnetOutBlock(spatial_dims=3, in_channels=embed_dim, out_channels=2)  # type: ignore

    def forward(self, x):
        out = self.transformer(x)  # (B, n_patch, hidden)

        # stage 4 features
        cab5 = self.CAB5(out[-1])
        enc5 = self.res_botneck(cab5)  # B, 5,5,5,2048

        dec4 = self.decoder5(enc5)  # B, 10,10,10,1024
        cab4 = self.CAB4(out[-2])
        enc4 = self.encoder4(cab4)  # skip features should be twice the di

        # stage 3 features
        dec3 = self.decoder4(dec4, enc4)
        cab3 = self.CAB3(out[-3])
        enc3 = self.encoder3(cab3)  # skip features

        # stage 2 features
        dec2 = self.decoder3(dec3, enc3)
        cab2 = self.CAB2(out[-4])
        enc2 = self.encoder2(cab2)  # skip features

        # stage 1 features
        dec1 = self.decoder2(dec2, enc2)
        cab1 = self.CAB1(out[-5])
        enc1 = self.encoder1(cab1)  # skip features


        dec0 = self.decoder1(dec1, enc1)
        enc0 = self.encoder0(x)

        head = self.decoder0(dec0, enc0)

        logits = self.out(head)

        return logits

class SwinUNETR_dualModalityFusion_OutSum(nn.Module):
    def __init__(self, config):
        super(SwinUNETR_dualModalityFusion_OutSum, self).__init__()
        if_convskip = config.if_convskip
        self.if_convskip = if_convskip
        if_transskip = config.if_transskip
        self.if_transskip = if_transskip
        embed_dim = config.embed_dim
        self.transformer = SwinTransformer_wDualModalityFeatureTalk_OutSum_5stageOuts(patch_size=config.patch_size,
                                           in_chans=int(config.in_chans/2),
                                           embed_dim=config.embed_dim,
                                           depths=config.depths,
                                           num_heads=config.num_heads,
                                           window_size=config.window_size,
                                           mlp_ratio=config.mlp_ratio,
                                           qkv_bias=config.qkv_bias,
                                           drop_rate=config.drop_rate,
                                           drop_path_rate=config.drop_path_rate,
                                           ape=config.ape,
                                           spe=config.spe,
                                           patch_norm=config.patch_norm,
                                           use_checkpoint=config.use_checkpoint,
                                           out_indices=config.out_indices,
                                           pat_merg_rf=config.pat_merg_rf,
                                           pos_embed_method=config.pos_embed_method,
                                           )

        self.encoder0 = UnetrBasicBlock(
            spatial_dims=3,
            in_channels=config.in_chans,
            out_channels=embed_dim*1,
            kernel_size=3,
            stride=1,
            norm_name='instance',
            res_block=True,
        )

        self.encoder1 = UnetrBasicBlock(
            spatial_dims=3,
            in_channels=embed_dim *1,
            out_channels=embed_dim *1,
            kernel_size=3,
            stride=1,
            norm_name='instance',
            res_block=True,
        )

        self.encoder2 = UnetResBlock(
            spatial_dims=3,
            in_channels=embed_dim * 2,
            out_channels=embed_dim * 2,
            kernel_size=3,
            stride=1,
            norm_name='instance',
        )

        self.encoder3 = UnetResBlock(
            spatial_dims=3,
            in_channels=embed_dim * 4,
            out_channels=embed_dim * 4,
            kernel_size=3,
            stride=1,
            norm_name='instance',
        )

        self.encoder4 = UnetResBlock(
            spatial_dims=3,
            in_channels=embed_dim * 8,
            out_channels=embed_dim * 8,
            kernel_size=3,
            stride=1,
            norm_name='instance',
        )

        self.res_botneck = UnetResBlock(
                spatial_dims=3,
                in_channels=embed_dim*16,
                out_channels=embed_dim*16,
                kernel_size=3,
                stride=1,
                norm_name='instance',
        )

        self.decoder5 = UnetrPrUpBlock(
            spatial_dims=3,
            in_channels=embed_dim*16,
            out_channels=embed_dim*8,
            num_layer=0,
            kernel_size=3,
            stride=1,
            upsample_kernel_size=2,
            norm_name='instance',
            conv_block=True,
            res_block=True,
        )

        self.decoder4 = SWINUnetrUpBlock(
            spatial_dims=3,
            in_channels=embed_dim*8,
            out_channels=embed_dim*4,
            kernel_size=3,
            upsample_kernel_size=2,
            norm_name='instance',
            res_block=True,
        )

        self.decoder3 = SWINUnetrUpBlock(
            spatial_dims=3,
            in_channels=embed_dim * 4,
            out_channels=embed_dim * 2,
            kernel_size=3,
            upsample_kernel_size=2,
            norm_name='instance',
            res_block=True,
        )
        self.decoder2 = SWINUnetrUpBlock(
            spatial_dims=3,
            in_channels=embed_dim * 2,
            out_channels=embed_dim * 1,
            kernel_size=3,
            upsample_kernel_size=2,
            norm_name='instance',
            res_block=True,
        )

        self.decoder1 = SWINUnetrUpBlock(
            spatial_dims=3,
            in_channels=embed_dim * 1,
            out_channels=embed_dim *1 ,
            kernel_size=3,
            upsample_kernel_size=2,
            norm_name='instance',
            res_block=True,
        )

        self.decoder0 = SWINUnetrUpBlock(
            spatial_dims=3,
            in_channels=embed_dim * 1,
            out_channels=embed_dim * 1,
            kernel_size=3,
            upsample_kernel_size=1,
            norm_name='instance',
            res_block=True,
        )

        self.out = UnetOutBlock(spatial_dims=3, in_channels=embed_dim, out_channels=2)  # type: ignore


    def forward(self, x):

        out = self.transformer(x)  # (B, n_patch, hidden)

        #stage 4 features
        enc5 = self.res_botneck(out[-1]) # B, 5,5,5,2048
        dec4 = self.decoder5(enc5) #B, 10,10,10,1024
        enc4 = self.encoder4(out[-2]) #skip features should be twice the di


        #stage 3 features
        dec3 = self.decoder4(dec4,enc4)
        enc3 = self.encoder3(out[-3]) #skip features

        #stage 2 features
        dec2 = self.decoder3(dec3,enc3)
        enc2 = self.encoder2(out[-4]) #skip features

        # stage 1 features
        dec1 = self.decoder2(dec2, enc2)
        enc1 = self.encoder1(out[-5])  # skip features

        dec0 = self.decoder1(dec1, enc1)
        enc0 = self.encoder0(x)

        head = self.decoder0(dec0, enc0)

        logits = self.out(head)

        return logits


class SwinUNETR_RandomSpatialFusion(nn.Module):
    def __init__(self, config):
        super(SwinUNETR_RandomSpatialFusion, self).__init__()
        if_convskip = config.if_convskip
        self.if_convskip = if_convskip
        if_transskip = config.if_transskip
        self.if_transskip = if_transskip
        embed_dim = config.embed_dim
        self.transformer = SwinTransformer_wRandomSpatialFeatureTalk_wCrossModalUpdating_5stageOuts(patch_size=config.patch_size,
                                           in_chans=int(config.in_chans/2),
                                           embed_dim=config.embed_dim,
                                           depths=config.depths,
                                           num_heads=config.num_heads,
                                           window_size=config.window_size,
                                           mlp_ratio=config.mlp_ratio,
                                           qkv_bias=config.qkv_bias,
                                           drop_rate=config.drop_rate,
                                           drop_path_rate=config.drop_path_rate,
                                           ape=config.ape,
                                           spe=config.spe,
                                           patch_norm=config.patch_norm,
                                           use_checkpoint=config.use_checkpoint,
                                           out_indices=config.out_indices,
                                           pat_merg_rf=config.pat_merg_rf,
                                           pos_embed_method=config.pos_embed_method,
                                           )

        self.encoder0 = UnetrBasicBlock(
            spatial_dims=3,
            in_channels=config.in_chans,
            out_channels=embed_dim*2,
            kernel_size=3,
            stride=1,
            norm_name='instance',
            res_block=True,
        )

        self.encoder1 = UnetrBasicBlock(
            spatial_dims=3,
            in_channels=embed_dim *2,
            out_channels=embed_dim *2,
            kernel_size=3,
            stride=1,
            norm_name='instance',
            res_block=True,
        )

        self.encoder2 = UnetResBlock(
            spatial_dims=3,
            in_channels=embed_dim * 4,
            out_channels=embed_dim * 4,
            kernel_size=3,
            stride=1,
            norm_name='instance',
        )

        self.encoder3 = UnetResBlock(
            spatial_dims=3,
            in_channels=embed_dim * 8,
            out_channels=embed_dim * 8,
            kernel_size=3,
            stride=1,
            norm_name='instance',
        )

        self.encoder4 = UnetResBlock(
            spatial_dims=3,
            in_channels=embed_dim * 16,
            out_channels=embed_dim * 16,
            kernel_size=3,
            stride=1,
            norm_name='instance',
        )

        self.res_botneck = UnetResBlock(
                spatial_dims=3,
                in_channels=embed_dim*32,
                out_channels=embed_dim*32,
                kernel_size=3,
                stride=1,
                norm_name='instance',
        )

        self.decoder5 = UnetrPrUpBlock(
            spatial_dims=3,
            in_channels=embed_dim*32,
            out_channels=embed_dim*16,
            num_layer=0,
            kernel_size=3,
            stride=1,
            upsample_kernel_size=2,
            norm_name='instance',
            conv_block=True,
            res_block=True,
        )

        self.decoder4 = SWINUnetrUpBlock(
            spatial_dims=3,
            in_channels=embed_dim*16,
            out_channels=embed_dim*8,
            kernel_size=3,
            upsample_kernel_size=2,
            norm_name='instance',
            res_block=True,
        )

        self.decoder3 = SWINUnetrUpBlock(
            spatial_dims=3,
            in_channels=embed_dim * 8,
            out_channels=embed_dim * 4,
            kernel_size=3,
            upsample_kernel_size=2,
            norm_name='instance',
            res_block=True,
        )
        self.decoder2 = SWINUnetrUpBlock(
            spatial_dims=3,
            in_channels=embed_dim * 4,
            out_channels=embed_dim * 2,
            kernel_size=3,
            upsample_kernel_size=2,
            norm_name='instance',
            res_block=True,
        )

        self.decoder1 = SWINUnetrUpBlock(
            spatial_dims=3,
            in_channels=embed_dim * 2,
            out_channels=embed_dim *2 ,
            kernel_size=3,
            upsample_kernel_size=2,
            norm_name='instance',
            res_block=True,
        )

        self.decoder0 = SWINUnetrUpBlock(
            spatial_dims=3,
            in_channels=embed_dim * 2,
            out_channels=embed_dim * 1,
            kernel_size=3,
            upsample_kernel_size=1,
            norm_name='instance',
            res_block=True,
        )

        self.out = UnetOutBlock(spatial_dims=3, in_channels=embed_dim, out_channels=2)  # type: ignore


    def forward(self, x):



        out = self.transformer(x)  # (B, n_patch, hidden)

        #stage 4 features
        enc5 = self.res_botneck(out[-1]) # B, 5,5,5,2048
        dec4 = self.decoder5(enc5) #B, 10,10,10,1024
        enc4 = self.encoder4(out[-2]) #skip features should be twice the di

        #stage 3 features
        dec3 = self.decoder4(dec4,enc4)
        enc3 = self.encoder3(out[-3]) #skip features

        #stage 2 features
        dec2 = self.decoder3(dec3,enc3)
        enc2 = self.encoder2(out[-4]) #skip features

        # stage 1 features
        dec1 = self.decoder2(dec2, enc2)
        enc1 = self.encoder1(out[-5])  # skip features

        dec0 = self.decoder1(dec1, enc1)
        enc0 = self.encoder0(x)

        head = self.decoder0(dec0, enc0)

        logits = self.out(head)

        return logits

CONFIGS = {
    'Swin-Net-v0': configs.get_3DSwinNetV0_config(),
    #'Swin-Net-v01': configs.get_3DSwinNetV01_config(),
    'Swin-Net-v02': configs.get_3DSwinNetV02_config(),
    'Swin-Net-v03': configs.get_3DSwinNetV03_config(),
    'Swin-Net-v04': configs.get_3DSwinNetV04_config(),
    'Swin-Net-v05': configs.get_3DSwinNetV05_config(),
    'Swin-Net-v06': configs.get_3DSwinNetV06_config(),
    'Swin-Net-hecktor-v01': configs.get_3DSwinNet_hecktor2021_V01_config(),
    'Swin-Net-hecktor-v02': configs.get_3DSwinNet_hecktor2021_V02_config(),
    'Swin-Net-hecktor-v03': configs.get_3DSwinNet_hecktor2021_V03_config(),
    'Swin-Net-hecktor-v01-ape': configs.get_3DSwinNetNoPosEmbd_config(),
    'Swin-Net-MGHHNData-v01-ape': configs.get_3DSwinNetV01_NoPosEmd_config(),
    'SwinUNETR-hecktor-v01': configs.get_3DSwinUNETR_hecktor2021_V01_config(),
    'SwinUNETR-hecktor-v02': configs.get_3DSwinUNETR_hecktor2021_V02_config(),
    'SwinUNETR_CMFF-hecktor-v01': configs.get_3DSwinUNETR_CMFF_hecktor2021_V01_config(),
    'SwinUNETR_CMFF-hecktor-v02': configs.get_3DSwinUNETR_CMFF_hecktor2021_V02_config(),
    'SwinUNETR_CMFF-hecktor-v03': configs.get_3DSwinUNETR_CMFF_hecktor2021_V03_config(),
    'SwinUNETR_CMFF-hecktor-v04': configs.get_3DSwinUNETR_CMFF_hecktor2021_V04_config(),
    'SwinUNETR_CMFF-hecktor-v05': configs.get_3DSwinUNETR_CMFF_hecktor2021_V05_config(),
    'SwinUNETR_CMFF-hecktor-v06': configs.get_3DSwinUNETR_CMFF_hecktor2021_V06_config()

}