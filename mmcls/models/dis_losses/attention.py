# Copyright (c) OpenMMLab. All rights reserved.
import itertools

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn.bricks.drop import build_dropout
from mmengine.model import BaseModule
from mmengine.model.weight_init import trunc_normal_
from mmengine.utils import digit_version
import math

from mmcls.registry import MODELS
from mmcls.models.utils.helpers import to_2tuple
from mmcls.models.utils.layer_scale import LayerScale

# After pytorch v1.10.0, use torch.meshgrid without indexing
# will raise extra warning. For more details,
# refers to https://github.com/pytorch/pytorch/issues/50276
if digit_version(torch.__version__) >= digit_version('1.10.0'):
    from functools import partial
    torch_meshgrid = partial(torch.meshgrid, indexing='ij')
else:
    torch_meshgrid = torch.meshgrid


class MultiheadPosAttention(BaseModule):
    """Multi-head Attention Module.

    This module implements multi-head attention that supports different input
    dims and embed dims. And it also supports a shortcut from ``value``, which
    is useful if input dims is not the same with embed dims.

    Args:
        embed_dims (int): The embedding dimension.
        num_heads (int): Parallel attention heads.
        input_dims (int, optional): The input dimension, and if None,
            use ``embed_dims``. Defaults to None.
        attn_drop (float): Dropout rate of the dropout layer after the
            attention calculation of query and key. Defaults to 0.
        proj_drop (float): Dropout rate of the dropout layer after the
            output projection. Defaults to 0.
        dropout_layer (dict): The dropout config before adding the shortcut.
            Defaults to ``dict(type='Dropout', drop_prob=0.)``.
        qkv_bias (bool): If True, add a learnable bias to q, k, v.
            Defaults to True.
        qk_scale (float, optional): Override default qk scale of
            ``head_dim ** -0.5`` if set. Defaults to None.
        proj_bias (bool) If True, add a learnable bias to output projection.
            Defaults to True.
        v_shortcut (bool): Add a shortcut from value to output. It's usually
            used if ``input_dims`` is different from ``embed_dims``.
            Defaults to False.
        init_cfg (dict, optional): The Config for initialization.
            Defaults to None.
    """

    def __init__(self,
                 embed_dims,
                 num_heads,
                 hw_shapes=(7,7),
                 input_dims=None,
                 attn_drop=0.,
                 proj_drop=0.,
                 dropout_layer=dict(type='Dropout', drop_prob=0.),
                 qkv_bias=True,
                 qk_scale=None,
                 proj_bias=True,
                 v_shortcut=False,
                 use_layer_scale=False,
                 init_cfg=None,
                 pos_dims=None):
        super(MultiheadPosAttention, self).__init__(init_cfg=init_cfg)

        self.input_dims = input_dims or embed_dims
        self.pos_dim = pos_dims
        self.embed_dims = embed_dims
        self.num_heads = num_heads
        self.v_shortcut = v_shortcut

        self.head_dims = embed_dims // num_heads
        self.scale = qk_scale or self.head_dims**-0.5

        self.q = nn.Linear(self.pos_dim, embed_dims, bias=qkv_bias)
        self.k = nn.Linear(self.input_dims, embed_dims, bias=qkv_bias)
        self.v = nn.Linear(self.input_dims, embed_dims, bias=qkv_bias)

        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(embed_dims, embed_dims, bias=proj_bias)
        self.proj_drop = nn.Dropout(proj_drop)

        self.out_drop = build_dropout(dropout_layer)

        self.attn_mask = self.get_window_mask(hw_shapes)

        if use_layer_scale:
            self.gamma1 = LayerScale(embed_dims)
        else:
            self.gamma1 = nn.Identity()

    def forward(self, x, pos_emb):
        B, N, _ = x.shape
        N_out = pos_emb.shape[1]

        q = self.q(pos_emb).reshape(B, N_out, self.num_heads, self.head_dims).permute(0, 2, 1, 3)
        k = self.k(x).reshape(B, N, self.num_heads, self.head_dims).permute(0, 2, 1, 3)
        v = self.v(x).reshape(B, N, self.num_heads, self.head_dims).permute(0, 2, 1, 3)


        attn = (q @ k.transpose(-2, -1)) * self.scale

        if self.attn_mask is not None:
            mask = self.attn_mask.expand(B, self.num_heads, N_out, N)
            attn = torch.mul(attn, mask)
        
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N_out, self.embed_dims)
        x = self.proj(x)
        x = self.out_drop(self.gamma1(self.proj_drop(x)))

        if self.v_shortcut:
            x = v.squeeze(1) + x
        return x
    
    def get_window_mask(x, hw_shapes, window_shapes=(7,7)):
        if hw_shapes == window_shapes:
            return 
        H_win, W_win = window_shapes
        H, W = hw_shapes

        N = H//H_win

        assert H_win * N == H and W_win * N == W

        ones = torch.ones(H_win, W_win)
        mask_list = []
        for i in range(N):
            for j in range(N):
                mask = torch.zeros(H, W)
                mask[i*H_win:(i+1)*H_win, j*W_win:(j+1)*W_win] = ones
                mask_list.append(mask)
        mask = torch.stack(mask_list, dim=0).contiguous().view(N,N,1,1,H,W)
        # shape[N*N,  H, W]
        mask = mask.expand(N, N, H_win, W_win, H, W).permute(0,2,1,3,4,5).contiguous()
        mask = mask.view(H*W,H*W).cuda()

        return mask


class WindowMultiheadPosAttention(BaseModule):
    """Multi-head Attention Module.

    This module implements multi-head attention that supports different input
    dims and embed dims. And it also supports a shortcut from ``value``, which
    is useful if input dims is not the same with embed dims.

    Args:
        embed_dims (int): The embedding dimension.
        num_heads (int): Parallel attention heads.
        input_dims (int, optional): The input dimension, and if None,
            use ``embed_dims``. Defaults to None.
        attn_drop (float): Dropout rate of the dropout layer after the
            attention calculation of query and key. Defaults to 0.
        proj_drop (float): Dropout rate of the dropout layer after the
            output projection. Defaults to 0.
        dropout_layer (dict): The dropout config before adding the shortcut.
            Defaults to ``dict(type='Dropout', drop_prob=0.)``.
        qkv_bias (bool): If True, add a learnable bias to q, k, v.
            Defaults to True.
        qk_scale (float, optional): Override default qk scale of
            ``head_dim ** -0.5`` if set. Defaults to None.
        proj_bias (bool) If True, add a learnable bias to output projection.
            Defaults to True.
        v_shortcut (bool): Add a shortcut from value to output. It's usually
            used if ``input_dims`` is different from ``embed_dims``.
            Defaults to False.
        init_cfg (dict, optional): The Config for initialization.
            Defaults to None.
    """

    def __init__(self,
                 embed_dims,
                 num_heads,
                 window_shapes=(1,1),
                 input_dims=None,
                 attn_drop=0.,
                 proj_drop=0.,
                 softmax_scale=5.,
                 dropout_layer=dict(type='Dropout', drop_prob=0.),
                 qkv_bias=True,
                 qk_scale=None,
                 proj_bias=True,
                 v_shortcut=False,
                 use_layer_scale=False,
                 init_cfg=None,
                 pos_dims=None
                 ):
        super(WindowMultiheadPosAttention, self).__init__(init_cfg=init_cfg)

        self.input_dims = input_dims or embed_dims
        self.pos_dim = pos_dims
        self.embed_dims = embed_dims
        self.num_heads = num_heads
        self.v_shortcut = v_shortcut

        self.head_dims = embed_dims // num_heads
        self.scale = qk_scale or self.head_dims**-0.5
        self.softmax_scale = softmax_scale

        self.q = nn.Linear(self.pos_dim, embed_dims, bias=qkv_bias)
        self.k = nn.Linear(self.input_dims, embed_dims, bias=qkv_bias)
        self.v = nn.Linear(self.input_dims, embed_dims, bias=qkv_bias)

        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(embed_dims, embed_dims, bias=proj_bias)
        self.proj_drop = nn.Dropout(proj_drop)

        self.out_drop = build_dropout(dropout_layer)

        self.window_shapes = window_shapes

        if use_layer_scale:
            self.gamma1 = LayerScale(embed_dims)
        else:
            self.gamma1 = nn.Identity()

    def forward(self, x, pos_emb):
        B, N, _ = x.shape
        N_out = pos_emb.shape[1]
        N_windows = self.window_shapes[0] * self.window_shapes[1]

        q = self.q(pos_emb).reshape(B, N_out, self.num_heads, self.head_dims).permute(0, 2, 1, 3)
        k = self.k(x).reshape(B, N, self.num_heads, self.head_dims).permute(0, 2, 1, 3)
        v = self.v(x).reshape(B, N, self.num_heads, self.head_dims).permute(0, 2, 1, 3)

        #######
        #  [BS, n_heads, n_q, token_dims]
        #  [BS, n_heads, n_kv, token_dims]

        #  [BS, n_heads*n_windows, n_q/n_window, token_dims]
        #  [BS, n_heads*n_windows, n_kv/n_window, token_dims]

        #  [BS, n_heads*n_windows, n_q/n_window, n_kv/n_window]
        #  [BS, n_heads*n_windows, n_kv/n_window, token_dims]
        #######
        if N_windows > 1:
            q = self.separate_tokens(q, self.window_shapes)
            k = self.separate_tokens(k, self.window_shapes)
            v = self.separate_tokens(v, self.window_shapes)

        attn = (q @ k.transpose(-2, -1)) * self.scale * self.softmax_scale
        
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).view(B, self.num_heads, N_windows, N_out//N_windows, self.head_dims)
        x = x.view(B, self.num_heads, N_out, self.head_dims).transpose(1, 2).reshape(B, N_out, self.embed_dims)

        x = self.proj(x)
        x = self.out_drop(self.gamma1(self.proj_drop(x)))

        if self.v_shortcut:
            x = v.squeeze(1) + x
        return x
    

    def separate_tokens(self, x, window_shapes=(2,2)):
        BS, num_heads, num_tokens, head_dims = x.shape
        H = W = int(math.sqrt(num_tokens))
        num_win_h, num_win_w = window_shapes

        x = x.view(BS, num_heads, num_win_h, H//num_win_h, num_win_w, W//num_win_w, head_dims).permute(0,1,2,4,3,5,6)
        x = x.contiguous().view(BS, num_heads*num_win_h*num_win_w, -1, head_dims)

        return x



        

        





    # def _get_gaussian_mask(self, hw_shapes):
    #     H, W = hw_shapes



    # def gaussian(
    #         M: int,
    #         *,
    #         std: float = 1.0,
    #         sym: bool = True,
    #         dtype: Optional[torch.dtype] = None,
    #         layout: torch.layout = torch.strided,
    #         device: Optional[torch.device] = None,
    #         requires_grad: bool = False
    # ) -> Tensor:
    #     if dtype is None:
    #         dtype = torch.get_default_dtype()


    #     if std <= 0:
    #         raise ValueError(f'Standard deviation must be positive, got: {std} instead.')

    #     if M == 0:
    #         return torch.empty((0,), dtype=dtype, layout=layout, device=device, requires_grad=requires_grad)

    #     start = -(M if not sym and M > 1 else M - 1) / 2.0

    #     constant = 1 / (std * sqrt(2))

    #     k = torch.linspace(start=start * constant,
    #                     end=(start + (M - 1)) * constant,
    #                     steps=M,
    #                     dtype=dtype,
    #                     layout=layout,
    #                     device=device,
    #                     requires_grad=requires_grad)

    #     return torch.exp(-k ** 2)








