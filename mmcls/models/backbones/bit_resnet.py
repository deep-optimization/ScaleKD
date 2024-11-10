# Copyright (c) OpenMMLab. All rights reserved.
import torch
import torch.nn as nn
import torch.utils.checkpoint as cp
from mmcv.cnn import (build_activation_layer,
                      build_norm_layer)
from mmcv.cnn.bricks import DropPath
from mmengine.model import BaseModule
from mmengine.model.weight_init import constant_init
from mmengine.utils.dl_utils.parrots_wrapper import _BatchNorm

from mmcls.registry import MODELS
from .base_backbone import BaseBackbone

eps = 1.0e-5


from collections import OrderedDict  # pylint: disable=g-importing-member

import torch
import torch.nn as nn
import torch.nn.functional as F


class StdConv2d(nn.Conv2d):

  def forward(self, x):
    w = self.weight
    v, m = torch.var_mean(w, dim=[1, 2, 3], keepdim=True, unbiased=False)
    w = (w - m) / torch.sqrt(v + 1e-10)
    return F.conv2d(x, w, self.bias, self.stride, self.padding,
                    self.dilation, self.groups)


def conv3x3(cin, cout, stride=1, groups=1, bias=False):
  return StdConv2d(cin, cout, kernel_size=3, stride=stride,
                   padding=1, bias=bias, groups=groups)


def conv1x1(cin, cout, stride=1, bias=False):
  return StdConv2d(cin, cout, kernel_size=1, stride=stride,
                   padding=0, bias=bias)


def tf2th(conv_weights):
  """Possibly convert HWIO to OIHW."""
  if conv_weights.ndim == 4:
    conv_weights = conv_weights.transpose([3, 2, 0, 1])
  return torch.from_numpy(conv_weights)


class PreActBottleneck(nn.Module):
  """Pre-activation (v2) bottleneck block.

  Follows the implementation of "Identity Mappings in Deep Residual Networks":
  https://github.com/KaimingHe/resnet-1k-layers/blob/master/resnet-pre-act.lua

  Except it puts the stride on 3x3 conv when available.
  """

  def __init__(self, cin, cout=None, cmid=None, stride=1, drop_path_rate=0.0):
    super().__init__()
    cout = cout or cin
    cmid = cmid or cout//4

    self.gn1 = nn.GroupNorm(32, cin)
    self.conv1 = conv1x1(cin, cmid)
    self.gn2 = nn.GroupNorm(32, cmid)
    self.conv2 = conv3x3(cmid, cmid, stride)  # Original code has it on conv1!!
    self.gn3 = nn.GroupNorm(32, cmid)
    self.conv3 = conv1x1(cmid, cout)
    self.relu = nn.ReLU(inplace=True)

    self.drop_path = DropPath(drop_prob=drop_path_rate
                                  ) if drop_path_rate > 1e-10 else nn.Identity()

    if (stride != 1 or cin != cout):
      # Projection also with pre-activation according to paper.
      self.downsample = nn.Sequential(
        nn.AvgPool2d(kernel_size=stride, stride=stride, padding=0), 
        conv1x1(cin, cout),
        )

  def forward(self, x):
    out = self.relu(self.gn1(x))

    # Residual branch
    residual = x
    if hasattr(self, 'downsample'):
      residual = self.downsample(out)

    # Unit's branch
    out = self.conv1(out)
    out = self.conv2(self.relu(self.gn2(out)))
    out = self.conv3(self.relu(self.gn3(out)))

    out = self.drop_path(out)

    return out + residual


@MODELS.register_module()
class BiTResNet(BaseBackbone):
  """Implementation of Pre-activation (v2) ResNet mode."""
  arch_settings = {
        50:  (3, 4, 6, 3),
        101: (3, 4, 23, 3),
        152: (3, 8, 36, 3),
   }
  def __init__(self, depth, width_factor=1, drop_path_rate=0.0, 
                zero_head=False, 
                init_cfg=None,
                # init_cfg=[
                #      dict(type='Kaiming', layer=['Conv2d']),
                #      dict(
                #          type='Constant',
                #          val=1,
                #          layer=['_BatchNorm', 'GroupNorm'])
                 ):
    super(BiTResNet, self).__init__(init_cfg)
    if depth not in self.arch_settings:
        raise KeyError(f'invalid depth {depth} for resnet')
    self.depth = depth
    block_units = self.arch_settings[depth]
    wf = width_factor  # shortcut 'cause we'll use it a lot.

    # The following will be unreadable if we split lines.
    # pylint: disable=line-too-long
    # self.root = nn.Sequential(OrderedDict([
    #     ('conv', StdConv2d(3, 64*wf, kernel_size=7, stride=2, padding=3, bias=False)),
    #     ('pad', nn.ConstantPad2d(1, 0)),
    #     ('pool', nn.MaxPool2d(kernel_size=3, stride=2, padding=0)),
    #     # The following is subtly not the same!
    #     # ('pool', nn.MaxPool2d(kernel_size=3, stride=2, padding=1)),
    # ]))

    self.root = nn.Sequential(OrderedDict([
        ('conv1', StdConv2d(3, 64*wf//2, kernel_size=3, stride=2, padding=1, bias=False)),
        ('conv2', StdConv2d(64*wf//2, 64*wf//2, kernel_size=3, stride=1, padding=1, bias=False)),
        ('conv3', StdConv2d(64*wf//2, 64*wf, kernel_size=3, stride=1, padding=1, bias=False)),
        ('pad', nn.ConstantPad2d(1, 0)),
        ('pool', nn.MaxPool2d(kernel_size=3, stride=2, padding=0)),
        # The following is subtly not the same!
        # ('pool', nn.MaxPool2d(kernel_size=3, stride=2, padding=1)),
    ]))

    self.body = nn.Sequential(OrderedDict([
        ('block1', nn.Sequential(OrderedDict(
            [('unit01', PreActBottleneck(cin=64*wf, cout=256*wf, cmid=64*wf, drop_path_rate=drop_path_rate))] +
            [(f'unit{i:02d}', PreActBottleneck(cin=256*wf, cout=256*wf, cmid=64*wf, drop_path_rate=drop_path_rate)) for i in range(2, block_units[0] + 1)],
        ))),
        ('block2', nn.Sequential(OrderedDict(
            [('unit01', PreActBottleneck(cin=256*wf, cout=512*wf, cmid=128*wf, stride=2, drop_path_rate=drop_path_rate))] +
            [(f'unit{i:02d}', PreActBottleneck(cin=512*wf, cout=512*wf, cmid=128*wf, drop_path_rate=drop_path_rate)) for i in range(2, block_units[1] + 1)],
        ))),
        ('block3', nn.Sequential(OrderedDict(
            [('unit01', PreActBottleneck(cin=512*wf, cout=1024*wf, cmid=256*wf, stride=2, drop_path_rate=drop_path_rate))] +
            [(f'unit{i:02d}', PreActBottleneck(cin=1024*wf, cout=1024*wf, cmid=256*wf, drop_path_rate=drop_path_rate)) for i in range(2, block_units[2] + 1)],
        ))),
        ('block4', nn.Sequential(OrderedDict(
            [('unit01', PreActBottleneck(cin=1024*wf, cout=2048*wf, cmid=512*wf, stride=2, drop_path_rate=drop_path_rate))] +
            [(f'unit{i:02d}', PreActBottleneck(cin=2048*wf, cout=2048*wf, cmid=512*wf, drop_path_rate=drop_path_rate)) for i in range(2, block_units[3] + 1)],
        ))),
    ]))

    self.norm = nn.GroupNorm(32, 2048*wf)

  def forward(self, x):
    x = self.root(x)
    outs = []
    for i in range(4):
      x = self.body[i](x)
      if i == 3:
        x = self.norm(x)
      outs.append(x)

    return tuple(outs)
