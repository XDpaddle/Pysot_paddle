# Copyright (c) SenseTime. All Rights Reserved.

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

# import torch
# import torch.nn.functional as F
import paddle
import paddle.nn.functional as F


def xcorr_slow(x, kernel):
    """for loop to calculate cross correlation, slow version
    """
    batch = x.size()[0]
    out = []
    for i in range(batch):
        px = x[i]
        pk = kernel[i]
        # px = px.view(1, px.size()[0], px.size()[1], px.size()[2])
        px = px.reshape([1, px.size()[0], px.size()[1], px.size()[2]])
        # pk = pk.view(-1, px.size()[1], pk.size()[1], pk.size()[2])
        pk = pk.reshape([-1, px.size()[1], pk.size()[1], pk.size()[2]])
        po = F.conv2d(px, pk)
        out.append(po)
    out = paddle.concat(out, 0)
    return out


def xcorr_fast(x, kernel):
    """group conv2d to calculate cross correlation, fast version
    """
    batch = kernel.size()[0]
    # pk = kernel.view(-1, x.size()[1], kernel.size()[2], kernel.size()[3])
    pk = kernel.reshape([-1, x.size()[1], kernel.size()[2], kernel.size()[3]])
    # px = x.view(1, -1, x.size()[2], x.size()[3])
    px = x.reshape([1, -1, x.size()[2], x.size()[3]])
    po = F.conv2d(px, pk, groups=batch)
    # po = po.view(batch, -1, po.size()[2], po.size()[3])
    po = po.reshape([batch, -1, po.size()[2], po.size()[3]])
    return po


def xcorr_depthwise(x, kernel):
    """depthwise cross correlation
    """
    batch = kernel.shape[0]
    channel = kernel.shape[1]
    # x = x.contiguous().view(1, batch*channel, x.size(2), x.size(3))
    x = x.reshape([1, batch*channel, x.shape[2], x.shape[3]])
    # kernel = kernel.contiguous().view(batch*channel, 1, kernel.size(2), kernel.size(3))
    kernel = kernel.reshape([batch*channel, 1, kernel.shape[2], kernel.shape[3]])
    out = F.conv2d(x, kernel, groups=batch*channel)
    # out = out.contiguous().view(batch, channel, out.size(2), out.size(3))
    out = out.reshape([batch, channel, out.shape[2], out.shape[3]])
    return out
