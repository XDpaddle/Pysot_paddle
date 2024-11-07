# Copyright (c) SenseTime. All Rights Reserved.

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

# import torch
# import torch.nn.functional as F
import paddle
import paddle.nn.functional as F

def get_cls_loss(pred, label, select):
    # if len(select.size()) == 0 or \
    #         select.size() == torch.Size([0]):  # 
    if len(select.shape) == 0 or \
            select.shape == [0]:
        return 0
    # pred = torch.index_select(pred, 0, select)
    pred = paddle.index_select(pred, select,0)
    # label = torch.index_select(label, 0, select)
    label = paddle.index_select(label, select, 0)
    return F.nll_loss(pred, label)


def select_cross_entropy_loss(pred, label):
    # pred = pred.view(-1, 2)
    pred = pred.reshape([-1, 2])
    # label = label.view(-1)
    label = label.reshape([-1])
    # pos = label.data.eq(1).nonzero().squeeze().cuda()
    pos = label.detach().equal(1).nonzero().squeeze()
    # neg = label.data.eq(0).nonzero().squeeze().cuda()
    neg = label.detach().equal(0).nonzero().squeeze()
    loss_pos = get_cls_loss(pred, label, pos)
    loss_neg = get_cls_loss(pred, label, neg)
    return loss_pos * 0.5 + loss_neg * 0.5


def weight_l1_loss(pred_loc, label_loc, loss_weight):
    # b, _, sh, sw = pred_loc.size()
    b, _, sh, sw = pred_loc.shape
    # pred_loc = pred_loc.view(b, 4, -1, sh, sw)
    pred_loc = pred_loc.reshape([b, 4, -1, sh, sw])
    diff = (pred_loc - label_loc).abs()
    # diff = diff.sum(dim=1).view(b, -1, sh, sw)
    diff = diff.sum(axis=1).reshape([b, -1, sh, sw])
    loss = diff * loss_weight
    # return loss.sum().div(b)
    return loss.sum().divide(paddle.to_tensor([b],dtype=paddle.float32))
