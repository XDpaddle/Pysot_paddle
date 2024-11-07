# Copyright (c) SenseTime. All Rights Reserved.

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

# import torch
# import torch.nn as nn
# import torch.nn.functional as F
import paddle
import paddle.nn as nn
import paddle.nn.functional as F

from pysot.core.config import cfg
from pysot.models_car.loss_car import make_siamcar_loss_evaluator
from pysot.models.backbone import get_backbone
from pysot.models.head.car_head import CARHead
from pysot.models.neck import get_neck
from ..utils.location_grid import compute_locations
from pysot.utils.xcorr import xcorr_depthwise

from pysot.models_car.BiFPN.bifpn import BiFPNModule
from pysot.models_car.Attention import get_self_attention
from pysot.models_car.positional_encoding.builder import build_position_embedding

class ModelBuilder(nn.Layer):
    def __init__(self):
        super(ModelBuilder, self).__init__()

        # build backbone
        self.backbone = get_backbone(cfg.BACKBONE.TYPE,
                                     **cfg.BACKBONE.KWARGS)

        # build adjust layer
        if cfg.ADJUST.ADJUST:
            self.neck = get_neck(cfg.ADJUST.TYPE,
                                 **cfg.ADJUST.KWARGS)

        #-------------------------------------------------------------上采样以及attention--------------------------------
        self.bifpn = BiFPNModule(channels=256,
                                 levels=3,
                                 )
        # self.upsampling = nn.Sequential(
        #     paddle.nn.Conv2DTranspose(256, 256, kernel_size=7, padding=0, stride=1),
        # )
        # build attention layer
        class BasicBlock(nn.Layer):
            def __init__(self, in_channel, out_channel, stride=1, **kwargs):
                super(BasicBlock, self).__init__()
                self.conv1 = nn.Conv2D(in_channels=in_channel, out_channels=out_channel,
                                       kernel_size=3, stride=stride, padding=1, bias_attr=False)
                #
                self.bn1 = nn.BatchNorm2D(out_channel)
                #
                self.relu = nn.ReLU()
                self.conv2 = nn.Conv2D(in_channels=out_channel, out_channels=out_channel,
                                       kernel_size=3, stride=1, padding=1, bias_attr=False)
                self.bn2 = nn.BatchNorm2D(out_channel)

            def forward(self, x, y):
                identity = y
                out = self.conv1(x)
                out = self.bn1(out)
                out = self.relu(out)
                #-----------------------------
                out = self.conv2(out)
                out = self.bn2(out)
                #
                out += identity
                out = self.relu(out)

                return out



        # self.Resnet = BasicBlock(256, 256)
        # self.self_attention_1 = get_self_attention(cfg)
        # self.self_attention_2 = get_self_attention(cfg)

        # self.pos_t, self.pos_s = build_position_embedding([7, 7], [31, 31], 256)
        #---------------------------------------------------------------------------------------------------------------

        # build car head
        self.car_head = CARHead(cfg, 256)


        # build response map
        self.xcorr_depthwise = xcorr_depthwise

        # build loss
        self.loss_evaluator = make_siamcar_loss_evaluator(cfg)

        self.down = nn.Conv2DTranspose(256 * 3, 256, 1, 1)
        # self.down1 = nn.Conv2DTranspose(256 * 3, 256, 1)
        # self.down2 = nn.Conv2DTranspose(256 * 3, 256, 1)


    def template(self, z):
        zf = self.backbone(z)
        if cfg.ADJUST.ADJUST:
            zf = self.neck(zf)

        self.zf = zf


    # def track(self, x):
    #     xf = self.backbone(x)
    #     if cfg.ADJUST.ADJUST:
    #         xf = self.neck(xf)          # xf:list:3 ;[1, 256, 31, 31];[1, 256, 31, 31];[1, 256, 31, 31]

    #     # features_3 = self.xcorr_depthwise(xf[2], self.zf[2])
    #     features_11, features_12, features_13 = self.zf[0], self.zf[1], self.zf[2]    # template
    #     features_21, features_22, features_23 = xf[0], xf[1], xf[2]    # search
    #     re1 = [features_11, features_12, features_13]   # zf:list:3 ;[1, 256, 7, 7];[1, 256, 7, 7];[1, 256, 7, 7]
    #     re2 = [features_21, features_22, features_23]   # xf:list:3 ;[1, 256, 31, 31];[1, 256, 31, 31];[1, 256, 31, 31]
    #     features_merge_1 = self.bifpn(re1)
    #     features_merge_2 = self.bifpn(re2)
    #     #
    #     features_1 = self.xcorr_depthwise(features_merge_2[0], features_merge_1[0])  # features:[1, 256, 25, 25]
    #     features_2 = self.xcorr_depthwise(features_merge_2[1], features_merge_1[1])
    #     features_3 = self.xcorr_depthwise(features_merge_2[2], features_merge_1[2])
    #     features_1 = self.upsampling(features_1)
    #     features_2 = self.upsampling(features_2)
    #     features_3 = self.upsampling(features_3)
    #     # 加softmax
    #     features_1 = F.softmax(features_1, axis=1)
    #     features_2 = F.softmax(features_2, axis=1)
    #     features_3 = F.softmax(features_3, axis=1)

    #     # 上采样后与原先的search相乘
    #     features_1 = features_1 * features_merge_2[0]
    #     features_2 = features_2 * features_merge_2[1]
    #     features_3 = features_3 * features_merge_2[2]

    #     features_21 = self.Resnet(features_merge_2[0], features_1)
    #     features_22 = self.Resnet(features_merge_2[1], features_2)
    #     features_23 = self.Resnet(features_merge_2[2], features_3)
    #     features_merge_2 = [features_21, features_22, features_23]
    #     # weight
    #     # weight = F.softmax(self.weight, 0)

    #     # features = torch.cat([features_merge[0], features_merge[1], features_merge[2]], 1)      # 12/5;19:20
    #     # features = self.down(features)
    #     # for i in range(len(xf)-1):                                # 11/29;20:49
    #     #     features_new = self.xcorr_depthwise(xf[i+1], self.zf[i+1])
    #     #     features = torch.cat([features, features_new], 1)

    #     # features = self.down(features)          # input:features:[1, 768, 25, 25]          output:features:[1, 256, 25, 25]  # 11/29;20:49  12/5;19:21

    #     # ----------------------------------------------------------上采样-----------------------------------------------

    #     template_p = self.pos_t().unsqueeze(0)
    #     search_p = self.pos_s().unsqueeze(0)

    #     # attn_out_x = self.self_attention(features_merge_1, features_merge_2, template_p, search_p)          # features_merge_1:list3[1, 256, 25, 25]; features_merge_2:list3[1, 256, 31, 31]
    #     # attn_out_x = self.self_attention_1(features_merge_1, template_p)          # features_merge_1:list3[1, 256, 7, 7]; features_merge_2:list3[1, 256, 31, 31]
    #     # attn_out_y = self.self_attention_2(features_merge_2, search_p)

    #     # x = torch.transpose(attn_out_x, 1, 2)
    #     # y = torch.transpose(attn_out_y, 1, 2)
    #     # out_1 = x.copy()
    #     # out_2 = y.copy()
    #     # out_11 = x[:, :, :49].reshape(x.shape[0], 256, 7, 7)
    #     # out_12 = x[:, :, 49:98].reshape(x.shape[0], 256, 7, 7)
    #     # out_13 = x[:, :, 98:147].reshape(x.shape[0], 256, 7, 7)
    #     # out_21 = y[:, :, :961].reshape(y.shape[0], 256, 31, 31)
    #     # out_22 = y[:, :, 961:1922].reshape(y.shape[0], 256, 31, 31)
    #     # out_23 = y[:, :, 1922:2883].reshape(y.shape[0], 256, 31, 31)
    #     # out_template = torch.cat([out_11, out_12, out_13], 1)
    #     # out_search = torch.cat([out_21, out_22, out_23], 1)

    #     out_template = paddle.concat([features_merge_1[0], features_merge_1[1], features_merge_1[2]], 1)
    #     out_search = paddle.concat([features_merge_2[0], features_merge_2[1], features_merge_2[2]], 1)
    #     out_template = self.down1(out_template)
    #     out_search = self.down2(out_search)
    #     # out_template = x[:, :, :49].reshape(x.shape[0], 256, 7, 7)
    #     # out_search = y[:, :, :961].reshape(x.shape[0], 256, 31, 31)
    #     # features = torch.cat([out[0], out[1], out[2]], 1)

    #     # features = self.down(features)
    #     features = self.xcorr_depthwise(out_search, out_template)
    #     cls, loc, cen = self.car_head(features)
    #     #---------------------------------------------------------------------------------------------------------------

    #     # cls, loc, cen = self.car_head(features)              # 11/28;16:38  12/5;19:23
    #     # cls, loc, cen = self.car_head(features)
    #     return {
    #             'cls': cls,
    #             'loc': loc,
    #             'cen': cen
    #            }

    def track(self, x):
        xf = self.backbone(x)
        if cfg.ADJUST.ADJUST:
            xf = self.neck(xf)          # xf:list:3 ;[1, 256, 31, 31];[1, 256, 31, 31];[1, 256, 31, 31]

        features_1 = self.xcorr_depthwise(xf[0], self.zf[0])           # features:[1, 256, 25, 25]
        features_2 = self.xcorr_depthwise(xf[1], self.zf[1])
        features_3 = self.xcorr_depthwise(xf[2], self.zf[2])
        re = [features_1, features_2, features_3]
        features_merge = self.bifpn(re)
        # weight
        # weight = F.softmax(self.weight, 0)

        # s = 0
        # for i in range(3):
        #     s += weight[i] * features_merge[i]

        features = paddle.concat([features_merge[0], features_merge[1], features_merge[2]], 1)
        features = self.down(features)
        # for i in range(len(xf)-1):                                # 11/29;20:49
        #     features_new = self.xcorr_depthwise(xf[i+1], self.zf[i+1])
        #     features = torch.cat([features, features_new], 1)

        # features = self.down(features)          # input:features:[1, 768, 25, 25]           output:features:[1, 256, 25, 25]  # 11/29;20:49

        # ----------------------------------------------------------上采样-----------------------------------------------
        # features_up = self.upsampling(features)        # output:features:[1, 256, 50, 50]

        # template_p = self.pos_t().unsqueeze(0)
        # B_up, C_up, W_up, H_up = features_up.shape
        # B, C, W, H = features.shape
        # x_1 = features.reshape(B, C, -1).permute(0, 2, 1)
        # x_2 = features_up.reshape(B_up, C_up, -1).permute(0, 2, 1)
        # x = self.self_attention(x_1, x_2, template_p)
        # cls, loc, cen = self.car_head(x)
        #---------------------------------------------------------------------------------------------------------------

        cls, loc, cen = self.car_head(features)              # 11/28;16:38
        # cls, loc, cen = self.car_head(features)
        return {
                'cls': cls,
                'loc': loc,
                'cen': cen
               }


    def log_softmax(self, cls):
        b, a2, h, w = cls.size()
        # cls = cls.view(b, 2, a2//2, h, w)
        cls = cls.reshape([b, 2, a2//2, h, w])
        # cls = cls.permute(0, 2, 3, 4, 1).contiguous()
        cls = cls.transpose([0, 2, 3, 4, 1])
        cls = F.log_softmax(cls, axis=4)
        return cls

    def forward(self, data):
        """ only used in training
        """
        template = data['template']
        search = data['search']
        label_cls = data['label_cls']
        label_loc = data['bbox']

        # get feature
        zf = self.backbone(template)
        xf = self.backbone(search)
        if cfg.ADJUST.ADJUST:
            zf = self.neck(zf)
            xf = self.neck(xf)


        # features = self.xcorr_depthwise(xf[0], zf[0])         # features:[1, 256, 25, 25]
        # re = [features_1, features_2, features_3]
        # features_merge = self.bifpn(re)
        features_11, features_12, features_13 = zf[0], zf[1], zf[2]  # template
        features_21, features_22, features_23 = xf[0], xf[1], xf[2]  # search
        re1 = [features_11, features_12, features_13]  # zf:list:3 ;[1, 256, 25, 25];[1, 256, 25, 25];[1, 256, 25, 25]
        re2 = [features_21, features_22, features_23]  # xf:list:3 ;[1, 256, 31, 31];[1, 256, 31, 31];[1, 256, 31, 31]
        features_merge_1 = self.bifpn(re1)
        features_merge_2 = self.bifpn(re2)

        features_1 = self.xcorr_depthwise(features_merge_2[0], features_merge_1[0])  # features:[1, 256, 25, 25]
        features_2 = self.xcorr_depthwise(features_merge_2[1], features_merge_1[1])
        features_3 = self.xcorr_depthwise(features_merge_2[2], features_merge_1[2])
        features_1 = self.upsampling(features_1)
        features_2 = self.upsampling(features_2)
        features_3 = self.upsampling(features_3)

        # 加softmax
        features_1 = F.softmax(features_1, axis=1)
        features_2 = F.softmax(features_2, axis=1)
        features_3 = F.softmax(features_3, axis=1)
        # 上采样后与原先的search相乘
        features_1 = features_1 * features_merge_2[0]
        features_2 = features_2 * features_merge_2[1]
        features_3 = features_3 * features_merge_2[2]

        features_21 = self.Resnet(features_merge_2[0], features_1)
        features_22 = self.Resnet(features_merge_2[1], features_2)
        features_23 = self.Resnet(features_merge_2[2], features_3)
        features_merge_2 = [features_21, features_22, features_23]
        # features = torch.cat([features_merge[0], features_merge[1], features_merge[2]], 1)

        # for i in range(len(xf)-1):
        #     features_new = self.xcorr_depthwise(xf[i+1], zf[i+1])
        #     features = torch.cat([features, features_new], 1)
        # features = self.down(features)
        # ----------------------------------------------------------上采样-----------------------------------------------
        template_p = self.pos_t().unsqueeze(0)
        search_p = self.pos_s().unsqueeze(0)

        # attn_out_x = self.self_attention(features_merge_1, features_merge_2, template_p, search_p)
        # attn_out_x = self.self_attention_1(features_merge_1, template_p)   # features_merge_1:list3[1, 256, 7, 7]; features_merge_2:list3[1, 256, 31, 31]
        # attn_out_y = self.self_attention_2(features_merge_2, search_p)

        # x = torch.transpose(attn_out_x, 1, 2)
        # y = torch.transpose(attn_out_y, 1, 2)
        # out_1 = x.copy()
        # out_2 = y.copy()
        # out_11 = x[:, :, :49].reshape(x.shape[0], 256, 7, 7)
        # out_12 = x[:, :, 49:98].reshape(x.shape[0], 256, 7, 7)
        # out_13 = x[:, :, 98:147].reshape(x.shape[0], 256, 7, 7)
        # out_21 = y[:, :, :961].reshape(y.shape[0], 256, 31, 31)
        # out_22 = y[:, :, 961:1922].reshape(y.shape[0], 256, 31, 31)
        # # out_23 = y[:, :, 1922:2883].reshape(y.shape[0], 256, 31, 31)
        # out_template = torch.cat([out_11, out_12, out_13], 1)
        # out_search = torch.cat([out_21, out_22, out_23], 1)

        out_template = paddle.concat([features_merge_1[0], features_merge_1[1], features_merge_1[2]], 1)
        out_search = paddle.concat([features_merge_2[0], features_merge_2[1], features_merge_2[2]], 1)
        out_template = self.down1(out_template)
        out_search = self.down2(out_search)

        # out_template = x[:, :, :49].reshape(x.shape[0], 256, 7, 7)
        # out_search = y[:, :, :961].reshape(x.shape[0], 256, 31, 31)
        # features = torch.cat([out[0], out[1], out[2]], 1)

        # features = self.down(features)
        features = self.xcorr_depthwise(out_search, out_template)
        cls, loc, cen = self.car_head(features)
        # ---------------------------------------------------------------------------------------------------------------

        # cls, loc, cen = self.car_head(features)              # 11/28;16:25
        # cls, loc, cen = self.car_head(features)
        locations = compute_locations(cls, cfg.TRACK.STRIDE)
        cls = self.log_softmax(cls)
        cls_loss, loc_loss, cen_loss = self.loss_evaluator(
            locations,
            cls,
            loc,
            cen, label_cls, label_loc
        )

        # get loss
        outputs = {}
        outputs['total_loss'] = cfg.TRAIN.CLS_WEIGHT * cls_loss + \
            cfg.TRAIN.LOC_WEIGHT * loc_loss + cfg.TRAIN.CEN_WEIGHT * cen_loss
        outputs['cls_loss'] = cls_loss
        outputs['loc_loss'] = loc_loss
        outputs['cen_loss'] = cen_loss
        return outputs
