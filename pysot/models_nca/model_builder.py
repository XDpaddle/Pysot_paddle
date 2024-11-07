# Copyright (c) SenseTime. All Rights Reserved.

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import paddle
import paddle.nn as nn
import paddle.nn.functional as F

from pysot.core.config import cfg
# from pysot.models_nca.loss_car import make_siamcar_loss_evaluator
from pysot.models.backbone import get_backbone
from pysot.models.head.car_head import CARHead
from pysot.models.neck import get_neck
from ..utils.location_grid import compute_locations
from pysot.utils.xcorr import xcorr_depthwise
from pysot.models_nca.nl.non_local_dot_product import NONLocalBlock2D
from pysot.models_nca.bifpn import BiFPNModule

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

        # build car head
        self.car_head = CARHead(cfg, 256)

        # build response map
        self.xcorr_depthwise = xcorr_depthwise

        # build loss
        # self.loss_evaluator = make_siamcar_loss_evaluator(cfg)

        # self.down = nn.ConvTranspose2d(256 * 3, 256, 1, 1)
        self.down = nn.Conv2DTranspose(256 * 3, 256, 1, 1)
        
        self.bifpn = BiFPNModule(channels= 256,
                          levels = 3,
                      )
        
        self.nl_1 = NONLocalBlock2D(in_channels=256)
        # self.nl_2 = NONLocalBlock2D(in_channels=256)
        # self.nl_3 = NONLocalBlock2D(in_channels=256)
        
        self.up_1 = nn.Conv2DTranspose(256, 256, 7)
        # self.up_2 = nn.ConvTranspose2d(256, 256, 7)        
        # self.up_3 = nn.ConvTranspose2d(256, 256, 7)
 

    def template(self, z):
        zf = self.backbone(z)
        if cfg.ADJUST.ADJUST:
            zf = self.neck(zf)
        self.zf = zf

    def track(self, x):
        xf = self.backbone(x)
        if cfg.ADJUST.ADJUST:
            xf = self.neck(xf)

        features = self.xcorr_depthwise(xf[0],self.zf[0])
        for i in range(len(xf)-1):
            features_new = self.xcorr_depthwise(xf[i+1],self.zf[i+1])
            features = paddle.concat([features,features_new],1)
        features = self.down(features)
        features = self.nl_1(features)

        cls, loc, cen = self.car_head(features)
        return {
                'cls': cls,
                'loc': loc,
                'cen': cen
               }

    def log_softmax(self, cls):
        b, a2, h, w = cls.size()
        cls = cls.view(b, 2, a2//2, h, w)
        cls = cls.permute(0, 2, 3, 4, 1).contiguous()
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

        # cross_at_new = self.xcorr_depthwise(xf[0],zf[0])
        # cross_at_new = self.up_1(self.nl_1(cross_at_new))
        # cross_at_new = F.softmax(cross_at_new,dim=1)
        # xf_cls_new = xf[0]*cross_at_new + xf[0] 
        # cross_at_temp = [cross_at_new, cross_at_new, cross_at_new]
        # xf_cls = [xf_cls_new, xf_cls_new, xf_cls_new]
        # for i in range(len(xf)-1):
        #     cross_at_temp[i+1] = self.xcorr_depthwise(xf[i+1],zf[i+1])
        #     cross_at_temp[i+1] = self.up_1(self.nl_1(cross_at_temp[i+1]))
        #     cross_at_temp[i+1] = F.softmax(cross_at_temp[i+1],dim=1)
        #     xf_cls[i+1] = xf[i+1]*cross_at_temp[i+1] + xf[i+1] 
        
        # # cross_at_temp[2] = self.xcorr_depthwise(xf[2],zf[2])
        # # cross_at_temp[2] = self.up_3(self.nl_3(cross_at_temp[2]))
        # # cross_at_temp[2] = F.softmax(cross_at_temp[2],dim=1)
        # # xf_cls[2] = xf[2]*cross_at_temp[2] + xf[2] 
            
        
        features_new = self.xcorr_depthwise(xf[0],zf[0])
        features_temp = [features_new, features_new, features_new]    
        for i in range(len(xf)-1):
            features_temp[i+1] = self.xcorr_depthwise(xf[i+1],zf[i+1])
        features_merge = self.bifpn(features_temp)
        features_new = features_merge[0]
        for i in range(len(features_merge)-1):
            features_new = paddle.concat([features_new,features_merge[i+1]],1)
        
        features = self.xcorr_depthwise(xf[0],zf[0])
        # for i in range(len(xf)-1):
        #     features_new = self.xcorr_depthwise(xf[i+1],zf[i+1])
        #     features = torch.cat([features,features_new],1)


        features = self.down(features_new)        
        # features = self.down(features)
        # features = self.nl_1(features)

        cls, loc, cen = self.car_head(features)
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
