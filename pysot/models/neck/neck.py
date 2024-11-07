# Copyright (c) SenseTime. All Rights Reserved.

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

# import torch.nn as nn
import paddle
import paddle.nn as nn


class AdjustLayer(nn.Layer):
    def __init__(self, in_channels, out_channels, center_size=7):
        super(AdjustLayer, self).__init__()
        self.downsample = nn.Sequential(
            nn.Conv2D(in_channels, out_channels, kernel_size=1, bias_attr=False),
            nn.BatchNorm2D(out_channels),
            )
        self.center_size = center_size

    def forward(self, x):
        x = self.downsample(x)
        if x.shape[3] < 20:
            l = (x.shape[3] - self.center_size) // 2
            r = l + self.center_size
            x = x[:, :, l:r, l:r]
        return x


class AdjustAllLayer(nn.Layer):
    def __init__(self, in_channels, out_channels, center_size=7):
        super(AdjustAllLayer, self).__init__()
        self.num = len(out_channels)
        if self.num == 1:
            self.downsample = AdjustLayer(in_channels[0],
                                          out_channels[0],
                                          center_size)
        else:
            for i in range(self.num):
                self.add_sublayer('downsample'+str(i+2),
                                AdjustLayer(in_channels[i],
                                            out_channels[i],
                                            center_size))

    def forward(self, features):
        if self.num == 1:
            return self.downsample(features)
        else:
            out = []
            for i in range(self.num):
                adj_layer = getattr(self, 'downsample'+str(i+2))
                out.append(adj_layer(features[i]))
            return out
        

if __name__ == '__main__':  # 测试
    in_channels = [512, 1024, 2048]
    out_channels = [256, 256, 256]
    net = AdjustAllLayer(in_channels, out_channels)
    features = [paddle.rand([1,512,15,15]), paddle.rand([1,1024,15,15]), paddle.rand([1,2048,15,15])]
    a = net(features)
    for n,m in net.named_sublayers():
        print(n)
    b = 1
