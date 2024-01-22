import paddle.nn as nn
from paddleseg.cvlibs import param_init
import math

def init_weights(model):
    # for m in model.modules():
    #     if isinstance(m, nn.Conv2d):
    #         nn.init.kaiming_normal_(m.weight.data,
    #                                 mode='fan_out',
    #                                 nonlinearity='relu')
    #     elif isinstance(m, nn.BatchNorm2d):
    #         m.weight.data.fill_(1)
    #         m.bias.data.zero_()

    for m in model.named_sublayers():
        if isinstance(m[1], nn.Conv2D):
            n = m[1]._kernel_size[0] * m[1]._kernel_size[1] * m[1]._out_channels
            # m.weight.data.normal_(0, math.sqrt(2. / n))
            param_init.normal_init(m[1].weight, mean=0, std=math.sqrt(2. / n))
            if m[1].bias is not None:
                param_init.constant_init(m[1].bias,value=0)
            # param_init.kaiming_uniform(m[1].weight, nonlinearity="relu")
            # if m[1].bias is not None:
            #     param_init.constant_init(m[1].bias,value=0)
        elif isinstance(m[1], nn.BatchNorm2D):
            # m.weight.data.fill_(1)
            # m.bias.data.zero_()
            param_init.constant_init(m[1].weight,value=1)
            param_init.constant_init(m[1].bias,value=0)
