# -----------------------------------------------------------------------------
# Copyright (c) Microsoft
# Licensed under the MIT License.
# Written by Zhipeng Zhang (zhangzhipeng2017@ia.ac.cn)
# ------------------------------------------------------------------------------
import math
# import torch
# import torch.nn as nn
# import torch.nn.functional as F

import paddle
import paddle.nn as nn
import paddle.nn.functional as F

from collections import OrderedDict
from functools import partial
import collections
import re
import numpy as np
import warnings
from paddleseg.cvlibs import param_init


eps = 1e-5

# -------------
# Single Layer
# -------------
def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2D(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)

def conv3x3NP(in_planes, out_planes, stride=1):
    """3x3 convolution without padding"""
    return nn.Conv2D(in_planes, out_planes, kernel_size=3, stride=stride, bias=False)



def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2D(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


def down(in_planes, out_planes):
    """downsampling at the output layer"""
    return nn.Sequential(nn.Conv2D(in_planes, out_planes, kernel_size=1),
            nn.BatchNorm2D(out_planes))

def down_spatial(in_planes, out_planes):
    """downsampling 21*21 to 5*5 (21-5)//4+1=5"""
    return nn.Sequential(nn.Conv2D(in_planes, out_planes, kernel_size=5, stride=4),
            nn.BatchNorm2D(out_planes))


# -------------------------------
# Several Kinds Bottleneck Blocks
# -------------------------------

class Bottleneck(nn.Layer):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None, dilation=1):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2D(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2D(planes)
        padding = 2 - stride
        if downsample is not None and dilation > 1:
            dilation = dilation // 2
            padding = dilation

        assert stride == 1 or dilation == 1, \
            "stride and dilation must have one equals to zero at least"

        if dilation > 1:
            padding = dilation
        self.conv2 = nn.Conv2D(planes, planes, kernel_size=3, stride=stride,
                               padding=padding, bias=False, dilation=dilation)
        self.bn2 = nn.BatchNorm2D(planes)
        self.conv3 = nn.Conv2D(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2D(planes * 4)
        # self.relu = nn.ReLU(inplace=True)
        self.relu = nn.ReLU()
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual

        out = self.relu(out)

        return out

class Bottleneck_BIG_CI(nn.Layer):
    """
    Bottleneck with center crop layer, double channels in 3*3 conv layer in shortcut branch
    """
    expansion = 4

    def __init__(self, inplanes, planes, last_relu, stride=1, downsample=None, dilation=1):
        super(Bottleneck_BIG_CI, self).__init__()
        self.conv1 = nn.Conv2D(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2D(planes)

        padding = 1
        if abs(dilation - 2) < eps: padding = 2
        if abs(dilation - 3) < eps: padding = 3

        self.conv2 = nn.Conv2D(planes, planes*2, kernel_size=3, stride=stride, padding=padding, bias=False, dilation=dilation)
        self.bn2 = nn.BatchNorm2D(planes*2)
        self.conv3 = nn.Conv2D(planes*2, planes * self.expansion, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2D(planes * self.expansion)
        # self.relu = nn.ReLU(inplace=True)
        self.relu = nn.ReLU()
        self.downsample = downsample
        self.stride = stride
        self.last_relu = last_relu

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual

        if self.last_relu:  # feature out no relu
            out = self.relu(out)

        out = self.center_crop(out)  # in-layer crop

        return out

    def center_crop(self, x):
        """
        center crop layer. crop [1:-2] to eliminate padding influence.
        Crop 1 element around the tensor
        input x can be a Variable or Tensor
        """
        return x[:, :, 1:-1, 1:-1].contiguous()

# ---------------------
# Modified ResNet
# ---------------------
class ResNet_plus2(nn.Layer):
    def __init__(self, block, layers, used_layers, online=False):
        self.inplanes = 64
        super(ResNet_plus2, self).__init__()
        self.conv1 = nn.Conv2D(3, 64, kernel_size=7, stride=2, padding=0,  # 3
                               bias=False)
        self.bn1 = nn.BatchNorm2D(64)
        # self.relu = nn.ReLU(inplace=True)
        self.relu = nn.ReLU()
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)

        self.feature_size = 128 * block.expansion
        self.used_layers = used_layers
        self.layer3_use = True if 3 in used_layers else False
        self.layer4_use = True if 4 in used_layers else False

        if self.layer3_use:
            if online:
                self.layer3 = self._make_layer(block, 256, layers[2], stride=1, dilation=2, update=True)
                self.layeronline = self._make_layer(block, 256, layers[2], stride=2)
            else:
                self.layer3 = self._make_layer(block, 256, layers[2], stride=1, dilation=2)

            self.feature_size = (256 + 128) * block.expansion
        else:
            self.layer3 = lambda x: x  # identity

        if self.layer4_use:
            self.layer4 = self._make_layer(block, 512, layers[3], stride=1, dilation=4)  # 7x7, 3x3
            self.feature_size = 512 * block.expansion
        else:
            self.layer4 = lambda x: x  # identity

        # for m in self.modules():
        #     if isinstance(m, nn.Conv2d):
        #         n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
        #         m.weight.data.normal_(0, math.sqrt(2. / n))
        #     elif isinstance(m, nn.BatchNorm2d):
        #         m.weight.data.fill_(1)
        #         m.bias.data.zero_()
        for m in self.named_sublayers():
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

    def _make_layer(self, block, planes, blocks, stride=1, dilation=1, update=False):
        downsample = None
        dd = dilation
        if stride != 1 or self.inplanes != planes * block.expansion:
            if stride == 1 and dilation == 1:
                downsample = nn.Sequential(
                    nn.Conv2D(self.inplanes, planes * block.expansion,
                              kernel_size=1, stride=stride, bias=False),
                    nn.BatchNorm2D(planes * block.expansion),
                )
            else:
                if dilation > 1:
                    dd = dilation // 2
                    padding = dd
                else:
                    dd = 1
                    padding = 0
                downsample = nn.Sequential(
                    nn.Conv2D(self.inplanes, planes * block.expansion,
                              kernel_size=3, stride=stride, bias=False,
                              padding=padding, dilation=dd),
                    nn.BatchNorm2D(planes * block.expansion),
                )

        layers = []
        layers.append(block(self.inplanes, planes, stride=stride,
                            downsample=downsample, dilation=dilation))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes, dilation=dilation))

        if update: self.inplanes = int(self.inplanes / 2)  # for online
        return nn.Sequential(*layers)

    def forward(self, x, online=False):
        x = self.conv1(x)
        x = self.bn1(x)
        x_ = self.relu(x)
        x = self.maxpool(x_)

        p1 = self.layer1(x)
        p2 = self.layer2(p1)

        if online: return self.layeronline(p2)
        p3 = self.layer3(p2)
        p4 = self.layer4(p3)

        return [p1,p2], [p2,p3,p4]


class ResNet(nn.Layer):
    """
    ResNet with 22 layer utilized in CVPR2019 paper.
    Usage: ResNet(Bottleneck_CI, [3, 4], [True, False], [False, True], 64, [64, 128])
    """

    def __init__(self, block, layers, last_relus, s2p_flags, firstchannels=64, channels=[64, 128], dilation=1):
        self.inplanes = firstchannels
        self.stage_len = len(layers)
        super(ResNet, self).__init__()
        self.conv1 = nn.Conv2D(3, firstchannels, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2D(firstchannels)
        # self.relu = nn.ReLU(inplace=True)
        self.relu = nn.ReLU()
        self.maxpool = nn.MaxPool2D(kernel_size=2, stride=2)

        # stage2
        if s2p_flags[0]:
            self.layer1 = self._make_layer(block, channels[0], layers[0], stride2pool=True, last_relu=last_relus[0])
        else:
            self.layer1 = self._make_layer(block, channels[0], layers[0], last_relu=last_relus[0])

        # stage3
        if s2p_flags[1]:
            self.layer2 = self._make_layer(block, channels[1], layers[1], stride2pool=True, last_relu=last_relus[1], dilation=dilation)
        else:
            self.layer2 = self._make_layer(block, channels[1], layers[1], last_relu=last_relus[1], dilation=dilation)

        # for m in self.modules():
        #     if isinstance(m, nn.Conv2d):
        #         nn.init.kaiming_normal(m.weight, mode='fan_out')
        #     elif isinstance(m, nn.BatchNorm2d):
        #         nn.init.constant(m.weight, 1)
        #         nn.init.constant(m.bias, 0)
        for m in self.named_sublayers():
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

    def _make_layer(self, block, planes, blocks, last_relu, stride=1, stride2pool=False, dilation=1):
        """
        :param block:
        :param planes:
        :param blocks:
        :param stride:
        :param stride2pool: translate (3,2) conv to (3, 1)conv + (2, 2)pool
        :return:
        """
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2D(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2D(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, last_relu=True, stride=stride, downsample=downsample, dilation=dilation))
        if stride2pool:
            layers.append(self.maxpool)
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            if i == blocks - 1:
                layers.append(block(self.inplanes, planes, last_relu=last_relu, dilation=dilation))
            else:
                layers.append(block(self.inplanes, planes, last_relu=True, dilation=dilation))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)     # stride = 2
        x = self.bn1(x)
        x = self.relu(x)
        x = self.center_crop7(x)
        x = self.maxpool(x)   # stride = 4

        x = self.layer1(x)
        x = self.layer2(x)    # stride = 8

        return x

    def center_crop7(self, x):
        """
        Center crop layer for stage1 of resnet. (7*7)
        input x can be a Variable or Tensor
        """

        return x[:, :, 2:-2, 2:-2].contiguous()

# ----------------------
# Modules used by ATOM
# ----------------------
class FeatureBase:
    """Base feature class.
    args:
        fparams: Feature specific parameters.
        pool_stride: Amount of average pooling to apply do downsample the feature map.
        output_size: Alternatively, specify the output size of the feature map. Adaptive average pooling will be applied.
        normalize_power: The power exponent for the normalization. None means no normalization (default).
        use_for_color: Use this feature for color images.
        use_for_gray: Use this feature for grayscale images.
    """
    def __init__(self, fparams = None, pool_stride = None, output_size = None, normalize_power = None, use_for_color = True, use_for_gray = True):
        self.fparams = fparams
        self.pool_stride = 1 if pool_stride is None else pool_stride
        self.output_size = output_size
        self.normalize_power = normalize_power
        self.use_for_color = use_for_color
        self.use_for_gray = use_for_gray

    def initialize(self):
        pass

    def free_memory(self):
        pass

    def dim(self):
        raise NotImplementedError

    def stride(self):
        raise NotImplementedError

    def size(self, im_sz):
        if self.output_size is None:
            return im_sz // self.stride()
        if isinstance(im_sz, paddle.Tensor):
            return torch.Tensor([self.output_size[0], self.output_size[1]])
        return self.output_size

    def extract(self, im):
        """Performs feature extraction."""
        raise NotImplementedError

    def get_feature(self, im: paddle.Tensor):
        """Get the feature. Generally, call this function.
        args:
            im: image patch as a torch.Tensor.
        """

        # Return empty tensor if it should not be used
        is_color = im.shape[1] == 3
        if is_color and not self.use_for_color or not is_color and not self.use_for_gray:
            return torch.Tensor([])

        # Extract feature
        feat = self.extract(im)

        # Pool/downsample
        if self.output_size is not None:
            feat = F.adaptive_avg_pool2d(feat, self.output_size)
        elif self.pool_stride != 1:
            feat = F.avg_pool2d(feat, self.pool_stride, self.pool_stride)

        # Normalize
        if self.normalize_power is not None:
            feat /= (torch.sum(feat.abs().view(feat.shape[0],1,1,-1)**self.normalize_power, dim=3, keepdim=True) /
                     (feat.shape[1]*feat.shape[2]*feat.shape[3]) + 1e-10)**(1/self.normalize_power)

        return feat


class MultiFeatureBase(FeatureBase):
    """Base class for features potentially having multiple feature blocks as output (like CNNs).
    See FeatureBase for more info.
    """
    def size(self, im_sz):
        if self.output_size is None:
            return TensorList([im_sz // s for s in self.stride()])
        if isinstance(im_sz, torch.Tensor):
            return TensorList([im_sz // s if sz is None else torch.Tensor([sz[0], sz[1]]) for sz, s in zip(self.output_size, self.stride())])

    def get_feature(self, im: paddle.Tensor):
        """Get the feature. Generally, call this function.
        args:
            im: image patch as a torch.Tensor.
        """

        # Return empty tensor if it should not be used
        is_color = im.shape[1] == 3
        if is_color and not self.use_for_color or not is_color and not self.use_for_gray:
            return torch.Tensor([])

        feat_list = self.extract(im)

        output_sz = [None]*len(feat_list) if self.output_size is None else self.output_size

        # Pool/downsample
        for i, (sz, s) in enumerate(zip(output_sz, self.pool_stride)):
            if sz is not None:
                feat_list[i] = F.adaptive_avg_pool2d(feat_list[i], sz)
            elif s != 1:
                feat_list[i] = F.avg_pool2d(feat_list[i], s, s)

        # Normalize
        if self.normalize_power is not None:
            for feat in feat_list:
                feat /= (torch.sum(feat.abs().view(feat.shape[0],1,1,-1)**self.normalize_power, dim=3, keepdim=True) /
                         (feat.shape[1]*feat.shape[2]*feat.shape[3]) + 1e-10)**(1/self.normalize_power)

        return feat_list


# -----------------
# Efficient net
# -----------------
#  --------------------------------------
# Efficient net -- lighter and stronger
# -------------------------------------
class Identity(nn.Layer):
    def __init__(self, ):
        super(Identity, self).__init__()

    def forward(self, input):
        return input

class Conv2dStaticSamePadding(nn.Conv2D):
    """ 2D Convolutions like TensorFlow, for a fixed image size"""

    def __init__(self, in_channels, out_channels, kernel_size, image_size=None, **kwargs):
        super().__init__(in_channels, out_channels, kernel_size, **kwargs)
        self.stride = self.stride if len(self.stride) == 2 else [self.stride[0]] * 2

        # Calculate padding based on image size and save it
        assert image_size is not None
        ih, iw = image_size if type(image_size) == list else [image_size, image_size]
        kh, kw = self.weight.size()[-2:]
        sh, sw = self.stride
        oh, ow = math.ceil(ih / sh), math.ceil(iw / sw)
        pad_h = max((oh - 1) * self.stride[0] + (kh - 1) * self.dilation[0] + 1 - ih, 0)
        pad_w = max((ow - 1) * self.stride[1] + (kw - 1) * self.dilation[1] + 1 - iw, 0)
        if pad_h > 0 or pad_w > 0:
            self.static_padding = nn.ZeroPad2d((pad_w // 2, pad_w - pad_w // 2, pad_h // 2, pad_h - pad_h // 2))
        else:
            self.static_padding = Identity()

    def forward(self, x):
        x = self.static_padding(x)
        x = F.conv2d(x, self.weight, self.bias, self.stride, self.padding, self.dilation, self.groups)
        return x


def get_same_padding_conv2d(image_size=None):
    """ Chooses static padding if you have specified an image size, and dynamic padding otherwise.
        Static padding is necessary for ONNX exporting of models. """
    if image_size is None:
        return Conv2dDynamicSamePadding
    else:
        return partial(Conv2dStaticSamePadding, image_size=image_size)


class Conv2dDynamicSamePadding(nn.Conv2D):
    """ 2D Convolutions like TensorFlow, for a dynamic image size """

    def __init__(self, in_channels, out_channels, kernel_size, stride=1, dilation=1, groups=1, bias=True):
        super().__init__(in_channels, out_channels, kernel_size, stride, 0, dilation, groups, bias)
        self.stride = self.stride if len(self.stride) == 2 else [self.stride[0]] * 2

    def forward(self, x):
        ih, iw = x.size()[-2:]
        kh, kw = self.weight.size()[-2:]
        sh, sw = self.stride
        oh, ow = math.ceil(ih / sh), math.ceil(iw / sw)
        pad_h = max((oh - 1) * self.stride[0] + (kh - 1) * self.dilation[0] + 1 - ih, 0)
        pad_w = max((ow - 1) * self.stride[1] + (kw - 1) * self.dilation[1] + 1 - iw, 0)
        if pad_h > 0 or pad_w > 0:
            x = F.pad(x, [pad_w // 2, pad_w - pad_w // 2, pad_h // 2, pad_h - pad_h // 2])
        return F.conv2d(x, self.weight, self.bias, self.stride, self.padding, self.dilation, self.groups)

def round_filters(filters, global_params):
    """ Calculate and round number of filters based on depth multiplier. """
    multiplier = global_params.width_coefficient
    if not multiplier:
        return filters
    divisor = global_params.depth_divisor
    min_depth = global_params.min_depth
    filters *= multiplier
    min_depth = min_depth or divisor
    new_filters = max(min_depth, int(filters + divisor / 2) // divisor * divisor)
    if new_filters < 0.9 * filters:  # prevent rounding by more than 10%
        new_filters += divisor
    return int(new_filters)

def round_repeats(repeats, global_params):
    """ Round number of filters based on depth multiplier. """
    multiplier = global_params.depth_coefficient
    if not multiplier:
        return repeats
    return int(math.ceil(multiplier * repeats))

# class SwishImplementation(torch.autograd.Function):
#     @staticmethod
#     def forward(ctx, i):
#         result = i * torch.sigmoid(i)
#         ctx.save_for_backward(i)
#         return result

#     @staticmethod
#     def backward(ctx, grad_output):
#         i = ctx.saved_variables[0]
#         sigmoid_i = torch.sigmoid(i)
#         return grad_output * (sigmoid_i * (1 + i * (1 - sigmoid_i)))


class MemoryEfficientSwish(nn.Layer):
    def forward(self, x):
        return SwishImplementation.apply(x)

def drop_connect(inputs, p, training):
    """ Drop connect. """
    if not training: return inputs
    batch_size = inputs.shape[0]
    keep_prob = 1 - p
    random_tensor = keep_prob
    random_tensor += torch.rand([batch_size, 1, 1, 1], dtype=inputs.dtype, device=inputs.device)
    binary_tensor = torch.floor(random_tensor)
    output = inputs / keep_prob * binary_tensor
    return output



class MBConvBlock(nn.Layer):
    """
    Mobile Inverted Residual Bottleneck Block
    Args:
        block_args (namedtuple): BlockArgs, see above
        global_params (namedtuple): GlobalParam, see above
    Attributes:
        has_se (bool): Whether the block contains a Squeeze and Excitation layer.
    """

    def __init__(self, block_args, global_params):
        super().__init__()
        self._block_args = block_args
        self._bn_mom = 1 - global_params.batch_norm_momentum
        self._bn_eps = global_params.batch_norm_epsilon
        self.has_se = (self._block_args.se_ratio is not None) and (0 < self._block_args.se_ratio <= 1)
        self.id_skip = block_args.id_skip  # skip connection and drop connect

        # Get static or dynamic convolution depending on image size
        Conv2d = get_same_padding_conv2d(image_size=global_params.image_size)

        # Expansion phase
        inp = self._block_args.input_filters  # number of input channels
        oup = self._block_args.input_filters * self._block_args.expand_ratio  # number of output channels
        if self._block_args.expand_ratio != 1:
            self._expand_conv = Conv2d(in_channels=inp, out_channels=oup, kernel_size=1, bias=False)
            self._bn0 = nn.BatchNorm2d(num_features=oup, momentum=self._bn_mom, eps=self._bn_eps)

        # Depthwise convolution phase
        k = self._block_args.kernel_size
        s = self._block_args.stride
        self._depthwise_conv = Conv2d(
            in_channels=oup, out_channels=oup, groups=oup,  # groups makes it depthwise
            kernel_size=k, stride=s, bias=False)
        self._bn1 = nn.BatchNorm2d(num_features=oup, momentum=self._bn_mom, eps=self._bn_eps)

        # Squeeze and Excitation layer, if desired
        if self.has_se:
            num_squeezed_channels = max(1, int(self._block_args.input_filters * self._block_args.se_ratio))
            self._se_reduce = Conv2d(in_channels=oup, out_channels=num_squeezed_channels, kernel_size=1)
            self._se_expand = Conv2d(in_channels=num_squeezed_channels, out_channels=oup, kernel_size=1)

        # Output phase
        final_oup = self._block_args.output_filters
        self._project_conv = Conv2d(in_channels=oup, out_channels=final_oup, kernel_size=1, bias=False)
        self._bn2 = nn.BatchNorm2d(num_features=final_oup, momentum=self._bn_mom, eps=self._bn_eps)
        self._swish = MemoryEfficientSwish()

    def forward(self, inputs, drop_connect_rate=None):
        """
        :param inputs: input tensor
        :param drop_connect_rate: drop connect rate (float, between 0 and 1)
        :return: output of block
        """

        # Expansion and Depthwise Convolution
        x = inputs
        if self._block_args.expand_ratio != 1:
            x = self._swish(self._bn0(self._expand_conv(inputs)))
        x = self._swish(self._bn1(self._depthwise_conv(x)))

        # Squeeze and Excitation
        if self.has_se:
            x_squeezed = F.adaptive_avg_pool2d(x, 1)
            x_squeezed = self._se_expand(self._swish(self._se_reduce(x_squeezed)))
            x = torch.sigmoid(x_squeezed) * x

        x = self._bn2(self._project_conv(x))

        # Skip connection and drop connect
        input_filters, output_filters = self._block_args.input_filters, self._block_args.output_filters
        if self.id_skip and self._block_args.stride == 1 and input_filters == output_filters:
            if drop_connect_rate:
                x = drop_connect(x, p=drop_connect_rate, training=self.training)
            x = x + inputs  # skip connection
        return x

    def set_swish(self, memory_efficient=True):
        """Sets swish function as memory efficient (for training) or standard (for export)"""
        self._swish = MemoryEfficientSwish() if memory_efficient else Swish()


class BlockDecoder(object):
    """ Block Decoder for readability, straight from the official TensorFlow repository """

    @staticmethod
    def _decode_block_string(block_string):
        """ Gets a block through a string notation of arguments. """
        assert isinstance(block_string, str)

        ops = block_string.split('_')
        options = {}
        for op in ops:
            splits = re.split(r'(\d.*)', op)
            if len(splits) >= 2:
                key, value = splits[:2]
                options[key] = value

        # Check stride
        assert (('s' in options and len(options['s']) == 1) or
                (len(options['s']) == 2 and options['s'][0] == options['s'][1]))

        return BlockArgs(
            kernel_size=int(options['k']),
            num_repeat=int(options['r']),
            input_filters=int(options['i']),
            output_filters=int(options['o']),
            expand_ratio=int(options['e']),
            id_skip=('noskip' not in block_string),
            se_ratio=float(options['se']) if 'se' in options else None,
            stride=[int(options['s'][0])])

    @staticmethod
    def _encode_block_string(block):
        """Encodes a block to a string."""
        args = [
            'r%d' % block.num_repeat,
            'k%d' % block.kernel_size,
            's%d%d' % (block.strides[0], block.strides[1]),
            'e%s' % block.expand_ratio,
            'i%d' % block.input_filters,
            'o%d' % block.output_filters
        ]
        if 0 < block.se_ratio <= 1:
            args.append('se%s' % block.se_ratio)
        if block.id_skip is False:
            args.append('noskip')
        return '_'.join(args)

    @staticmethod
    def decode(string_list):
        """
        Decodes a list of string notations to specify blocks inside the network.
        :param string_list: a list of strings, each string is a notation of block
        :return: a list of BlockArgs namedtuples of block args
        """
        assert isinstance(string_list, list)
        blocks_args = []
        for block_string in string_list:
            blocks_args.append(BlockDecoder._decode_block_string(block_string))
        return blocks_args

    @staticmethod
    def encode(blocks_args):
        """
        Encodes a list of BlockArgs to a list of strings.
        :param blocks_args: a list of BlockArgs namedtuples of block args
        :return: a list of strings, each string is a notation of block
        """
        block_strings = []
        for block in blocks_args:
            block_strings.append(BlockDecoder._encode_block_string(block))
        return block_strings
    
    
def conv_ws_2d(input,
               weight,
               bias=None,
               stride=1,
               padding=0,
               dilation=1,
               groups=1,
               eps=1e-5):
    c_in = weight.size(0)
    weight_flat = weight.view(c_in, -1)
    mean = weight_flat.mean(dim=1, keepdim=True).view(c_in, 1, 1, 1)
    std = weight_flat.std(dim=1, keepdim=True).view(c_in, 1, 1, 1)
    weight = (weight - mean) / (std + eps)
    return F.conv2d(input, weight, bias, stride, padding, dilation, groups)


class ConvWS2d(nn.Conv2D):
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 stride=1,
                 padding=0,
                 dilation=1,
                 groups=1,
                 bias=True,
                 eps=1e-5):
        super(ConvWS2d, self).__init__(
            in_channels,
            out_channels,
            kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=groups,
            bias=bias)
        self.eps = eps

    def forward(self, x):
        return conv_ws_2d(x, self.weight, self.bias, self.stride, self.padding,
                          self.dilation, self.groups, self.eps)


conv_cfg = {
    'Conv': nn.Conv2D,
    'ConvWS': ConvWS2d,
    # TODO: octave conv
}


def build_conv_layer(cfg, *args, **kwargs):
    """ Build convolution layer
    Args:
        cfg (None or dict): cfg should contain:
            type (str): identify conv layer type.
            layer args: args needed to instantiate a conv layer.
    Returns:
        layer (nn.Module): created conv layer
    """
    if cfg is None:
        cfg_ = dict(type='Conv')
    else:
        assert isinstance(cfg, dict) and 'type' in cfg
        cfg_ = cfg.copy()

    layer_type = cfg_.pop('type')
    if layer_type not in conv_cfg:
        raise KeyError('Unrecognized norm type {}'.format(layer_type))
    else:
        conv_layer = conv_cfg[layer_type]

    layer = conv_layer(*args, **kwargs, **cfg_)

    return layer


norm_cfg = {
    # format: layer_type: (abbreviation, module)
    'BN': ('bn', nn.BatchNorm2D),
    'SyncBN': ('bn', nn.SyncBatchNorm),
    'GN': ('gn', nn.GroupNorm),
    # and potentially 'SN'
}


def build_norm_layer(cfg, num_features, postfix=''):
    """ Build normalization layer
    Args:
        cfg (dict): cfg should contain:
            type (str): identify norm layer type.
            layer args: args needed to instantiate a norm layer.
            requires_grad (bool): [optional] whether stop gradient updates
        num_features (int): number of channels from input.
        postfix (int, str): appended into norm abbreviation to
            create named layer.
    Returns:
        name (str): abbreviation + postfix
        layer (nn.Module): created norm layer
    """
    assert isinstance(cfg, dict) and 'type' in cfg
    cfg_ = cfg.copy()

    layer_type = cfg_.pop('type')
    if layer_type not in norm_cfg:
        raise KeyError('Unrecognized norm type {}'.format(layer_type))
    else:
        abbr, norm_layer = norm_cfg[layer_type]
        if norm_layer is None:
            raise NotImplementedError

    assert isinstance(postfix, (int, str))
    name = abbr + str(postfix)

    requires_grad = cfg_.pop('requires_grad', True)
    cfg_.setdefault('eps', 1e-5)
    if layer_type != 'GN':
        layer = norm_layer(num_features, **cfg_)
        if layer_type == 'SyncBN':
            layer._specify_ddp_gpu_num(1)
    else:
        assert 'num_groups' in cfg_
        layer = norm_layer(num_channels=num_features, **cfg_)

    for param in layer.parameters():
        param.requires_grad = requires_grad

    return name, layer

class ConvModule(nn.Layer):
    """A conv block that contains conv/norm/activation layers.
    Args:
        in_channels (int): Same as nn.Conv2d.
        out_channels (int): Same as nn.Conv2d.
        kernel_size (int or tuple[int]): Same as nn.Conv2d.
        stride (int or tuple[int]): Same as nn.Conv2d.
        padding (int or tuple[int]): Same as nn.Conv2d.
        dilation (int or tuple[int]): Same as nn.Conv2d.
        groups (int): Same as nn.Conv2d.
        bias (bool or str): If specified as `auto`, it will be decided by the
            norm_cfg. Bias will be set as True if norm_cfg is None, otherwise
            False.
        conv_cfg (dict): Config dict for convolution layer.
        norm_cfg (dict): Config dict for normalization layer.
        activation (str or None): Activation type, "ReLU" by default.
        inplace (bool): Whether to use inplace mode for activation.
        order (tuple[str]): The order of conv/norm/activation layers. It is a
            sequence of "conv", "norm" and "act". Examples are
            ("conv", "norm", "act") and ("act", "conv", "norm").
    """

    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 stride=1,
                 padding=0,
                 dilation=1,
                 groups=1,
                 bias='auto',
                 conv_cfg=None,
                 norm_cfg=None,
                 activation='relu',
                 inplace=True,
                 order=('conv', 'norm', 'act')):
        super(ConvModule, self).__init__()
        assert conv_cfg is None or isinstance(conv_cfg, dict)
        assert norm_cfg is None or isinstance(norm_cfg, dict)
        self.conv_cfg = conv_cfg
        self.norm_cfg = norm_cfg
        self.activation = activation
        self.inplace = inplace
        self.order = order
        assert isinstance(self.order, tuple) and len(self.order) == 3
        assert set(order) == set(['conv', 'norm', 'act'])

        self.with_norm = norm_cfg is not None
        self.with_activatation = activation is not None
        # if the conv layer is before a norm layer, bias is unnecessary.
        if bias == 'auto':
            bias = False if self.with_norm else True
        self.with_bias = bias

        if self.with_norm and self.with_bias:
            warnings.warn('ConvModule has norm and bias at the same time')

        # build convolution layer
        self.conv = build_conv_layer(
            conv_cfg,
            in_channels,
            out_channels,
            kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=groups,
            bias_attr =bias)
        # export the attributes of self.conv to a higher level for convenience
        self.in_channels = self.conv._in_channels
        self.out_channels = self.conv._out_channels
        self.kernel_size = self.conv._kernel_size
        self.stride = self.conv._stride
        self.padding = self.conv._padding
        self.dilation = self.conv._dilation
        # self.transposed = self.conv.transposed
        # self.output_padding = self.conv.output_padding
        self.groups = self.conv._groups

        # build normalization layers
        if self.with_norm:
            # norm layer is after conv layer
            if order.index('norm') > order.index('conv'):
                norm_channels = out_channels
            else:
                norm_channels = in_channels
            self.norm_name, norm = build_norm_layer(norm_cfg, norm_channels)
            self.add_sublayer(self.norm_name, norm)

        # build activation layer
        if self.with_activatation:
            # TODO: introduce `act_cfg` and supports more activation layers
            if self.activation not in ['relu']:
                raise ValueError('{} is currently not supported.'.format(
                    self.activation))
            if self.activation == 'relu':
                # self.activate = nn.ReLU(inplace=inplace)
                self.activate = nn.ReLU()

    @property
    def norm(self):
        return getattr(self, self.norm_name)

    def forward(self, x, activate=True, norm=True):
        for layer in self.order:
            if layer == 'conv':
                x = self.conv(x)
            elif layer == 'norm' and norm and self.with_norm:
                x = self.norm(x)
            elif layer == 'act' and activate and self.with_activatation:
                x = self.activate(x)
        return x


# def xavier_init(module, gain=1, bias=0, distribution='normal'):
#     assert distribution in ['uniform', 'normal']
#     if distribution == 'uniform':
#         nn.init.xavier_uniform_(module.weight, gain=gain)
#     else:
#         nn.init.xavier_normal_(module.weight, gain=gain)
#     if hasattr(module, 'bias'):
#         nn.init.constant_(module.bias, bias)


# def normal_init(module, mean=0, std=1, bias=0):
#     nn.init.normal_(module.weight, mean, std)
#     if hasattr(module, 'bias'):
#         nn.init.constant_(module.bias, bias)


# def uniform_init(module, a=0, b=1, bias=0):
#     nn.init.uniform_(module.weight, a, b)
#     if hasattr(module, 'bias'):
#         nn.init.constant_(module.bias, bias)


# def kaiming_init(module,
#                  mode='fan_out',
#                  nonlinearity='relu',
#                  bias=0,
#                  distribution='normal'):
#     assert distribution in ['uniform', 'normal']
#     if distribution == 'uniform':
#         nn.init.kaiming_uniform_(
#             module.weight, mode=mode, nonlinearity=nonlinearity)
#     else:
#         nn.init.kaiming_normal_(
#             module.weight, mode=mode, nonlinearity=nonlinearity)
#     if hasattr(module, 'bias'):
#         nn.init.constant_(module.bias, bias)


def bias_init_with_prob(prior_prob):
    """ initialize conv/fc bias value according to giving probablity"""
    bias_init = float(-np.log((1 - prior_prob) / prior_prob))
    return bias_init

