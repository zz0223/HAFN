import torch.nn.functional as F
import torch.nn as nn
import torch

def weights_init(init_type='gaussian'):
    def init_fun(m):
        classname = m.__class__.__name__
        if (classname.find('Conv') == 0 or classname.find(
                'Linear') == 0) and hasattr(m, 'weight'):
            if init_type == 'gaussian':
                nn.init.normal_(m.weight, 0.0, 0.02)
            elif init_type == 'xavier':
                nn.init.xavier_normal_(m.weight, gain=math.sqrt(2))
            elif init_type == 'kaiming':
                nn.init.kaiming_normal_(m.weight, a=0, mode='fan_in')
            elif init_type == 'orthogonal':
                nn.init.orthogonal_(m.weight, gain=math.sqrt(2))
            elif init_type == 'default':
                pass
            else:
                assert 0, "Unsupported initialization: {}".format(init_type)
            if hasattr(m, 'bias') and m.bias is not None:
                nn.init.constant_(m.bias, 0.0)

    return init_fun


class Cvi(nn.Module):
    def __init__(self, in_channels, out_channels, before=None, after=False, kernel_size=4, stride=2,
                 padding=1, dilation=1, groups=1, bias=False):
        super(Cvi, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias)
        self.conv.apply(weights_init('gaussian'))

        if after=='BN':
            self.after = nn.BatchNorm2d(out_channels)
        elif after=='Tanh':
            self.after = torch.tanh
        elif after=='sigmoid':
            self.after = torch.sigmoid

        if before=='ReLU':
            self.before = nn.ReLU(inplace=True)
        elif before=='LReLU':
            self.before = nn.LeakyReLU(negative_slope=0.2, inplace=True)

    def forward(self, x):

        if hasattr(self, 'before'):
            x = self.before(x)

        x = self.conv(x)

        if hasattr(self, 'after'):
            x = self.after(x)

        return x


class CvTi(nn.Module):
    def __init__(self, in_channels, out_channels, before=None, after=False, kernel_size=4, stride=2,
                 padding=1, dilation=1, groups=1, bias=False):
        super(CvTi, self).__init__()
        # with errors: TypeError: conv_transpose2d(): argument 'output_padding' (position 6) must be tuple of ints, not tuple
        # self.conv = nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride, padding, bias)
        self.conv = nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride, padding, bias=True)
        self.conv.apply(weights_init('gaussian'))

        if after=='BN':
            self.after = nn.BatchNorm2d(out_channels)
        elif after=='Tanh':
            self.after = torch.tanh
        elif after=='sigmoid':
            self.after = torch.sigmoid

        if before=='ReLU':
            self.before = nn.ReLU(inplace=True)
        elif before=='LReLU':
            self.before = nn.LeakyReLU(negative_slope=0.2, inplace=True)

    def forward(self, x):

        if hasattr(self, 'before'):
            x = self.before(x)

        x = self.conv(x)

        if hasattr(self, 'after'):
            x = self.after(x)

        return x

class UNet(nn.Module):
    def __init__(self, input_channels=3, output_channels=1):
        super(UNet, self).__init__()

        self.Cv0 = Cvi(input_channels, 64)
        self.Cv00 = ResidualCbamBlock(64, norm=None, cbam_reduction=2)

        self.Cv1 = Cvi(64, 128, before='LReLU', after='BN', dilation=1)
        self.Cv11 = ResidualCbamBlock(128, norm=None, cbam_reduction=4)
        self.Cv111 = ResidualCbamBlock(128, norm=None, cbam_reduction=4)

        self.Cv2 = Cvi(128, 256, before='LReLU', after='BN', dilation=1)
        #self.Cv22 = ResidualCbamBlock(256, norm=None, cbam_reduction=8)
        #self.Cv222 = ResidualCbamBlock(256, norm=None, cbam_reduction=8)

        #self.Cv3 = Cvi(256, 512, before='LReLU', after='BN', dilation=1)

        #self.Cv4 = Cvi(512, 512, before='LReLU', after='BN', dilation=1)
        self.res1 = ResidualCbamBlock(256, norm=None, cbam_reduction=8)

        #self.Cv5 = Cvi(512, 512, before='LReLU', dilation=1)
        self.res2 = ResidualCbamBlock(256, norm=None, cbam_reduction=8)

        #self.CvT6 = CvTi(512, 512, before='ReLU', after='BN', dilation=1)
        self.res3 = ResidualCbamBlock(256, norm=None, cbam_reduction=8)

        #self.CvT7 = CvTi(1024, 512, before='ReLU', after='BN', dilation=1)

        #self.CvT8 = CvTi(1024, 256, before='ReLU', after='BN', dilation=1)

        self.CvT9 = CvTi(512, 128, before='ReLU', after='BN', dilation=1)

        self.CvT10 = CvTi(256, 64, before='ReLU', after='BN', dilation=1)

        self.CvT11 = CvTi(128, output_channels, before='ReLU', after='Tanh', dilation=1)

    def forward(self, input):
        # encoder
        x0 = self.Cv0(input)
        x0 = self.Cv00(x0)
        x1 = self.Cv1(x0)
        x1 = self.Cv11(x1)
        x1 = self.Cv111(x1)
        x2 = self.Cv2(x1)

        #x3 = self.Cv3(x2)


        x4_1 = self.res1(x2)
        x4_2 = self.res2(x4_1)
        x4_3 = self.res3(x4_2)
        x4_4 = self.res3(x4_3)
        x4_5 = self.res3(x4_4)
        
        # decoder
        #x6 = self.CvT6(x5)

        #cat2 = torch.cat([x4_5, x3], dim=1)
        #x8 = self.CvT8(cat2)

        cat3 = torch.cat([x4_5, x2], dim=1)
        x9 = self.CvT9(cat3)

        cat4 = torch.cat([x9, x1], dim=1)
        x10 = self.CvT10(cat4)

        cat5 = torch.cat([x10, x0], dim=1)
        out = self.CvT11(cat5)

        return out


#****************************************************************************************************************
class Conv2DLayer(nn.Sequential):
    def __init__(self, in_channels, out_channels, k_size, stride, padding=None, dilation=1, norm=None, act=None,
                 bias=False):
        super(Conv2DLayer, self).__init__()
        # use default padding value or (kernel size // 2) * dilation value
        if padding is not None:
            padding = padding
        else:
            padding = dilation * (k_size - 1) // 2

        self.add_module('conv2d',
                        nn.Conv2d(in_channels, out_channels, k_size, stride, padding, dilation=dilation, bias=bias))
        if norm is not None:
            self.add_module('norm', norm(out_channels))
        if act is not None:
            self.add_module('act', act)


class SElayer(nn.Module):
    # The SE_layer(Channel Attention.) implement, reference to:
    # Squeeze-and-Excitation Networks
    def __init__(self, channel, reduction=16):
        super(SElayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.se = nn.Sequential(
            nn.Linear(channel, channel // reduction),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.shape
        y = self.avg_pool(x).view(b, c)
        y = self.se(y).view(b, c, 1, 1)

        return x * y


class ResidualBlock(nn.Module):
    # The ResBlock implements: the conv & skip connections here.
    # Original Resnet paper: https://arxiv.org/pdf/1512.03385.pdf.
    # Which contains SE-layer implements.

    def __init__(self, channel, norm=nn.BatchNorm2d, dilation=1, bias=False, se_reduction=None, res_scale=1,
                 act=nn.ReLU(True)):
        super(ResidualBlock, self).__init__()

        self.conv1 = Conv2DLayer(channel, channel, k_size=3, stride=1, dilation=dilation, norm=norm, act=act, bias=bias)
        self.conv2 = Conv2DLayer(channel, channel, k_size=3, stride=1, dilation=dilation, norm=norm, act=None,
                                 bias=None)
        self.se_layer = None
        self.res_scale = res_scale
        if se_reduction is not None:
            self.se_layer = SElayer(channel, se_reduction)

    def forward(self, x):
        res = x
        x = self.conv1(x)
        x = self.conv2(x)
        if self.se_layer:
            x = self.se_layer(x)
        x = x * self.res_scale
        out = x + res
        return out


class ChannelAttention(nn.Module):
    # The channel attention block
    # Original relize of CBAM module.
    # Sigma(MLP(F_max^c) + MLP(F_avg^c)) -> output channel attention feature.
    def __init__(self, channel, reduction=16):
        super(ChannelAttention, self).__init__()

        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.fc_1 = nn.Conv2d(channel, channel // reduction, 1, bias=False)
        self.relu = nn.ReLU(True)
        self.fc_2 = nn.Conv2d(channel // reduction, channel, 1, bias=False)

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_output = self.fc_2(self.relu(self.fc_1(self.avg_pool(x))))
        max_output = self.fc_2(self.relu(self.fc_1(self.max_pool(x))))
        out = avg_output + max_output
        return self.sigmoid(out)


class SpatialAttention(nn.Module):
    # The spatial attention block.
    # Simgoid(conv([F_max^s; F_avg^s])) -> output spatial attention feature.
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()
        assert kernel_size in [3, 7], 'kernel size must be 3 or 7.'
        padding_size = 1 if kernel_size == 3 else 3

        self.conv = nn.Conv2d(2, 1, padding=padding_size, bias=False, kernel_size=kernel_size)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)

        pool_out = torch.cat([avg_out, max_out], dim=1)
        x = self.conv(pool_out)
        return self.sigmoid(x)


class CBAMlayer(nn.Module):
    # THe CBAM module(Channel & Spatial Attention feature) implement
    # reference from paper: CBAM(Convolutional Block Attention Module)
    def __init__(self, channel, reduction=16):
        super(CBAMlayer, self).__init__()
        self.channel_layer = ChannelAttention(channel, reduction)
        self.spatial_layer = SpatialAttention()

    def forward(self, x):
        x = self.channel_layer(x) * x
        x = self.spatial_layer(x) * x
        return x


class ResidualCbamBlock(nn.Module):
    # The ResBlock which contain CBAM attention module.

    def __init__(self, channel, norm=nn.BatchNorm2d, dilation=1, bias=False, cbam_reduction=None, act=nn.ReLU(True)):
        super(ResidualCbamBlock, self).__init__()

        self.conv1 = Conv2DLayer(channel, channel, k_size=3, stride=1, dilation=dilation, norm=norm, act=act, bias=bias)
        self.conv2 = Conv2DLayer(channel, channel, k_size=3, stride=1, dilation=dilation, norm=norm, act=None,
                                 bias=None)
        self.cbam_layer = None
        if cbam_reduction is not None:
            self.cbam_layer = CBAMlayer(channel, cbam_reduction)

    def forward(self, x):
        res = x
        x = self.conv1(x)
        x = self.conv2(x)
        if self.cbam_layer:
            x = self.cbam_layer(x)

        out = x + res
        return out