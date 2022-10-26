import math
import numpy as np
from torch import nn, cat, add
import torch.nn.functional as F

class asym_bottleneck_module(nn.Module):
    def __init__(self, in_channels, out_channels, p_drop=0.25, compress_factor=2):
        super(asym_bottleneck_module, self).__init__()
        self.bottleneck_conv = nn.Sequential(
            nn.Conv3d(in_channels=in_channels, out_channels=in_channels//compress_factor, kernel_size=(1,1,1)),
            #nn.BatchNorm3d(in_channels//compress_factor),
            nn.ReLU(inplace=True),
            nn.Dropout3d(p=p_drop),
            nn.Conv3d(in_channels=in_channels//compress_factor, out_channels=in_channels//compress_factor, kernel_size=(3,1,1), padding=(1,0,0)),
            nn.Conv3d(in_channels=in_channels//compress_factor, out_channels=in_channels//compress_factor, kernel_size=(1,3,1), padding=(0,1,0)),
            nn.Conv3d(in_channels=in_channels//compress_factor, out_channels=in_channels//compress_factor, kernel_size=(1,1,3), padding=(0,0,1)),
            #nn.BatchNorm3d(in_channels//compress_factor),
            nn.ReLU(inplace=True),
            nn.Dropout3d(p=p_drop),
            nn.Conv3d(in_channels=in_channels//compress_factor, out_channels=out_channels, kernel_size=(1,1,1)),
            #nn.BatchNorm3d(out_channels),
            nn.ReLU(inplace=True),
            nn.Dropout3d(p=p_drop),
        )
        if in_channels != out_channels:
            self.res_conv = nn.Sequential(
                nn.Conv3d(in_channels=in_channels, out_channels=out_channels, kernel_size=(1,1,1)),

            )
        else:
            self.res_conv = None
    def forward(self, x):
        if self.res_conv is not None:
            return self.bottleneck_conv(x) + self.res_conv(x)
        else:
            return self.bottleneck_conv(x) + x

class asym_module(nn.Module):
    def __init__(self, in_channels, out_channels, p_drop=0.25, compress_factor=2):
        super(asym_module, self).__init__()
        self.bottleneck_conv = nn.Sequential(
            nn.Conv3d(in_channels=in_channels, out_channels=out_channels, kernel_size=(3,1,1), padding=(1,0,0)),
            nn.Conv3d(in_channels=out_channels, out_channels=out_channels, kernel_size=(1,3,1), padding=(0,1,0)),
            nn.Conv3d(in_channels=out_channels, out_channels=out_channels, kernel_size=(1,1,3), padding=(0,0,1)),
        )
        if in_channels != out_channels:
            self.res_conv = nn.Sequential(
                nn.Conv3d(in_channels=in_channels, out_channels=out_channels, kernel_size=(1,1,1)),
            )
        else:
            self.res_conv = None
    def forward(self, x):
        if self.res_conv is not None:
            return self.bottleneck_conv(x) + self.res_conv(x)
        else:
            return self.bottleneck_conv(x) + x

class conv_bias1(nn.Module):
    def __init__(self, in_ch, out_ch, bias_size=1):
        super(conv_bias1, self).__init__()
        self.conv = nn.Conv3d(in_ch, out_ch, 3, padding=1)
        self.merge = nn.Conv3d(out_ch, bias_size, 1)
        self.gn = nn.GroupNorm(out_ch // 4, out_ch)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.gn(x)
        x = self.relu(x)
        x_bias = self.merge(x)
        return x_bias, x

class conv_bias2(nn.Module):
    def __init__(self, in_ch, out_ch, bias_size=1):
        super(conv_bias2, self).__init__()
        self.conv = asym_bottleneck_module(in_ch,out_ch)
        self.merge = nn.Conv3d(out_ch, bias_size, 1)
        self.gn = nn.GroupNorm(out_ch // 4, out_ch)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.gn(x)
        x = self.relu(x)
        x_bias = self.merge(x)
        return x_bias, x

class DenseBiasNet_base(nn.Module):
    #depth=(8, 8, 16, 16, 32, 32, 64, 64, 128, 128, 64, 64, 32, 32, 16, 16, 8, 8)
    def __init__(self, n_channels, depth=(16, 16, 32, 32, 64, 64, 128, 128, 256, 256, 128, 128, 64, 64, 32, 32, 16, 16),bias=1):
        super(DenseBiasNet_base, self).__init__()
        self.depth = depth
        self.conv0 = conv_bias1(n_channels, depth[0], bias_size=bias)

        self.conv1 = conv_bias2(depth[0], depth[1], bias_size=bias)

        in_chan = bias
        self.conv2 = conv_bias2(depth[1] + in_chan, depth[2], bias_size=bias)

        in_chan = in_chan + bias
        self.conv3 = conv_bias2(depth[2] + in_chan, depth[3], bias_size=bias)

        in_chan = in_chan + bias
        self.conv4 = conv_bias2(depth[3] + in_chan, depth[4], bias_size=bias)

        in_chan = in_chan + bias
        self.conv5 = conv_bias2(depth[4] + in_chan, depth[5], bias_size=bias)

        in_chan = in_chan + bias
        self.conv6 = conv_bias1(depth[5] + in_chan, depth[6], bias_size=bias)

        in_chan = in_chan + bias
        self.conv7 = conv_bias1(depth[6] + in_chan, depth[7], bias_size=bias)

        in_chan = in_chan + bias
        self.conv8 = conv_bias1(depth[7] + in_chan, depth[8], bias_size=bias)

        in_chan = in_chan + bias
        self.conv9 = conv_bias1(depth[8] + in_chan, depth[9], bias_size=bias)

        in_chan = in_chan + bias
        self.conv10 = conv_bias1(depth[9] + in_chan, depth[10], bias_size=bias)

        in_chan = in_chan + bias
        self.conv11 = conv_bias1(depth[10] + in_chan, depth[11], bias_size=bias)

        in_chan = in_chan + bias
        self.conv12 = conv_bias2(depth[11] + in_chan, depth[12], bias_size=bias)

        in_chan = in_chan + bias
        self.conv13 = conv_bias2(depth[12] + in_chan, depth[13], bias_size=bias)

        in_chan = in_chan + bias
        self.conv14 = conv_bias2(depth[13] + in_chan, depth[14], bias_size=bias)

        in_chan = in_chan + bias
        self.conv15 = conv_bias2(depth[14] + in_chan, depth[15], bias_size=bias)

        in_chan = in_chan + bias
        self.conv16 = conv_bias2(depth[15] + in_chan, depth[16], bias_size=bias)

        in_chan = in_chan + bias
        self.conv17 = conv_bias2(depth[16] + in_chan, depth[17], bias_size=bias)

        self.up_1_0_0 = nn.ConvTranspose3d(bias, bias, kernel_size=2, stride=2)
        self.up_1_1_0 = nn.ConvTranspose3d(bias, bias, kernel_size=2, stride=2)
        self.up_2_0_1 = nn.ConvTranspose3d(bias, bias, kernel_size=2, stride=2)
        self.up_2_0_0 = nn.ConvTranspose3d(bias, bias, kernel_size=2, stride=2)
        self.up_2_1_1 = nn.ConvTranspose3d(bias, bias, kernel_size=2, stride=2)
        self.up_2_1_0 = nn.ConvTranspose3d(bias, bias, kernel_size=2, stride=2)
        self.up_3_0_2 = nn.ConvTranspose3d(bias, bias, kernel_size=2, stride=2)
        self.up_3_0_1 = nn.ConvTranspose3d(bias, bias, kernel_size=2, stride=2)
        self.up_3_0_0 = nn.ConvTranspose3d(bias, bias, kernel_size=2, stride=2)
        self.up_3_1_2 = nn.ConvTranspose3d(bias, bias, kernel_size=2, stride=2)
        self.up_3_1_1 = nn.ConvTranspose3d(bias, bias, kernel_size=2, stride=2)
        self.up_3_1_0 = nn.ConvTranspose3d(bias, bias, kernel_size=2, stride=2)
        self.up_4_0_3 = nn.ConvTranspose3d(bias, bias, kernel_size=2, stride=2)
        self.up_4_0_2 = nn.ConvTranspose3d(bias, bias, kernel_size=2, stride=2)
        self.up_4_0_1 = nn.ConvTranspose3d(bias, bias, kernel_size=2, stride=2)
        self.up_4_0_0 = nn.ConvTranspose3d(bias, bias, kernel_size=2, stride=2)
        self.up_4_1_3 = nn.ConvTranspose3d(bias, bias, kernel_size=2, stride=2)
        self.up_4_1_2 = nn.ConvTranspose3d(bias, bias, kernel_size=2, stride=2)
        self.up_4_1_1 = nn.ConvTranspose3d(bias, bias, kernel_size=2, stride=2)
        self.up_4_1_0 = nn.ConvTranspose3d(bias, bias, kernel_size=2, stride=2)
        self.up_3_2_2 = nn.ConvTranspose3d(bias, bias, kernel_size=2, stride=2)
        self.up_3_2_1 = nn.ConvTranspose3d(bias, bias, kernel_size=2, stride=2)
        self.up_3_2_0 = nn.ConvTranspose3d(bias, bias, kernel_size=2, stride=2)
        self.up_3_3_2 = nn.ConvTranspose3d(bias, bias, kernel_size=2, stride=2)
        self.up_3_3_1 = nn.ConvTranspose3d(bias, bias, kernel_size=2, stride=2)
        self.up_3_3_0 = nn.ConvTranspose3d(bias, bias, kernel_size=2, stride=2)
        self.up_2_2_1 = nn.ConvTranspose3d(bias, bias, kernel_size=2, stride=2)
        self.up_2_2_0 = nn.ConvTranspose3d(bias, bias, kernel_size=2, stride=2)
        self.up_2_3_1 = nn.ConvTranspose3d(bias, bias, kernel_size=2, stride=2)
        self.up_2_3_0 = nn.ConvTranspose3d(bias, bias, kernel_size=2, stride=2)
        self.up_1_2_0 = nn.ConvTranspose3d(bias, bias, kernel_size=2, stride=2)
        self.up_1_3_0 = nn.ConvTranspose3d(bias, bias, kernel_size=2, stride=2)

        self.down_0_0_1 = nn.Conv3d(bias, bias, kernel_size=2, stride=2)
        self.down_0_0_2 = nn.Conv3d(bias, bias, kernel_size=2, stride=2)
        self.down_0_0_3 = nn.Conv3d(bias, bias, kernel_size=2, stride=2)
        self.down_0_0_4 = nn.Conv3d(bias, bias, kernel_size=2, stride=2)

        self.down_0_1_1 = nn.Conv3d(bias, bias, kernel_size=2, stride=2)
        self.down_0_1_2 = nn.Conv3d(bias, bias, kernel_size=2, stride=2)
        self.down_0_1_3 = nn.Conv3d(bias, bias, kernel_size=2, stride=2)
        self.down_0_1_4 = nn.Conv3d(bias, bias, kernel_size=2, stride=2)

        self.down_1_0_2 = nn.Conv3d(bias, bias, kernel_size=2, stride=2)
        self.down_1_0_3 = nn.Conv3d(bias, bias, kernel_size=2, stride=2)
        self.down_1_0_4 = nn.Conv3d(bias, bias, kernel_size=2, stride=2)

        self.down_1_1_2 = nn.Conv3d(bias, bias, kernel_size=2, stride=2)
        self.down_1_1_3 = nn.Conv3d(bias, bias, kernel_size=2, stride=2)
        self.down_1_1_4 = nn.Conv3d(bias, bias, kernel_size=2, stride=2)

        self.down_2_0_3 = nn.Conv3d(bias, bias, kernel_size=2, stride=2)
        self.down_2_0_4 = nn.Conv3d(bias, bias, kernel_size=2, stride=2)

        self.down_2_1_3 = nn.Conv3d(bias, bias, kernel_size=2, stride=2)
        self.down_2_1_4 = nn.Conv3d(bias, bias, kernel_size=2, stride=2)

        self.down_3_0_4 = nn.Conv3d(bias, bias, kernel_size=2, stride=2)

        self.down_3_1_4 = nn.Conv3d(bias, bias, kernel_size=2, stride=2)

        self.maxpooling = nn.MaxPool3d(2)
        self.up = nn.Upsample(scale_factor=2, mode='trilinear', align_corners=True)
    def forward(self, x):
        # block0
        x_bias_0_0_0, x = self.conv0(x)
        x_bias_0_1_0, x = self.conv1(x)

        x_bias_0_0_1 = self.down_0_0_1(x_bias_0_0_0)
        x_bias_0_0_2 = self.down_0_0_2(x_bias_0_0_1)
        x_bias_0_0_3 = self.down_0_0_3(x_bias_0_0_2)
        x_bias_0_0_4 = self.down_0_0_4(x_bias_0_0_3)

        x_bias_0_1_1 = self.down_0_1_1(x_bias_0_1_0)
        x_bias_0_1_2 = self.down_0_1_2(x_bias_0_1_1)
        x_bias_0_1_3 = self.down_0_1_3(x_bias_0_1_2)
        x_bias_0_1_4 = self.down_0_1_4(x_bias_0_1_3)

        # block1
        x = self.maxpooling(x)
        x_bias_1_0_1, x = self.conv2(cat([x, x_bias_0_0_1], dim=1))
        x_bias_1_1_1, x = self.conv3(cat([x, x_bias_0_0_1, x_bias_0_1_1], dim=1))

        x_bias_1_0_0 = self.up_1_0_0(x_bias_1_0_1)
        x_bias_1_0_2 = self.down_1_0_2(x_bias_1_0_1)
        x_bias_1_0_3 = self.down_1_0_3(x_bias_1_0_2)
        x_bias_1_0_4 = self.down_1_0_4(x_bias_1_0_3)

        x_bias_1_1_0 = self.up_1_1_0(x_bias_1_1_1)
        x_bias_1_1_2 = self.down_1_1_2(x_bias_1_1_1)
        x_bias_1_1_3 = self.down_1_1_3(x_bias_1_1_2)
        x_bias_1_1_4 = self.down_1_1_4(x_bias_1_1_3)

        # block2
        x = self.maxpooling(x)
        x_bias_2_0_2, x = self.conv4(cat([x, x_bias_0_0_2, x_bias_0_1_2, x_bias_1_0_2], dim=1))
        x_bias_2_1_2, x = self.conv5(cat([x, x_bias_0_0_2, x_bias_0_1_2, x_bias_1_0_2, x_bias_1_1_2], dim=1))

        x_bias_2_0_1 = self.up_2_0_1(x_bias_2_0_2)
        x_bias_2_0_0 = self.up_2_0_0(x_bias_2_0_1)
        x_bias_2_0_3 = self.down_2_0_3(x_bias_2_0_2)
        x_bias_2_0_4 = self.down_2_0_4(x_bias_2_0_3)

        x_bias_2_1_1 = self.up_2_1_1(x_bias_2_1_2)
        x_bias_2_1_0 = self.up_2_1_0(x_bias_2_1_1)
        x_bias_2_1_3 = self.down_2_1_3(x_bias_2_1_2)
        x_bias_2_1_4 = self.down_2_1_4(x_bias_2_1_3)

        # block3
        x = self.maxpooling(x)
        x_bias_3_0_3, x = self.conv6(
            cat([x, x_bias_0_0_3, x_bias_0_1_3, x_bias_1_0_3, x_bias_1_1_3, x_bias_2_0_3], dim=1))
        x_bias_3_1_3, x = self.conv7(cat([x, x_bias_0_0_3, x_bias_0_1_3, x_bias_1_0_3, x_bias_1_1_3, x_bias_2_0_3,
                                          x_bias_2_1_3], dim=1))

        x_bias_3_0_2 = self.up_3_0_2(x_bias_3_0_3)
        x_bias_3_0_1 = self.up_3_0_1(x_bias_3_0_2)
        x_bias_3_0_0 = self.up_3_0_0(x_bias_3_0_1)
        x_bias_3_0_4 = self.down_3_0_4(x_bias_3_0_3)

        x_bias_3_1_2 = self.up_3_1_2(x_bias_3_1_3)
        x_bias_3_1_1 = self.up_3_1_1(x_bias_3_1_2)
        x_bias_3_1_0 = self.up_3_1_0(x_bias_3_1_1)
        x_bias_3_1_4 = self.down_3_1_4(x_bias_3_1_3)

        # block4
        x = self.maxpooling(x)
        x_bias_4_0_4, x = self.conv8(
            cat([x, x_bias_0_0_4, x_bias_0_1_4, x_bias_1_0_4, x_bias_1_1_4, x_bias_2_0_4, x_bias_2_1_4, x_bias_3_0_4],
                dim=1))
        x_bias_4_1_4, x = self.conv9(cat([x, x_bias_0_0_4, x_bias_0_1_4, x_bias_1_0_4, x_bias_1_1_4, x_bias_2_0_4,
                                          x_bias_2_1_4, x_bias_3_0_4, x_bias_3_1_4], dim=1))

        x_bias_4_0_3 = self.up_4_0_3(x_bias_4_0_4)
        x_bias_4_0_2 = self.up_4_0_2(x_bias_4_0_3)
        x_bias_4_0_1 = self.up_4_0_1(x_bias_4_0_2)
        x_bias_4_0_0 = self.up_4_0_0(x_bias_4_0_1)

        x_bias_4_1_3 = self.up_4_1_3(x_bias_4_1_4)
        x_bias_4_1_2 = self.up_4_1_2(x_bias_4_1_3)
        x_bias_4_1_1 = self.up_4_1_1(x_bias_4_1_2)
        x_bias_4_1_0 = self.up_4_1_0(x_bias_4_1_1)

        # block5
        x = self.up(x)
        x_bias_3_2_3, x = self.conv10(
            cat([x, x_bias_0_0_3, x_bias_0_1_3, x_bias_1_0_3, x_bias_1_1_3, x_bias_2_0_3, x_bias_2_1_3, x_bias_3_0_3,
                 x_bias_3_1_3, x_bias_4_0_3], dim=1))
        x_bias_3_3_3, x = self.conv11(
            cat([x, x_bias_0_0_3, x_bias_0_1_3, x_bias_1_0_3, x_bias_1_1_3, x_bias_2_0_3, x_bias_2_1_3, x_bias_3_0_3,
                 x_bias_3_1_3, x_bias_4_0_3, x_bias_4_1_3], dim=1))

        x_bias_3_2_2 = self.up_3_2_2(x_bias_3_2_3)
        x_bias_3_2_1 = self.up_3_2_1(x_bias_3_2_2)
        x_bias_3_2_0 = self.up_3_2_0(x_bias_3_2_1)

        x_bias_3_3_2 = self.up_3_3_2(x_bias_3_3_3)
        x_bias_3_3_1 = self.up_3_3_1(x_bias_3_3_2)
        x_bias_3_3_0 = self.up_3_3_0(x_bias_3_3_1)

        # block6
        x = self.up(x)
        x_bias_2_2_2, x = self.conv12(cat([x, x_bias_0_0_2, x_bias_0_1_2, x_bias_1_0_2, x_bias_1_1_2, x_bias_2_0_2,
                                          x_bias_2_1_2, x_bias_3_0_2, x_bias_3_1_2, x_bias_4_0_2, x_bias_4_1_2,
                                          x_bias_3_2_2], dim=1))
        x_bias_2_3_2, x = self.conv13(cat([x, x_bias_0_0_2, x_bias_0_1_2, x_bias_1_0_2, x_bias_1_1_2, x_bias_2_0_2,
                                          x_bias_2_1_2, x_bias_3_0_2, x_bias_3_1_2, x_bias_4_0_2, x_bias_4_1_2,
                                          x_bias_3_2_2, x_bias_3_3_2], dim=1))

        x_bias_2_2_1 = self.up_2_2_1(x_bias_2_2_2)
        x_bias_2_2_0 = self.up_2_2_0(x_bias_2_2_1)

        x_bias_2_3_1 = self.up_2_3_1(x_bias_2_3_2)
        x_bias_2_3_0 = self.up_2_3_0(x_bias_2_3_1)

        # block7
        x = self.up(x)
        x_bias_1_2_1, x = self.conv14(cat([x, x_bias_0_0_1, x_bias_0_1_1, x_bias_1_0_1, x_bias_1_1_1, x_bias_2_0_1,
                                           x_bias_2_1_1, x_bias_3_0_1, x_bias_3_1_1, x_bias_4_0_1, x_bias_4_1_1,
                                           x_bias_3_2_1, x_bias_3_3_1, x_bias_2_2_1], dim=1))
        x_bias_1_3_1, x = self.conv15(cat([x, x_bias_0_0_1, x_bias_0_1_1, x_bias_1_0_1, x_bias_1_1_1, x_bias_2_0_1,
                                           x_bias_2_1_1, x_bias_3_0_1, x_bias_3_1_1, x_bias_4_0_1, x_bias_4_1_1,
                                           x_bias_3_2_1, x_bias_3_3_1, x_bias_2_2_1, x_bias_2_3_1], dim=1))

        x_bias_1_2_0 = self.up_1_2_0(x_bias_1_2_1)
        x_bias_1_3_0 = self.up_1_3_0(x_bias_1_3_1)

        # block8
        x = self.up(x)
        x_bias_0_2_0, x = self.conv16(cat([x, x_bias_0_0_0, x_bias_0_1_0, x_bias_1_0_0, x_bias_1_1_0, x_bias_2_0_0,
                                           x_bias_2_1_0, x_bias_3_0_0, x_bias_3_1_0, x_bias_4_0_0, x_bias_4_1_0,
                                           x_bias_3_2_0, x_bias_3_3_0, x_bias_2_2_0, x_bias_2_3_0, x_bias_1_2_0],
                                          dim=1))

        x_bias_0_3_0, x = self.conv17(cat([x, x_bias_0_0_0, x_bias_0_1_0, x_bias_1_0_0, x_bias_1_1_0, x_bias_2_0_0,
                                           x_bias_2_1_0, x_bias_3_0_0, x_bias_3_1_0, x_bias_4_0_0, x_bias_4_1_0,
                                           x_bias_3_2_0, x_bias_3_3_0, x_bias_2_2_0, x_bias_2_3_0, x_bias_1_2_0,
                                           x_bias_1_3_0], dim=1))

        return x

class DenseBiasNet(nn.Module):
    def __init__(self, n_channels, n_classes, depth=(16, 16, 32, 32, 64, 64, 128, 128, 256, 256, 128, 128, 64, 64, 32, 32, 16, 16), bias=4):
        super(DenseBiasNet, self).__init__()
        self.densebisanet = DenseBiasNet_base(n_channels, depth, bias)
        self.out_conv = nn.Conv3d(depth[-1], n_classes, 1)
        self.softmax = nn.Softmax(dim=1)
    def forward(self, x):
        Z = x.size()[2]
        Y = x.size()[3]
        X = x.size()[4]
        diffZ = (16 - x.size()[2] % 16) % 16
        diffY = (16 - x.size()[3] % 16) % 16
        diffX = (16 - x.size()[4] % 16) % 16

        x = F.pad(x, [diffX // 2, diffX - diffX // 2,
                      diffY // 2, diffY - diffY // 2,
                      diffZ // 2, diffZ - diffZ // 2])

        x = self.densebisanet(x)
        x = self.out_conv(x)
        x = self.softmax(x)
        return x[:, :, diffZ//2: Z+diffZ//2, diffY//2: Y+diffY//2, diffX // 2:X + diffX // 2]
