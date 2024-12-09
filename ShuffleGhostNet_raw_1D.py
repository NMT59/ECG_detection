import math

import torch
import torch.nn as nn
import torch.nn.functional as F


####################################################################################################
# custom ghost net v1
class SE1DLayer(nn.Module):
    def __init__(self, channel, reduction=16):
        super(SE1DLayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1)
        return x * y


class ECA1DLayer(nn.Module):
    def __init__(self, channel, k_size=3):
        super(ECA1DLayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.conv = nn.Conv1d(1, 1, kernel_size=k_size, padding=(k_size - 1) // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # feature descriptor on the global spatial information
        b, c, _ = x.size()
        y = self.avg_pool(x).view(b, 1, c)

        # Two different branches of ECA module
        y = self.conv(y).transpose(-1, -2)

        # Multi-scale information fusion
        y = self.sigmoid(y)

        return x * y.expand_as(x)


class DepthwiseSeparableConvolution1D(nn.Module):
    def __init__(self, nin, nout, kernel_size=3, stride=1):
        super(DepthwiseSeparableConvolution1D, self).__init__()
        self.depthwise = nn.Sequential(
            nn.Conv1d(nin, nin, kernel_size=kernel_size, stride=stride, padding=(kernel_size - 1) // 2,
                      groups=nin,
                      bias=False),
            nn.BatchNorm1d(nin),
            nn.LeakyReLU(inplace=True),
        )
        self.pointwise = nn.Sequential(
            nn.Conv1d(nin, nout, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm1d(nout),
            nn.LeakyReLU(inplace=True),
        )

    def forward(self, x):
        x = self.depthwise(x)
        x = self.pointwise(x)

        return x


class GhostModule(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=1, ratio=2, dw_size=3, stride=1, use_act=True):
        super(GhostModule, self).__init__()
        self.out_channels = out_channels
        init_channels = math.ceil(out_channels / ratio)
        new_channels = init_channels * (ratio - 1)

        self.primary_conv = nn.Sequential(
            nn.Conv1d(in_channels, init_channels, kernel_size=kernel_size, stride=stride,
                      padding=(kernel_size - 1) // 2, bias=False),
            nn.BatchNorm1d(init_channels),
            # nn.PReLU() if use_act else nn.Sequential(),
            nn.LeakyReLU(inplace=True) if use_act else nn.Sequential(),
        )

        self.cheap_operation = nn.Sequential(
            nn.Conv1d(init_channels, new_channels, kernel_size=dw_size, stride=1, padding=(dw_size - 1) // 2,
                      groups=init_channels, bias=False),
            nn.BatchNorm1d(new_channels),
            # nn.PReLU() if use_act else nn.Sequential(),
            nn.LeakyReLU(inplace=True) if use_act else nn.Sequential(),
        )

    def forward(self, x):
        x1 = self.primary_conv(x)
        x2 = self.cheap_operation(x1)
        # print(x.shape, x1.shape, x2.shape)
        out = torch.cat([x1, x2], dim=1)
        # print(out.shape, self.out_channels, x.shape, x1.shape, x2.shape)

        return out[:, :self.out_channels, :]
        # return out


class ShuffleGhostBottleneck(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channel, kernel_size, stride, use_se=False, shuffle=False):
        super(ShuffleGhostBottleneck, self).__init__()
        assert stride in [1, 2]

        self.shuffle = ShuffleBlock(groups=2) if shuffle else nn.Sequential()

        self.conv = nn.Sequential(
            # pw
            GhostModule(in_channels, hidden_channels, kernel_size=1, use_act=True),

            # dw
            nn.Sequential(
                nn.Conv1d(hidden_channels, hidden_channels, kernel_size=kernel_size, stride=stride,
                          padding=(kernel_size - 1) // 2, groups=hidden_channels, bias=False),
                nn.BatchNorm1d(hidden_channels),
            ) if stride == 2 else nn.Sequential(),

            # Squeeze-and-Excite
            ECA1DLayer(hidden_channels) if use_se else nn.Sequential(),

            # pw-linear
            GhostModule(hidden_channels, out_channel, kernel_size=1, use_act=False),
        )

        if in_channels == out_channel and stride == 1:
            self.shortcut = lambda x: x
        else:
            self.shortcut = DepthwiseSeparableConvolution1D(in_channels, out_channel, kernel_size=1, stride=stride)

        self.act_fn = nn.LeakyReLU(inplace=True)

    def forward(self, x):
        residual = self.shortcut(x)
        x = self.conv(self.shuffle(x))

        return x + residual


class ShuffleBlock(nn.Module):
    def __init__(self, groups):
        super(ShuffleBlock, self).__init__()
        self.groups = groups

    def forward(self, x):
        # print(x.size())
        N, C, L = x.size()
        g = self.groups

        return x.view(N, g, C // g, L).permute(0, 2, 1, 3).reshape(N, C, L)


class NetworkCore(nn.Module):
    def __init__(self, n_classes=26):
        super(NetworkCore, self).__init__()
        self.projection = nn.Sequential(
            nn.Conv1d(12, 32, kernel_size=15, stride=2, padding=7, bias=False),
            nn.BatchNorm1d(32),
            nn.LeakyReLU(inplace=True),
        )

        self.layers = nn.Sequential(
            # stage 1
            ShuffleGhostBottleneck(32, 48, 32, kernel_size=3, stride=2, use_se=False),
            ShuffleGhostBottleneck(32, 64, 32, kernel_size=3, stride=1, use_se=True, shuffle=True),
            # stage 2
            ShuffleGhostBottleneck(32, 96, 64, kernel_size=3, stride=2, use_se=True),
            ShuffleGhostBottleneck(64, 128, 64, kernel_size=3, stride=1, use_se=True, shuffle=True),
            ShuffleGhostBottleneck(64, 160, 64, kernel_size=3, stride=1, use_se=False),
            ShuffleGhostBottleneck(64, 192, 64, kernel_size=3, stride=1, use_se=True, shuffle=True),
            # stage 3
            ShuffleGhostBottleneck(64, 144, 96, kernel_size=5, stride=2, use_se=True),
            ShuffleGhostBottleneck(96, 192, 96, kernel_size=5, stride=1, use_se=True, shuffle=True),
            ShuffleGhostBottleneck(96, 240, 96, kernel_size=5, stride=1, use_se=False),
            ShuffleGhostBottleneck(96, 248, 96, kernel_size=5, stride=1, use_se=True, shuffle=True),
            # stage 4
            ShuffleGhostBottleneck(96, 192, 128, kernel_size=3, stride=2, use_se=True),
            ShuffleGhostBottleneck(128, 256, 128, kernel_size=3, stride=1, use_se=True, shuffle=True),
            ShuffleGhostBottleneck(128, 320, 128, kernel_size=3, stride=1, use_se=False),
            ShuffleGhostBottleneck(128, 384, 128, kernel_size=3, stride=1, use_se=True, shuffle=True),
            # stage 5
            ShuffleGhostBottleneck(128, 512, 256, kernel_size=5, stride=2, use_se=True),
            ShuffleGhostBottleneck(256, 512, 256, kernel_size=5, stride=1, use_se=True, shuffle=True),
        )

        self.mha = nn.MultiheadAttention(256, 8)
        self.pool = nn.AdaptiveMaxPool1d(output_size=1)

        self.mlp_head = nn.Linear(256, n_classes)

    def forward(self, x):
        x = self.projection(x)
        x = self.layers(x)
        x = F.dropout(x, p=0.5, training=self.training)
        x = x.squeeze(2).permute(2, 0, 1)
        x, s = self.mha(x, x, x)
        x = x.permute(1, 2, 0)
        x = self.pool(x).squeeze(2)
        x = self.mlp_head(x)

        return x


def init_weights(m):
    if isinstance(m, nn.Linear):
        # we use xavier_uniform following official JAX ViT:
        torch.nn.init.xavier_uniform_(m.weight)
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)
    elif isinstance(m, nn.LayerNorm):
        nn.init.constant_(m.bias, 0)
        nn.init.constant_(m.weight, 1.0)
    elif isinstance(m, nn.Conv1d) or isinstance(m, nn.ConvTranspose1d):
        w = m.weight.data
        torch.nn.init.xavier_uniform_(w.view([w.shape[0], -1]))


if __name__ == '__main__':
    x = torch.rand(2, 12, 8192)
    model = NetworkCore(n_classes=10)
    print(sum(p.numel() for p in model.parameters() if p.requires_grad))
    outputs = model(x)
    print(outputs.shape)
