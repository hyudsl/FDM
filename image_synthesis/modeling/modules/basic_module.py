import torch
import torch.nn as nn
import torch.nn.functional as F

class UpSample(nn.Module):
    def __init__(self, in_channels, out_channels, activate_before='none', activate_after='none', upsample_type='deconv'):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.activate_before = activate_before
        self.activate_after = activate_after
        self.upsample_type = upsample_type 
        if self.upsample_type == 'deconv':
            self.deconv = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=4, stride=2, padding=1)
        else:
            assert self.upsample_type in ['bilinear', 'nearest'], 'upsample {} not implemented!'.format(self.upsample_type)
            self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1)

    def forward(self, x):

        if self.activate_before == 'relu':
            x = F.relu(x)
        elif self.activate_before == 'none':
            pass
        else:
            raise NotImplementedError

        if self.upsample_type == 'deconv':
            x = self.deconv(x)
        else:
            x = F.interpolate(x, scale_factor=2.0, mode=self.upsample_type)
            x = self.conv(x)

        if self.activate_after == 'relu':
            x = F.relu(x)
        elif self.activate_after == 'none':
            pass
        else:
            raise NotImplementedError
        return x

class DownSample(nn.Module):
    def __init__(self, in_channels, out_channels, activate_before='none', activate_after='none', downsample_type='conv'):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.activate_before = activate_before
        self.activate_after = activate_after
        self.downsample_type = downsample_type
        if self.downsample_type == 'conv':
            self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=4, stride=2, padding=1) 
        else:
            assert self.downsample_type in ['bilinear', 'nearest', 'maxpool', 'avgpool'], 'upsample {} not implemented!'.format(self.downsample_type)
            self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1)

    def forward(self, x):

        if self.activate_before == 'relu':
            x = F.relu(x)
        elif self.activate_before == 'none':
            pass
        else:
            raise NotImplementedError

        if self.downsample_type != 'conv':
            if self.downsample_type in ['nearest', 'bilinear']:
                x = F.interpolate(x, scale_factor=2.0, mode=self.downsample_type)
            elif self.downsample_type == 'maxpool':
                x = torch.max_pool2d(x, kernel_size=2, stride=2, padding=0, dilation=1)
            elif self.downsample_type == 'avgpool':
                x = torch.avg_pool2d(x, kernel_size=2, stride=2, padding=0, dilation=1)
        x = self.conv(x)

        if self.activate_after == 'relu':
            x = F.relu(x)
        elif self.activate_after == 'none':
            pass
        else:
            raise NotImplementedError
        return x
    
class LinearResBlock(nn.Module):
    def __init__(self, in_channel, channel):
        super().__init__()

        self.layers = nn.Sequential(
            nn.Linear(in_channel, channel),
            nn.ReLU(inplace=True),
            nn.Linear(channel, in_channel),
        )
        self.out_channels = in_channel
        self.in_channels = in_channel

    def forward(self, x):
        out = self.layers(x)
        out = out + x

        out = F.relu(out)

        return out
    
class ConvResBlock(nn.Module):
    def __init__(self, in_channel, channel):
        super().__init__()

        self.conv = nn.Sequential(
            nn.Conv2d(in_channel, channel, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(channel, in_channel, 1)
        )

        self.out_channels = in_channel
        self.in_channels = in_channel

    def forward(self, x):
        out = self.conv(x)
        out = out + x

        out = F.relu(out)
        
        return out

class ConvResBlock_(nn.Module):
    def __init__(self, in_ch, middle_ch, kernel_size=3, stride=1, padding=1, layer_num=2):
        super().__init__()
        layers = []

        _in_ch = in_ch
        _out_ch = middle_ch
        for i in range(layer_num):
            if i == layer_num-1:
                _out_ch = in_ch

            layers.append(nn.Conv2d(_in_ch, _out_ch, kernel_size, stride, padding))
            layers.append(nn.ReLU())

            _in_ch = middle_ch

        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        out = self.layers(x)
        out = out + x

        return out
