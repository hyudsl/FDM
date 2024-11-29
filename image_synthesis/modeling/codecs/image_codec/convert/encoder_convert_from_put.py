from copy import deepcopy
import torch.nn as nn
import torch.nn.functional as F

from image_synthesis.modeling.modules.basic_module import DownSample, LinearResBlock, ConvResBlock
from image_synthesis.modeling.modules.original_module_backup import LinearResBlock_ORG


class PatchEncoder(nn.Module):
    def __init__(self, *, 
                in_ch=3, 
                res_ch=256, 
                out_ch=256,
                stride=8,
                ):
        super().__init__()
        in_dim = in_ch * stride * stride 
        self.stride = stride
        self.out_channels = out_ch

        self.pre_layers = nn.Sequential(
            nn.Linear(in_dim, res_ch),
        )

        self._pre_layers = nn.Sequential(
            nn.Linear(in_dim, res_ch),
            nn.ReLU(inplace=True)
        )

        self.res_layers = nn.Sequential(
            LinearResBlock_ORG(256, 256//2),
            LinearResBlock_ORG(256, 256//2),
            LinearResBlock_ORG(256, 256//2),
            LinearResBlock_ORG(256, 256//2),
            LinearResBlock_ORG(256, 256//2),
            LinearResBlock_ORG(256, 256//2),
            LinearResBlock_ORG(256, 256//2),
            LinearResBlock_ORG(256, 256//2)
        )

        self._res_layers = nn.Sequential(
            LinearResBlock(256, 256//2),
            LinearResBlock(256, 256//2),
            LinearResBlock(256, 256//2),
            LinearResBlock(256, 256//2),
            LinearResBlock(256, 256//2),
            LinearResBlock(256, 256//2),
            LinearResBlock(256, 256//2),
            LinearResBlock(256, 256//2)
        )
        
        self.post_layers = nn.Sequential(
            nn.ReLU(inplace=True),
            nn.Linear(res_ch, out_ch),
            nn.ReLU(inplace=True)
        )

        self._post_layers = nn.Sequential(
            nn.Linear(res_ch, out_ch),
            nn.ReLU(inplace=True)
        )

    def convert(self):
        self._pre_layers[0] = self.pre_layers[0]
        self.pre_layers = deepcopy(self._pre_layers)

        for i in range(len(self.res_layers)):
            self._res_layers[i].layers[0] = self.res_layers[i].layers[1]
            self._res_layers[i].layers[2] = self.res_layers[i].layers[3]
        self.res_layers = deepcopy(self._res_layers)

        self._post_layers[0] = self.post_layers[1]
        self.post_layers = deepcopy(self._post_layers)

    def forward(self, x):
        """
        x: [B, 3, H, W]

        """
        in_size = [x.shape[-2], x.shape[-1]]
        out_size = [s//self.stride for s in in_size]

        x = F.unfold(x, kernel_size=(self.stride, self.stride), stride=(self.stride, self.stride)) # B x 3*patch_size^2 x L
        x = x.permute(0, 2, 1).contiguous() # B x L x 3*patch_size

        x = self.pre_layers(x)
        x = self.res_layers(x)
        x = self.post_layers(x)

        x = x.permute(0, 2, 1).contiguous() # B x C x L
        x = F.fold(x, output_size=out_size, kernel_size=(1,1), stride=(1,1))

        return x


class PatchConvEncoder(nn.Module):
    def __init__(self, *, 
                in_ch=3, 
                res_ch=256, 
                out_ch,
                num_res_block=2, 
                num_res_block_before_resolution_change=0,
                res_block_bottleneck=2,
                stride=8,
                downsample_layer='downsample'):
        super().__init__()
        self.stride = stride
        self.out_channels = out_ch
        self.num_res_block_before_resolution_change = num_res_block_before_resolution_change

        # downsample with stride
        pre_layers = []
        in_ch_ = in_ch
        out_ch_ = 64
        while stride > 1:
            stride = stride // 2
            if stride == 1:
                out_ch_ = res_ch
            for i in range(self.num_res_block_before_resolution_change):
                pre_layers.append(
                    ConvResBlock(in_ch_, in_ch_//res_block_bottleneck)
                )
            if downsample_layer == 'downsample':
                pre_layers.append(DownSample(in_ch_, out_ch_, activate_before='none', activate_after='relu', downsample_type='conv'))
            elif downsample_layer == 'conv':
                pre_layers.append(nn.Conv2d(in_ch_, out_ch_, kernel_size=4, stride=2, padding=1))
                if stride != 1:
                    pre_layers.append(nn.ReLU(inplace=True))
            else:
                raise RuntimeError('{} not impleted!'.format(downsample_layer))
            in_ch_ = out_ch_
            out_ch_ = 2 * in_ch_
        self.pre_layers = nn.Sequential(*pre_layers)

        res_layers = []
        for i in range(num_res_block):
            res_layers.append(ConvResBlock(res_ch, res_ch//res_block_bottleneck))
        if len(res_layers) > 0:
            self.res_layers = nn.Sequential(*res_layers)
        else:
            self.res_layers = nn.Identity()

        post_layers = [
            nn.ReLU(inplace=True),
            nn.Conv2d(res_ch, out_ch, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
        ]
        self.post_layers = nn.Sequential(*post_layers)

    def forward(self, x):
        """
        x: [B, 3, H, W]

        """
        x = self.pre_layers(x)
        x = self.res_layers(x)
        x = self.post_layers(x)
        return x
