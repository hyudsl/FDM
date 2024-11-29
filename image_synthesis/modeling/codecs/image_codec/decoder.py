import torch.nn as nn
import torch.nn.functional as F

from image_synthesis.modeling.modules.basic_module import UpSample, ConvResBlock


class EncoderInPatchDecoder(nn.Module):
    def __init__(self):
        super().__init__()

        self.layers = nn.Sequential(
            nn.Conv2d(  3,  64, 3, 1, 1),
            nn.Conv2d( 64, 128, 4, 2, 1),
            nn.Conv2d(128, 256, 4, 2, 1),
            nn.Conv2d(256, 256, 4, 2, 1),
        )
        
    def forward(self, x):
        out = {}

        for l in range(len(self.layers)):# layer in self.layers:
            layer = self.layers[l]
            x = layer(x)
            x = F.relu(x)
            if not isinstance(layer, (ConvResBlock,)):
                out[str(tuple(x.shape))] = x # before activation, because other modules perform activativation first

        return out

class PatchDecoder(nn.Module):
    def __init__(self, *, 
                 in_ch,
                 res_ch,
                 out_ch=3,
                 upsample_type='deconv'
                 ):
        super().__init__()
        self.in_channels = in_ch
        self.upsample_type = upsample_type

        self.pre_layers = nn.Sequential(
            nn.Conv2d(in_ch, res_ch, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True)
        )

        self.res_layers = nn.Sequential(
            ConvResBlock(256, 256//2),
            ConvResBlock(256, 256//2),
            ConvResBlock(256, 256//2),
            ConvResBlock(256, 256//2),
            ConvResBlock(256, 256//2),
            ConvResBlock(256, 256//2),
            ConvResBlock(256, 256//2),
            ConvResBlock(256, 256//2)
        )

        # upsampling in middle layers
        post_layer_in_ch = 64

        self.up_layers = nn.Sequential(
            UpSample(256,              256, activate_before='none', activate_after='relu', upsample_type=self.upsample_type),
            UpSample(256,              128, activate_before='none', activate_after='relu', upsample_type=self.upsample_type),
            UpSample(128, post_layer_in_ch, activate_before='none', activate_after='relu', upsample_type=self.upsample_type)
        )

        self.encoder = EncoderInPatchDecoder()

        self.post_layers = nn.Sequential(
            nn.Conv2d(post_layer_in_ch, out_ch, kernel_size=3, stride=1, padding=1),
        )

    def forward(self, input, masked_image=None, mask=None):
        x = self.pre_layers(input)
        x = self.res_layers(x)

        mask = mask.to(x)
        im_x = self.encoder(masked_image)
        for l in range(len(self.up_layers)):
            if isinstance(self.up_layers[l], UpSample):
                x_ = im_x[str(tuple(x.shape))]
                mask_ = F.interpolate(mask, size=x.shape[-2:], mode='nearest')
                x = x * (1-mask_) + x_ * mask_
            x = self.up_layers[l](x)
        x = x * (1-mask) + im_x[str(tuple(x.shape))] * mask
        x = self.post_layers(x)

        return x
