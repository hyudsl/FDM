import torch
import torch.nn as nn

from image_synthesis.modeling.modules.basic_module import ConvResBlock

class FDM(nn.Module):
    def __init__(self, e_dim):
        super().__init__()

        self.e_dim = e_dim

        in_ch = e_dim+1 # emb_channel + mask

        self.mix = nn.Sequential(
            nn.Conv2d(in_ch, e_dim, 3, 1, 1),
            nn.ReLU(inplace=True),
            ConvResBlock(e_dim, e_dim//2),
            ConvResBlock(e_dim, e_dim//2),
            ConvResBlock(e_dim, e_dim//2),
            ConvResBlock(e_dim, e_dim//2),
            ConvResBlock(e_dim, e_dim//2),
            ConvResBlock(e_dim, e_dim//2),
            ConvResBlock(e_dim, e_dim//2),
            ConvResBlock(e_dim, e_dim//2),
            nn.Conv2d(e_dim, e_dim, 3, 1, 1)
        )

    def forward(self, qv, masked_patch_idx):
        mix_input = torch.cat([qv, masked_patch_idx], dim=1)
        pv = self.mix(mix_input) # BxCxHxW
        pv = pv*masked_patch_idx

        return pv
