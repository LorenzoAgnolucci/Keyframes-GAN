from torch import nn as nn
from torch.nn.utils import spectral_norm
import numpy as np

from basicsr.utils.registry import ARCH_REGISTRY


@ARCH_REGISTRY.register()
class MultiScaleDiscriminator(nn.Module):
    def __init__(self, input_ch, base_ch=64, num_layers=4, num_discriminators=4):
        super().__init__()

        self.D_pool = nn.ModuleList()
        for i in range(num_discriminators):
            netD = NLayerDiscriminator(input_ch, base_ch, depth=num_layers)
            self.D_pool.append(netD)

        self.downsample = nn.AvgPool2d(3, stride=2, padding=[1, 1], count_include_pad=False)

    def forward(self, input):
        results = []
        for netd in self.D_pool:
            output = netd(input)
            results.append(output)
            # Downsample input
            input = self.downsample(input)
        return results


class NLayerDiscriminator(nn.Module):
    def __init__(self,
                 input_ch=3,
                 base_ch=64,
                 max_ch=512,
                 depth=3,
                 ):
        super().__init__()
        self.input_ch = input_ch
        cout = max_ch

        self.model = []
        self.model.append(spectral_norm(nn.Conv2d(input_ch, base_ch, 3, stride=1, padding=1, padding_mode="circular")))
        self.model.append(nn.BatchNorm2d(base_ch, affine=True))
        self.model.append(nn.LeakyReLU(0.2, True))
        for i in range(depth):
            cin = min(base_ch * 2 ** i, max_ch)
            cout = min(base_ch * 2 ** (i + 1), max_ch)
            self.model.append(spectral_norm(nn.Conv2d(cin, cout, 3, stride=2, padding=1, padding_mode="circular")))
            self.model.append(nn.BatchNorm2d(cout, affine=True))
            self.model.append(nn.LeakyReLU(0.2, True))
        self.model = nn.Sequential(*self.model)
        self.score_out = nn.Conv2d(cout, 1, 3, stride=1, padding=1)

    def forward(self, x):
        for idx, m in enumerate(self.model):
            x = m(x)
        x = self.score_out(x)
        return x
