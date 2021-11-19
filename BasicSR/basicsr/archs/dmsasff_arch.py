import torch
from torch import nn
import numpy as np

from basicsr.utils.registry import ARCH_REGISTRY
from basicsr.archs.arch_util import default_init_weights
from .vgg_arch import VGGFeatureExtractor
from .dfdnet_util import Blur, MSDilationBlock, UpResBlock
import torch.nn.utils.spectral_norm as SpectralNorm



class SFTUpBlock(nn.Module):
    """Spatial feature transform (SFT) with upsampling block."""

    def __init__(self, in_channel, out_channel, kernel_size=3, padding=1):
        super(SFTUpBlock, self).__init__()
        self.conv1 = nn.Sequential(
            Blur(in_channel),
            SpectralNorm(nn.Conv2d(in_channel, out_channel, kernel_size, padding=padding, padding_mode="circular")),
            nn.LeakyReLU(0.04, True),
            # The official codes use two LeakyReLU here, so 0.04 for equivalent
        )
        self.convup = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            SpectralNorm(nn.Conv2d(out_channel, out_channel, kernel_size, padding=padding, padding_mode="circular")),
            nn.LeakyReLU(0.2, True),
        )

        # for SFT scale and shift
        self.scale_block = nn.Sequential(
            SpectralNorm(nn.Conv2d(in_channel, out_channel, 3, 1, 1, padding_mode="circular")), nn.LeakyReLU(0.2, True),
            SpectralNorm(nn.Conv2d(out_channel, out_channel, 3, 1, 1, padding_mode="circular")))
        self.shift_block = nn.Sequential(
            SpectralNorm(nn.Conv2d(in_channel, out_channel, 3, 1, 1, padding_mode="circular")), nn.LeakyReLU(0.2, True),
            SpectralNorm(nn.Conv2d(out_channel, out_channel, 3, 1, 1, padding_mode="circular")), nn.Sigmoid())
        # The official codes use sigmoid for shift block, do not know why

    def forward(self, x, updated_feat):
        out = self.conv1(x)
        # SFT
        scale = self.scale_block(updated_feat)
        shift = self.shift_block(updated_feat)
        out = out * scale + shift
        # upsample
        out = self.convup(out)
        return out

def AdaIN(compressed_features, reference_features, eps=1e-5):
    """
    Implements Adaptive Instance Normalization for style transfer

    :param compressed_features: features extracted from compressed reference_image (that constitute the style)
    :param reference_features: features extracted from reference reference_image
    :param eps: small value to avoid division by zero
    :return: normalized reference features
    """

    B, C, H, W = compressed_features.shape
    compressed_features = compressed_features.view(B, C, -1)
    compressed_mean = torch.mean(compressed_features, dim=2).view(B, C, 1)
    compressed_std = (torch.std(compressed_features, dim=2) + eps).view(B, C, 1)

    B, C, H, W = reference_features.shape
    reference_features = reference_features.view(B, C, -1)
    reference_mean = torch.mean(reference_features, dim=2).view(B, C, 1)
    reference_std = (torch.std(reference_features, dim=2) + eps).view(B, C, 1)

    adain = compressed_std * (reference_features - reference_mean) / reference_std + compressed_mean
    adain = adain.view(B, C, H, W)

    return adain


class ModifiedASFFBlock(nn.Module):
    """

    Adaptive Spatial Feature Fusion block of EASFFNet

    """

    def __init__(self, num_channels):
        super(ModifiedASFFBlock, self).__init__()
        self.attention_compressed_conv = nn.Conv2d(num_channels, num_channels // 2, 1, 1)
        self.attention_reference_conv = nn.Conv2d(num_channels, num_channels // 2, 1, 1)
        self.attention_landmark_conv = nn.Conv2d(num_channels, num_channels // 2, 1, 1)

        self.attention_concatenated_conv1 = nn.Conv2d(num_channels // 2 * 3, num_channels, 3, 1, padding=1, bias=False,
                                                      padding_mode="circular")
        self.attention_concatenated_bn1 = nn.BatchNorm2d(num_channels)
        self.attention_concatenated_conv2 = nn.Conv2d(num_channels, num_channels, 3, 1, padding=1, bias=False,
                                                      padding_mode="circular")
        self.attention_concatenated_bn2 = nn.BatchNorm2d(num_channels)

        self.compressed_conv1 = nn.Conv2d(num_channels, num_channels, 3, 1, padding=1, padding_mode="circular")
        self.compressed_bn = nn.BatchNorm2d(num_channels)
        self.compressed_conv2 = nn.Conv2d(num_channels, num_channels, 1, 1)

        self.reference_conv1 = nn.Conv2d(num_channels, num_channels, 3, 1, padding=1, padding_mode="circular")
        self.reference_bn = nn.BatchNorm2d(num_channels)
        self.reference_conv2 = nn.Conv2d(num_channels, num_channels, 1, 1)

        self.lrelu = nn.LeakyReLU(0.2, inplace=True)

        default_init_weights([self.attention_concatenated_conv1, self.attention_concatenated_bn1,
                              self.attention_concatenated_conv2, self.attention_concatenated_bn2,
                              self.compressed_conv1, self.compressed_bn, self.compressed_conv2,
                              self.reference_conv1, self.reference_bn, self.reference_conv2], 0.1)

    def forward(self, compressed, reference, landmark):
        attention_compressed = self.attention_compressed_conv(compressed)
        attention_reference = self.attention_reference_conv(reference)
        attention_landmark = self.attention_landmark_conv(landmark)

        attention_concatenated = torch.cat((attention_compressed, attention_reference, attention_landmark), dim=1)
        attention_mask = self.lrelu(
            self.attention_concatenated_bn1(self.attention_concatenated_conv1(attention_concatenated)))
        attention_mask = self.lrelu(self.attention_concatenated_bn2(self.attention_concatenated_conv2(attention_mask)))

        compressed_feat = self.compressed_conv2(self.lrelu(self.compressed_bn(self.compressed_conv1(compressed))))
        reference_feat = self.reference_conv2(self.lrelu(self.reference_bn(self.reference_conv1(reference))))

        out = torch.add(torch.mul(torch.sub(reference_feat, compressed_feat), attention_mask), compressed_feat)
        return out


# @ARCH_REGISTRY.register()
class DMSASFFNet(nn.Module):
    """
    Network for artifact reduction with exemplar reference_image

    DMSASFFNet: Deep Multi-Scale Adaptive Spatial Feature Fusion Network

    """

    def __init__(self):
        super(DMSASFFNet, self).__init__()
        self.num_feat = 64

        self.vgg_layers = ['relu2_2', 'relu3_4', 'relu4_4', 'conv5_4']
        channel_sizes = [128, 256, 512, 512]
        self.feature_sizes = np.array([256, 128, 64, 32])

        # vgg face extractor
        self.vgg_extractor = VGGFeatureExtractor(
            layer_name_list=self.vgg_layers,
            vgg_type='vgg19',
            use_input_norm=True,
            range_norm=True,
            requires_grad=False)

        self.MASFF_blocks = nn.ModuleDict()
        for i, vgg_layer in enumerate(self.vgg_layers):
            self.MASFF_blocks[f"{i}"] = ModifiedASFFBlock(channel_sizes[i])

        # multi scale dilation block
        self.multi_scale_dilation = MSDilationBlock(self.num_feat * 8, dilation=[4, 3, 2, 1])

        # upsampling and reconstruction
        self.upsample0 = SFTUpBlock(self.num_feat * 8, self.num_feat * 8)
        self.upsample1 = SFTUpBlock(self.num_feat * 8, self.num_feat * 4)
        self.upsample2 = SFTUpBlock(self.num_feat * 4, self.num_feat * 2)
        self.upsample3 = SFTUpBlock(self.num_feat * 2, self.num_feat)
        self.upsample4 = nn.Sequential(
            SpectralNorm(nn.Conv2d(self.num_feat, self.num_feat, 3, 1, 1, padding_mode="circular")), nn.LeakyReLU(0.2, True), UpResBlock(self.num_feat),
            UpResBlock(self.num_feat), nn.Conv2d(self.num_feat, 3, kernel_size=3, stride=1, padding=1, padding_mode="circular"))

    def forward(self, compressed, reference, landmark):
        vgg_compressed_features = self.vgg_extractor(compressed)
        vgg_reference_features = self.vgg_extractor(reference)
        vgg_landmark_features = self.vgg_extractor(landmark)

        feature_fusion = []
        for i, vgg_layer in enumerate(self.vgg_layers):
            vgg_compressed_feat = vgg_compressed_features[vgg_layer]
            vgg_reference_feat = AdaIN(vgg_compressed_feat, vgg_reference_features[vgg_layer])
            vgg_landmark_feat = vgg_landmark_features[vgg_layer]

            feature_fusion.append(self.MASFF_blocks[f"{i}"](vgg_compressed_feat, vgg_reference_feat, vgg_landmark_feat))

        dilated_vgg_compressed_feat = self.multi_scale_dilation(vgg_compressed_features['conv5_4'])

        upsampled_feat = self.upsample0(dilated_vgg_compressed_feat, feature_fusion[3])
        upsampled_feat = self.upsample1(upsampled_feat, feature_fusion[2])
        upsampled_feat = self.upsample2(upsampled_feat, feature_fusion[1])
        upsampled_feat = self.upsample3(upsampled_feat, feature_fusion[0])
        out = self.upsample4(upsampled_feat)

        return out + compressed

    # VGG layer relu2_2 shape: torch.Size([1, 128, 256, 256])
    # VGG layer relu3_4 shape: torch.Size([1, 256, 128, 128])
    # VGG layer relu4_4 shape: torch.Size([1, 512, 64, 64])
    # VGG layer conv5_4 shape: torch.Size([1, 512, 32, 32])