import torch
from torch import nn

from basicsr.utils.registry import ARCH_REGISTRY
from basicsr.archs.arch_util import default_init_weights
from .arch_util import make_layer


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

class LandmarkFeatureExtraction(nn.Module):
    """

    Feature extraction from landmark binary reference_image (1 X 256 X 256) in EASFFNet

    """

    def __init__(self):
        super(LandmarkFeatureExtraction, self).__init__()
        self.conv1 = nn.Conv2d(1, 64, 9, 1, padding=4, bias=False, padding_mode="circular")
        self.conv2 = nn.Conv2d(64, 64, 3, 1, padding=1, bias=False, padding_mode="circular")
        self.conv3 = nn.Conv2d(64, 64, 7, 1, padding=3, bias=False, padding_mode="circular")
        self.conv4 = nn.Conv2d(64, 128, 3, 1, padding=1, bias=False, padding_mode="circular")
        self.conv5 = nn.Conv2d(128, 128, 5, 1, padding=2, bias=False, padding_mode="circular")
        self.conv6 = nn.Conv2d(128, 128, 3, 1, padding=1, bias=False, padding_mode="circular")
        self.conv7 = nn.Conv2d(128, 128, 3, 1, padding=1, bias=False, padding_mode="circular")
        self.conv8 = nn.Conv2d(128, 128, 3, 1, padding=1, bias=False, padding_mode="circular")
        self.conv9 = nn.Conv2d(128, 128, 3, 1, padding=1, bias=False, padding_mode="circular")
        self.conv10 = nn.Conv2d(128, 128, 3, 1, padding=1, bias=False, padding_mode="circular")

        self.lrelu = nn.LeakyReLU(0.2, inplace=True)

        default_init_weights([self.conv1, self.conv2, self.conv3, self.conv4, self.conv5, self.conv6, self.conv7,
                              self.conv8, self.conv9, self.conv10], 0.1)

    def forward(self, x):
        out = self.lrelu(self.conv1(x))
        out = self.lrelu(self.conv2(out))
        out = self.lrelu(self.conv3(out))
        out = self.lrelu(self.conv4(out))
        out = self.lrelu(self.conv5(out))
        out = self.lrelu(self.conv6(out))
        out = self.lrelu(self.conv7(out))
        out = self.lrelu(self.conv8(out))
        out = self.lrelu(self.conv9(out))
        out = self.lrelu(self.conv10(out))
        return out

class ASFFBlock(nn.Module):
    """

    Adaptive Spatial Feature Fusion block of EASFFNet

    """

    def __init__(self):
        super(ASFFBlock, self).__init__()
        self.attention_compressed_conv = nn.Conv2d(128, 64, 1, 1)
        self.attention_reference_conv = nn.Conv2d(128, 64, 1, 1)
        self.attention_landmark_conv = nn.Conv2d(128, 64, 1, 1)
        
        self.attention_concatenated_conv1 = nn.Conv2d(192, 128, 3, 1, padding=1, bias=False, padding_mode="circular")
        self.attention_concatenated_bn1 = nn.BatchNorm2d(128)
        self.attention_concatenated_conv2 = nn.Conv2d(128, 128, 3, 1, padding=1, bias=False, padding_mode="circular")
        self.attention_concatenated_bn2 = nn.BatchNorm2d(128)

        self.compressed_conv1 = nn.Conv2d(128, 128, 3, 1, padding=1, padding_mode="circular")
        self.compressed_bn = nn.BatchNorm2d(128)
        self.compressed_conv2 = nn.Conv2d(128, 128, 1, 1)

        self.reference_conv1 = nn.Conv2d(128, 128, 3, 1, padding=1, padding_mode="circular")
        self.reference_bn = nn.BatchNorm2d(128)
        self.reference_conv2 = nn.Conv2d(128, 128, 1, 1)

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
        attention_mask = self.lrelu(self.attention_concatenated_bn1(self.attention_concatenated_conv1(attention_concatenated)))
        attention_mask = self.lrelu(self.attention_concatenated_bn2(self.attention_concatenated_conv2(attention_mask)))

        compressed_feat = self.compressed_conv2(self.lrelu(self.compressed_bn(self.compressed_conv1(compressed))))
        reference_feat = self.reference_conv2(self.lrelu(self.reference_bn(self.reference_conv1(reference))))

        out = torch.add(torch.mul(torch.sub(reference_feat, compressed_feat), attention_mask), compressed_feat)
        return out

class ResidualDenseBlock(nn.Module):
    """Residual Dense Block.

    Used in RRDB block in ESRGAN.

    Args:
        num_feat (int): Channel number of intermediate features.
        num_grow_ch (int): Channels for each growth.
    """

    def __init__(self, num_feat=64, num_grow_ch=32):
        super(ResidualDenseBlock, self).__init__()
        self.conv1 = nn.Conv2d(num_feat, num_grow_ch, 3, 1, 1, padding_mode="circular")
        self.conv2 = nn.Conv2d(num_feat + num_grow_ch, num_grow_ch, 3, 1, 1, padding_mode="circular")
        self.conv3 = nn.Conv2d(num_feat + 2 * num_grow_ch, num_grow_ch, 3, 1, 1, padding_mode="circular")
        self.conv4 = nn.Conv2d(num_feat + 3 * num_grow_ch, num_grow_ch, 3, 1, 1, padding_mode="circular")
        self.conv5 = nn.Conv2d(num_feat + 4 * num_grow_ch, num_feat, 3, 1, 1, padding_mode="circular")

        self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)

        # initialization
        default_init_weights([self.conv1, self.conv2, self.conv3, self.conv4, self.conv5], 0.1)

    def forward(self, x):
        x1 = self.lrelu(self.conv1(x))
        x2 = self.lrelu(self.conv2(torch.cat((x, x1), 1)))
        x3 = self.lrelu(self.conv3(torch.cat((x, x1, x2), 1)))
        x4 = self.lrelu(self.conv4(torch.cat((x, x1, x2, x3), 1)))
        x5 = self.conv5(torch.cat((x, x1, x2, x3, x4), 1))
        # Empirically, we use 0.2 to scale the residual for better performance
        return x5 * 0.2 + x


class RRDB(nn.Module):
    """Residual in Residual Dense Block.

    Used in RRDB-Net in ESRGAN.

    Args:
        num_feat (int): Channel number of intermediate features.
        num_grow_ch (int): Channels for each growth.
    """

    def __init__(self, num_feat, num_grow_ch=32):
        super(RRDB, self).__init__()
        self.rdb1 = ResidualDenseBlock(num_feat, num_grow_ch)
        self.rdb2 = ResidualDenseBlock(num_feat, num_grow_ch)
        self.rdb3 = ResidualDenseBlock(num_feat, num_grow_ch)

    def forward(self, x):
        out = self.rdb1(x)
        out = self.rdb2(out)
        out = self.rdb3(out)
        # Emperically, we use 0.2 to scale the residual for better performance
        return out * 0.2 + x


# @ARCH_REGISTRY.register()
class EASFFNet(nn.Module):
    """
    Network for artifact reduction with exemplar reference_image

    EASFFNet: Enhanced Adaptive Spatial Feature Fusion Network

    """

    def __init__(self):
        super(EASFFNet, self).__init__()
        num_in_ch = 3
        num_out_ch = 3
        self.num_feat = 64
        self.num_block = 5
        self.num_grow_ch = 32

        self.compressed_conv_first = nn.Conv2d(num_in_ch, self.num_feat, 3, 1, 1, padding_mode="circular")
        self.reference_conv_first = nn.Conv2d(num_in_ch, self.num_feat, 3, 1, 1, padding_mode="circular")
        self.compressed_body = make_layer(RRDB, self.num_block, num_feat=self.num_feat, num_grow_ch=self.num_grow_ch)
        self.reference_body = make_layer(RRDB, self.num_block, num_feat=self.num_feat, num_grow_ch=self.num_grow_ch)
        self.compressed_conv_body = nn.Conv2d(self.num_feat, self.num_feat, 3, 1, 1, padding_mode="circular")
        self.reference_conv_body = nn.Conv2d(self.num_feat, self.num_feat, 3, 1, 1, padding_mode="circular")
        self.compressed_128 = nn.Conv2d(self.num_feat, 128, 3, 1, 1, padding_mode="circular")
        self.reference_128 = nn.Conv2d(self.num_feat, 128, 3, 1, 1, padding_mode="circular")

        self.landmark_extraction = LandmarkFeatureExtraction()

        self.ASFF1 = ASFFBlock()
        self.ASFF2 = ASFFBlock()
        self.ASFF3 = ASFFBlock()
        self.ASFF4 = ASFFBlock()

        self.conv_hr = nn.Conv2d(128, self.num_feat, 3, 1, 1, padding_mode="circular")
        self.conv_last = nn.Conv2d(self.num_feat, num_out_ch, 3, 1, 1, padding_mode="circular")

        self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)

    def forward(self, compressed, reference, landmark):
        compressed_feat = self.compressed_conv_first(compressed)
        compressed_feat = compressed_feat + self.compressed_conv_body(self.compressed_body(compressed_feat))

        reference_feat = self.reference_conv_first(reference)
        reference_feat = reference_feat + self.reference_conv_body(self.reference_body(reference_feat))

        compressed_feat = self.compressed_128(compressed_feat)
        reference_feat = self.reference_128(reference_feat)

        landmark_feat = self.landmark_extraction(landmark)

        reference_feat = AdaIN(compressed_feat, reference_feat)

        feature_fusion = self.ASFF1(compressed_feat, reference_feat, landmark_feat)
        feature_fusion = self.ASFF2(feature_fusion, reference_feat, landmark_feat)
        feature_fusion = self.ASFF3(feature_fusion, reference_feat, landmark_feat)
        feature_fusion = self.ASFF4(feature_fusion, reference_feat, landmark_feat)

        out = self.conv_last(self.lrelu(self.conv_hr(feature_fusion)))
        return compressed + out
