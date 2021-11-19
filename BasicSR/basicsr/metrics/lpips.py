import numpy as np
import lpips
from torchvision.transforms.functional import normalize

from basicsr.metrics.metric_util import reorder_image
from basicsr.utils.registry import METRIC_REGISTRY
from basicsr.utils import img2tensor


# @METRIC_REGISTRY.register()
def calculate_lpips(img1, img2, lpips_net, input_order='HWC'):
    """Calculate LPIPS (Learned Perceptual Image Patch Similarity).
    Ref: https://richzhang.github.io/PerceptualSimilarity/
    Args:
        img1 (ndarray): Images with range [0, 255].
        img2 (ndarray): Images with range [0, 255].
        input_order (str): Whether the input order is 'HWC' or 'CHW'.
            Default: 'HWC'.
    Returns:
        float: LPIPS result.
    """

    assert img1.shape == img2.shape, (f'Image shapes are different: {img1.shape}, {img2.shape}.')
    if input_order not in ['HWC', 'CHW']:
        raise ValueError(f'Wrong input_order {input_order}. Supported input_orders are ' '"HWC" and "CHW"')

    # lpips_alex = lpips.LPIPS(net='alex').cuda()  # RGB, normalized to [-1,1]

    mean = [0.5, 0.5, 0.5]
    std = [0.5, 0.5, 0.5]
    img1 = reorder_image(img1, input_order=input_order).astype(np.float64)
    img2 = reorder_image(img2, input_order=input_order).astype(np.float64)

    img1, img2 = img2tensor([img1, img2], bgr2rgb=True, float32=True)

    normalize(img1, mean, std, inplace=True)
    normalize(img2, mean, std, inplace=True)

    lpips_val = lpips_net(img1.unsqueeze(0).cuda(), img2.unsqueeze(0).cuda())

    return lpips_val.squeeze().cpu().detach().numpy()
