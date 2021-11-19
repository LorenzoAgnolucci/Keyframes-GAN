# Heavily based on https://learnopencv.com/image-quality-assessment-brisque/

import numpy as np
import cv2
from basicsr.metrics.metric_util import reorder_image, to_y_channel


def calculate_brisque(img, crop_border, input_order='HWC', convert_to='y'):
    C = 1 / 255
    mu = cv2.GaussianBlur(img, (7, 7), 7 / 6)  # Same as MATLAB implementation
    squared_mu = mu ** 2
    sigma = cv2.GaussianBlur(img ** 2, (7, 7), 7 / 6)
    sigma = (sigma - squared_mu) ** 0.5
    structdis = (img - mu) / (sigma + C)

    # indices to calculate pair-wise products (H, V, D1, D2)
    shifts = [[0, 1], [1, 0], [1, 1], [-1, 1]]
    # calculate pairwise components in each orientation
    for itr_shift in range(1, len(shifts) + 1):
        original_array = structdis
        req_shift = shifts[itr_shift - 1]  # shifting index

        # create affine matrix (to shift the image)
        M = np.float32([[1, 0, req_shift[1]], [0, 1, req_shift[0]]])
        shift_array = cv2.warpAffine(original_array, M, (structdis.shape[1], structdis.shape[0]))