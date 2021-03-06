import cv2
import math
import numpy as np
import random

# -------------------------------------------------------------------- #
# --------------------------- blur kernels --------------------------- #
# -------------------------------------------------------------------- #


def sigma_matrix2(sig_x, sig_y, theta):
    """Calculate the rotated sigma matrix (two dimensional matrix).

    Args:
        sig_x (float):
        sig_y (float):
        theta (float): Radian measurement.

    Returns:
        ndarray: Rotated sigma matrix.
    """
    D = np.array([[sig_x**2, 0], [0, sig_y**2]])
    U = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
    return np.dot(U, np.dot(D, U.T))


def mesh_grid(kernel_size):
    """Generate the mesh grid, centering at zero.

    Args:
        kernel_size (int):

    Returns:
        xy (ndarray): with the shape (kernel_size, kernel_size, 2)
        xx (ndarray): with the shape (kernel_size, kernel_size)
        yy (ndarray): with the shape (kernel_size, kernel_size)
    """
    ax = np.arange(-kernel_size // 2 + 1., kernel_size // 2 + 1.)
    xx, yy = np.meshgrid(ax, ax)
    xy = np.hstack((xx.reshape((kernel_size * kernel_size, 1)), yy.reshape(kernel_size * kernel_size,
                                                                           1))).reshape(kernel_size, kernel_size, 2)
    return xy, xx, yy


def pdf2(sigma_matrix, grid):
    """Calculate PDF of the bivariate Gaussian distribution.

    Args:
        sigma_matrix (ndarray): with the shape (2, 2)
        grid (ndarray): generated by :func:`mesh_grid`,
            with the shape (K, K, 2), K is the kernel size.

    Returns:
        kernel (ndarrray): un-normalized kernel.
    """
    inverse_sigma = np.linalg.inv(sigma_matrix)
    kernel = np.exp(-0.5 * np.sum(np.dot(grid, inverse_sigma) * grid, 2))
    return kernel


def mass_center_shift(kernel_size, kernel):
    """Calculate the shift of the mass center of a kenrel.

    Args:
        kernel_size (int):
        kernel (ndarray): normalized kernel.

    Returns:
        delta_h (float):
        delta_w (float):
    """
    ax = np.arange(-kernel_size // 2 + 1., kernel_size // 2 + 1.)
    col_sum, row_sum = np.sum(kernel, axis=0), np.sum(kernel, axis=1)
    delta_h = np.dot(row_sum, ax)
    delta_w = np.dot(col_sum, ax)
    return delta_h, delta_w


def bivariate_anisotropic_Gaussian(kernel_size, sig_x, sig_y, theta, grid=None):
    """Generate a bivariate anisotropic Gaussian kernel.

    Args:
        kernel_size (int):
        sig_x (float):
        sig_y (float):
        theta (float): Radian measurement.
        grid (ndarray, optional): generated by :func:`mesh_grid`,
            with the shape (K, K, 2), K is the kernel size. Default: None

    Returns:
        kernel (ndarray): normalized kernel.
    """
    if grid is None:
        grid, _, _ = mesh_grid(kernel_size)
    sigma_matrix = sigma_matrix2(sig_x, sig_y, theta)
    kernel = pdf2(sigma_matrix, grid)
    kernel = kernel / np.sum(kernel)
    return kernel


def bivariate_isotropic_Gaussian(kernel_size, sig, grid=None):
    """Generate a bivariate isotropic Gaussian kernel.

    Args:
        kernel_size (int):
        sig (float):
        grid (ndarray, optional): generated by :func:`mesh_grid`,
            with the shape (K, K, 2), K is the kernel size. Default: None

    Returns:
        kernel (ndarray): normalized kernel.
    """
    if grid is None:
        grid, _, _ = mesh_grid(kernel_size)
    sigma_matrix = np.array([[sig**2, 0], [0, sig**2]])
    kernel = pdf2(sigma_matrix, grid)
    kernel = kernel / np.sum(kernel)
    return kernel


def random_bivariate_anisotropic_Gaussian(kernel_size,
                                          sigma_x_range,
                                          sigma_y_range,
                                          rotation_range,
                                          noise_range=None,
                                          strict=False):
    """Randomly generate bivariate anisotropic Gaussian kernels.

    Args:
        kernel_size (int):
        sigma_x_range (tuple): [0.6, 5]
        sigma_y_range (tuple): [0.6, 5]
        rotation range (tuple): [-math.pi, math.pi]
        noise_range(tuple, optional): multiplicative kernel noise,
            [0.75, 1.25]. Default: None

    Returns:
        kernel (ndarray):
    """
    assert kernel_size % 2 == 1, 'Kernel size must be an odd number.'
    assert sigma_x_range[0] < sigma_x_range[1], 'Wrong sigma_x_range.'
    assert sigma_y_range[0] < sigma_y_range[1], 'Wrong sigma_y_range.'
    assert rotation_range[0] < rotation_range[1], 'Wrong rotation_range.'
    sigma_x = np.random.uniform(sigma_x_range[0], sigma_x_range[1])
    sigma_y = np.random.uniform(sigma_y_range[0], sigma_y_range[1])
    if strict:
        sigma_max = np.max([sigma_x, sigma_y])
        sigma_min = np.min([sigma_x, sigma_y])
        sigma_x, sigma_y = sigma_max, sigma_min
    rotation = np.random.uniform(rotation_range[0], rotation_range[1])

    kernel = bivariate_anisotropic_Gaussian(kernel_size, sigma_x, sigma_y, rotation)

    # add multiplicative noise
    if noise_range is not None:
        assert noise_range[0] < noise_range[1], 'Wrong noise range.'
        noise = np.random.uniform(noise_range[0], noise_range[1], size=kernel.shape)
        kernel = kernel * noise
    kernel = kernel / np.sum(kernel)
    if strict:
        return kernel, sigma_x, sigma_y, rotation
    else:
        return kernel


def random_bivariate_isotropic_Gaussian(kernel_size, sigma_range, noise_range=None, strict=False):
    """Randomly generate bivariate isotropic Gaussian kernels.

    Args:
        kernel_size (int):
        sigma_range (tuple): [0.6, 5]
        noise_range(tuple, optional): multiplicative kernel noise,
            [0.75, 1.25]. Default: None

    Returns:
        kernel (ndarray):
    """
    assert kernel_size % 2 == 1, 'Kernel size must be an odd number.'
    assert sigma_range[0] < sigma_range[1], 'Wrong sigma_x_range.'
    sigma = np.random.uniform(sigma_range[0], sigma_range[1])

    kernel = bivariate_isotropic_Gaussian(kernel_size, sigma)

    # add multiplicative noise
    if noise_range is not None:
        assert noise_range[0] < noise_range[1], 'Wrong noise range.'
        noise = np.random.uniform(noise_range[0], noise_range[1], size=kernel.shape)
        kernel = kernel * noise
    kernel = kernel / np.sum(kernel)
    if strict:
        return kernel, sigma
    else:
        return kernel


def random_mixed_kernels(kernel_list,
                         kernel_prob,
                         kernel_size=21,
                         sigma_x_range=[0.6, 5],
                         sigma_y_range=[0.6, 5],
                         rotation_range=[-math.pi, math.pi],
                         beta_range=[0.5, 8],
                         noise_range=None):
    """Randomly generate mixed kernels.

    Args:
        kernel_list (tuple): a list name of kenrel types,
            support ['iso', 'aniso']
        kernel_prob (tuple): corresponding kernel probability for each
            kernel type
        kernel_size (int):
        sigma_x_range (tuple): [0.6, 5]
        sigma_y_range (tuple): [0.6, 5]
        rotation range (tuple): [-math.pi, math.pi]
        beta_range (tuple): [0.5, 8]
        noise_range(tuple, optional): multiplicative kernel noise,
            [0.75, 1.25]. Default: None

    Returns:
        kernel (ndarray):
    """
    kernel_type = random.choices(kernel_list, kernel_prob)[0]
    if kernel_type == 'iso':
        kernel = random_bivariate_isotropic_Gaussian(kernel_size, sigma_x_range, noise_range=noise_range)
    elif kernel_type == 'aniso':
        kernel = random_bivariate_anisotropic_Gaussian(
            kernel_size, sigma_x_range, sigma_y_range, rotation_range, noise_range=noise_range)

    # add multiplicative noise
    if noise_range is not None:
        assert noise_range[0] < noise_range[1], 'Wrong noise range.'
        noise = np.random.uniform(noise_range[0], noise_range[1], size=kernel.shape)
        kernel = kernel * noise
    kernel = kernel / np.sum(kernel)
    return kernel


# ------------------------------------------------------------- #
# --------------------------- noise --------------------------- #
# ------------------------------------------------------------- #

# ----------------------- Gaussian Noise ----------------------- #


def generate_gaussian_noise(img, sigma=10, gray_noise=False):
    """Generate Gaussian noise.

    Args:
        img (Numpy array): Input reference_image, shape (h, w, c), range [0, 1], float32.
        sigma (float): Noise scale (measured in range 255). Default: 10.

    Returns:
        (Numpy array): Returned noisy reference_image, shape (h, w, c), range[0, 1],
            float32.
    """
    noise = np.float32(np.random.randn(*(img.shape))) * sigma / 255.
    if gray_noise:
        noise = np.float32(np.random.randn(*(img.shape[0:2]))) * sigma / 255.
        noise = np.expand_dims(noise, axis=2).repeat(3, axis=2)
    else:
        noise = np.float32(np.random.randn(*(img.shape))) * sigma / 255.
    return noise


def add_gaussian_noise(img, sigma=10, clip=True, rounds=False, gray_noise=False):
    """Add Gaussian noise.

    Args:
        img (Numpy array): Input reference_image, shape (h, w, c), range [0, 1], float32.
        sigma (float): Noise scale (measured in range 255). Default: 10.

    Returns:
        (Numpy array): Returned noisy reference_image, shape (h, w, c), range[0, 1],
            float32.
    """
    noise = generate_gaussian_noise(img, sigma, gray_noise)
    out = img + noise
    if clip and rounds:
        out = np.clip((out * 255.0).round(), 0, 255) / 255.
    elif clip:
        out = np.clip(out, 0, 1)
    elif rounds:
        out = (out * 255.0).round() / 255.
    return out


# ----------------------- Random Gaussian Noise ----------------------- #
def random_generate_gaussian_noise(img, sigma_range=(0, 10), gray_prob=0):
    sigma = np.random.uniform(sigma_range[0], sigma_range[1])
    if np.random.uniform() < gray_prob:
        gray_noise = True
    else:
        gray_noise = False
    return generate_gaussian_noise(img, sigma, gray_noise)


def random_add_gaussian_noise(img, sigma_range=(0, 1.0), gray_prob=0, clip=True, rounds=False):
    noise = random_generate_gaussian_noise(img, sigma_range, gray_prob)
    out = img + noise
    if clip and rounds:
        out = np.clip((out * 255.0).round(), 0, 255) / 255.
    elif clip:
        out = np.clip(out, 0, 1)
    elif rounds:
        out = (out * 255.0).round() / 255.
    return out


# ------------------------------------------------------------------------ #
# --------------------------- JPEG compression --------------------------- #
# ------------------------------------------------------------------------ #


def add_jpg_compression(img, quality=90):
    """Add JPG compression artifacts.

    Args:
        img (Numpy array): Input reference_image, shape (h, w, c), range [0, 1], float32.
        quality (float): JPG compression quality. 0 for lowest quality, 100 for
            best quality. Default: 90.

    Returns:
        (Numpy array): Returned reference_image after JPG, shape (h, w, c), range[0, 1],
            float32.
    """
    img = np.clip(img, 0, 1)
    encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), quality]
    _, encimg = cv2.imencode('.jpg', img * 255., encode_param)
    img = np.float32(cv2.imdecode(encimg, 1)) / 255.
    return img


def random_add_jpg_compression(img, quality_range=(90, 100)):
    """Randomly add JPG compression artifacts.

    Args:
        img (Numpy array): Input reference_image, shape (h, w, c), range [0, 1], float32.
        quality_range (tuple[float] | list[float]): JPG compression quality
            range. 0 for lowest quality, 100 for best quality.
            Default: (90, 100).

    Returns:
        (Numpy array): Returned reference_image after JPG, shape (h, w, c), range[0, 1],
            float32.
    """
    quality = np.random.uniform(quality_range[0], quality_range[1])
    return add_jpg_compression(img, quality)
