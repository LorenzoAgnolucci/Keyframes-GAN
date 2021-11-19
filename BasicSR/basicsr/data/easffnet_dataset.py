import numpy as np
import torch
from torch.utils import data as data
import torch.nn.functional as F
import os
import random
import albumentations as A
import dlib
import cv2
from torchvision.utils import save_image


from basicsr.data.data_util import paired_paths_from_lmdb, paired_paths_from_meta_info_file
from basicsr.data.transforms import augment
from basicsr.utils import FileClient, imfrombytes, img2tensor, scandir
from basicsr.utils.registry import DATASET_REGISTRY


@DATASET_REGISTRY.register()
class EASFFNetDataset(data.Dataset):
    """EASFFNet reference_image dataset for reference_image restoration.

    Read compressed, reference, landmark and gt images.

    (Does not support modes different than "folder")

    There are three modes:
    1. 'lmdb': Use lmdb files.
        If opt['io_backend'] == lmdb.
    2. 'meta_info_file': Use meta information file to generate paths.
        If opt['io_backend'] != lmdb and opt['meta_info_file'] is not None.
    3. 'folder': Scan folders to generate paths.
        The rest.

    Args:
        opt (dict): Config for train datasets. It contains the following keys:
            dataroot_compressed (str): Data root path for compressed images.
            dataroot_reference (str): Data root path for reference images.
            dataroot_landmark (str): Data root path for landmark images.
            dataroot_gt (str): Data root path for gt images.
            meta_info_file (str): Path for meta information file.
            io_backend (dict): IO backend type and other kwarg.
            filename_tmpl (str): Template for each filename. Note that the
                template excludes the file extension. Default: '{}'.
            gt_size (int): Cropped patched size for gt patches.
            use_flip (bool): Use horizontal flips.
            use_rot (bool): Use rotation (use vertical flip and transposing h
                and w for implementation).

            scale (bool): Scale, which will be added automatically.
            phase (str): 'train' or 'val'.
    """

    def __init__(self, opt):
        super(EASFFNetDataset, self).__init__()
        self.opt = opt
        # file client (io backend)
        self.file_client = None
        self.io_backend_opt = opt['io_backend']
        self.mean = opt['mean'] if 'mean' in opt else None
        self.std = opt['std'] if 'std' in opt else None
        self.scale = self.opt['scale']
        self.offset = self.opt['offset']

        self.compressed_folder = opt['dataroot_compressed']
        self.reference_folder = opt['dataroot_reference']
        self.landmark_folder = opt['dataroot_landmark']
        self.gt_folder = opt['dataroot_gt']
        if 'filename_tmpl' in opt:
            self.filename_tmpl = opt['filename_tmpl']
        else:
            self.filename_tmpl = '{}'

        if self.io_backend_opt['type'] == 'lmdb':
            self.io_backend_opt['db_paths'] = [self.lq_folder, self.gt_folder]
            self.io_backend_opt['client_keys'] = ['lq', 'gt']
            self.paths = paired_paths_from_lmdb([self.lq_folder, self.gt_folder], ['lq', 'gt'])
        elif 'meta_info_file' in self.opt and self.opt['meta_info_file'] is not None:
            self.paths = paired_paths_from_meta_info_file([self.lq_folder, self.gt_folder], ['lq', 'gt'],
                                                          self.opt['meta_info_file'], self.filename_tmpl)
        else:
            self.paths = paths_from_folder([self.compressed_folder, self.reference_folder,
                                            self.landmark_folder, self.gt_folder], self.offset,
                                           ['compressed', 'reference', 'landmark', 'gt'])

        self.face_detector = dlib.cnn_face_detection_model_v1(
            "/homes/students_home/lagnolucci/artifact_reduction/dlib_weights.dat")
        self.landmark_detector = dlib.shape_predictor(
            "/homes/students_home/lagnolucci/artifact_reduction/dlib_shape_predictor_68_face_landmarks.dat")

    def __getitem__(self, index):
        if self.file_client is None:
            self.file_client = FileClient(self.io_backend_opt.pop('type'), **self.io_backend_opt)

        # Load compressed, reference, landmark and gt images. Dimension order: HWC; channel order: BGR;
        # reference_image range: [0, 1], float32.
        compressed_path = self.paths[index]['compressed_path']
        img_bytes = self.file_client.get(compressed_path, 'compressed')
        img_compressed = imfrombytes(img_bytes, float32=True)
        # img_compressed = img_compressed * 2 - 1
        reference_path = self.paths[index]['reference_path']
        img_bytes = self.file_client.get(reference_path, 'reference')
        img_reference = imfrombytes(img_bytes, float32=True)
        # img_reference = img_reference * 2 - 1
        landmark_path = self.paths[index]['landmark_path']
        img_bytes = self.file_client.get(landmark_path, 'landmark')
        img_landmark = imfrombytes(img_bytes, flag="grayscale", float32=True)
        img_landmark = np.expand_dims(img_landmark, 2)
        # img_landmark = img_landmark * 2 - 1
        gt_path = self.paths[index]['gt_path']
        img_bytes = self.file_client.get(gt_path, 'gt')
        img_gt = imfrombytes(img_bytes, float32=True)
        # img_gt = img_gt * 2 - 1

        # augmentation for training
        if self.opt['phase'] == 'train':
            gt_size = self.opt['gt_size']

            transform = A.Compose([
                A.RandomCrop(width=gt_size, height=gt_size, p=1),
                A.ShiftScaleRotate(shift_limit=(-0.13, 0.13), scale_limit=(0.0, 0.0), rotate_limit=(0, 0), p=0.2),
                A.RandomRotate90(p=0.2),
                A.CoarseDropout(max_holes=1, max_height=64, max_width=64, min_holes=1, min_height=16, min_width=16, p=0.2)
            ],
                additional_targets={
                    "compressed": "image",
                    "reference": "image",
                    "landmark": "image",
                    "gt": "image"
                }
            )

            transformed = transform(image=img_compressed, reference=img_reference, landmark=img_landmark, gt=img_gt)

            img_compressed = transformed['image']
            img_reference = transformed['reference']
            img_landmark = transformed['landmark']
            img_gt = transformed['gt']

        print("Img landmark shape: ", img_landmark.shape, type(img_landmark))
        # BGR to RGB, HWC to CHW, numpy to tensor
        img_compressed, img_reference, img_landmark, img_gt = img2tensor([img_compressed, img_reference, img_landmark, img_gt], bgr2rgb=True, float32=True)
        print("Img landmark shape: ", img_landmark.shape, type(img_landmark))
        # Resize images
        if self.opt['phase'] == 'val' or self.opt['phase'] == 'test':
            img_compressed = F.interpolate(img_compressed.unsqueeze(0), (256, 256), mode='bilinear', align_corners=False).squeeze(0)
            img_reference = F.interpolate(img_reference.unsqueeze(0), (256, 256), mode='bilinear', align_corners=False).squeeze(0)
            img_gt = F.interpolate(img_gt.unsqueeze(0), (256, 256), mode='bilinear', align_corners=False).squeeze(0)
            # Landmark reference_image should be already 1 X 256 X 256
            if img_landmark.shape[-3:] != (1, 256, 256):
                print("Wrong landmark reference_image size: ", img_landmark.shape, landmark_path)
                img_landmark = F.interpolate(img_landmark.unsqueeze(0), (256, 256), mode='nearest').squeeze(0)
        print("Img landmark shape: ", img_landmark.shape, type(img_landmark))

        return {'compressed': img_compressed, 'reference': img_reference, 'landmark': img_landmark, 'gt': img_gt,
                'compressed_path': compressed_path, 'reference_path': reference_path, 'landmark_path': landmark_path, 'gt_path': gt_path}

    def __len__(self):
        return len(self.paths)

def paths_from_folder(folders, offset, keys):
    """Generate paths from folders.

    Args:
        folders (list[str]): A list of folder path. The order of list should
            be [compressed_folder, reference_folder, landmark_folder, gt_folder].
        keys (list[str]): A list of keys identifying folders. The order should
            be in consistent with folders, e.g., ['compressed', 'reference', 'landmark', 'gt'].

    Returns:
        list[str]: Returned path list.
    """
    assert len(folders) == 4, (f'The len of folders should be 2 with [compressed_folder, reference_folder,'
                               f' landmark_folder, gt_folder]. But got {len(folders)}')
    assert len(keys) == 4, (f'The len of keys should be 4 with [compressed_key, reference_key, landmark_key, gt_key].'
                            f' But got {len(keys)}')
    compressed_folder, reference_folder, landmark_folder, gt_folder = folders
    compressed_key, reference_key, landmark_key, gt_key = keys

    compressed_paths = list(scandir(compressed_folder))
    reference_paths = list(scandir(reference_folder))
    landmark_paths = list(scandir(landmark_folder))
    gt_paths = list(scandir(gt_folder))
    assert len(compressed_paths) == len(reference_paths) == len(landmark_paths) == len(gt_paths),\
        (f'{compressed_key}, {reference_key}, {landmark_key} and {gt_key} datasets have different number of images:'
         f' {len(compressed_paths)}, {len(reference_paths)}, {len(landmark_paths)} {len(gt_paths)}.')

    paths = []
    filename_difference = f"_o_{offset}"
    for gt_path in gt_paths:
        basename, ext = os.path.splitext(os.path.basename(gt_path))
        gt_path = os.path.join(gt_folder, gt_path)
        
        compressed_name = f'{basename}{filename_difference}{ext}'
        compressed_path = os.path.join(compressed_folder, compressed_name)
        assert compressed_name in compressed_paths, f'{compressed_name} is not in ' f'{compressed_key}_paths.'

        reference_name = f'{basename}{ext}'
        reference_path = os.path.join(reference_folder, reference_name)
        assert reference_name in reference_paths, f'{reference_name} is not in ' f'{reference_key}_paths.'

        landmark_name = f'{basename}{filename_difference}{ext}'
        landmark_path = os.path.join(landmark_folder, landmark_name)
        assert landmark_name in landmark_paths, f'{landmark_name} is not in ' f'{landmark_key}_paths.'

        paths.append(dict([(f'{compressed_key}_path', compressed_path),
                           (f'{reference_key}_path', reference_path),
                           (f'{landmark_key}_path', landmark_path),
                           (f'{gt_key}_path', gt_path)]))
    return paths


def random_crop(imgs_compressed, imgs_reference, imgs_landmark, imgs_gt, gt_patch_size, scale=1):
    """
    Random crop.

    It crops lists of compressed, reference, landmark and gt images with corresponding locations.

    Args:
        imgs_compressed (list[ndarray] | ndarray): compressed images. Note that all images
            should have the same shape. If the input is an ndarray, it will
            be transformed to a list containing itself.
        imgs_reference (list[ndarray] | ndarray): reference images. Note that all images
            should have the same shape. If the input is an ndarray, it will
            be transformed to a list containing itself.
        imgs_landmark (list[ndarray] | ndarray): landmark images. Note that all images
            should have the same shape. If the input is an ndarray, it will
            be transformed to a list containing itself.
        imgs_gt (list[ndarray] | ndarray): GT images. Note that all images
            should have the same shape. If the input is an ndarray, it will
            be transformed to a list containing itself.
        gt_patch_size (int): GT patch size.
        scale (int): Scale factor.

    Returns:
        list[ndarray] | ndarray: compressed images, reference images, landmark images and GT images. If returned results
            only have one element, just return ndarray.
    """

    if not isinstance(imgs_compressed, list):
        imgs_compressed = [imgs_compressed]
    if not isinstance(imgs_reference, list):
        imgs_reference = [imgs_reference]
    if not isinstance(imgs_landmark, list):
        imgs_landmark = [imgs_landmark]
    if not isinstance(imgs_gt, list):
        imgs_gt = [imgs_gt]

    h_compressed, w_compressed, _ = imgs_compressed[0].shape
    h_gt, w_gt, _ = imgs_gt[0].shape
    compressed_patch_size = gt_patch_size // scale

    if h_gt != h_compressed * scale or w_gt != w_compressed * scale:
        raise ValueError(f'Scale mismatches. GT ({h_gt}, {w_gt}) is not {scale}x ',
                         f'multiplication of LQ ({h_compressed}, {w_compressed}).')
    if h_compressed < compressed_patch_size or w_compressed < compressed_patch_size:
        raise ValueError(f'LQ ({h_compressed}, {w_compressed}) is smaller than patch size '
                         f'({compressed_patch_size}, {compressed_patch_size}). ')

    # randomly choose top and left coordinates for compressed patch
    top = random.randint(0, h_compressed - compressed_patch_size)
    left = random.randint(0, w_compressed - compressed_patch_size)

    # crop compressed, reference and landmark patch
    imgs_compressed = [v[top:top + compressed_patch_size, left:left + compressed_patch_size, ...] for v in imgs_compressed]
    imgs_reference = [v[top:top + compressed_patch_size, left:left + compressed_patch_size, ...] for v in imgs_reference]
    imgs_landmark = [v[top:top + compressed_patch_size, left:left + compressed_patch_size, ...] for v in imgs_landmark]

    # crop corresponding gt patch
    top_gt, left_gt = int(top * scale), int(left * scale)
    imgs_gt = [v[top_gt:top_gt + gt_patch_size, left_gt:left_gt + gt_patch_size, ...] for v in imgs_gt]
    
    if len(imgs_compressed) == 1:
        imgs_compressed = imgs_compressed[0]
    if len(imgs_reference) == 1:
        imgs_reference = imgs_reference[0]
    if len(imgs_landmark) == 1:
        imgs_landmark = imgs_landmark[0]
    if len(imgs_gt) == 1:
        imgs_gt = imgs_gt[0]
    return imgs_compressed, imgs_reference, imgs_landmark, imgs_gt
