import hashlib
import math
import re
import typing

import cv2 as cv
import numpy as np
import torch.utils.data


def project_points(points, projection_matrix, view_matrix, width, height):
    p_3d_cam = np.concatenate((points, np.ones_like(points[:, :1])), axis=-1).T
    p_2d_proj = np.matmul(projection_matrix, p_3d_cam)
    p_2d_ndc = p_2d_proj[:-1, :] / p_2d_proj[-1, :]
    p_2d_ndc = p_2d_ndc.T
    x = p_2d_ndc[:, 1]
    y = p_2d_ndc[:, 0]
    pixels = np.copy(p_2d_ndc)
    pixels[:, 0] = ((1 + x) * 0.5) * width
    pixels[:, 1] = ((1 + y) * 0.5) * height
    pixels = pixels.astype(int)
    return pixels


def distance_between_point_and_plane(x1, y1, z1, a, b, c, d):
    d = abs((a * x1 + b * y1 + c * z1 + d))
    e = math.sqrt(a * a + b * b + c * c)
    return d / e


def get_params_hash(*args, **kwargs):
    """Returns a sha1 hash for the given list of parameters (useful for caching)."""
    # by default, will use the repr of all params but remove the 'at 0x00000000' addresses
    clean_str = re.sub(r" at 0x[a-fA-F\d]+", "", str(args) + str(kwargs))
    return hashlib.sha1(clean_str.encode()).hexdigest()


def get_obj_center_crop(
        sample_frame: typing.Dict,
        instance_idx: int,
        crop_size: typing.Tuple[int, int],
):
    assert sample_frame["INSTANCE_NUM"] > instance_idx
    assert crop_size[0] > 0 and crop_size[1] > 0
    tl = (int(round(sample_frame["CENTROID_2D_IM"][instance_idx][0] - crop_size[0] / 2)),
          int(round(sample_frame["CENTROID_2D_IM"][instance_idx][1] - crop_size[1] / 2)))
    br = (tl[0] + crop_size[0], tl[1] + crop_size[1])
    return safe_crop(sample_frame["IMAGE"], tl, br, force_copy=True)


def is_frame_blurry(
        frame: np.ndarray,
        nz_gradmag_threshold: float = 0.05,  # need at least 5% non-zero grad mag (default)
        return_nz_grad_mag: bool = False,
):
    grad_mags = np.abs(cv.Laplacian(frame, cv.CV_64F, dst=None)).max(axis=2).flatten()
    grad_mag_hist = np.histogram(grad_mags, bins=20, density=True)
    grad_mag_hist_norm = grad_mag_hist[0] / grad_mag_hist[0].sum()
    nonzero_grad_mag_sum = grad_mag_hist_norm[1:].sum()
    if return_nz_grad_mag:
        return nonzero_grad_mag_sum < nz_gradmag_threshold, nonzero_grad_mag_sum
    else:
        return nonzero_grad_mag_sum < nz_gradmag_threshold


def safe_crop(image, tl, br, bordertype=cv.BORDER_CONSTANT, borderval=0, force_copy=False):
    """Safely crops a region from within an image, padding borders if needed.

    Args:
        image: the image to crop (provided as a numpy array).
        tl: a tuple or list specifying the (x,y) coordinates of the top-left crop corner.
        br: a tuple or list specifying the (x,y) coordinates of the bottom-right crop corner.
        bordertype: border copy type to use when the image is too small for the required crop size.
            See ``cv2.copyMakeBorder`` for more information.
        borderval: border value to use when the image is too small for the required crop size. See
            ``cv2.copyMakeBorder`` for more information.
        force_copy: defines whether to force a copy of the target image region even when it can be
            avoided.

    Returns:
        The cropped image.
    """
    if not isinstance(image, np.ndarray):
        raise AssertionError("expected input image to be numpy array")
    if isinstance(tl, tuple):
        tl = list(tl)
    if isinstance(br, tuple):
        br = list(br)
    if not isinstance(tl, list) or not isinstance(br, list):
        raise AssertionError("expected tl/br coords to be provided as tuple or list")
    if tl[0] < 0 or tl[1] < 0 or br[0] > image.shape[1] or br[1] > image.shape[0]:
        image = cv.copyMakeBorder(image, max(-tl[1], 0), max(br[1] - image.shape[0], 0),
                                  max(-tl[0], 0), max(br[0] - image.shape[1], 0),
                                  borderType=bordertype, value=borderval)
        if tl[0] < 0:
            br[0] -= tl[0]
            tl[0] = 0
        if tl[1] < 0:
            br[1] -= tl[1]
            tl[1] = 0
        return image[tl[1]:br[1], tl[0]:br[0], ...]
    if force_copy:
        return np.copy(image[tl[1]:br[1], tl[0]:br[0], ...])
    return image[tl[1]:br[1], tl[0]:br[0], ...]


def get_label_color_mapping(idx):
    """Returns the PASCAL VOC color triplet for a given label index."""

    # https://gist.github.com/wllhf/a4533e0adebe57e3ed06d4b50c8419ae
    def bitget(byteval, ch):
        return (byteval & (1 << ch)) != 0

    r = g = b = 0
    for j in range(8):
        r = r | (bitget(idx, 0) << 7 - j)
        g = g | (bitget(idx, 1) << 7 - j)
        b = b | (bitget(idx, 2) << 7 - j)
        idx = idx >> 3
    return np.array([r, g, b], dtype=np.uint8)


class ConstantRandomOrderSampler(torch.utils.data.Sampler[int]):
    """
    Samples elements based on a random but constant order picked on construction.

    Args:
        data_source (Dataset): dataset to sample from
    """
    data_source: typing.Sized

    def __init__(self, data_source):
        self.sample_idxs = np.random.permutation(len(data_source))
        self.data_source = data_source

    def __iter__(self):
        assert len(self.data_source) == len(self.sample_idxs)
        return iter(self.sample_idxs)

    def __len__(self) -> int:
        return len(self.data_source)
