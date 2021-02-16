import os
from skimage import io
import numpy as np
import cv2
from csl_common.utils import cropping, geometry
from csl_common.utils.io_utils import makedirs
import numbers

FILE_EXT_CROPS = '.jpg'


def get_bbox_non_black_area(img, tol=20):
    img_filtered = cv2.medianBlur(img, 5)
    mask =  img_filtered.max(axis=2) > tol
    inds = np.nonzero(mask)
    t, b = inds[0].min(), inds[0].max()
    l, r = inds[1].min(), inds[1].max()
    return np.array([l,t,r,b])


def get_roi_from_bbox(bbox, crop_size, margin):

    if bbox is None:
        return bbox

    l, t, r, b = bbox
    w = r - l
    h = b - t

    # set width of bbox same as height
    size = w if w > h else h
    cx = (r + l) / 2
    cy = (t + b) / 2
    l_new, r_new = cx - size / 2, cx + size / 2
    t_new, b_new = cy - size / 2, cy + size / 2

    if l_new > r_new:
        l_new, r_new = r_new, l_new

    bbox = np.array([l_new, t_new, r_new, b_new], dtype=np.float32)

    # extend bbox so that (final resized) crops will contain border area specified by margin
    scalef = (crop_size + margin) / crop_size
    crop_roi = geometry.scaleBB(bbox, scalef, scalef, typeBB=2)
    return crop_roi


class ImageLoader():
    def __init__(self, fullsize_img_dir, img_size,  border_mode='black'):
        assert border_mode in ['black', 'edge', 'mirror']
        self.fullsize_img_dir = fullsize_img_dir
        self.border_mode = border_mode
        if not isinstance(img_size, numbers.Number):
            raise ValueError("img_size must be a scalar value. "
                             "Only square crops are supported at the moment.")
        self.image_size = int(img_size)

    def load_image(self, filename):
        """ Load original image from dataset """
        img_path = os.path.join(self.fullsize_img_dir, filename)
        try:
            img = io.imread(img_path)
        except:
            raise IOError("\tError: Could not load image {}".format(img_path))
        if len(img.shape) == 2 or img.shape[2] == 1:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
        if img.shape[2] == 4:
            print(filename, "converting RGBA to RGB...")
            img = cv2.cvtColor(img, cv2.COLOR_RGBA2RGB)
        assert img.shape[2] == 3, "{}, invalid format: {}".format(img_path, img.shape)
        return img


class CachedCropLoader(ImageLoader):
    def __init__(self, fullsize_img_dir, cropped_img_root, crop_type, margin,
                 use_cache=True, median_blur_crop=False, **kwargs):

        if crop_type not in ['fullsize', 'tight']:
            raise ValueError(f"Invalid crop_type {crop_type}.")

        super().__init__(fullsize_img_dir, **kwargs)

        self.cropped_img_root = cropped_img_root
        self.median_blur_crop = median_blur_crop
        self.crop_type = crop_type
        self.use_cache = use_cache
        self.margin = int(margin)
        self.roi_size = self.image_size + self.margin

    def _cache_filepath(self, filename, id, aligned):
        imgsize_dirname = str(self.image_size)
        crop_dir = self.crop_type
        if not aligned:
            crop_dir += '_noalign'
        filename_noext = os.path.splitext(filename)[0]
        # adding an id is necessary in case there is more than one crop in an image
        if id is not None:
            filename_noext += '.{:07d}'.format(id)
        return os.path.join(self.cropped_img_root, imgsize_dirname, crop_dir, filename_noext + FILE_EXT_CROPS)

    def _load_cached_image(self, filename, id=None, aligned=False):
        is_cached = False
        if self.crop_type=='fullsize':
            img = self.load_image(filename)
        else:
            cache_filepath = self._cache_filepath(filename, id, aligned)
            if self.use_cache and os.path.isfile(cache_filepath):
                try:
                    img = io.imread(cache_filepath)
                except:
                    print("\tError: Could load not cropped image {}!".format(cache_filepath))
                    print("\tDeleting file and loading fullsize image.")
                    os.remove(cache_filepath)
                    img = self.load_image(filename)
                is_cached = True
            else:
                img = self.load_image(filename)

        assert isinstance(img, np.ndarray)
        return img, is_cached

    def load_crop(self, filename, bb=None, landmarks=None, id=None, aligned=False, mode='bounding_box'):
        assert mode in ['bounding_box', 'landmarks']
        assert mode != 'landmarks' or landmarks is not None
        assert mode == 'landmarks' or not aligned

        img, is_cached_crop = self._load_cached_image(filename, id, aligned)

        if self.crop_type == 'fullsize':
            return img

        if mode == 'bounding_box':
            if bb is None:
                bb = np.array([0, 0, img.shape[1] - 1, img.shape[0] - 1])
                if not is_cached_crop:
                    bb = get_bbox_non_black_area(img)

            roi = get_roi_from_bbox(bb, crop_size=self.image_size, margin=self.margin)
            self._cropper = cropping.FaceCrop(img, output_size=self.roi_size, bbox=roi,
                                              img_already_cropped=is_cached_crop)
        else:
            self._cropper = cropping.FaceCrop(img, output_size=self.roi_size, landmarks=landmarks,
                                              align_face_orientation=aligned,
                                              img_already_cropped=is_cached_crop)

        try:
            crop = self._cropper.apply_to_image(border_mode=self.border_mode)
            if self.use_cache and not is_cached_crop:
                cache_filepath = self._cache_filepath(filename, id, aligned)
                makedirs(cache_filepath)
                io.imsave(cache_filepath, crop)
        except cv2.error:
            print('Could not crop image {}.'.format(filename))
            crop = img  # fallback to fullsize image

        if self.median_blur_crop:
            crop = cv2.medianBlur(crop, ksize=3)
        return np.clip(crop, a_min=0, a_max=255)
