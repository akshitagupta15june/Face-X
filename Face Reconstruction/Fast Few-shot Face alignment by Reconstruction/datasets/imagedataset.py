import os
import sys
import math
import numbers

import numpy as np
from csl_common.utils import transforms as csl_tf, geometry
from csl_common.utils import image_loader
from csl_common.utils.image_loader import CachedCropLoader
from torchvision import transforms as tf
import torchvision.datasets as tdv
import pandas as pd



class ImageDataset(tdv.VisionDataset):

    def __init__(self, root, fullsize_img_dir, image_size, output_size=None, cache_root=None, train=True,
                 transform=None, target_transform=None, crop_type='tight', color=True, start=None, max_samples=None,
                 use_cache=True, test_split='fullset', crop_source='bb_ground_truth', loader=None,
                 roi_background='black', crop_dir='crops', roi_margin=None, median_blur_crop=False,
                 **kwargs):

        print("Setting up dataset {}...".format(self.__class__.__name__))

        if not isinstance(image_size, numbers.Number):
            raise FileNotFoundError(f"Image size must be scalar number (image_size={image_size}).")

        if not os.path.exists(root):
            raise FileNotFoundError(f"Invalid dataset root path: '{root}'")

        if cache_root is not None and not os.path.exists(cache_root):
            raise FileNotFoundError(f"Invalid dataset cache path: '{cache_root}'")

        if not os.path.exists(root):
            raise FileNotFoundError(f"Image directory not found: '{root}'")

        self.fullsize_img_dir = fullsize_img_dir
        self.root = root
        self.cache_root = cache_root if cache_root is not None else self.root

        self.image_size = image_size
        if output_size is not None:
            self.output_size = output_size
        else:
            self.output_size = image_size

        if roi_margin is None:
            # crop size equals input diagonal, so images can be fully rotated
            self.roi_size = geometry.get_diagonal(image_size)
            self.margin = self.roi_size - self.image_size
        else:
            self.roi_size = image_size + roi_margin
            self.margin = roi_margin

        self.crop_dir = crop_dir
        self.test_split = test_split
        self.split = 'train' if train else self.test_split
        self.train = train
        self.use_cache = use_cache
        self.crop_source = crop_source
        self.crop_type = crop_type
        self.start = start
        self.max_samples = max_samples
        self.color = color

        self.annotations = self._load_annotations(self.split)
        self._init()
        self._select_index_range()

        transforms = [csl_tf.CenterCrop(self.output_size)]
        transforms += [csl_tf.ToTensor()]
        transforms += [csl_tf.Normalize()]
        self.crop_to_tensor = tf.Compose(transforms)

        if loader is not None:
            self.loader = loader
        else:
            self.loader = CachedCropLoader(fullsize_img_dir,
                                           self.cropped_img_dir,
                                           img_size=self.image_size,
                                           margin=self.margin,
                                           use_cache=self.use_cache,
                                           crop_type=crop_type,
                                           border_mode=roi_background,
                                           median_blur_crop=median_blur_crop)

        super().__init__(root, transform=transform, target_transform=target_transform)


    @property
    def cropped_img_dir(self):
        return os.path.join(self.cache_root, self.crop_dir, self.crop_source)

    def _init(self):
        pass

    def _load_annotations(self, split):
        raise NotImplementedError

    @property
    def name(self):
        return self.__class__.__name__

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, idx):
        raise NotImplementedError

    def filter_labels(self, label_dict):
        import collections
        print("Applying filter to labels: {}".format(label_dict))
        for k, v in label_dict.items():
            if isinstance(v, collections.Sequence):
                selected_rows = self.annotations[k].isin(v)
            else:
                selected_rows = self.annotations[k] == v
            self.annotations = self.annotations[selected_rows]
        print("  Number of images: {}".format(len(self.annotations)))

    def _select_index_range(self):
        st,nd = 0, None
        if self.start is not None:
            st = self.start
        if self.max_samples is not None:
            nd = st + self.max_samples
        self.annotations = self.annotations[st:nd]

    @property
    def labels(self):
        return NotImplemented

    def _get_image_roi_from_bbox(self, bbox):
        return image_loader.get_roi_from_bbox(bbox, crop_size=self.image_size, margin=self.margin)

    def get_sample(self, filename, bb=None, landmarks_for_crop=None, id=None):
        image_roi = self._get_image_roi_from_bbox(bb)
        try:
            image  = self.loader.load_crop(filename, bb=image_roi, id=id)
        except:
            print('Could not load image {}'.format(filename))
            raise

        if self.transform is not None:
            image = self.transform(image)
        target = self.target_transform(image.copy()) if self.target_transform else None

        if self.crop_type != 'fullsize':
            image = self.crop_to_tensor(image)
            if target is not None:
                target = self.crop_to_tensor(target)

        sample = ({ 'image': image,
                    'fnames': filename,
                    'bb': bb if bb is not None else [0,0,0,0]})

        if target is not None:
            sample['target'] = target

        return sample



class ImageFolderDataset(ImageDataset):
    def __init__(self, root, fullsize_img_dir, image_size, **kwargs):
        super().__init__(root, fullsize_img_dir, image_size, **kwargs)

    def _find_classes(self, dir):
        """
        Finds the class folders in a dataset.

        Args:
            dir (string): Root directory path.

        Returns:
            tuple: (classes, class_to_idx) where classes are relative to (dir), and class_to_idx is a dictionary.

        Ensures:
            No class is a subdirectory of another.
        """
        if sys.version_info >= (3, 5):
            # Faster and available in Python 3.5 and above
            classes = [d.name for d in os.scandir(dir) if d.is_dir()]
        else:
            classes = [d for d in os.listdir(dir) if os.path.isdir(os.path.join(dir, d))]
        classes.sort()
        class_to_idx = {classes[i]: i for i in range(len(classes))}
        return classes, class_to_idx

    def _load_annotations(self, split):
        def strip_root(fullpath, root):
            return os.path.relpath(fullpath, root)

        from torchvision.datasets import folder
        classes, class_to_idx = self._find_classes(self.root)
        samples = folder.make_dataset(self.root, class_to_idx, extensions=folder.IMG_EXTENSIONS)
        self.classes = classes
        self.class_to_ids = class_to_idx
        fnames = [strip_root(s[0], self.root) for s in samples]
        labels = [s[1] for s in samples]
        ann = pd.DataFrame({'fname': fnames, 'label': labels})
        return ann
