import numbers
import random

import cv2
import numpy as np
import skimage.transform
from torchvision.transforms import functional as F


class CenterCrop(object):
    """Like tf.CenterCrop, but works works on numpy arrays instead of PIL images."""

    def __init__(self, size):
        if isinstance(size, numbers.Number):
            self.size = (int(size), int(size))
        else:
            self.size = size

    def __crop_image(self, img):
        t = int((img.shape[0] - self.size[0]) / 2)
        l = int((img.shape[1] - self.size[1]) / 2)
        b = t + self.size[0]
        r = l + self.size[1]
        return img[t:b, l:r]

    def __call__(self, sample):
        if isinstance(sample, dict):
            img, landmarks, pose = sample['image'], sample['landmarks'], sample['pose']
            if landmarks is not None:
                landmarks[...,0] -= int((img.shape[0] - self.size[0]) / 2)
                landmarks[...,1] -= int((img.shape[1] - self.size[1]) / 2)
                landmarks[landmarks < 0] = 0
            return {'image': self.__crop_image(img), 'landmarks': landmarks, 'pose': pose}
        else:
            return self.__crop_image(sample)


    def __repr__(self):
        return self.__class__.__name__ + '(size={})'.format(self.size)


class RandomRotation(object):
    """Rotate the image by angle.

    Like tf.RandomRotation, but works works on numpy arrays instead of PIL images.

    """

    def __init__(self, degrees):
        if isinstance(degrees, numbers.Number):
            if degrees < 0:
                raise ValueError("If degrees is a single number, it must be positive.")
            self.degrees = (-degrees, degrees)
        else:
            if len(degrees) != 2:
                raise ValueError("If degrees is a sequence, it must be of len 2.")
            self.degrees = degrees

    @staticmethod
    def get_params(degrees):
        angle = random.uniform(degrees[0], degrees[1])
        return angle

    def __call__(self, sample):
        image, landmarks, pose = sample['image'], sample['landmarks'], sample['pose']
        angle = self.get_params(self.degrees)
        h, w = image.shape[:2]
        center = (w//2, h//2)
        M = calc_rotation_matrix(center, angle)
        img_rotated = rotate_image(image, M)
        if landmarks is not None:
            landmarks = rotate_landmarks(landmarks, M).astype(np.float32)
            pose_rotated = pose
            pose_rotated[2] -= np.deg2rad(angle).astype(np.float32)
        return {'image': img_rotated, 'landmarks': landmarks, 'pose': pose}

    def __repr__(self):
        format_string = self.__class__.__name__ + '(degrees={0}'.format(self.degrees)
        format_string += ')'
        return format_string


class Rescale(object):
    """Rescale the image in a sample to a given size.

    Args:
        output_size (tuple or int): Desired output size. If tuple, output is
            matched to output_size. If int, smaller of image edges is matched
            to output_size keeping aspect ratio the same.
    """

    def __init__(self, output_size):
        assert isinstance(output_size, (numbers.Number, tuple))
        self.output_size = output_size

    def __call__(self, sample):
        image, landmarks, pose = sample['image'], sample['landmarks'], sample['pose']

        h, w = image.shape[:2]

        if isinstance(self.output_size, numbers.Number):
            if h > w:
                new_h, new_w = self.output_size * h / w, self.output_size
            else:
                new_h, new_w = self.output_size, self.output_size * w / h
        else:
            new_h, new_w = self.output_size

        new_h, new_w = int(new_h), int(new_w)

        img = cv2.resize(image, dsize=(new_h, new_w))

        # h and w are swapped for landmarks because for images,
        # x and y axes are axis 1 and 0 respectively
        if landmarks is not None:
            landmarks = landmarks * [new_w / w, new_h / h]
            landmarks = landmarks.astype(np.float32)

        return {'image': img, 'landmarks': landmarks, 'pose': pose}


class RandomCrop(object):
    """Crop randomly the image in a sample.

    Args:
        output_size (tuple or int): Desired output size. If int, square crop
            is made.
    """

    def __init__(self, output_size):
        assert isinstance(output_size, (int, tuple))
        if isinstance(output_size, int):
            self.output_size = (output_size, output_size)
        else:
            assert len(output_size) == 2
            self.output_size = output_size

    def __call__(self, sample):
        image, landmarks, pose = sample['image'], sample['landmarks'], sample['pose']

        h, w = image.shape[:2]
        new_h, new_w = self.output_size

        top = np.random.randint(0, h - new_h)
        left = np.random.randint(0, w - new_w)

        image = image[top: top + new_h,
                      left: left + new_w]

        if landmarks is not None:
            landmarks = landmarks - [left, top]
            landmarks = landmarks.astype(np.float32)

        return {'image': image, 'landmarks': landmarks, 'pose': pose}


class RandomResizedCrop(object):
    """Crop randomly the image in a sample.

    Args:
        output_size (tuple or int): Desired output size. If int, square crop
            is made.
    """

    def __init__(self, output_size, p=1.0, scale=(1.0, 1.0), keep_aspect=True):
        assert isinstance(output_size, (int, tuple))
        if isinstance(output_size, int):
            self.output_size = (output_size, output_size)
        else:
            assert len(output_size) == 2
            self.output_size = output_size

        self.scale = scale
        self.p = p
        self.keep_aspect = keep_aspect

    def __call__(self, sample):
        image, landmarks, pose = sample['image'], sample['landmarks'], sample['pose']

        h, w = image.shape[:2]
        s_x = random.uniform(*self.scale)
        if self.keep_aspect:
            s_y = s_x
        else:
            s_y = random.uniform(*self.scale)
        new_w, new_h = int(self.output_size[0] * s_x), int(self.output_size[1] * s_y)

        top = np.random.randint(0, h - new_h)
        left = np.random.randint(0, w - new_w)

        image = image[top: top + new_h,
                left: left + new_w]

        landmarks = landmarks - [left, top]

        image = cv2.resize(image, dsize=self.output_size)
        landmarks /= [s_x, s_y]

        return {'image': image, 'landmarks': landmarks.astype(np.float32), 'pose': pose.astype(np.float32)}


class RandomHorizontalFlip(object):
    """Horizontally flip the given numpy array randomly with a given probability.

    Args:
        p (float): probability of the image being flipped. Default value is 0.5
    """
    lm_left_to_right_98 = {
        # outline
        0:32,
        1:31,
        2:30,
        3:29,
        4:28,
        5:27,
        6:26,
        7:25,
        8:24,

        9:23,
        10:22,
        11:21,
        12:20,
        13:19,
        14:18,
        15:17,
        16:16,

        #eyebrows
        33:46,
        34:45,
        35:44,
        36:43,
        37:42,
        38:50,
        39:49,
        40:48,
        41:47,

        #nose
        51:51,
        52:52,
        53:53,
        54:54,

        55:59,
        56:58,
        57:57,

        #eyes
        60:72,
        61:71,
        62:70,
        63:69,
        64:68,
        65:75,
        66:74,
        67:73,
        96:97,

        #mouth outer
        76:82,
        77:81,
        78:80,
        79:79,
        87:83,
        86:84,
        85:85,

        #mouth inner
        88:92,
        89:91,
        90:90,
        95:93,
        94:94,
    }

    lm_left_to_right_68 = {
        # outline
        0:16,
        1:15,
        2:14,
        3:13,
        4:12,
        5:11,
        6:10,
        7:9,
        8:8,

        #eyebrows
        17:26,
        18:25,
        19:24,
        20:23,
        21:22,

        #nose
        27:27,
        28:28,
        29:29,
        30:30,

        31:35,
        32:34,
        33:33,

        #eyes
        36:45,
        37:44,
        38:43,
        39:42,
        40:47,
        41:46,

        #mouth outer
        48:54,
        49:53,
        50:52,
        51:51,
        57:57,
        58:56,
        59:55,

        #mouth inner
        60:64,
        61:63,
        62:62,
        66:66,
        67:65,
    }

    # AFLW
    lm_left_to_right_21 = {
        0:5,
        1:4,
        2:3,
        6:11,
        7:10,
        8:9,

        12:16,
        13:15,
        14:14,
        17:19,
        18:18,
        20:20
    }

    # AFLW without ears
    lm_left_to_right_19 = {
        0:5,
        1:4,
        2:3,
        6:11,
        7:10,
        8:9,

        12:14,
        13:13,
        15:17,
        16:16,
        18:18
    }

    lm_left_to_right_5 = {
        0:1,
        2:2,
        3:4,
    }

    lm_left_to_right_38 = {
        # eye brows
        0: 5,
        1: 4,
        2: 3,

        # eyes
        12: 24,
        13: 23,
        14: 22,
        15: 21,
        16: 20,
        17: 27,
        18: 26,
        19: 25,

        # nose
        6: 6,
        7: 7,
        8: 8,
        9: 11,
        10: 10,

        # mouth
        28: 34,
        29: 33,
        30: 32,
        31: 31,
        36: 36,
        37: 37
    }

    # DeepFashion full body fashion landmarks
    lm_left_to_right_8 = {
        0:1,
        2:3,
        4:5,
        6:7,
    }

    def __init__(self, p=0.5):

        def build_landmark_flip_map(left_to_right):
            map = left_to_right
            right_to_left = {v:k for k,v in map.items()}
            map.update(right_to_left)
            return map

        self.p = p

        self.lm_flip_map_98 = build_landmark_flip_map(self.lm_left_to_right_98)
        self.lm_flip_map_68 = build_landmark_flip_map(self.lm_left_to_right_68)
        self.lm_flip_map_21 = build_landmark_flip_map(self.lm_left_to_right_21)
        self.lm_flip_map_19 = build_landmark_flip_map(self.lm_left_to_right_19)
        self.lm_flip_map_5 = build_landmark_flip_map(self.lm_left_to_right_5)
        self.lm_flip_map_8 = build_landmark_flip_map(self.lm_left_to_right_8)
        self.lm_flip_map_38 = build_landmark_flip_map(self.lm_left_to_right_38)


    def __call__(self, sample):
        if random.random() < self.p:
            if isinstance(sample, dict):
                img, landmarks, pose = sample['image'], sample['landmarks'], sample['pose']
                # flip image
                flipped_img = np.fliplr(img).copy()
                # flip landmarks
                non_zeros = landmarks[:,0] > 0
                landmarks[non_zeros, 0] *= -1
                landmarks[non_zeros, 0] += img.shape[1]
                landmarks_new = landmarks.copy()
                if len(landmarks) == 21:
                    lm_flip_map = self.lm_flip_map_21
                elif len(landmarks) == 19:
                    lm_flip_map = self.lm_flip_map_19
                elif len(landmarks) == 68:
                    lm_flip_map = self.lm_flip_map_68
                elif len(landmarks) == 5:
                    lm_flip_map = self.lm_flip_map_5
                elif len(landmarks) == 98:
                    lm_flip_map = self.lm_flip_map_98
                elif len(landmarks) == 8:
                    lm_flip_map = self.lm_flip_map_8
                elif len(landmarks) == 38:
                    lm_flip_map = self.lm_flip_map_38
                else:
                    raise ValueError('Invalid landmark format.')
                for i in range(len(landmarks)):
                    landmarks_new[i] = landmarks[lm_flip_map[i]]
                # flip pose
                if pose is not None:
                    pose[1] *= -1
                return {'image': flipped_img, 'landmarks': landmarks_new, 'pose': pose}

            return np.fliplr(sample).copy()
        return sample

    def __repr__(self):
        return self.__class__.__name__ + '(p={})'.format(self.p)


class RandomAffine(object):
    """Random affine transformation of the image keeping center invariant

    Args:
        degrees (sequence or float or int): Range of degrees to select from.
            If degrees is a number instead of sequence like (min, max), the range of degrees
            will be (-degrees, +degrees). Set to 0 to deactivate rotations.
        translate (tuple, optional): tuple of maximum absolute fraction for horizontal
            and vertical translations. For example translate=(a, b), then horizontal shift
            is randomly sampled in the range -img_width * a < dx < img_width * a and vertical shift is
            randomly sampled in the range -img_height * b < dy < img_height * b. Will not translate by default.
        scale (tuple, optional): scaling factor interval, e.g (a, b), then scale is
            randomly sampled from the range a <= scale <= b. Will keep original scale by default.
        shear (sequence or float or int, optional): Range of degrees to select from.
            If degrees is a number instead of sequence like (min, max), the range of degrees
            will be (-degrees, +degrees). Will not apply shear by default
        resample ({PIL.Image.NEAREST, PIL.Image.BILINEAR, PIL.Image.BICUBIC}, optional):
            An optional resampling filter. See `filters`_ for more information.
            If omitted, or if the image has mode "1" or "P", it is set to PIL.Image.NEAREST.
        fillcolor (int): Optional fill color for the area outside the transform in the output image. (Pillow>=5.0.0)

    .. _filters: https://pillow.readthedocs.io/en/latest/handbook/concepts.html#filters

    """

    def __init__(self, degrees=0, translate=None, scale=None, shear=None, resample=False, fillcolor=0, keep_aspect=True):
        if isinstance(degrees, numbers.Number):
            if degrees < 0:
                raise ValueError("If degrees is a single number, it must be positive.")
            self.angle_range = (-degrees, degrees)
        else:
            assert isinstance(degrees, (tuple, list)) and len(degrees) == 2, \
                "degrees should be a list or tuple and it must be of length 2."
            self.angle_range = degrees

        if translate is not None:
            assert isinstance(translate, (tuple, list)) and len(translate) == 2, \
                "translate should be a list or tuple and it must be of length 2."
            for t in translate:
                if not (0.0 <= t <= 1.0):
                    raise ValueError("translation values should be between 0 and 1")
        self.translate = translate

        if scale is not None:
            assert isinstance(scale, (tuple, list)) and len(scale) == 2, \
                "scale should be a list or tuple and it must be of length 2."
            for s in scale:
                if s <= 0:
                    raise ValueError("scale values should be positive")
        self.scale_range = scale

        if shear is not None:
            if isinstance(shear, numbers.Number):
                if shear < 0:
                    raise ValueError("If shear is a single number, it must be positive.")
                self.shear_range = (-shear, shear)
            else:
                assert isinstance(shear, (tuple, list)) and len(shear) == 2, \
                    "shear should be a list or tuple and it must be of length 2."
                self.shear_range = shear
        else:
            self.shear_range = shear

        self.resample = resample
        self.fillcolor = fillcolor
        self.keep_aspect = keep_aspect

    # @staticmethod
    # def get_params(degrees, translate, scale_range, shears, img_size, keep_aspect):
    def get_params(self, img_size):
        """Get parameters for affine transformation

        Returns:
            sequence: params to be passed to the affine transformation
        """
        angle = random.uniform(self.angle_range[0], self.angle_range[1])

        if self.translate is not None:
            max_dx = self.translate[0] * img_size[0]
            max_dy = self.translate[1] * img_size[1]
            translations = (-np.round(random.uniform(-max_dx, max_dx)),
                            -np.round(random.uniform(-max_dy, max_dy)))
        else:
            translations = (0, 0)

        if self.scale_range is not None:
            scale_x = random.uniform(self.scale_range[0], self.scale_range[1])
            if self.keep_aspect:
                scale_y = scale_x
            else:
                scale_y = random.uniform(self.scale_range[0], self.scale_range[1])
        else:
            scale_x, scale_y = 1.0, 1.0

        if self.shear_range is not None:
            shear = random.uniform(self.shear_range[0], self.shear_range[1])
        else:
            shear = 0.0

        return angle, translations, (scale_x, scale_y), shear

    def _get_full_matrix(self, angle, translations, scales, shear, img_size):
        M = skimage.transform.AffineTransform(
            rotation=np.deg2rad(angle),
            translation=translations,
            shear=np.deg2rad(shear),
            scale=scales,
        )
        t = skimage.transform.AffineTransform(translation=-np.array(img_size[::-1])/2)
        return skimage.transform.AffineTransform(matrix=t._inv_matrix.dot(M.params.dot(t.params)))

    def __call__(self, sample):
        if isinstance(sample, dict):
            img, landmarks, pose = sample['image'], sample['landmarks'], sample['pose']
        else:
            img = sample

        angle, translations, scale, shear = self.get_params(img.shape[:2])
        M = self._get_full_matrix(angle, translations, scale, shear, img.shape[:2])
        img_new = transform_image(img, M)

        if isinstance(sample, dict):
            if landmarks is None:
                landmarks_new = None
            else:
                landmarks_new = transform_landmarks(landmarks, M).astype(np.float32)
            return {'image': img_new, 'landmarks': landmarks_new, 'pose': pose}
        else:
            return img_new

    def get_matrix(self, img_size):
        if isinstance(img_size, numbers.Number):
            self.degrees = (img_size, img_size)
        else:
            assert isinstance(img_size, (tuple, list)) and len(img_size) == 2, \
                "img_size should be a list or tuple and it must be of length 2."
        return self.get_params(img_size)

    def __repr__(self):
        s = f'{self.__class__.__name__}(degrees={self.angle_range}'
        if self.translate is not None:
            s += f', translate={self.translate}'
        if self.scale_range is not None:
            s += f', scale={self.scale_range}'
        if self.shear_range is not None:
            s += f', shear={self.shear_range}'
        if self.resample > 0:
            s += f', resample={self.resample}'
        if self.fillcolor != 0:
            s += f', fillcolor={self.fillcolor}'
        s += ')'
        return s


class RandomLowQuality(object):
    """Reduce image quality by as encoding as low quality jpg.

    Args:
        p (float): probability of the image being recoded. Default value is 0.2
        qmin (float): min jpg quality
        qmax (float): max jpg quality
    """

    def __init__(self, p=0.5, qmin=8, qmax=25):
        self.p = p
        self.qmin = qmin
        self.qmax = qmax

    def _encode(self, img, q):
        return cv2.imencode('.jpg', img, params=[int(cv2.IMWRITE_JPEG_QUALITY), q])

    def _recode(self, img, q):
        return cv2.imdecode(self._encode(img, q)[1], flags=cv2.IMREAD_COLOR)

    def __call__(self, img):
        """
        Args:
            img (PIL Image): Image to be recoded .

        Returns:
            PIL Image: Randomly flipped image.
        """
        if random.random() < self.p:
            return self._recode(img, random.randint(self.qmin, self.qmax))
        return img

    def __repr__(self):
        return self.__class__.__name__ + '(p={})'.format(self.p)


class RandomOcclusion(object):
    def __init__(self, img_size, crop_border):
        self.img_size = img_size
        self.crop_border = crop_border
        self.bkg_size = self.crop_border
        self.max_occ_size = self.img_size
        self.min_occ_size = self.max_occ_size // 10

    def __add_occlusions(self, img):

        cx = random.randint(self.bkg_size, self.bkg_size+self.img_size)
        cy = random.randint(self.bkg_size, self.bkg_size+self.img_size)

        w_half = min(img.shape[1]-cx-1, random.randint(self.min_occ_size, self.max_occ_size)) // 2
        h_half = min(img.shape[0]-cy-1, random.randint(self.min_occ_size, self.max_occ_size)) // 2
        w_half = min(cx, w_half)
        h_half = min(cy, h_half)

        l = 0
        t = random.randint(h_half+1, self.img_size)

        r = self.bkg_size
        b = min(img.shape[0]-1, t+2*h_half)

        cutout = img[t:b, l:r]
        dst_shape = (2*h_half, 2*w_half)

        if cutout.shape[:2] != dst_shape:
            try:
                cutout = cv2.resize(cutout, dsize=dst_shape[::-1], interpolation=cv2.INTER_CUBIC)
            except:
                print('resize error', img.shape, dst_shape, cutout.shape[:2], cy, cx, h_half, w_half)

        try:
            cutout = cv2.blur(cutout, ksize=(5,5))
            img[cy-h_half:cy+h_half, cx-w_half:cx+w_half] = cutout
        except:
            print(img.shape, dst_shape, cutout.shape[:2], cy, cx, h_half, w_half)
        # plt.imshow(img)
        # plt.show()
        return img

    def __call__(self, sample):
        if isinstance(sample, dict):
            img, landmarks, pose = sample['image'], sample['landmarks'], sample['pose']
            return {'image': self.__add_occlusions(img), 'landmarks': landmarks, 'pose': pose}
        else:
            return self.__add_occlusions(sample)

    def __repr__(self):
        return self.__class__.__name__ + '(size={})'.format(self.size)


def calc_rotation_matrix(center, degrees):
    return cv2.getRotationMatrix2D(tuple(center), degrees, 1.0)


def rotate_image(img, M):
    return cv2.warpAffine(img, M, img.shape[:2][::-1], flags=cv2.INTER_CUBIC)


def rotate_landmarks(lms, M):
    _lms_hom = np.hstack((lms, np.ones((lms.shape[0], 1))))  # make landmarks homogeneous
    return M.dot(_lms_hom.T).T  # apply transformation


def transform_image(img, M):
    return cv2.warpAffine(img, M.params[:2], img.shape[:2][::-1], flags=cv2.INTER_CUBIC)


def transform_landmarks(lms, M):
    _lms_hom = np.hstack((lms, np.ones((lms.shape[0], 1))))  # make landmarks homogeneous
    # t = skimage.transform.AffineTransform(translation=-np.array(img.shape[:2][::-1])/2)
    # m = t._inv_matrix.dot(M.params.dot(t.params))
    # return M.params.dot(_lms_hom.T).T[:,:2]
    return M(lms)


class Normalize(object):
    """Normalize a tensor image with mean and standard deviation.
    Given mean: ``(M1,...,Mn)`` and std: ``(S1,..,Sn)`` for ``n`` channels, this transform
    will normalize each channel of the input ``torch.*Tensor`` i.e.
    ``input[channel] = (input[channel] - mean[channel]) / std[channel]``

    Args:
        mean (sequence): Sequence of means for each channel.
        std (sequence): Sequence of standard deviations for each channel.
    """

    def __init__(self, mean=None, std=None):
        if mean is None:
            mean = [0.518, 0.418, 0.361] # VGGFace(2) means
        if std is None:
            std = [1, 1, 1]
        self.mean = mean
        self.std = std

    def __call__(self, sample):
        """
        Args:
            tensor (Tensor): Tensor image of size (C, H, W) to be normalized.

        Returns:
            Tensor: Normalized Tensor image.
        """
        if isinstance(sample, dict):
            sample['image'] = F.normalize(sample['image'], self.mean, self.std)
        else:
            sample = F.normalize(sample, self.mean, self.std)
        return sample


    def __repr__(self):
        return self.__class__.__name__ + '(mean={0}, std={1})'.format(self.mean, self.std)


class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, sample):
        if isinstance(sample, dict):
            image, landmarks, pose = sample['image'], sample['landmarks'], sample['pose']

            # swap color axis because
            # numpy image: H x W x C
            # torch image: C X H X W
            # image = image.transpose((2, 0, 1))
            return {'image': F.to_tensor(image),
                    'landmarks': landmarks,
                    'pose': pose}
        else:
            return F.to_tensor(sample)
            # return torch.from_numpy(sample)