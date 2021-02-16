import os
import cv2
import numpy as np

import torch

# from utils import cropping as fp
from csl_common.utils import nn, cropping
from csl_common import utils

from landmarks import fabrec
from torchvision import transforms as tf
from landmarks import lmvis

snapshot_dir = os.path.join('.')

INPUT_SIZE = 256

transforms = [utils.transforms.CenterCrop(INPUT_SIZE)]
transforms += [utils.transforms.ToTensor()]
transforms += [utils.transforms.Normalize([0.518, 0.418, 0.361], [1, 1, 1])]
crop_to_tensor = tf.Compose(transforms)


def load_image(im_dir, fname):
    from skimage import io
    img_path = os.path.join(im_dir, fname)
    img = io.imread(img_path)
    if img is None:
        raise IOError("\tError: Could not load image {}!".format(img_path))
    if len(img.shape) == 2 or img.shape[2] == 1:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    if img.shape[2] == 4:
        print(fname, "converting RGBA to RGB...")
        img = cv2.cvtColor(img, cv2.COLOR_RGBA2RGB)
    assert img.shape[2] == 3, "{}, invalid format: {}".format(img_path, img.shape)
    return img


def detect_in_crop(net, crop):
    with torch.no_grad():
        X_recon, lms_in_crop, X_lm_hm = net.detect_landmarks(crop)
    lms_in_crop = utils.nn.to_numpy(lms_in_crop.reshape(1, -1, 2))
    return X_recon, lms_in_crop, X_lm_hm


def test_crop(net, input_image, gt_landmarks, bb_for_crop=None, lms_for_crop=None, align=False, scale=1.0):
    assert bb_for_crop is not None or lms_for_crop is not None

    cropper = cropping.FaceCrop(input_image, bbox=bb_for_crop, landmarks=lms_for_crop,
                                align_face_orientation=align, scale=scale,
                                output_size=(INPUT_SIZE, INPUT_SIZE))
    crop = cropper.apply_to_image()
    landmarks = cropper.apply_to_landmarks(gt_landmarks)[0]

    item = {'image': crop, 'landmarks': landmarks, 'pose': None}
    item = crop_to_tensor(item)

    images = nn.atleast4d(item['image']).cuda()
    X_recon, lms, X_lm_hm = detect_in_crop(net, images)

    # lmvis.visualize_batch(images, landmarks, X_recon, X_lm_hm, lms, wait=0, clean=True)
    lmvis.visualize_batch_CVPR(images, landmarks, X_recon, X_lm_hm, lms, wait=0,
                               horizontal=True, show_recon=True, radius=2, draw_wireframes=True)


if __name__ == '__main__':

    model = './data/models/snapshots/demo'
    net = fabrec.load_net(model, num_landmarks=98)
    net.eval()

    im_dir = './images'
    img0 = 'ada.jpg'

    with torch.no_grad():

        img = load_image(im_dir, img0)

        scalef = 0.65
        bb0 = [0,0] + list(img.shape[:2][::-1])
        bb = utils.geometry.scaleBB(bb0, scalef, scalef, typeBB=2)
        test_crop(net, img, gt_landmarks=None, bb_for_crop=bb)


