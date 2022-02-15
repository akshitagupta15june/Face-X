"""Test dog mask application for a single, static image."""

import argparse
import cv2
import numpy as np


def apply_mask(face: np.array, mask: np.array) -> np.array:
    """Add the mask to the provided face, and return the face with mask."""
    mask_h, mask_w, _ = mask.shape
    face_h, face_w, _ = face.shape

    # Resize the mask to fit on face
    factor = min(face_h / mask_h, face_w / mask_w)
    new_mask_w = int(factor * mask_w)
    new_mask_h = int(factor * mask_h)
    new_mask_shape = (new_mask_w, new_mask_h)
    resized_mask = cv2.resize(mask, new_mask_shape)

    # Add mask to face - ensure mask is centered
    face_with_mask = face.copy()
    non_white_pixels = (resized_mask < 250).all(axis=2)
    off_h = int((face_h - new_mask_h) / 2)
    off_w = int((face_w - new_mask_w) / 2)
    face_with_mask[off_h: off_h+new_mask_h, off_w: off_w+new_mask_w][non_white_pixels] = \
         resized_mask[non_white_pixels]

    return face_with_mask


def main():
    args = argparse.ArgumentParser('Dog Mask for single assets')
    args.add_argument('--mask', default='assets/dog.png')
    args = args.parse_args()

    face = cv2.imread('assets/child.png')
    mask = cv2.imread(args.mask)
    face_with_mask = apply_mask(face, mask)
    cv2.imwrite('outputs/child_with_dog_mask.png', face_with_mask)

if __name__ == '__main__':
    main()