# -*- coding: utf-8 -*-
# @Author  : Jiaxiang Shang
# @Email   : jiaxiang.shang@gmail.com
# @Time    : 8/11/20 8:46 PM

import numpy as np
import cv2
import math
from skimage import transform as trans

std_224_bfm09 = [
    81.672401, 88.470589,
    141.862671, 88.462921,
    112.000000, 132.863434,
    87.397392, 153.562943,
    136.007263, 153.552078
]
std_224_bfm09 = np.array(std_224_bfm09)
std_224_bfm09 = np.reshape(std_224_bfm09, [-1, 2])

def cvrt_300w_to_CelebA(lm68):
    l_eye = np.mean(lm68[37 - 1:43 - 1], axis=0)

    r_eye = np.mean(lm68[43 - 1:49 - 1], axis=0)

    nose = lm68[34 - 1]
    l_m = lm68[49 - 1]
    r_m = lm68[55 - 1]

    return [l_eye, r_eye, nose, l_m, r_m]

def inverse_affine_warp_overlay(m_inv, image_ori, image_now, image_mask_now):
    from skimage import transform as trans
    tform = trans.SimilarityTransform(m_inv)
    M = tform.params[0:2, :]

    image_now_cv = cv2.cvtColor(image_now, cv2.COLOR_RGB2BGR)
    image_mask_now_cv = cv2.cvtColor(image_mask_now, cv2.COLOR_RGB2BGR)



    img_now_warp = cv2.warpAffine(image_now_cv, M, (image_ori.shape[1], image_ori.shape[0]), flags=cv2.INTER_LINEAR,
                             borderMode=cv2.BORDER_REPLICATE)
    image_mask_now_warp = cv2.warpAffine(image_mask_now_cv, M, (image_ori.shape[1], image_ori.shape[0]), flags=cv2.INTER_LINEAR,
                             borderMode=cv2.BORDER_REPLICATE)

    image_ori_back = (1.0 - image_mask_now_warp) * image_ori
    image_ori_back = image_ori_back.astype(np.uint8)
    image_ori_back = np.clip(image_ori_back, 0, 255)
    # if 1:
    #     cv2.imshow("Image Debug", image_ori_back)
    #     k = cv2.waitKey(0) & 0xFF
    #     if k == 27:
    #         cv2.destroyAllWindows()

    img_now_warp = img_now_warp * image_mask_now_warp
    img_now_warp = img_now_warp.astype(np.uint8)
    img_now_warp = np.clip(img_now_warp, 0, 255)

    img_replace = img_now_warp + image_ori_back
    img_replace = np.clip(img_replace, 0, 255)


    img_replace = img_replace.astype(np.uint8)
    img_replace = np.clip(img_replace, 0, 255)

    return img_replace

def crop_align_affine_transform(lm2d, image, crop_size, std_landmark):
    lm_celebA = cvrt_300w_to_CelebA(lm2d)
    # Transform
    std_points = np.array(std_landmark) * (crop_size / 224.0)

    tform = trans.SimilarityTransform()
    tform.estimate(np.array(lm_celebA), std_points)
    M = tform.params[0:2, :]

    rot_angle = tform.rotation * 180.0 / (math.pi)
    #print(rot_angle, tform.translation)

    img_warped = cv2.warpAffine(image, M, (crop_size, crop_size), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)

    h_lm2d_home = np.concatenate([lm2d, np.ones([lm2d.shape[0], 1])], axis=1)
    lm_trans = np.matmul(M, np.array(np.transpose(h_lm2d_home)))
    lm_trans = np.transpose(lm_trans)

    return lm_trans, img_warped, tform

