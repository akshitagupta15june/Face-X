
# system
from __future__ import print_function

import os
import sys

# third party
import math
import numpy as np
import cv2
from skimage import transform as trans

# 0.75, 4700 intrinsic
# main pose [0.000000, 0.000000, 3.141593, 0.17440447, 9.1053238, 4994.3359]

std_224_bfm09 = [
    81.672401, 88.470589,
    141.862671, 88.462921,
    112.000000, 132.863434,
    87.397392, 153.562943,
    136.007263, 153.552078
]
std_224_bfm09 = np.array(std_224_bfm09)
std_224_bfm09 = np.reshape(std_224_bfm09, [-1, 2])

std_224_bfm09_lmFull = [
    42.820927, 96.560013,
    44.740612, 115.502319,
    48.336060, 131.956711,
    51.973404, 147.542618,
    56.942982, 163.242767,
    66.128014, 176.873581,
    77.121262, 185.345123,
    91.350266, 192.333847,
    112.111984, 195.973877,
    132.907944, 192.150299,
    147.110550, 185.404709,
    158.049088, 177.110962,
    167.180313, 163.636917,
    172.164627, 148.013474,
    175.628494, 132.314041,
    178.904556, 115.763123,
    180.747192, 96.624603,
    57.339161, 75.401939,
    65.075607, 69.420296,
    75.346153, 67.477211,
    84.851959, 68.512169,
    93.684509, 70.445839,
    129.837448, 70.238472,
    138.728607, 68.199989,
    148.288986, 67.130638,
    158.729416, 69.066544,
    166.466049, 75.241402,
    112.025818, 88.439606,
    112.079277, 100.463402,
    112.175385, 112.699623,
    112.155502, 123.107132,
    100.997696, 130.704987,
    105.334106, 131.744461,
    112.000000, 132.863434,
    118.604187, 131.729279,
    122.843803, 130.633392,
    70.887840, 88.658264,
    76.957016, 84.771416,
    85.494408, 84.991066,
    93.396942, 89.300255,
    86.316574, 91.344498,
    76.981613, 91.758080,
    129.938248, 89.322220,
    137.879929, 84.897110,
    146.571060, 84.798187,
    152.894745, 88.773125,
    146.685852, 91.653938,
    137.206253, 91.332932,
    87.397392, 153.562943,
    95.851616, 147.984695,
    106.369308, 144.729660,
    111.973000, 145.595917,
    117.534439, 144.735779,
    128.142181, 147.902420,
    136.007263, 153.552078,
    127.221329, 157.526154,
    119.790115, 160.680283,
    112.020966, 160.912857,
    104.267090, 160.673126,
    96.875687, 157.533768,
    89.817505, 152.756683,
    104.776360, 150.597717,
    111.898491, 150.485580,
    119.044411, 150.646561,
    134.418503, 152.868683,
    119.012489, 152.582932,
    111.951797, 152.893265,
    104.901955, 152.522141
]
std_224_bfm09_lmFull = np.array(std_224_bfm09_lmFull)
std_224_bfm09_lmFull = np.reshape(std_224_bfm09_lmFull, [-1, 2])

# 0.75, 800 intrinsic
# main pose [0.000000, 0.000000, -3.141593  0.1744 9.1053  929.1698]
std_224_bfm09_800 = [
    81.774864, 88.538139,
    141.755737, 88.535439,
    112.000000, 133.284698,
    87.324623, 153.685867,
    136.077255, 153.673218
]


DLIB_TO_CELEA_INDEX = [36, 42, 33, 48, 54]

def cvrt_PRN_to_DY(lm68):
    if isinstance(lm68, np.ndarray) == True:
        lm68 = lm68.tolist()

    l_1_17_syn = lm68[1-1:17][::-1]

    l_18_27_syn = lm68[18-1:27][::-1]

    l_28_31 = lm68[28-1:31]

    l_32_36_syn = lm68[32-1:36][::-1]

    l_37_40_syn = lm68[43-1:46][::-1]

    l_41_42_syn = lm68[47-1:48][::-1]

    l_43_46_syn = lm68[37-1:40][::-1]

    l_47_48_syn = lm68[41-1:42][::-1]

    l_49_55_syn = lm68[49-1:55][::-1]

    l_56_60_syn = lm68[56-1:60][::-1]

    l_61_65_syn = lm68[61-1:65][::-1]

    l_66_68_syn = lm68[66-1:68][::-1]

    lm = l_1_17_syn + l_18_27_syn + l_28_31 + \
         l_32_36_syn + l_37_40_syn + l_41_42_syn + l_43_46_syn + l_47_48_syn + \
         l_49_55_syn + l_56_60_syn + l_61_65_syn + l_66_68_syn

    return lm

def cvrt_300w_to_CelebA(lm68):
    l_eye = np.mean(lm68[37 - 1:43 - 1], axis=0)

    r_eye = np.mean(lm68[43 - 1:49 - 1], axis=0)

    nose = lm68[34 - 1]
    l_m = lm68[49 - 1]
    r_m = lm68[55 - 1]

    return [l_eye, r_eye, nose, l_m, r_m]

def cvrt_Now_to_CelebA(lm7):
    l_eye = (lm7[0]+lm7[1])/2.0

    r_eye = (lm7[2]+lm7[3])/2.0

    return np.concatenate([np.array([l_eye, r_eye]), lm7[4:]])


def cvrt_300w_to_Now(lm68):
    l_eye_out = lm68[37 - 1]
    l_eye_in = lm68[40 - 1]

    r_eye_in = lm68[43 - 1]
    r_eye_out = lm68[46 - 1]

    nose = lm68[34 - 1]
    l_m = lm68[49 - 1]
    r_m = lm68[55 - 1]

    return [l_eye_out, l_eye_in, r_eye_in, r_eye_out, nose, l_m, r_m]

def crop_align_affine_transform(h_lm2d, image, crop_size, std_landmark):
    lm_celebA = np.array(h_lm2d.get_lm())
    if lm_celebA.shape[0] != 5:
        lm_celebA = cvrt_300w_to_CelebA(lm_celebA)
    # Transform
    std_points = np.array(std_landmark) * (crop_size / 224.0)

    tform = trans.SimilarityTransform()
    tform.estimate(np.array(lm_celebA), std_points)
    M = tform.params[0:2, :]

    rot_angle = tform.rotation * 180.0 / (math.pi)
    #print(rot_angle, tform.translation)

    img_warped = cv2.warpAffine(image, M, (crop_size, crop_size), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)

    lm_trans = np.matmul(M, np.array(np.transpose(h_lm2d.get_lm_homo())))
    lm_trans = np.transpose(lm_trans)

    return lm_trans, img_warped, tform

def crop_align_affine_transform_5(h_lm2d, image, crop_size, std_224_bfm09):
    lm_celebA = np.array(h_lm2d.get_lm())
    if lm_celebA.shape[0] != 5:
        lm_celebA = cvrt_300w_to_CelebA(lm_celebA)
    # Transform
    std_points = np.array(std_224_bfm09) * (crop_size / 224.0)

    tform = trans.SimilarityTransform()
    tform.estimate(np.array(lm_celebA), std_points)
    M = tform.params[0:2, :]

    rot_angle = tform.rotation * 180.0 / (math.pi)
    #print(tform.scale)

    img_warped = cv2.warpAffine(image, M, (crop_size, crop_size), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)

    lm_trans = np.matmul(M, np.array(np.transpose(h_lm2d.get_lm_homo())))
    lm_trans = np.transpose(lm_trans)

    return lm_trans, img_warped, rot_angle

# need more robust to detect very bad lm detection sequence
def crop_align_affine_transform_68(h_lm2d, image, crop_size, std_224_bfm09_lmFull):
    lm_celebA = h_lm2d.get_lm()
    # Transform
    std_points = np.array(std_224_bfm09_lmFull) * (crop_size / 224.0)

    tform = trans.SimilarityTransform()
    tform.estimate(np.array(lm_celebA), std_points)
    M = tform.params[0:2, :]

    rot_angle = tform.rotation * 180.0 / (math.pi)
    #print(rot_angle, tform.translation)

    img_warped = cv2.warpAffine(image, M, (crop_size, crop_size), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)

    lm_trans = np.matmul(M, np.array(np.transpose(h_lm2d.get_lm_homo())))
    lm_trans = np.transpose(lm_trans)

    return lm_trans, img_warped, rot_angle

def crop_bbox_affine_transform(bbox, image, crop_size):
    contour = np.array(
        [[bbox[0], bbox[1]],
         [bbox[2], bbox[1]],
         [bbox[2], bbox[3]],
         [bbox[0], bbox[3]]]
    )
    contour_image = np.array(
        [[0.0, 0.0],
         [crop_size, 0.0],
         [crop_size, crop_size],
         [0.0, crop_size]]
    )
    tform = trans.SimilarityTransform()
    tform.estimate(contour, contour_image)
    M = tform.params[0:2, :]
    img_warped = cv2.warpAffine(image, M, (crop_size, crop_size), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)

    return img_warped