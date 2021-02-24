#
import tensorflow as tf
import numpy as np
import math


"""
From 
"""
def Quaternion2Mat(quat):
    """
    :param quat: 4
    :return: 3x3
    """
    quat = tf.squeeze(quat)
    w = quat[0]
    x = quat[1]
    y = quat[2]
    z = quat[3]

    val00 = 1 - 2 * y * y - 2 * z * z
    val01 = 2 * x * y - 2 * z * w
    val02 = 2 * x * z + 2 * y * w
    val10 = 2 * x * y + 2 * z * w

    val11 = 1 - 2 * x * x - 2 * z * z
    val12 = 2 * y * z - 2 * x * w
    val20 = 2 * x * z - 2 * y * w
    val21 = 2 * y * z + 2 * x * w
    val22 = 1 - 2 * x * x - 2 * y * y
    rotation = tf.stack([val00, val01, val02, val10, val11, val12, val20, val21, val22], axis=0)
    rotation = tf.reshape(rotation, shape=[3,3])
    return rotation

def CenterOfPoints(points):
    center = tf.reduce_mean(points, axis=1) # 3
    return center

def UncenterPoints(points):
    point_num = points.get_shape().as_list()[1]
    center = CenterOfPoints(points)    # 2
    center = tf.expand_dims(center, axis=-1)    # 2x1
    center_tile = tf.tile(center, [1, point_num])
    u_points = points - center_tile
    return u_points

def ScaleFromPoints(left_points, right_points):
    """
    Compute relative scale from left points to right points
    :param left_points: 3xN
    :param right_points: 3xN
    :return:
    """
    lefts = UncenterPoints(left_points)  # 3xN
    rights = UncenterPoints(right_points)

    ## Compute scale
    left_norm_square = tf.reduce_sum(tf.square(tf.norm(lefts, axis=0)))
    right_norm_square = tf.reduce_sum(tf.square(tf.norm(rights, axis=0)))
    scale = tf.sqrt(right_norm_square / left_norm_square)
    return scale

def TransformFromPointsTF(left_points, right_points):
    """
    Tensorflow implementatin of aligning left points to right points
    :param left_points: 3xN
    :param right_points: 3xN
    :return:
    """

    lefts = UncenterPoints(left_points)    # 3xN
    rights = UncenterPoints(right_points)
    # lefts = left_points
    # rights = right_points

    ## Compute scale
    left_norm_square = tf.reduce_sum(tf.square(tf.norm(lefts, axis=0)))
    right_norm_square = tf.reduce_sum(tf.square(tf.norm(rights, axis=0)))
    scale = tf.sqrt(right_norm_square / (left_norm_square+1e-6))

    ## Compute rotation
    #rights = tf.Print(rights, [rights], message='rights', summarize=2 * 68)
    M = tf.matmul(lefts, rights, transpose_b=True)  # 3x3
    #M = tf.Print(M, [M.shape, M], message="M", summarize=64)

    N00 = M[0, 0] + M[1, 1] + M[2, 2]
    N11 = M[0, 0] - M[1, 1] - M[2, 2]
    N22 = -M[0, 0] + M[1, 1] - M[2, 2]
    N33 = -M[0, 0] - M[1, 1] + M[2, 2]

    N01 = M[1, 2] - M[2, 1]
    N10 = M[1, 2] - M[2, 1]
    N02 = M[2, 0] - M[0, 2]
    N20 = M[2, 0] - M[0, 2]

    N03 = M[0, 1] - M[1, 0]
    N30 = M[0, 1] - M[1, 0]
    N12 = M[0, 1] + M[1, 0]
    N21 = M[0, 1] + M[1, 0]

    N13 = M[0, 2] + M[2, 0]
    N31 = M[0, 2] + M[2, 0]
    N23 = M[1, 2] + M[2, 1]
    N32 = M[1, 2] + M[2, 1]
    N = tf.stack([N00,N01,N02,N03,N10,N11,N12,N13,N20,N21,N22,N23,N30,N31,N32,N33], axis=0)
    N = tf.reshape(N, [4,4])

    #N = tf.Print(N, [N.shape, N], message="N", summarize=64)

    eigen_vals, eigen_vecs = tf.self_adjoint_eig(N)
    quaternion = tf.squeeze((tf.slice(eigen_vecs, [0, 3], [4, 1])))    # 4
    #quaternion = tf_render.Print(quaternion, [quaternion], message='quaternion', summarize=4)
    rotation = Quaternion2Mat(quaternion)   # 3x3

    ## Compute translation
    left_center = CenterOfPoints(left_points)
    right_center = CenterOfPoints(right_points)
    rot_left_center = tf.squeeze(tf.matmul(rotation, tf.expand_dims(left_center, axis=-1))) # 3
    translation = right_center - scale * rot_left_center

    return scale, rotation, translation

#
def lm2d_trans(lm_src, lm_tar):
    filler_mtx = tf.constant([0.0, 0.0, 1.0], shape=[1, 3])
    list_trans_mtx = []
    for b in range(lm_src.shape[0]):
        filler_z = tf.constant([0.0], shape=[1, 1])
        filler_z = tf.tile(filler_z, multiples=[lm_src.shape[1], 1])
        b_src = lm_src[b]
        b_src = tf.concat([b_src, filler_z], axis=1)
        b_src = tf.transpose(b_src)
        b_tar = lm_tar[b]
        b_tar = tf.concat([b_tar, filler_z], axis=1)
        b_tar = tf.transpose(b_tar)

        #b_src = tf.Print(b_src, [b_src], message='b_src', summarize=2 * 68)
        # b_tar = tf_render.Print(b_tar, [b_tar], message='b_tar', summarize=16)
        s, rot_mat, translation = TransformFromPointsTF(b_src, b_tar)

        # s = tf_render.Print(s, [s, s.shape], message='s', summarize=1)

        # rot_mat = tf_render.Print(rot_mat, [rot_mat], message='rot_mat', summarize=9)
        # translation = tf_render.Print(translation, [translation], message='translation', summarize=3)
        rot_mat = rot_mat[0:2, 0:2] * s
        translation = translation[0:2]
        translation = tf.expand_dims(translation, axis=-1)

        ext_mat = tf.concat([rot_mat, translation], axis=1)
        ext_mat = tf.concat([ext_mat, filler_mtx], axis=0)
        list_trans_mtx.append(ext_mat)

    trans_mtx = tf.stack(list_trans_mtx)
    return trans_mtx
