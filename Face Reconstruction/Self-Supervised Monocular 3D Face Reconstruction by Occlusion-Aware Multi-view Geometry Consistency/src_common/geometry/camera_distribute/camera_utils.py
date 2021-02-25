
# system
from __future__ import print_function

# python lib
import math
from copy import deepcopy


# tf_render
import tensorflow as tf

# self

# tianwei
from src_common.geometry.geo_utils import get_ext_inv, get_relative_pose, projective_inverse_warp, pose_vec2rt, pose_vec2mat, mat2pose_vec, \
    fundamental_matrix_from_rt, reprojection_error

"""
Without bp
"""
def build_train_graph_3dmm_frustrum(intrinsic, near=3000.0, far=7000.0):
    batch_size = intrinsic.shape[0]
    # def build_train_graph_3dmm_frustrum(self, intrinsic, near=0.1, far=2000.0):
    # intrinsic
    focal_len_x = tf.slice(intrinsic, [0, 0, 0], [-1, 1, 1])
    focal_len_x = tf.squeeze(focal_len_x, axis=-1)

    focal_len_y = tf.slice(intrinsic, [0, 1, 1], [-1, 1, 1])
    focal_len_y = tf.squeeze(focal_len_y, axis=-1)

    u = tf.slice(intrinsic, [0, 0, 2], [-1, 1, 1])
    u = tf.squeeze(u, axis=-1)

    v = tf.slice(intrinsic, [0, 1, 2], [-1, 1, 1])
    v = tf.squeeze(v, axis=-1)

    #
    near = tf.reshape(tf.constant(near), shape=[1, 1])
    far = tf.reshape(tf.constant(far), shape=[1, 1])
    near = tf.tile(near, [batch_size, 1])
    far = tf.tile(far, [batch_size, 1])

    #
    mtx_frustrum = projectionFrustrumMatrix_batch(focal_len_x, focal_len_y, u, v, near, far)

    return mtx_frustrum

def build_train_graph_3dmm_camera(intrinsic, pose_6dof):
    if isinstance(pose_6dof, list) == False:
        pose_6dof = [pose_6dof]

    list_ext = []
    list_proj = []
    list_mv = []
    list_eye = []

    for i in range(len(pose_6dof)):
        mtx_ext = pose_vec2mat(pose_6dof[i], False)
        mtx_rot = tf.slice(mtx_ext, [0, 0, 0], [-1, 3, 3])
        mtx_t = tf.slice(pose_6dof[i], [0, 3], [-1, 3])
        mtx_t = tf.expand_dims(mtx_t, -1)

        # ext
        mtx_proj = project_batch(intrinsic, mtx_rot, mtx_t)
        #
        mtx_mv = modelViewMatrix_batch(mtx_rot, mtx_t)
        #
        mtx_eye = ext_to_eye_batch(mtx_rot, mtx_t)

        list_ext.append(mtx_ext)
        list_proj.append(mtx_proj)
        list_mv.append(mtx_mv)
        list_eye.append(mtx_eye)

    return list_ext, list_proj, list_mv, list_eye

"""
With bp
"""
def project_batch(mtx_intrinsic, rot_batch, t_batch):
    batch_size = mtx_intrinsic.shape[0]

    M = tf.matmul(mtx_intrinsic, rot_batch)
    p4 = tf.matmul(mtx_intrinsic, t_batch)
    proj = tf.concat([M, p4], axis=2)

    r4 = tf.constant([0., 0., 0., 1.], shape=[1, 1, 4])
    r4 = tf.tile(r4, [batch_size, 1, 1])
    proj = tf.concat([proj, r4], axis=1)

    return proj

def project3d_batch(pt_batch, mtx_proj_batch):
    batch_size = pt_batch.shape[0]

    homo_batch = tf.ones([batch_size, tf.shape(pt_batch)[1], 1], dtype=tf.float32)
    pt_batch_homo = tf.concat([pt_batch, homo_batch], axis=2)
    pt_batch_homo_trans = tf.transpose(pt_batch_homo, perm=[0, 2, 1])
    pt_batch_homo_2d_trans = tf.matmul(mtx_proj_batch, pt_batch_homo_trans)
    pt_batch_homo_2d = tf.transpose(pt_batch_homo_2d_trans, perm=[0, 2, 1])

    pt_batch_homo_2d_main = pt_batch_homo_2d[:, :, 0:2]
    pt_batch_homo_2d_w = pt_batch_homo_2d[:, :, 2]
    pt_batch_homo_2d_w = tf.expand_dims(pt_batch_homo_2d_w, -1)
    pt_batch_homo_2d_normal = pt_batch_homo_2d_main / (pt_batch_homo_2d_w + 1e-6)

    return pt_batch_homo_2d_normal

def ext_to_eye_batch(rot_batch, t_batch):
    #mtx_t_trans = tf_render.expand_dims(t_batch, 1)
    t_batch = tf.transpose(t_batch, perm=[0, 2, 1])
    eye_trans = - tf.matmul(t_batch, rot_batch)
    eye = tf.squeeze(eye_trans, axis=1)
    return eye

def modelViewMatrix_batch(rot_batch, t_batch):
    batch_size = rot_batch.shape[0]

    mtx_inv = tf.constant(
        [
            [1.,  0.,  0.],
            [0., -1.,  0.],
            [0.,  0., -1.]
        ], shape=[1, 3, 3]
    )
    mtx_inv = tf.tile(mtx_inv, [batch_size, 1, 1])

    # Inv rotate
    rot_inv = tf.matmul(mtx_inv, rot_batch)
    c4 = tf.constant([0., 0., 0.], shape=[1, 3, 1])
    c4 = tf.tile(c4, [batch_size, 1, 1])
    rot_inv = tf.concat([rot_inv, c4], axis=2)

    r4 = tf.constant([0., 0., 0., 1.], shape=[1, 1, 4])
    r4 = tf.tile(r4, [batch_size, 1, 1])
    rot_inv = tf.concat([rot_inv, r4], axis=1)

    eye_inv = -ext_to_eye_batch(rot_batch, t_batch)
    eye_inv_trans = tf.expand_dims(eye_inv, axis=-1)
    trans_id_inv = tf.eye(3, batch_shape=[batch_size])
    trans_inv = tf.concat([trans_id_inv, eye_inv_trans], axis=2)
    trans_inv = tf.concat([trans_inv, r4], axis=1)

    mv = tf.matmul(rot_inv, trans_inv)

    return mv

"""
Theory: https://www.scratchapixel.com/lessons/3d-basic-rendering/perspective-and-orthographic-projection-matrix/opengl-perspective-projection-matrix
"""
def projectionFrustrumMatrix_batch(focal_len_x, focal_len_y, u, v,  near, far):
    image_width_batch = 2 * u
    image_height_batch = 2 * v

    # From triangle similarity
    width = image_width_batch * near / focal_len_x
    height = image_height_batch * near / focal_len_y

    right = width - (u * near / focal_len_x)
    left = right - width

    top = v * near / focal_len_y
    bottom = top - height

    vertical_range = right - left
    p00 = 2 * near / vertical_range
    p02 = (right + left) / vertical_range

    horizon_range = top-bottom
    p11 = 2 * near / horizon_range
    p12 = (top + bottom) / horizon_range

    depth_range = far - near
    p_22 = -(far + near) / depth_range
    p_23 = -2.0 * (far * near / depth_range)

    zero_fill = tf.zeros_like(p00)
    minus_one_fill = tf.ones_like(p00)

    r1 = tf.stack([p00, zero_fill, p02, zero_fill], axis=2)
    r2 = tf.stack([zero_fill, p11, p12, zero_fill], axis=2)
    r3 = tf.stack([zero_fill, zero_fill, p_22, p_23], axis=2)
    r4 = tf.stack([zero_fill, zero_fill, -minus_one_fill, zero_fill], axis=2)

    P = tf.concat([r1, r2, r3, r4], axis=1, name='mtx_fustrum_batch')

    return P