
# system
from __future__ import print_function

# python lib
import math
from copy import deepcopy
import numpy as np

# tf_render
import tensorflow as tf

# self
from rotation import RotationMtxBatch
from tfmatchd.face.common.format_helper import batch_size_extract

"""
***************************************     Theory     ***************************************
"""

"""
Proj:
    mtx_proj
    M, p4(no need now)
    mtx_intrinsic, mtx_rot, mtx_t
Intrinsic:    
    mtx_intrinsic
    focal_len_x, focal_len_y, u, v
Ext:
    mtx_rot, mtx_t
    camera_center
    
"""

"""
***************************************     Code     ***************************************
"""

"""
numpy or tensor
no grad
"""
class IntrinsicMtxBatch(object):
    """
    0. batch
    1. broadcast support
    """
    def __init__(self, u, v, focal_len_x=None, focal_len_y=None, fov_x=None, fov_y=None):
        """
        :param u:
        :param v:
        :param focal_len_x:
        :param focal_len_y:
        :param fov_x: field of view
                            *
                        *   ~
                    *       |
                *           |   u
            *   ) fov_x     |
        *   *   *   *   *   ~
        {__________________}
            focal_len_x

        """

        # Read information
        if isinstance(u, tf.Tensor) == False:
            self.u = tf.convert_to_tensor(u, dtype=tf.float32)
            self.v = tf.convert_to_tensor(v, dtype=tf.float32)
        else:
            self.u = u
            self.v = v
        self.aspect_ratio = self.u / self.v

        if focal_len_x is not None:
            self._create_focal_len(focal_len_x, focal_len_y)
        else:
            assert fov_x is not None or fov_y is not None

            if fov_x is not None:
                self._create_fov_x(fov_x)
            else:
                self._create_fov_y(fov_y)

        # Normalize [batch_size, data]
        self.u = tf.reshape(self.u, [-1, 1])
        self.v = tf.reshape(self.v, [-1, 1])
        self.aspect_ratio = tf.reshape(self.aspect_ratio, [-1, 1])
        self.focal_len_x = tf.reshape(self.focal_len_x, [-1, 1])
        self.focal_len_y = tf.reshape(self.focal_len_y, [-1, 1])

        # 0. Batch
        batch_size = batch_size_extract(self.u, self.v, self.focal_len_x, self.focal_len_y)
        assert batch_size is not None

        # 1. Broadcast
        if self.u.shape[0] == 1:
            self.u = tf.tile(self.u, [batch_size, 1])
            self.v = tf.tile(self.v, [batch_size, 1])

        if self.focal_len_x.shape[0] == 1:
            self.focal_len_x = tf.tile(self.focal_len_x, [batch_size, 1])
            self.focal_len_y = tf.tile(self.focal_len_y, [batch_size, 1])

        zeros = tf.zeros_like(self.focal_len_x, dtype=tf.float32)
        r1 = tf.stack([self.focal_len_x, zeros,            self.u], axis=1)
        r1 = tf.reshape(r1, [batch_size, 3])
        r2 = tf.stack([zeros,            self.focal_len_y, self.v], axis=1)
        r2 = tf.reshape(r2, [batch_size, 3])
        r3 = tf.constant([0., 0., 1.], shape=[1, 3])
        r3 = tf.tile(r3, [batch_size, 1])
        self.mtx_intrinsic = tf.stack([r1, r2, r3], axis=1) # [batch, r, c] r:axis=1

    def _create_focal_len(self, focal_len_x, focal_len_y):
        if isinstance(focal_len_x, tf.Tensor) == False:
            self.focal_len_x = tf.convert_to_tensor(focal_len_x, dtype=tf.float32)
            self.focal_len_y = tf.convert_to_tensor(focal_len_y, dtype=tf.float32)
        else:
            self.focal_len_x = focal_len_x
            self.focal_len_y = focal_len_y

    def _create_fov_x(self, fov_x):
        if isinstance(fov_x, tf.Tensor) == False:
            self.fov_x = tf.convert_to_tensor(fov_x, dtype=tf.float32)
        self.focal_len_x = (1.0 * self.u) / tf.tan(fov_x * (math.pi / 360.0))
        self.focal_len_y = self.focal_len_x * self.aspect_ratio

    def _create_fov_y(self, fov_y):
        if isinstance(fov_y, tf.Tensor) == False:
            self.fov_y = tf.convert_to_tensor(fov_y, dtype=tf.float32)
        self.focal_len_y = (1.0 * self.v) / tf.tan(fov_y * (math.pi / 360.0))
        self.focal_len_x = self.focal_len_y / self.aspect_ratio

    def Get_image_width(self):
        return self.u * 2.0

    def Get_image_height(self):
        return self.v * 2.0

    def Get_batch_mtx_intrinsic(self):
        return self.mtx_intrinsic

"""
tensor only
"""
class CameraMtxBatch(object):
    """
    0. batch
    1. broadcast support
    """
    def __init__(self, h_intrinsic, h_extenal):
        self.h_intrinsic = h_intrinsic
        self.h_extenal = h_extenal

        self.mtx_intrinsic = self.h_intrinsic.Get_batch_mtx_intrinsic()
        self.mtx_rot = self.h_extenal.rot_batch
        self.mtx_t = self.h_extenal.t_batch

        # 0. Batch
        self.batch_size = batch_size_extract(self.mtx_intrinsic, self.mtx_rot, self.mtx_t)
        assert self.batch_size is not None

        # 1. broadcast
        if self.mtx_intrinsic.shape[0] == 1:
            self.mtx_intrinsic = tf.tile(self.mtx_intrinsic, [self.batch_size, 1, 1])

        if self.mtx_rot.shape[0] == 1:
            self.mtx_rot = tf.tile(self.mtx_rot, [self.batch_size, 1, 1])
            self.mtx_t = tf.tile(self.mtx_t, [self.batch_size, 1])

        #
        self.mtx_proj = self._Cal_mtxProj()

    def _Cal_mtxProj(self):
        M = tf.matmul(self.mtx_intrinsic, self.mtx_rot)
        t_trans = tf.expand_dims(self.mtx_t, -1)
        p4 = tf.matmul(self.mtx_intrinsic, t_trans)
        ext = tf.concat([M, p4], axis=2)

        r4 = tf.constant([0., 0., 0., 1.], shape=[1, 1, 4])
        r4 = tf.tile(r4, [self.batch_size, 1, 1])
        ext = tf.concat([ext, r4], axis=1)

        return ext

    def Project(self, pt_batch, re_grad=False):
        homo_batch = tf.ones([self.batch_size, pt_batch.shape[1], 1])
        pt_batch_homo = tf.concat([pt_batch, homo_batch], axis=2)
        pt_batch_homo_trans = tf.transpose(pt_batch_homo, perm=[0, 2, 1])
        pt_batch_homo_2d_trans = tf.matmul(self.mtx_proj, pt_batch_homo_trans)
        pt_batch_homo_2d = tf.transpose(pt_batch_homo_2d_trans, perm=[0, 2, 1])

        pt_batch_homo_2d_main = pt_batch_homo_2d[:, :, 0:2]
        pt_batch_homo_2d_w = pt_batch_homo_2d[:, :, 2]
        pt_batch_homo_2d_w = tf.expand_dims(pt_batch_homo_2d_w, -1)
        pt_batch_homo_2d_normal = pt_batch_homo_2d_main / pt_batch_homo_2d_w

        return pt_batch_homo_2d_normal

    def Get_rot_t_batch(self):
        return self.mtx_rot, self.mtx_t

    def Get_eularAngle_rot_t_batch(self):
        eular_angle_rot = self.h_extenal.eular_rotMtx_batch(self.mtx_rot)
        return eular_angle_rot, self.mtx_t

    def Get_eye_batch(self):
        return self.h_extenal.Get_eye_batch()

if __name__ == "__main__":
    h_intrMtx = IntrinsicMtxBatch(np.random.random((16,1)), np.random.random((16,1)), 1000, 1000)
    h_intrMtx.Get_batch_mtx_intrinsic()