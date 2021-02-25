
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
from camera import IntrinsicMtxBatch, CameraMtxBatch

"""
param numpy
inherit tensor
"""

class CameraRender(CameraMtxBatch):

    def __init__(self, h_intrinsic, h_extenal, near=0.1, far=2000.0):
        super(CameraRender, self).__init__(h_intrinsic, h_extenal)

        self.h_intrinsic = h_intrinsic
        self.h_extenal = h_extenal

        self.focal_len_x = h_intrinsic.focal_len_x
        self.focal_len_y = h_intrinsic.focal_len_y

        self.u = h_intrinsic.u
        self.v = h_intrinsic.v

        self.image_width_batch = h_intrinsic.Get_image_width()
        self.image_height_batch = h_intrinsic.Get_image_height()

        #super(CameraRender, self)._Cal_mtxProj()
        self.near = tf.reshape(tf.constant(near), shape=[1, 1])
        self.far = tf.reshape(tf.constant(far), shape=[1, 1])
        self.near = tf.tile(self.near, [self.batch_size, 1])
        self.far = tf.tile(self.far, [self.batch_size, 1])
    #
    def Get_modelViewMatrix_batch(self, re_grad=False):
        mtx_inv = tf.constant(
            [
                [1.,  0.,  0.],
                [0., -1.,  0.],
                [0.,  0., -1.]
            ], shape=[1, 3, 3]
        )
        mtx_inv = tf.tile(mtx_inv, [self.batch_size, 1, 1])

        # Inv rotate
        rot_inv = tf.matmul(mtx_inv, self.mtx_rot)
        c4 = tf.constant([0., 0., 0.], shape=[1, 3, 1])
        c4 = tf.tile(c4, [self.batch_size, 1, 1])
        rot_inv = tf.concat([rot_inv, c4], axis=2)

        r4 = tf.constant([0., 0., 0., 1.], shape=[1, 1, 4])
        r4 = tf.tile(r4, [self.batch_size, 1, 1])
        rot_inv = tf.concat([rot_inv, r4], axis=1)

        eye_inv = -self.Get_eye_batch()
        eye_inv_trans = tf.expand_dims(eye_inv, axis=-1)
        trans_id_inv = tf.eye(3, batch_shape=[self.batch_size])
        trans_inv = tf.concat([trans_id_inv, eye_inv_trans], axis=2)
        trans_inv = tf.concat([trans_inv, r4], axis=1)

        mv = tf.matmul(rot_inv, trans_inv)

        return mv

    def Get_projectionFrustrumMatrix_batch(self, re_grad=False):
        # From triangle similarity
        width = self.image_width_batch * self.near / self.focal_len_x
        height = self.image_height_batch * self.near / self.focal_len_y

        right = width - (self.u * self.near / self.focal_len_x)
        left = right - width

        top = self.v * self.near / self.focal_len_y
        bottom = top - height

        vertical_range = right - left
        p00 = 2 * self.near / vertical_range
        p02 = (right + left) / vertical_range

        horizon_range = top-bottom
        p11 = 2 * self.near / horizon_range
        p12 = (top + bottom) / horizon_range

        depth_range = self.far - self.near
        p_22 = -(self.far + self.near) / depth_range
        p_23 = -2.0 * (self.far * self.near / depth_range)

        zero_fill = tf.zeros_like(p00)
        minus_one_fill = tf.ones_like(p00)

        r1 = tf.stack([p00, zero_fill, p02, zero_fill], axis=2)
        r2 = tf.stack([zero_fill, p11, p12, zero_fill], axis=2)
        r3 = tf.stack([zero_fill, zero_fill, p_22, p_23], axis=2, name='mtx_fustrum_r3_batch')
        r4 = tf.stack([zero_fill, zero_fill, -minus_one_fill, zero_fill], axis=2)

        P = tf.concat([r1, r2, r3, r4], axis=1, name='mtx_fustrum_batch')

        return P
