
# system
from __future__ import print_function

# python lib
import math
from copy import deepcopy
import numpy as np

# tf_render
import tensorflow as tf

# self
from rotation import RotationMtxBatch, ExtMtxBatch
from camera import IntrinsicMtxBatch, CameraMtxBatch
from camera_render import CameraRender
"""
param numpy
inherit tensor
no weight update
"""

class CameraAugment(CameraRender):

    def __init__(self, h_intrinsic, h_extenal,
                 centre_camera_rot,
                 roll_num=0, roll_max_angle=0,  #
                 pitch_num=0, pitch_max_angle=0,  #
                 yaw_num=0, yaw_max_angle=0,  #
                 near = 0.1, far = 2000.0
                 ):
        super(CameraAugment, self).__init__(h_intrinsic, h_extenal, near, far)

        self.image_width_batch = h_intrinsic.Get_image_width()
        self.image_height_batch = h_intrinsic.Get_image_height()

        #super(CameraRender, self)._Cal_mtxProj()

        self.centre_camera_rot = centre_camera_rot

        self.roll_num = roll_num
        self.roll_max_angle = roll_max_angle

        self.pitch_num = pitch_num
        self.pitch_max_angle = pitch_max_angle

        self.yaw_num = yaw_num
        self.yaw_max_angle = yaw_max_angle

    def Augment_Single_Random(self):
        cam = CameraRender(self.h_intrinsic, self.h_extenal)

        z_axis = self.h_extenal.Get_viewDirect_batch()
        y_axis = self.h_extenal.Get_upDirect_batch()
        x_axis= self.h_extenal.Get_rightDirect_batch()

        #
        psi_angle = tf.random_uniform(shape=[int(self.batch_size)])
        psi_angle = psi_angle * (2 * self.roll_max_angle) - self.roll_max_angle
        psi = psi_angle * math.pi / (180.)

        mtx_rel_rot = self.h_extenal.rotMtx_axisAngle_batch(z_axis, psi)
        mtx_rot_batch, mtx_t_batch = cam.h_extenal.rotate_batch(mtx_rel_rot, self.centre_camera_rot)
        # New
        h_ext_tmp = ExtMtxBatch.create_matrixExt_batch(mtx_rot_batch, mtx_t_batch)
        cam_psi = CameraRender(self.h_intrinsic, h_ext_tmp)

        #
        phi_angle = tf.random_uniform(shape=[1]) * (2 * self.pitch_max_angle) - self.pitch_max_angle
        phi = phi_angle * math.pi / (180.)

        mtx_rel_rot = self.h_extenal.rotMtx_axisAngle_batch(x_axis, phi)
        mtx_rot_batch, mtx_t_batch = cam_psi.h_extenal.rotate_batch(mtx_rel_rot, self.centre_camera_rot)
        # New
        h_ext_tmp = ExtMtxBatch.create_matrixExt_batch(mtx_rot_batch, mtx_t_batch)
        cam_phi = CameraRender(self.h_intrinsic, h_ext_tmp)

        #
        theta_angle = tf.random_uniform(shape=[1]) * (2 * self.yaw_max_angle) - self.yaw_max_angle
        theta = theta_angle * math.pi / (180.)

        mtx_rel_rot = self.h_extenal.rotMtx_axisAngle_batch(y_axis, theta)
        mtx_rot_batch, mtx_t_batch = cam_phi.h_extenal.rotate_batch(mtx_rel_rot, self.centre_camera_rot)
        # New
        h_ext_tmp = ExtMtxBatch.create_matrixExt_batch(mtx_rot_batch, mtx_t_batch)
        cam_th = CameraRender(self.h_intrinsic, h_ext_tmp)

        #
        rot, t = cam_th.Get_eularAngle_rot_t_batch()
        rot = tf.reverse(rot, axis=[1]) # rx, ry, rz, to, rz, ry, rx

        return tf.concat([rot, t], axis=1)

    def Augment_Average_Interval(self):
        self.list_cam = list()
        self.list_cam.append(CameraRender(self.h_intrinsic, self.h_extenal))

        z_axis = self.h_extenal.Get_viewDirect_batch()
        y_axis = self.h_extenal.Get_upDirect_batch()
        x_axis= self.h_extenal.Get_rightDirect_batch()

        list_cam_prev = []
        if self.roll_num != 0:
            for r in range(-self.roll_num, self.roll_num+1):
                if r == 0:
                    continue
                psi_angle = r * (self.roll_max_angle / (self.roll_num+1.))
                psi = psi_angle * math.pi / (180.)
                psi = tf.Variable([psi])

                for cam in self.list_cam:
                    # Rotate
                    mtx_rel_rot = self.h_extenal.rotMtx_axisAngle(z_axis, psi)
                    mtx_rot_batch, mtx_t_batch = cam.h_extenal.rotate_batch(mtx_rel_rot, self.centre_camera_rot)
                    # New
                    h_ext_tmp = ExtMtxBatch.create_matrixExt_batch(mtx_rot_batch, mtx_t_batch)
                    cam_aug = CameraRender(self.h_intrinsic, h_ext_tmp)
                    list_cam_prev.append(cam_aug)
            self.list_cam = self.list_cam + list_cam_prev

        list_cam_prev = []
        if self.pitch_num != 0:
            for p in range(-self.pitch_num, self.pitch_num+1):
                phi_angle = p * (self.pitch_max_angle / (self.pitch_num+1.))
                phi = phi_angle * math.pi / (180.)
                phi = tf.Variable([phi])

                for cam in self.list_cam:
                    # Rotate
                    mtx_rel_rot = self.h_extenal.rotMtx_axisAngle(x_axis, phi)
                    mtx_rot_batch, mtx_t_batch = cam.h_extenal.rotate_batch(mtx_rel_rot, self.centre_camera_rot)
                    # New
                    h_ext_tmp = ExtMtxBatch.create_matrixExt_batch(mtx_rot_batch, mtx_t_batch)
                    cam_aug = CameraRender(self.h_intrinsic, h_ext_tmp)
                    list_cam_prev.append(cam_aug)
            self.list_cam = self.list_cam + list_cam_prev

        list_cam_prev = []
        if self.yaw_num != 0:
            for y in range(-self.yaw_num, self.yaw_num+1):
                theta_angle = y * (self.yaw_max_angle / (self.yaw_num+1.))
                theta = theta_angle * math.pi / (180.)
                theta = tf.Variable([theta])

                for cam in self.list_cam:
                    # Rotate
                    mtx_rel_rot = self.h_extenal.rotMtx_axisAngle(y_axis, theta)
                    mtx_rot_batch, mtx_t_batch = cam.h_extenal.rotate_batch(mtx_rel_rot, self.centre_camera_rot)
                    # New
                    h_ext_tmp = ExtMtxBatch.create_matrixExt_batch(mtx_rot_batch, mtx_t_batch)
                    cam_aug = CameraRender(self.h_intrinsic, h_ext_tmp)
                    list_cam_prev.append(cam_aug)
            self.list_cam = self.list_cam + list_cam_prev
        if len(self.list_cam) > 1:
            self.list_cam = self.list_cam[1:]

    def Get_aug_mtxMV_batch(self): # Model View matrix
        list_mv = []
        for i in range(len(self.list_cam)):
            cam = self.list_cam[i]
            mv = cam.Get_modelViewMatrix_batch()
            list_mv.append(mv)
        mv_batch = tf.concat(list_mv, axis=0)
        return mv_batch

    def Get_aug_eye_batch(self):
        list_eye = []
        for i in range(len(self.list_cam)):
            cam = self.list_cam[i]
            eye = cam.Get_eye_batch()
            list_eye.append(eye)
        eye_batch = tf.concat(list_eye, axis=0)
        return eye_batch

    def Get_aug_eularAngle_rot_t_batch(self):
        list_rot = []
        list_t = []
        for i in range(len(self.list_cam)):
            cam = self.list_cam[i]
            mtx_param_rot, mtx_t  = cam.Get_eularAngle_rot_t_batch()
            list_rot.append(mtx_param_rot)
            list_t.append(mtx_t)
        param_rot_batch = tf.concat(list_rot, axis=0)
        t_batch = tf.concat(list_t, axis=0)
        return param_rot_batch, t_batch

    def Get_aug_proj_pt2d_batch(self, lm3d_batch):
        list_proj = []
        for i in range(len(self.list_cam)):
            cam = self.list_cam[i]
            proj = super(CameraRender, cam).Project(lm3d_batch)
            list_proj.append(proj)
        proj_batch = tf.concat(list_proj, axis=0)
        return proj_batch
