
# system
from __future__ import print_function

# python lib
from copy import deepcopy

# tf_render
import tensorflow as tf

# self

"""
numpy or tensor
"""
class RotationMtxBatch(object):
    """
    0. batch
    1. broadcast support
    """
    def __init__(self, **rotation_param):
        # Normalize [batch_size, data]
        #
        if rotation_param['type_rot'] == 'matrix':
            self.rot_batch = tf.reshape(rotation_param['data'], [-1, 3, 3])
        elif rotation_param['type_rot'] == 'quaternion':
            self.data = tf.reshape(rotation_param['data'], [-1, 4])
            self.rotMtx_quat_batch(self.data)
        elif rotation_param['type_rot'] == 'eularangle':
            pass
            # self.data = tf_render.reshape(rotation_param['data'], [-1, 3, 3])
            # self.rotMtx_quat_batch(self.data)
        self.batch_size = self.rot_batch.shape[0]

    @classmethod
    def create_matrixRot_batch(class_self, data_batch):
        return class_self(type_rot='matrix', data=data_batch)

    @classmethod
    def create_quaternion_batch(class_self, data_batch):
        return class_self(type_rot='quaternion', data=data_batch)

    def rotMtx_quat_batch(self, quat_tensor_batch, re_grad=False):
        rot_batch_list = []
        for i in range(quat_tensor_batch.shape[0]):
            quat = quat_tensor_batch[i, :]
            rot = self.rotMtx_quat(quat, re_grad)
            rot_batch_list.append(rot)
        self.rot_batch = tf.stack(rot_batch_list)

    def rotMtx_quat(self, quat_tensor, re_grad=False):
        # (*this)(0) = DT(1) - yy - zz;
        # (*this)(1) = xy - zw;
        # (*this)(2) = xz + yw;
        # (*this)(3) = xy + zw;
        # (*this)(4) = DT(1) - xx - zz;
        # (*this)(5) = yz - xw;
        # (*this)(6) = xz - yw;
        # (*this)(7) = yz + xw;
        # (*this)(8) = DT(1) - xx - yy;

        X = quat_tensor[0]
        Y = quat_tensor[1]
        Z = quat_tensor[2]
        W = quat_tensor[3]

        sqX = X * X
        sqY = Y * Y
        sqZ = Z * Z
        sqW = W * W

        invs = 1.0 / (sqX + sqY + sqZ + sqW)

        xy = X * Y
        zw = Z * W

        xz = X * Z
        yw = Y * W

        yz = Y * Z
        xw = X * W

        rot_r0 = tf.stack([1 - 2.0 * (sqY + sqZ),                 2 * (xy - zw),         2 * (xz + yw)])
        rot_r1 = tf.stack([        2 * (xy + zw),         1 - 2.0 * (sqX + sqZ),         2 * (yz - xw)])
        rot_r2 = tf.stack([        2 * (xz - yw),                 2 * (yz + xw),         1 - 2.0 * (sqX + sqY) ])

        rot_r0 = rot_r0.unsqueeze(0)
        rot_r1 = rot_r1.unsqueeze(0)
        rot_r2 = rot_r2.unsqueeze(0)

        rot = tf.stack([rot_r0, rot_r1, rot_r2], dim=1)

        tf.mul(rot, invs)

        return rot

    #
    def Get_viewDirect_batch(self):
        return self.rot_batch[:, 2, :]

    def Get_upDirect_batch(self):
        return self.rot_batch[:, 1, :]

    def Get_rightDirect_batch(self):
        return self.rot_batch[:, 0, :]

    #
    def rotMtx_eular_batch(self, eular_batch):
        rot_batch_list = []
        for i in range(eular_batch.shape[0]):
            eular = eular_batch[i, :]
            rot = self.rotMtx_eular(eular)
            rot_batch_list.append(rot)
        self.rot_batch = tf.stack(rot_batch_list)

    """
    /// \brief Mat2Euler
    /// \param R = Rx * Ry * Rz =
    ///	[                       cos(y)*cos(z),                       -cos(y)*sin(z),         sin(y)],
    ///	[cos(x)*sin(z) + cos(z)*sin(x)*sin(y), cos(x)*cos(z) - sin(x)*sin(y)*sin(z), -cos(y)*sin(x)],
    ///	[sin(x)*sin(z) - cos(x)*cos(z)*sin(y), cos(z)*sin(x) + cos(x)*sin(y)*sin(z),  cos(x)*cos(y)]
    ///
    /// Derivation:
    /// z = atan2(-r12, r11)
    /// y = asin(r13)
    /// x = atan2(-r23, r33)
    /// We only keep the zyx order. Problems arise when cos(y) is close to zero, because both of::
    /// z = atan2(cos(y)*sin(z), cos(y)*cos(z))
    /// x = atan2(cos(y)*sin(x), cos(x)*cos(y))
    /// will be close to atan2(0, 0), and highly unstable.
    ///
    /// We use the ``cy`` fix for numerical instability below is from: *Graphics
    /// Gems IV*, Paul Heckbert (editor), Academic Press, 1994, ISBN:
    /// 0123361559.  Specifically it comes from EulerAngles.c by Ken
    /// Shoemake, and deals with the case where cos(y) is close to zero:
    """
    def rotMtx_eular(self, euler_tensor):
        phi = euler_tensor[0] # x
        theta = euler_tensor[1] # y
        psi = euler_tensor[2] # z

        s_ph = tf.sin(phi) # x
        c_ph = tf.cos(phi)

        s_t = tf.sin(theta) # y
        c_t = tf.cos(theta)

        s_ps = tf.sin(psi) # z
        c_ps = tf.cos(psi)

        r1 = tf.concat([c_t * c_ps,                         -c_t * s_ps,                                s_t])
        r2 = tf.concat([c_ph * s_ps + c_ps * s_ph * s_t,    c_ph * c_ps - s_ph * s_t * s_ps,    -c_t * s_ph])
        r3 = tf.concat([s_ps * s_ph - c_ph * c_ps * s_t,    c_ps * s_ph + c_ph * s_t * s_ps,     c_t * c_ph])

        rot = tf.concat([r1, r2, r3])

        return rot

    def eular_rotMtx_batch(self, rot_batch):
        eular_batch_list = []
        for i in range(rot_batch.shape[0]):
            rot = rot_batch[i, :]
            eular = self.eular_rotMtx(rot)
            eular_batch_list.append(eular)
        eular_batch = tf.stack(eular_batch_list)
        return eular_batch

    def eular_rotMtx(self, rot):
        c_t_y = tf.sqrt(
            rot[2][2] * rot[2][2] + rot[1][2] * rot[1][2]
        )

        psi_z = tf.cond(tf.less(c_t_y, 1e-6), lambda : tf.atan2(rot[1][0], rot[1][1]), lambda : tf.atan2(rot[0][1], rot[0][0]))
        theta_y = tf.cond(tf.less(c_t_y, 1e-6), lambda : tf.atan2(rot[0][2], c_t_y), lambda : tf.atan2(rot[0][2], c_t_y))
        phi_x = tf.cond(tf.less(c_t_y, 1e-6), lambda : tf.zeros_like(theta_y), lambda : tf.atan2(rot[1][2], rot[2][2]))

        euler_tensor = tf.stack([phi_x, theta_y, psi_z])

        return euler_tensor

    #
    def rotMtx_axisAngle(self, axis_tensor, rad_tensor):
        if len(axis_tensor.shape) > 1:
            axis_tensor = tf.squeeze(axis_tensor)
        if len(rad_tensor.shape) > 1:
            rad_tensor = tf.squeeze(rad_tensor)

        c = tf.cos(rad_tensor)
        s = tf.sin(rad_tensor)

        x = axis_tensor[0]
        y = axis_tensor[1]
        z = axis_tensor[2]

        rot_r0 = tf.stack([       c + (1.0-c) * x * x,          (1.0-c) * x * y - s * z,           (1.0-c) * x * z + s * y ], axis=1)
        rot_r1 = tf.stack([ (1.0 - c) * x * y + s * z,            c + (1.0 - c) * y * y,         (1.0 - c) * y * z - s * x ], axis=1)
        rot_r2 = tf.stack([ (1.0 - c) * x * z - s * y,        (1.0 - c) * y * z + s * x,             c + (1.0 - c) * z * z ], axis=1)

        # rot_r0 = rot_r0.unsqueeze(0)
        # rot_r1 = rot_r1.unsqueeze(0)
        # rot_r2 = rot_r2.unsqueeze(0)

        rot = tf.stack([rot_r0, rot_r1, rot_r2], axis=1) # [batch, row, col] so axis=1

        return rot

    def rotMtx_axisAngle_batch(self, axis_tensor, rad_tensor):
        c = tf.cos(rad_tensor)
        s = tf.sin(rad_tensor)

        x = axis_tensor[:, 0]
        y = axis_tensor[:, 1]
        z = axis_tensor[:, 2]

        rot_r0 = tf.stack([       c + (1.0-c) * x * x,          (1.0-c) * x * y - s * z,           (1.0-c) * x * z + s * y ], axis=-1)
        rot_r1 = tf.stack([ (1.0 - c) * x * y + s * z,            c + (1.0 - c) * y * y,         (1.0 - c) * y * z - s * x ], axis=-1)
        rot_r2 = tf.stack([ (1.0 - c) * x * z - s * y,        (1.0 - c) * y * z + s * x,             c + (1.0 - c) * z * z ], axis=-1)

        # rot_r0 = rot_r0.unsqueeze(0)
        # rot_r1 = rot_r1.unsqueeze(0)
        # rot_r2 = rot_r2.unsqueeze(0)

        rot = tf.stack([rot_r0, rot_r1, rot_r2], axis=1) # [batch, row, col] so axis=1

        return rot




class ExtMtxBatch(RotationMtxBatch):
    def __init__(self, **ext_param):
        # Normalize [batch_size, data]
        #
        if ext_param['type_ext'] == 'matrix':
            self.data_rot = tf.reshape(ext_param['data_rot'], [-1, 3, 3])
            self.data_t = tf.reshape(ext_param['data_t'], [-1, 3])
            super(ExtMtxBatch, self).__init__(type_rot='matrix', data=self.data_rot)
            self.t_batch = self.data_t
        elif ext_param['type_ext'] == 'location':
            self.data = tf.reshape(ext_param['data'], [-1, 3, 3])
            # super(ExtMtxBatch, self).__init__(type_rot='data', data=self.data)
            self.rotMtx_location_batch(self.data)
        elif ext_param['type_ext'] == 'locationOpengl':
            self.data = tf.reshape(ext_param['data'], [-1, 3, 3])
            # super(ExtMtxBatch, self).__init__(type_rot='data', data=self.data)
            self.rotMtx_locationOpengl_batch(self.data)

    @classmethod
    def create_matrixExt_batch(class_self, rot_data_batch, t_data_batch):
        return class_self(type_ext='matrix', data_rot=rot_data_batch, data_t=t_data_batch)


    @classmethod
    def create_location_batch(class_self, data_batch):
        return class_self(type_ext='location', data=data_batch)
    @classmethod
    def create_locationOpengl_batch(class_self, data_batch):
        return class_self(type_ext='locationOpengl', data=data_batch)
    #
    def rotMtx_location_batch(self, eye_center_up_batch, re_grad=False):
        rot_batch_list = []
        t_batch_list = []
        for i in range(eye_center_up_batch.shape[0]):
            eye_center_up = eye_center_up_batch[i, :, :]
            rot, t = self.rotMtx_location(eye_center_up, re_grad)
            rot_batch_list.append(rot)
            t_batch_list.append(t)
        self.rot_batch = tf.stack(rot_batch_list)
        self.t_batch = tf.stack(t_batch_list)

    def rotMtx_location(self, eye_center_up, re_grad=False):
        eye = eye_center_up[0]
        center = eye_center_up[1]
        up = eye_center_up[2]

        view_dir = center - eye
        view_dir = tf.nn.l2_normalize(view_dir)

        down_dir = -up

        right_dir = tf.cross(down_dir, view_dir)
        right_dir = tf.nn.l2_normalize(right_dir)

        down_dir = tf.cross(view_dir, right_dir)

        rot = tf.stack([right_dir, down_dir, view_dir])
        eye_trans = tf.expand_dims(eye, -1)
        t_trans = -tf.matmul(rot, eye_trans)
        t = tf.transpose(tf.squeeze(t_trans))

        return rot, t

    #
    def rotMtx_locationOpengl_batch(self, eye_center_up_batch, re_grad=False):
        rot_batch_list = []
        t_batch_list = []
        for i in range(eye_center_up_batch.shape[0]):
            eye_center_up = eye_center_up_batch[i, :, :]
            rot, t = self.rotMtx_locationOpengl(eye_center_up, re_grad)
            rot_batch_list.append(rot)
            t_batch_list.append(t)
        self.rot_batch = tf.stack(rot_batch_list)
        self.t_batch = tf.stack(t_batch_list)

    def rotMtx_locationOpengl(self, eye_center_up, re_grad=False):
        eye = eye_center_up[0]
        center = eye_center_up[1]
        up = eye_center_up[2]

        view_dir = -(center - eye)
        view_dir = tf.nn.l2_normalize(view_dir)


        right_dir = tf.cross(up, view_dir)
        right_dir = tf.nn.l2_normalize(right_dir)

        up_dir = tf.cross(view_dir, right_dir)

        rot = tf.stack([right_dir, up_dir, view_dir])
        eye_trans = tf.expand_dims(eye, -1)
        t_trans = -tf.matmul(rot, eye_trans)
        t = tf.transpose(tf.squeeze(t_trans))

        return rot, t

    #
    def Apply_batch(self, v3d):
        if len(v3d.shape) < 3:
            v3d = tf.expand_dims(v3d, -1)
        v3d_rot = tf.matmul(self.rot_batch, v3d)
        v3d_rot = tf.squeeze(v3d_rot, -1)

        v3d_transform = v3d_rot + self.t_batch

        return v3d_transform

    def Get_ext_batch(self):
        t_batch_trans = tf.expand_dims(self.t_batch, axis=-1)
        ext_batch = tf.concat([self.rot_batch, t_batch_trans], axis=2)

        r4 = tf.constant([0., 0., 0., 1.], shape=[1, 1, 4])
        r4 = tf.tile(r4, [self.batch_size, 1, 1])
        ext_batch = tf.concat([ext_batch, r4], axis=1)

        return ext_batch
    #
    def Get_eye_batch(self):
        mtx_t_trans = tf.expand_dims(self.t_batch, 1)
        eye_trans = - tf.matmul(mtx_t_trans, self.rot_batch)
        eye = tf.squeeze(eye_trans, squeeze_dims=1)
        return eye

    # Same from pipline
    """
    Concatenate a rotation R1 around center c1(both in camera coordinate frame) before current camera external transformation [R, -Rc]

    |R1 -R1c1+c1| * |R -Rc| = |R1R -R1Rc-R1c1+c1|
    | 0     1   |   | 0  1|   |  0      1    |	|
    """
    def rotate_batch(self, rel_rot_batch, centre_mesh_batch):
        """
        :param rel_rot_batch:
        :param centre_mesh_batch: world xyz
        :return:
        """
        # Camera centre
        centre_mesh_cameraAxis_batch = self.Apply_batch(centre_mesh_batch)

        # Rotation
        r1_r = tf.matmul(rel_rot_batch, self.rot_batch)

        # Translation
        eye_trans = tf.expand_dims(self.Get_eye_batch(), -1)
        r1_r_c = tf.matmul(r1_r, eye_trans)
        r1_r_c = tf.squeeze(r1_r_c, squeeze_dims=-1)

        centre_mesh_cameraAxis_batch_trans = tf.expand_dims(centre_mesh_cameraAxis_batch, -1)
        r1_c1 = tf.matmul(rel_rot_batch, centre_mesh_cameraAxis_batch_trans)
        r1_c1 = tf.squeeze(r1_c1, squeeze_dims=-1)

        t = centre_mesh_cameraAxis_batch - r1_r_c - r1_c1


        return r1_r, t