
# system
from __future__ import print_function

import os
import sys

# third party

#

# self
_curr_path = os.path.abspath(__file__) # /home/..../face
_cur_dir = os.path.dirname(_curr_path) # ./
_tf_dir = os.path.dirname(_cur_dir) # ./
_tool_data_dir = os.path.dirname(_tf_dir) # ../
_deep_learning_dir = os.path.dirname(_tool_data_dir) # ../
print(_deep_learning_dir)
sys.path.append(_deep_learning_dir) # /home/..../pytorch3d

from tools_data.face_common.gafr_std_align import cvrt_300w_to_CelebA
from tools_data.face_common.faceIO import write_self_lm

from tfmatchd.face.gpmm.bfm09_tf import *
from tfmatchd.face.geometry.camera.rotation import *

def set_ext_mesh_nose_centre_radio(h_lrgp, intrinsic_mtx, mesh_tri, image_height, image_radio_face):
    idx_nose = h_lrgp.idx_lm68_np[34-1]
    idx_low = h_lrgp.idx_lm68_np[9-1]


    mesh_c = mesh_tri.vertices[idx_nose]  # mm
    mesh_low = mesh_tri.vertices[idx_low]

    # Cam
    max_xyz_model = vertex_y_max(mesh_tri)
    min_xyz_model = vertex_y_min(mesh_tri)
    y_mid = (max_xyz_model[1] + mesh_low[1]) / 2.0
    z_mid = (max_xyz_model[2] + mesh_low[2]) / 2.0

    k_eye_dis = intrinsic_mtx[4] * (max_xyz_model[1] - mesh_low[1]) / (image_height * image_radio_face)
    print(k_eye_dis)

    cam_front_eye = [mesh_c[0], y_mid, z_mid + k_eye_dis]
    cam_front_center = [mesh_c[0], y_mid, z_mid]
    cam_front_up = [0.0, 1.0, 0.0]

    ecu = [cam_front_eye, cam_front_center, cam_front_up]
    ecu = tf.constant(ecu)

    mtx_rot, t = ExtMtxBatch.create_location_batch(ecu).rotMtx_location(ecu)
    rot = RotationMtxBatch.create_matrixRot_batch(mtx_rot).eular_rotMtx(mtx_rot)
    rot = tf.expand_dims(rot, 0)
    t = tf.expand_dims(t, 0)

    # rx, ry, rz to rz, ry, rx
    rot = tf.reverse(rot, axis=[1])

    pose = tf.concat([rot, t], axis=1)

    return pose


if __name__ == '__main__':
    path_gpmm = '/home/jshang/SHANG_Data/ThirdLib/BFM2009/bfm09_dy_gyd_presplit.h5'
    h_lrgp = BFM_TF(path_gpmm, 80, 2)
    tri = h_lrgp.get_mesh_mean()
    #tri.show()
    tri.export("/home/jshang/SHANG_Data/ThirdLib/BFM2009/bfm09_mean.ply")

    """
    build graph
    """
    ver, ver_color, _, _ = h_lrgp.get_random_vertex_color_batch()
    ver_color = tf.cast(ver_color*255.0, dtype=tf.uint8)

    lm3d_mean = h_lrgp.get_lm3d_mean()
    lm3d_mean = tf.expand_dims(lm3d_mean, 0)

    # test camera
    from tfmatchd.face.deep_3dmm import build_train_graph_3dmm_frustrum, build_train_graph_3dmm_camera

    intrinsics_single_np = [800.000000, 0., 112.000000, 0., 800.000000, 112.000000, 0., 0., 1.]
    intrinsics_single = tf.constant(intrinsics_single_np, shape=[1, 3, 3])
    gpmm_frustrum = build_train_graph_3dmm_frustrum(intrinsics_single)


    # calculate main pose
    defined_pose_main = set_ext_mesh_nose_centre_radio(h_lrgp, intrinsics_single_np, tri, 224, 0.75)

    #defined_pose_main = tf.constant([0.000000, 0.000000, 3.141593, 0.17440447, 9.1053238, 5748.0352], shape=[1, 6]) now
    #defined_pose_main = tf.constant([0.000000, 0.000000, 3.141593, 0.088619, 8.519336, 5644.714844], shape=[1, 6]) old

    gpmm_tar_ext, gpmm_tar_proj, gpmm_tar_mv, gpmm_tar_eye = \
        build_train_graph_3dmm_camera(intrinsics_single, defined_pose_main)

    # test lm
    from tfmatchd.face.geometry.camera_distribute.camera_utils import project3d_batch

    lm2d = project3d_batch(lm3d_mean, gpmm_tar_proj[0])  # bs, ver_num, xy

    """
    run
    """
    sv = tf.train.Supervisor()
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    with sv.managed_session(config=config) as sess:
        fetches = {
            "defined_pose_main": defined_pose_main,
            "lm2d":lm2d
        }
        """
        *********************************************   Start Trainning   *********************************************
        """
        results = sess.run(fetches)

        defined_pose_main = results["defined_pose_main"]
        lm2d = results["lm2d"]


    # lm
    lm2d = lm2d[0]
    lm2d_5 = cvrt_300w_to_CelebA(lm2d)

    print(defined_pose_main)
    path_std_lm_pose = "./std_lm_pose.txt"
    with open(path_std_lm_pose, 'w') as f_std:
        write_self_lm(f_std, lm2d)
        write_self_lm(f_std, lm2d_5)
