from __future__ import division

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import sys
from shutil import copyfile

# tf
import numpy as np
import tensorflow as tf
import torch

# save result
import face_alignment
import cv2
import PIL.Image as pil
import matplotlib.pyplot as plt
import trimesh

# path
_curr_path = os.path.abspath(__file__) # /home/..../face
_cur_dir = os.path.dirname(_curr_path) # ./

# save result
from src_common.common.face_io import write_self_camera, write_self_lm
from tools.preprocess.detect_landmark import LM_detector_howfar
from tools.preprocess.crop_image_affine import *

# graph
from src_tfGraph.build_graph import MGC_TRAIN

flags = tf.app.flags

#
flags.DEFINE_string("dic_image", "data/test/", "Dataset directory")
flags.DEFINE_string("output_dir", "data/output_test_one", "Output directory")
flags.DEFINE_string("ckpt_file", "model/model-400000", "checkpoint file")
#flags.DEFINE_string("ckpt_file", "/home/jiaxiangshang/Downloads/202008/70_31_warpdepthepi_reg/model-400000", "checkpoint file")

#
flags.DEFINE_integer("batch_size", 1, "The size of of a sample batch")
flags.DEFINE_integer("img_width", 224, "Image(square) size")
flags.DEFINE_integer("img_height", 224, "Image(square) size")

# gpmm
flags.DEFINE_string("path_gpmm", "model/bfm09_trim_exp_uv_presplit.h5", "Dataset directory")
flags.DEFINE_integer("light_rank", 27, "3DMM coeffient rank")
flags.DEFINE_integer("gpmm_rank", 80, "3DMM coeffient rank")
flags.DEFINE_integer("gpmm_exp_rank", 64, "3DMM coeffient rank")

#
flags.DEFINE_boolean("flag_eval", True, "3DMM coeffient rank")
flags.DEFINE_boolean("flag_visual", True, "")
flags.DEFINE_boolean("flag_fore", False, "")

# visual
flags.DEFINE_boolean("flag_overlay_save", True, "")
flags.DEFINE_boolean("flag_overlayOrigin_save", True, "")
flags.DEFINE_boolean("flag_main_save", True, "")

FLAGS = flags.FLAGS

if __name__ == '__main__':
    FLAGS.dic_image = os.path.join(_cur_dir, FLAGS.dic_image)
    FLAGS.output_dir = os.path.join(_cur_dir, FLAGS.output_dir)

    FLAGS.ckpt_file = os.path.join(_cur_dir, FLAGS.ckpt_file)
    FLAGS.path_gpmm = os.path.join(_cur_dir, FLAGS.path_gpmm)
    
    
    if not os.path.exists(FLAGS.dic_image):
        print("Error: no dataset_dir found")

    if not os.path.exists(FLAGS.output_dir):
        os.makedirs(FLAGS.output_dir)
    print("Finish copy")

    """
    preprocess
    """
    lm_d_hf = LM_detector_howfar(lm_type=int(3), device='cpu', face_detector='sfd')

    """
    build graph
    """
    system = MGC_TRAIN(FLAGS)
    system.build_test_graph(
        FLAGS, img_height=FLAGS.img_height, img_width=FLAGS.img_width, batch_size=FLAGS.batch_size
    )

    """
    load model
    """
    test_var = tf.global_variables()#tf.model_variables()
    # this because we need using the
    test_var = [tv for tv in test_var if tv.op.name.find('VertexNormalsPreSplit') == -1]
    saver = tf.train.Saver([var for var in test_var])

    #config = tf.ConfigProto()
    config=tf.ConfigProto(device_count={'cpu':0})
    config.gpu_options.allow_growth = True
    with tf.Session(config=config) as sess:
        sess.run(tf.global_variables_initializer())
        sess.graph.finalize()
        saver.restore(sess, FLAGS.ckpt_file)
        #
        import time
        # preprocess
        path_image = os.path.join(FLAGS.dic_image, 'image04275.jpg')
        image_bgr = cv2.imread(path_image)
        image_rgb = image_bgr[..., ::-1]
        if image_bgr is None:
            print("Error: can not find ", path_image)
        with torch.no_grad():
            lm_howfar = lm_d_hf.lm_detection_howfar(image_bgr)
            lm_howfar = lm_howfar[:, :2]

        # face image align by landmark
        # we also provide a tools to generate 'std_224_bfm09'
        lm_trans, img_warped, tform = crop_align_affine_transform(lm_howfar, image_rgb, FLAGS.img_height, std_224_bfm09)
        image_rgb_b = img_warped[None, ...]
        # M_inv is used to back project the face reconstruction result to origin image
        M_inv = np.linalg.inv(tform.params)
        M = tform.params
        #print(np.matmul(M_inv, M))

        """
        Start
        """
        time_st = time.time()
        pred = system.inference(sess, image_rgb_b)
        time_end = time.time()
        print("Time each batch: ", time_end - time_st)

        # name
        dic_image, name_image = os.path.split(path_image)
        name_image_pure, _ = os.path.splitext(name_image)

        """
        Render
        """
        image_input = image_rgb_b

        """
        NP
        """
        b = 0
        vertex_shape = pred['vertex_shape'][0][b, :, :]
        vertex_color = pred['vertex_color'][0][b, :, :]
        vertex_color = np.clip(vertex_color, 0, 1)
        #vertex_color_rgba = np.concatenate([vertex_color, np.ones([vertex_color.shape[0], 1])], axis=1)
        vertex_color_ori = pred['vertex_color_ori'][0][b, :, :]
        vertex_color_ori = np.clip(vertex_color_ori, 0, 1)

        if FLAGS.flag_eval:
            mesh_tri = trimesh.Trimesh(
                vertex_shape.reshape(-1, 3),
                system.h_lrgp.h_curr.mesh_tri_np.reshape(-1, 3),
                vertex_colors=vertex_color.reshape(-1, 3),
                process=False
            )
            mesh_tri.visual.kind == 'vertex'

            path_mesh_save = os.path.join(FLAGS.output_dir, name_image_pure + ".ply")
            mesh_tri.export(path_mesh_save)
            """
            Landmark 3D
            """
            path_lm3d_save = os.path.join(FLAGS.output_dir, name_image_pure + "_lm3d.txt")
            lm_68 = vertex_shape[system.h_lrgp.h_curr.idx_lm68_np]

            write_self_lm(path_lm3d_save, lm_68)

            """
            Landmark 2D

            """
            lm2d = pred['lm2d'][0][b, :, :]
            path_lm2d_save = os.path.join(FLAGS.output_dir, name_image_pure + "_lm2d.txt")
            write_self_lm(path_lm2d_save, lm2d)

            """
            Pose
            """
            path_cam_save = os.path.join(FLAGS.output_dir, name_image_pure + "_cam.txt")

            pose = pred['gpmm_pose'][0][b, :]
            intrinsic = pred['gpmm_intrinsic'][b, :, :]

            write_self_camera(path_cam_save, FLAGS.img_width, FLAGS.img_height, intrinsic, pose)

        """
        Common visual
        """
        if FLAGS.flag_visual:
            # visual
            result_overlayMain_255 = pred['overlayMain_255'][0][b, :, :]
            result_overlayTexMain_255 = pred['overlayTexMain_255'][0][b, :, :]
            result_overlayGeoMain_255 = pred['overlayGeoMain_255'][0][b, :, :]
            result_overlayLightMain_255 = pred['overlayLightMain_255'][0][b, :, :]
            result_apper_mulPose_255 = pred['apper_mulPose_255'][0][b, :, :]

            result_overlay_255 = pred['overlay_255'][0][b, :, :]
            result_overlayTex_255 = pred['overlayTex_255'][0][b, :, :]
            result_overlayGeo_255 = pred['overlayGeo_255'][0][b, :, :]
            result_overlayLight_255 = pred['overlayLight_255'][0][b, :, :]

            # common
            visual_concat = np.concatenate([image_input[0], result_overlay_255, result_overlayGeo_255, result_apper_mulPose_255], axis=1)
            path_image_save = os.path.join(FLAGS.output_dir, name_image_pure + "_mulPoses.jpg")
            plt.imsave(path_image_save, visual_concat)

            if FLAGS.flag_overlayOrigin_save:
                gpmm_render_mask = pred['gpmm_render_mask'][0][b, :, :]
                gpmm_render_mask = np.tile(gpmm_render_mask, reps=(1, 1, 3))

                path_image_origin = os.path.join(dic_image, name_image_pure + ".jpg")
                image_origin = cv2.imread(path_image_origin)

                gpmm_render_overlay_wo = inverse_affine_warp_overlay(
                    M_inv, image_origin, result_overlay_255, gpmm_render_mask)
                gpmm_render_overlay_texture_wo = inverse_affine_warp_overlay(
                    M_inv, image_origin, result_overlayTex_255, gpmm_render_mask)
                gpmm_render_overlay_gary_wo = inverse_affine_warp_overlay(
                    M_inv, image_origin, result_overlayGeo_255, gpmm_render_mask)
                gpmm_render_overlay_illu_wo = inverse_affine_warp_overlay(
                    M_inv, image_origin, result_overlayLight_255, gpmm_render_mask)

                path_image_save = os.path.join(FLAGS.output_dir, name_image_pure + "_overlayOrigin.jpg")
                cv2.imwrite(path_image_save, gpmm_render_overlay_wo)
                path_image_save = os.path.join(FLAGS.output_dir, name_image_pure + "_overlayTexOrigin.jpg")
                # cv2.imwrite(path_image_save, gpmm_render_overlay_texture_wo)
                path_image_save = os.path.join(FLAGS.output_dir, name_image_pure + "_overlayGeoOrigin.jpg")
                cv2.imwrite(path_image_save, gpmm_render_overlay_gary_wo)
                path_image_save = os.path.join(FLAGS.output_dir, name_image_pure + "_overlayLightOrigin.jpg")
                # cv2.imwrite(path_image_save, gpmm_render_overlay_illu_wo)

            if FLAGS.flag_main_save:
                path_image_save = os.path.join(FLAGS.output_dir, name_image_pure + "_overlayMain.jpg")
                plt.imsave(path_image_save, result_overlayMain_255)
                path_image_save = os.path.join(FLAGS.output_dir, name_image_pure + "_overlayTexMain.jpg")
                #plt.imsave(path_image_gray_main_overlay, gpmm_render_overlay)
                path_image_save = os.path.join(FLAGS.output_dir, name_image_pure + "_overlayGeoMain.jpg")
                plt.imsave(path_image_save, result_overlayGeoMain_255)
                path_image_save = os.path.join(FLAGS.output_dir, name_image_pure + "_overlayLightMain.jpg")
                #cv2.imwrite(path_image_save, result_overlayLightMain_255)

            if FLAGS.flag_overlay_save:
                path_image_save = os.path.join(FLAGS.output_dir, name_image_pure + "_overlay.jpg")
                plt.imsave(path_image_save, result_overlay_255)
                path_image_save = os.path.join(FLAGS.output_dir, name_image_pure + "_overlayTex.jpg")
                plt.imsave(path_image_save, result_overlayTex_255)
                path_image_save = os.path.join(FLAGS.output_dir, name_image_pure + "_overlayGeo.jpg")
                plt.imsave(path_image_save, result_overlayGeo_255)
                path_image_save = os.path.join(FLAGS.output_dir, name_image_pure + "_overlayLight.jpg")
                plt.imsave(path_image_save, result_overlayLight_255)

