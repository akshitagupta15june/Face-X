
# system
from __future__ import print_function

# python lib
import numpy as np
# tf_render
import tensorflow as tf

from src_common.net.resnet_v1_3dmm import encoder_resnet50

from src_common.net.inception_resnet_v1 import identity_inference
from src_common.geometry.face_align.align_facenet import facenet_align

from src_common.geometry.camera_distribute.camera_utils import project3d_batch
from src_common.geometry.render.lighting import vertex_normals_pre_split_fixtopo
from src_common.geometry.render.api_tf_mesh_render import mesh_depthmap_camera, mesh_renderer_camera, mesh_renderer_camera_light, \
    tone_mapper



# net
def pred_encoder_coeff(opt, defined_pose_main, list_image, is_training=True):
    pred_rank = 2 * opt.gpmm_rank + opt.gpmm_exp_rank + 6 + 27

    with tf.name_scope("3dmm_coeff"):
        #
        list_gpmm = []
        list_gpmm_color = []
        list_gpmm_exp = []
        list_gpmm_pose = []
        list_gpmm_lighting = []

        for i in range(len(list_image)):

            pred_id_src, end_points_src = encoder_resnet50(list_image[i], pred_rank, is_training=is_training, reuse=tf.AUTO_REUSE)

            pred_3dmm_src = pred_id_src[:,                       : opt.gpmm_rank]
            pred_3dmm_color_src = pred_id_src[:,   opt.gpmm_rank : 2 * opt.gpmm_rank]
            pred_3dmm_exp_src = pred_id_src[:, 2 * opt.gpmm_rank : 2 * opt.gpmm_rank + opt.gpmm_exp_rank]

            list_gpmm.append(pred_3dmm_src)
            list_gpmm_color.append(pred_3dmm_color_src)
            list_gpmm_exp.append(pred_3dmm_exp_src)

            pred_pose_render_src = pred_id_src[:,  2 * opt.gpmm_rank + opt.gpmm_exp_rank : 2 * opt.gpmm_rank + opt.gpmm_exp_rank + 6]
            pred_pose_render_src = pred_pose_render_src + defined_pose_main
            pred_lighting_src = pred_id_src[:, 2 * opt.gpmm_rank + opt.gpmm_exp_rank + 6 : 2 * opt.gpmm_rank + opt.gpmm_exp_rank + 6 + 27]

            list_gpmm_pose.append(pred_pose_render_src)
            list_gpmm_lighting.append(pred_lighting_src)

        return list_gpmm, list_gpmm_color, list_gpmm_exp, list_gpmm_pose, list_gpmm_lighting

def pred_encoder_coeff_light(opt, defined_pose_main, list_image, is_training=True):
    pred_rank = 2 * opt.gpmm_rank + opt.gpmm_exp_rank + 6 + 27

    with tf.name_scope("3dmm_coeff"):
        #
        list_gpmm = []
        for i in range(len(list_image)):
            pred_id_src, end_points_src = encoder_resnet50(list_image[i], pred_rank, is_training=is_training, reuse=tf.AUTO_REUSE)
            list_gpmm.append(pred_id_src)
        return list_gpmm

# id
def pred_encoder_id(opt, gpmm_render_tar_align):
    list_gpmm_id_pred_tar = []
    for i in range(len(gpmm_render_tar_align)):
        gpmm_render_de = gpmm_render_tar_align[i] * 255.0

        # if opt.mode_depth_pixel_loss == 'clip':
        #     gpmm_render_de = tf.clip_by_value(gpmm_render_de, 0.0, 255.0)

        gpmm_render_de = facenet_image_process(gpmm_render_de)

        gpmm_id_pred_tar = pred_encoder_facenet(gpmm_render_de)

        #gpmm_id_pred_tar = tf.Print(gpmm_id_pred_tar, [tf.reduce_mean(gpmm_render_de), tf.reduce_mean(gpmm_id_pred_tar)], message='gpmm_id_pred')

        list_gpmm_id_pred_tar.append(gpmm_id_pred_tar[0])

    return list_gpmm_id_pred_tar

def pred_encoder_facenet(image):
    with tf.name_scope("3dmm_identity"):
        #
        list_gpmm_id = []

        prelogits, end_points = identity_inference(
            image, 0.8, phase_train=False,
            bottleneck_layer_size=512, weight_decay=0.0, reuse=tf.AUTO_REUSE
        )
        # list_gpmm_id.append(prelogits)
        embeddings = tf.nn.l2_normalize(prelogits, 1, 1e-10, name='embeddings')

        list_gpmm_id.append(embeddings)

        return list_gpmm_id

def decoder_similar(opt, defined_lm_facenet_align, render_img, list_img, lm2d, list_lm2d_gt):
    """
    Render image align, id
    """
    gpmm_render_align = facenet_align(render_img, lm2d, defined_lm_facenet_align, opt.img_height, opt.img_width)
    gpmm_render_tar_id = pred_encoder_id(opt, gpmm_render_align)

    """
    Ori image align, id
    """
    image_align = facenet_align(list_img, list_lm2d_gt, defined_lm_facenet_align, opt.img_height, opt.img_width)
    tgt_image_id = pred_encoder_id(opt, image_align)

    return gpmm_render_tar_id, tgt_image_id, gpmm_render_align, image_align

# decode mesh
def decoder_colorMesh(h_lrgp, list_gpmm, list_gpmm_color, list_gpmm_exp, list_gpmm_lighting, flag_sgl_mul=None):
    """
    :param list_gpmm:
    :param list_gpmm_color:
    :param list_gpmm_exp:
    :param list_gpmm_lighting:
    :return:
    list_gpmm_vertex:
    list_gpmm_vertexNormal: range (0, 1),
    list_gpmm_vertexColor: range (0, 1_NAN),
    list_gpmm_vertexShade: range (0, 1_NAN),
    list_gpmm_vertexColorOrigin: range (0, 1),
    """
    list_gpmm_vertex = []
    list_gpmm_vertexNormal = []
    list_gpmm_vertexColor = []
    list_gpmm_vertexShade = []
    list_gpmm_vertexColorOrigin = []

    for i in range(len(list_gpmm)):

        gpmm_vertex_src = h_lrgp.instance(list_gpmm[i], list_gpmm_exp[i])

        # gpmm_vertex_src = tf.Print(gpmm_vertex_src, [tf.reduce_mean(gpmm_vertex_src), tf.reduce_mean(list_gpmm[i]),
        #                                              tf.reduce_mean(list_gpmm_exp[i])], message='coeff, vertex')
        gpmm_vertexColor_ori = h_lrgp.instance_color(list_gpmm_color[i])

        gpmm_vertexNormal = vertex_normals_pre_split_fixtopo(
            gpmm_vertex_src, h_lrgp.h_curr.mesh_tri, h_lrgp.h_curr.mesh_vertex_refer_face,
            h_lrgp.h_curr.mesh_vertex_refer_face_index, h_lrgp.h_curr.mesh_vertex_refer_face_num
        )
        gpmm_vertexColor, pred_gpmm_vertexShade = \
            gpmm_lighting(list_gpmm_lighting[i], gpmm_vertexNormal, gpmm_vertexColor_ori)

        list_gpmm_vertex.append(gpmm_vertex_src)
        list_gpmm_vertexNormal.append(gpmm_vertexNormal)

        list_gpmm_vertexColor.append(gpmm_vertexColor)
        list_gpmm_vertexShade.append(pred_gpmm_vertexShade)
        list_gpmm_vertexColorOrigin.append(gpmm_vertexColor_ori)

    return list_gpmm_vertex, list_gpmm_vertexNormal, list_gpmm_vertexColor, list_gpmm_vertexShade, list_gpmm_vertexColorOrigin

def decoder_colorMesh_test(h_lrgp, dict_inter_comm, exp=True, full=False):
    """
    :param list_gpmm:
    :param list_gpmm_color:
    :param list_gpmm_exp:
    :param list_gpmm_lighting:
    :return:
    list_gpmm_vertex:
    list_gpmm_vertexNormal: range (0, 1),
    list_gpmm_vertexColor: range (0, 1_NAN),
    list_gpmm_vertexShade: range (0, 1_NAN),
    list_gpmm_vertexColorOrigin: range (0, 1),
    """
    list_gpmm_vertex = []
    list_gpmm_vertexNormal = []
    list_gpmm_vertexColor = []
    list_gpmm_vertexShade = []
    list_gpmm_vertexColorOrigin = []

    # parse
    list_gpmm = dict_inter_comm['pred_coeff_shape']
    list_gpmm_color = dict_inter_comm['pred_coeff_color']
    list_gpmm_exp = dict_inter_comm['pred_coeff_exp']
    list_gpmm_lighting = dict_inter_comm['pred_coeff_light']

    #
    for i in range(len(list_gpmm)):
        if full:
            if exp == True:
                gpmm_vertex_src = h_lrgp.instance_full(list_gpmm[i], list_gpmm_exp[i])
            else:
                gpmm_vertex_src = h_lrgp.instance_full(list_gpmm[i])

            gpmm_vertexColor_ori = h_lrgp.instance_color_full(list_gpmm_color[i])
        else:
            if exp == True:
                gpmm_vertex_src = h_lrgp.instance(list_gpmm[i], list_gpmm_exp[i])
            else:
                gpmm_vertex_src = h_lrgp.instance(list_gpmm[i])

            gpmm_vertexColor_ori = h_lrgp.instance_color(list_gpmm_color[i])

            gpmm_vertexNormal = vertex_normals_pre_split_fixtopo(
                gpmm_vertex_src, h_lrgp.h_curr.mesh_tri, h_lrgp.h_curr.mesh_vertex_refer_face,
                h_lrgp.h_curr.mesh_vertex_refer_face_index, h_lrgp.h_curr.mesh_vertex_refer_face_num
            )
            gpmm_vertexColor, pred_gpmm_vertexShade = \
                gpmm_lighting(list_gpmm_lighting[i], gpmm_vertexNormal, gpmm_vertexColor_ori)

            list_gpmm_vertexNormal.append(gpmm_vertexNormal)
            list_gpmm_vertexColor.append(gpmm_vertexColor)
            list_gpmm_vertexShade.append(pred_gpmm_vertexShade)

        list_gpmm_vertex.append(gpmm_vertex_src)
        list_gpmm_vertexColorOrigin.append(gpmm_vertexColor_ori)

    return list_gpmm_vertex, list_gpmm_vertexNormal, list_gpmm_vertexColor, list_gpmm_vertexShade, list_gpmm_vertexColorOrigin

def decoder_lm(h_lrgp, list_gpmm_vertex_tar_batch, list_mtx_proj_batch):
    with tf.name_scope("3dmm/lm"):
        list_lm2d = []
        for i in range(len(list_gpmm_vertex_tar_batch)):
            gpmm_vertex_tar_batch = list_gpmm_vertex_tar_batch[i]
            #
            lm3d = h_lrgp.get_lm3d_instance_vertex(h_lrgp.idx_lm68, gpmm_vertex_tar_batch)
            #
            lm2d = project3d_batch(lm3d, list_mtx_proj_batch[i])

            list_lm2d.append(lm2d)

        return list_lm2d

# render
def decoder_renderColorMesh(opt, h_lrgp, list_vertex, list_vertex_normal, list_vertexColor, mtx_perspect_frustrum,
                     list_mtx_model_view, list_cam_position, fore=1, tone=True):
    if isinstance(list_vertex, list)==False:
        list_vertex = [list_vertex]
        list_vertex_normal = [list_vertex_normal]
    if isinstance(list_vertexColor, list) == False:
        list_vertexColor = [list_vertexColor]
    if isinstance(list_mtx_model_view, list) == False:
        list_mtx_model_view = [list_mtx_model_view]
        list_cam_position = [list_cam_position]

    # render
    gpmm_render = []
    gpmm_render_mask = []
    gpmm_render_tri_ids = []
    for i in range(len(list_vertex)):
        if fore > 0:
            vertex_fore = tf.gather(list_vertex[i], h_lrgp.h_curr.idx_subTopo, axis=1)
            vertex_normal_fore = tf.gather(list_vertex_normal[i], h_lrgp.h_curr.idx_subTopo, axis=1)
            vertex_color_fore = tf.gather(list_vertexColor[i], h_lrgp.h_curr.idx_subTopo, axis=1)
            tri = h_lrgp.h_fore.mesh_tri
            # vertex_fore = tf.Print(vertex_fore, [tf.reduce_mean(vertex_fore), tf.reduce_mean(vertex_normal_fore),
            #                                      tf.reduce_mean(vertex_color_fore),
            #                                      tf.reduce_mean(list_mtx_model_view[0]),
            #                                      tf.reduce_mean(list_cam_position[0])], message='before render')
        else:
            vertex_fore = list_vertex[i]
            vertex_normal_fore = list_vertex_normal[i]
            vertex_color_fore = list_vertexColor[i]
            tri = h_lrgp.h_curr.mesh_tri

        if i < len(list_mtx_model_view):
            pred_render, pred_render_mask, pred_render_tri_ids = gpmm_render_image(
                opt, vertex_fore, tri, vertex_normal_fore, vertex_color_fore,
                mtx_perspect_frustrum, list_mtx_model_view[i], list_cam_position[i], tone
            )
        else:
            pred_render, pred_render_mask, pred_render_tri_ids = gpmm_render_image(
                opt, vertex_fore, tri, vertex_normal_fore, vertex_color_fore,
                mtx_perspect_frustrum, list_mtx_model_view[0], list_cam_position[0], tone
            )

        gpmm_render.append(pred_render)
        gpmm_render_mask.append(pred_render_mask)
        gpmm_render_tri_ids.append(pred_render_tri_ids)
    return gpmm_render, gpmm_render_mask, gpmm_render_tri_ids

def decoder_renderColorMesh_gary(opt, h_lrgp, list_vertex, list_vertex_normal, list_vertexColor, mtx_perspect_frustrum,
                     list_mtx_model_view, list_cam_position, fore=1, tone=True, background=10.0):
    if isinstance(list_vertex, list)==False:
        list_vertex = [list_vertex]
        list_vertex_normal = [list_vertex_normal]
    if isinstance(list_vertexColor, list) == False:
        list_vertexColor = [list_vertexColor]
    if isinstance(list_mtx_model_view, list) == False:
        list_mtx_model_view = [list_mtx_model_view]
        list_cam_position = [list_cam_position]
    # render
    gpmm_render = []
    gpmm_render_mask = []
    gpmm_render_tri_ids = []
    for i in range(len(list_vertex)):
        if fore > 0:
            vertex_fore = tf.gather(list_vertex[i], h_lrgp.h_curr.idx_subTopo, axis=1)
            vertex_normal_fore = tf.gather(list_vertex_normal[i], h_lrgp.h_curr.idx_subTopo, axis=1)
            vertex_color_fore = tf.gather(list_vertexColor[i], h_lrgp.h_curr.idx_subTopo, axis=1)
            tri = h_lrgp.h_fore.mesh_tri
        else:
            vertex_fore = list_vertex[i]
            vertex_normal_fore = list_vertex_normal[i]
            vertex_color_fore = list_vertexColor[i]
            tri = h_lrgp.h_curr.mesh_tri

        if i < len(list_mtx_model_view):
            pred_render, pred_render_mask, pred_render_tri_ids = gpmm_render_image_garyLight(
                opt, vertex_fore, tri, vertex_normal_fore, vertex_color_fore,
                mtx_perspect_frustrum, list_mtx_model_view[i], list_cam_position[i], tone, background
            )
        else:
            pred_render, pred_render_mask, pred_render_tri_ids = gpmm_render_image_garyLight(
                opt, vertex_fore, tri, vertex_normal_fore, vertex_color_fore,
                mtx_perspect_frustrum, list_mtx_model_view[0], list_cam_position[0], tone, background
            )

        gpmm_render.append(pred_render)
        gpmm_render_mask.append(pred_render_mask)
        gpmm_render_tri_ids.append(pred_render_tri_ids)
    return gpmm_render, gpmm_render_mask, gpmm_render_tri_ids

def decoder_depth(opt, h_lrgp, list_vertex, mtx_perspect_frustrum, list_mtx_ext, list_mtx_model_view, list_cam_position, fore=1):
    refine_depths = []
    refine_depths_mask = []
    with tf.name_scope("3dmm/depth"):
        for i in range(len(list_vertex)):
            if fore > 0:
                vertex_fore = tf.gather(list_vertex[i], h_lrgp.h_curr.idx_subTopo, axis=1)
                tri = h_lrgp.h_fore.mesh_tri
            else:
                vertex_fore = list_vertex[i]
                tri = h_lrgp.h_curr.mesh_tri

            if i < len(list_mtx_model_view):
                pred_render, mask = gpmm_generate_depthmap(
                    opt, vertex_fore, tri,
                    mtx_perspect_frustrum, list_mtx_ext[i], list_mtx_model_view[i], list_cam_position[i]
                )
            else:
                pred_render, mask = gpmm_generate_depthmap(
                    opt, vertex_fore, tri,
                    mtx_perspect_frustrum, list_mtx_ext[0], list_mtx_model_view[0], list_cam_position[0]
                )
            refine_depths.append(pred_render)
            refine_depths_mask.append(mask)
    return refine_depths, refine_depths_mask

#
def facenet_image_process(image_batch_float):
    list_img_std = []
    for b in range(image_batch_float.shape[0]):
        image_std = (tf.cast(image_batch_float[b], tf.float32) - 127.5) / 128.0
        # image_std = tf.image.per_image_standardization(image_batch_float[b])
        list_img_std.append(image_std)
    return tf.stack(list_img_std, axis=0)

# detail api
def gpmm_lighting(gamma, norm, face_texture):
    # compute vertex color using face_texture and SH function lighting approximation
    # input: face_texture with shape [1,N,3]
    # 	     norm with shape [1,N,3]
    #		 gamma with shape [1,27]
    # output: face_color with shape [1,N,3], RGB order, range from 0-1
    #		  lighting with shape [1,N,3], color under uniform texture, range from 0-1
    batch_size = face_texture.shape[0]
    num_vertex = face_texture.shape[1]

    init_lit = tf.constant([0.8, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    init_lit = tf.reshape(init_lit, [1, 1, 9])
    init_lit = tf.tile(init_lit, multiples=[batch_size, 3, 1])

    gamma = tf.reshape(gamma, [-1, 3, 9])
    gamma = gamma + init_lit

    # parameter of 9 SH function
    a0 = np.pi
    a1 = 2 * np.pi / np.sqrt(3.0)
    a2 = 2 * np.pi / np.sqrt(8.0)
    c0 = 1 / np.sqrt(4 * np.pi)
    c1 = np.sqrt(3.0) / np.sqrt(4 * np.pi)
    c2 = 3 * np.sqrt(5.0) / np.sqrt(12 * np.pi)

    Y0 = tf.tile(tf.reshape(a0 * c0, [1, 1, 1]), [batch_size, num_vertex, 1])
    Y0 = tf.cast(Y0, tf.float32)
    Y1 = tf.reshape(-a1 * c1 * norm[:, :, 1], [batch_size, num_vertex, 1])
    Y2 = tf.reshape(a1 * c1 * norm[:, :, 2], [batch_size, num_vertex, 1])
    Y3 = tf.reshape(-a1 * c1 * norm[:, :, 0], [batch_size, num_vertex, 1])
    Y4 = tf.reshape(a2 * c2 * norm[:, :, 0] * norm[:, :, 1], [batch_size, num_vertex, 1])
    Y5 = tf.reshape(-a2 * c2 * norm[:, :, 1] * norm[:, :, 2], [batch_size, num_vertex, 1])
    Y6 = tf.reshape(a2 * c2 * 0.5 / tf.sqrt(3.0) * (3 * tf.square(norm[:, :, 2]) - 1), [batch_size, num_vertex, 1])
    Y7 = tf.reshape(-a2 * c2 * norm[:, :, 0] * norm[:, :, 2], [batch_size, num_vertex, 1])
    Y8 = tf.reshape(a2 * c2 * 0.5 * (tf.square(norm[:, :, 0]) - tf.square(norm[:, :, 1])), [batch_size, num_vertex, 1])

    Y = tf.concat([Y0, Y1, Y2, Y3, Y4, Y5, Y6, Y7, Y8], axis=2)

    # Y shape:[batch,N,9].
    lit_r = tf.squeeze(tf.matmul(Y, tf.expand_dims(gamma[:, 0, :], 2)), 2)  # [batch,N,9] * [batch,9,1] = [batch,N]
    lit_g = tf.squeeze(tf.matmul(Y, tf.expand_dims(gamma[:, 1, :], 2)), 2)
    lit_b = tf.squeeze(tf.matmul(Y, tf.expand_dims(gamma[:, 2, :], 2)), 2)

    # shape:[batch,N,3]
    face_color = tf.stack(
        [lit_r * face_texture[:, :, 0], lit_g * face_texture[:, :, 1], lit_b * face_texture[:, :, 2]], axis=2)
    shade_color = tf.stack([lit_r, lit_g, lit_b], axis=2)

    # face_color = tf.clip_by_value(face_color, 0.0, 1.0)
    # shade_color = tf.clip_by_value(shade_color, 0.0, 1.0)

    return face_color, shade_color  # (0, Nan) (0, Nan)


def gpmm_face_replace(list_img, list_gpmm_render_img, list_gpmm_render_mask):
    if isinstance(list_img, list) == False:
        list_img = [list_img]
    if isinstance(list_gpmm_render_img, list) == False:
        list_gpmm_render_img = [list_gpmm_render_img]
    if isinstance(list_gpmm_render_mask, list) == False:
        list_gpmm_render_mask = [list_gpmm_render_mask]

    list_gpmm_render_img_replace = []
    for i in range(len(list_img)):
        img = list_img[i]
        gpmm_render_img = list_gpmm_render_img[i]
        gpmm_render_mask = list_gpmm_render_mask[i]
        gpmm_render_mask = tf.tile(gpmm_render_mask, multiples=[1, 1, 1, 3])

        img_replace = gpmm_render_img + (1.0 - gpmm_render_mask) * img

        list_gpmm_render_img_replace.append(img_replace)

    return list_gpmm_render_img_replace


def gpmm_render_image(opt, vertex, tri, vertex_normal, vertex_color, mtx_perspect_frustrum, mtx_model_view,
                      cam_position, tone=True):
    """
    :param vertex: [bs, num_ver, 3]
    :param tri: [bs, num_tri, 3] or [num_tri, 3]
    :param vertex_normal: [bs, num_ver, 3]
    :param vertex_color: [bs, num_ver, 3]
    :param mtx_perspect_frustrum: [bs, 4, 4]
    :param mtx_model_view: [bs, 4, 4]
    :param cam_position: [bs, 3]
    :return:
    render_image, shape=[batch_size, h, w, 3], dtype=tf_render.float32
    render_image_mask, shape=[batch_size, h, w, 1], dtype=tf_render.float32
    render_tri_ids, shape=[batch_size, h, w, 1], dtype=tf_render.int32
    """

    # manual light
    # light_positions = tf.constant([[0.0, 0.0, 1000.0]], shape=[1, 1, 3])
    # light_intensities = tf.constant([[1.0, 0.0, 0.0]], shape=[1, 1, 3])
    # ambient_color = tf.constant([[1.0, 1.0, 1.0]], shape=[1, 3])
    # ambient_color = tf.tile(ambient_color, [opt.batch_size, 1])

    if len(tri.shape) == 2:
        render_image, render_image_mask, render_tri_ids = \
            mesh_renderer_camera_light(vertex, tri, vertex_normal, vertex_color, mtx_model_view,
                                       mtx_perspect_frustrum, cam_position, opt.img_width, opt.img_height)
        if tone:
            tonemapped_renders = tf.concat(
                [
                    tone_mapper(render_image[:, :, :, 0:3], 0.7),
                    render_image[:, :, :, 3:4]
                ],
                axis=3)
        else:
            tonemapped_renders = tf.clip_by_value(render_image, 0.0, 100000.0)

    else:
        list_tonemapped_renders = []
        list_render_image_mask = []
        list_render_tri_ids = []
        for i in range(tri.shape[0]):  # bs
            render_image, render_image_mask, render_tri_ids = \
                mesh_renderer_camera_light(
                    vertex[i:i + 1, :, :], tri[i], vertex_normal[i:i + 1, :, :], vertex_color[i:i + 1, :, :],
                    mtx_model_view[i:i + 1, :, :], mtx_perspect_frustrum[i:i + 1, :, :], cam_position[i:i + 1, :],
                    opt.img_width, opt.img_height)

            if tone:
                tonemapped_renders = tf.concat(
                    [
                        tone_mapper(render_image[:, :, :, 0:3], 0.7),
                        render_image[:, :, :, 3:4]
                    ],
                    axis=3)
            else:
                tonemapped_renders = tf.clip_by_value(render_image, 0.0, 100000.0)

            list_tonemapped_renders.append(tonemapped_renders)
            list_render_image_mask.append(render_image_mask)
            list_render_tri_ids.append(render_tri_ids)

        tonemapped_renders = tf.concat(list_tonemapped_renders, axis=0)
        render_image_mask = tf.concat(list_render_image_mask, axis=0)
        render_tri_ids = tf.concat(list_render_tri_ids, axis=0)

    return tonemapped_renders[:, :, :, 0:3], render_image_mask, render_tri_ids

def gpmm_render_image_garyLight(opt, vertex, tri, vertex_normal, vertex_color, mtx_perspect_frustrum, mtx_model_view,
                      cam_position, tone=True, background=10.999):
    """
    :param vertex: [bs, num_ver, 3]
    :param tri: [bs, num_tri, 3] or [num_tri, 3]
    :param vertex_normal: [bs, num_ver, 3]
    :param vertex_color: [bs, num_ver, 3]
    :param mtx_perspect_frustrum: [bs, 4, 4]
    :param mtx_model_view: [bs, 4, 4]
    :param cam_position: [bs, 3]
    :return:
    render_image, shape=[batch_size, h, w, 3], dtype=tf_render.float32
    render_image_mask, shape=[batch_size, h, w, 1], dtype=tf_render.float32
    render_tri_ids, shape=[batch_size, h, w, 1], dtype=tf_render.int32
    """

    # manual light
    light_positions = tf.constant([[0.0, 0.0, 1000.0, -1000.0, 0.0, 1000.0, 1000.0, 0.0, 1000.0]], shape=[1, 3, 3])
    light_positions = tf.tile(light_positions, [opt.batch_size, 1, 1])
    light_intensities = tf.constant([[0.50, 0.50, 0.50]], shape=[1, 3, 3])
    light_intensities = tf.tile(light_intensities, [opt.batch_size, 1, 1])
    # ambient_color = tf.constant([[1.0, 1.0, 1.0]], shape=[1, 3])
    # ambient_color = tf.tile(ambient_color, [opt.batch_size, 1])

    if len(tri.shape) == 2:
        render_image, render_image_mask = \
            mesh_renderer_camera(vertex, tri, vertex_normal, vertex_color, mtx_model_view,
                                 mtx_perspect_frustrum, cam_position, light_positions, light_intensities,
                                 opt.img_width, opt.img_height, background=background)

        tonemapped_renders = tf.clip_by_value(render_image, 0.0, 100000.0)

    else:
        list_tonemapped_renders = []
        list_render_image_mask = []
        list_render_tri_ids = []
        for i in range(tri.shape[0]):  # bs
            render_image, render_image_mask = \
                mesh_renderer_camera(
                    vertex[i:i + 1, :, :], tri[i], vertex_normal[i:i + 1, :, :], vertex_color[i:i + 1, :, :],
                    mtx_model_view[i:i + 1, :, :], mtx_perspect_frustrum[i:i + 1, :, :], cam_position[i:i + 1, :],
                    light_positions, light_intensities, opt.img_width, opt.img_height, background=10.999)

            tonemapped_renders = tf.clip_by_value(render_image, 0.0, 100000.0)

            list_tonemapped_renders.append(tonemapped_renders)
            list_render_image_mask.append(render_image_mask)
            list_render_tri_ids.append(1)

        tonemapped_renders = tf.concat(list_tonemapped_renders, axis=0)
        render_image_mask = tf.concat(list_render_image_mask, axis=0)
        render_tri_ids = tf.concat(list_render_tri_ids, axis=0)

    return tonemapped_renders[:, :, :, 0:3], render_image_mask, render_image_mask


def gpmm_generate_depthmap(opt, mesh, tri, mtx_perspect_frustrum, mtx_ext, mtx_model_view, cam_position):
    depthmap, depthmap_mask = mesh_depthmap_camera(mesh, tri, mtx_ext, mtx_model_view, mtx_perspect_frustrum,
                                                   opt.img_width, opt.img_height)

    depthmap = depthmap * tf.squeeze(depthmap_mask, axis=-1)
    depthmap = tf.clip_by_value(depthmap, opt.depth_min, opt.depth_max)
    depthmap = tf.expand_dims(depthmap, axis=-1)

    return depthmap, depthmap_mask

