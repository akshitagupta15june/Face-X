
# system
from __future__ import print_function

# tf_render
import tensorflow as tf

# tianwei
from src_common.geometry.geo_utils import get_relative_pose, mat2pose_vec

# self
# jiaxiang
from src_common.geometry.render.tensor_render import *
from src_common.geometry.covisible import mm_covisible_tri
# python lib
#

"""
Multi-view decoder
"""
def decoder_warppose(opt, list_pred_pose):
    pred_pose_render = list_pred_pose[0]
    rel_pose_list = []
    for i in range(opt.num_source):
        rel_pose_l = get_relative_pose(pred_pose_render, list_pred_pose[1+i]) # careful of [tar, src]
        rel_pose_l = mat2pose_vec(rel_pose_l)
        #rel_pose_l = tf.Print(rel_pose_l, [rel_pose_l], message='rel_pose_l')
        rel_pose_list.append(rel_pose_l)
    #image_pred_poses = tf.stack(rel_pose_list, axis=1)
    return rel_pose_list

def decoder_covisible_map(opt, h_lrgp, gpmm_frustrum, dict_inter_common):
    # input
    list_tri_buffer = dict_inter_common['3dmm_render_tri_id']

    # return
    list_gpmm_covisibleMap = []

    """
    Here calculate consistence triangle
    """
    tri_ids_tar = list_tri_buffer[0]  # triangle id on image for per pixel
    for i in range(opt.num_source):
        # careful of [tar, src]
        tri_ids_src = list_tri_buffer[1+i]  # triangle id on image for per pixel

        tri_consistency = mm_covisible_tri(h_lrgp, tri_ids_tar, tri_ids_src)

        # fore render
        zbf_vertex_fore_tar = tf.gather(dict_inter_common['gpmm_vertex'][0], h_lrgp.h_curr.idx_subTopo, axis=1)
        zbf_vertex_normal_fore = tf.gather(dict_inter_common['gpmm_vertexNormal'][0], h_lrgp.h_curr.idx_subTopo, axis=1)
        zbf_vertex_color_fore = tf.gather(dict_inter_common['gpmm_vertexColor'][0], h_lrgp.h_curr.idx_subTopo, axis=1)

        _, zbuffer_mask, _ = gpmm_render_image(
            opt, zbf_vertex_fore_tar, tri_consistency, zbf_vertex_normal_fore, zbf_vertex_color_fore,
            gpmm_frustrum, dict_inter_common['pred_cam_mv'][0], dict_inter_common['pred_cam_eye'][0]
        )
        list_gpmm_covisibleMap.append(zbuffer_mask)
        # mid
        # _, zbuffer_mask, _ = gpmm_render_image(
        #     opt, dict_inter_common['gpmm_vertex'][0], tri_consistency, dict_inter_common['gpmm_vertexNormal'][0], dict_inter_common['gpmm_vertexColor'][0],
        #     gpmm_frustrum, dict_inter_common['pose_mv'][0], dict_inter_common['pose_eye'][0]
        # )
        # list_gpmm_covisibleMap.append(zbuffer_mask)
    return list_gpmm_covisibleMap

def decoder_align_depthMap(opt, list_depthMap, list_syn_depthMap, list_syn_mask):
    list_depth_align = []
    for i in range(len(list_depthMap)):
        visible_target_depth = list_depthMap[i]
        visible_source_depth = list_syn_depthMap[i]
        proj_mask = list_syn_mask[i]

        # radio
        visible_target_depth_mask = tf.multiply(visible_target_depth, proj_mask)
        visible_source_depth_mask = tf.multiply(visible_source_depth, proj_mask)

        # visible_tar_depth_value = tf.boolean_mask(visible_source_depth_mask, proj_mask)
        # visible_src_depth_value = tf.boolean_mask(visible_source_depth_mask, proj_mask)
        mean_target_depth = tf.reduce_sum(visible_target_depth_mask, axis=[1, 2]) / \
                            (tf.reduce_sum(proj_mask, axis=[1, 2]) + 1.0)
        mean_source_depth = tf.reduce_sum(visible_source_depth_mask, axis=[1, 2]) / \
                            (tf.reduce_sum(proj_mask, axis=[1, 2]) + 1.0)

        depth_ratio = mean_target_depth / (mean_source_depth + 1e-6)
        #depth_ratio = tf.Print(depth_ratio ,[depth_ratio], message='depth_ratio ')
        visible_source_depth_radio = list_syn_depthMap[i] * \
                                     tf.tile(tf.reshape(depth_ratio, [opt.batch_size, 1, 1, 1]), [1, opt.img_height, opt.img_width, 1])

        #pred_render_max = tf.reduce_max(visible_source_depth_radio)
        #pred_render_min = tf.reduce_min(visible_source_depth_radio)
        #visible_source_depth_radio = tf.Print(visible_source_depth_radio, [pred_render_max, pred_render_min], message='src align depthmap')
        list_depth_align.append(visible_source_depth_radio)

    return list_depth_align