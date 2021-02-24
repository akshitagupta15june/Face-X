# system
from __future__ import print_function

# tf_render
import tensorflow as tf
import tensorflow.contrib.slim as slim

# self
# jiaxiang
from src_common.common.visual_helper import pixel_error_heatmap
# tianwei
from src_common.geometry.geo_utils import fundamental_matrix_from_rt, reprojection_error

# common losses
def compute_3dmm_regular_l2_loss(pred_batch_list):
    gpmm_regular_loss = 0.0
    for i in range(len(pred_batch_list)):
        loss_gpmm_src_reg = tf.reduce_sum(tf.square(pred_batch_list[i]))
        gpmm_regular_loss += loss_gpmm_src_reg
    return gpmm_regular_loss

def compute_lm_eul_square_loss(pred_batch_list, gt_batch_list, weight=None):
    gpmm_lm_loss = 0.0
    for i in range(len(pred_batch_list)):
        lm2d = pred_batch_list[i]
        lm2d_gt = gt_batch_list[i]

        lm_loss = tf.reduce_sum(tf.square(lm2d - lm2d_gt), 2)
        if weight is not None:
            lm_loss = lm_loss * weight

        gpmm_lm_loss += tf.reduce_mean(lm_loss)
    return gpmm_lm_loss

def compute_3dmm_render_eul_masknorm_skin_loss(pred_batch_list, pred_mask_batch_list, pred_skin_batch_list, gt_batch):
    """
    :param pred_batch_list:
    :param pred_mask_batch_list:
    :param pred_skin_batch_list:
    :param gt_batch:
    :return:
    """
    if isinstance(pred_batch_list, list) == False:
        pred_batch_list = [pred_batch_list]
    if isinstance(pred_mask_batch_list, list) == False:
        pred_mask_batch_list = [pred_mask_batch_list]

    gpmm_pixel_loss = 0.0
    list_render_loss_error = []
    for i in range(len(pred_batch_list)):
        pred = pred_batch_list[i] * pred_skin_batch_list[i] # (0, 1) * (0, 1)
        gt = gt_batch[i] * pred_skin_batch_list[i]

        # l1
        curr_render_error = pred - gt
        curr_render_src_error = tf.reduce_sum(tf.square(curr_render_error), 3)  # bs, h, w, c
        curr_render_src_error = tf.sqrt(curr_render_src_error + 1e-6)
        curr_render_src_error = tf.expand_dims(curr_render_src_error, -1)  # bs, h, w, 1 # (0, 1) * (0, 1)

        list_render_loss_error.append(pixel_error_heatmap(curr_render_src_error))

        # loss
        # curr_render_mask_sum = pred_mask_batch_list[i]
        # curr_render_mask_sum = tf.reduce_sum(curr_render_mask_sum, axis=[1, 2, 3])

        curr_render_option_sum = pred_skin_batch_list[i] * pred_mask_batch_list[i]
        curr_render_option_sum = tf.reduce_sum(curr_render_option_sum, axis=[1, 2, 3])

        curr_render_src_error = tf.reduce_sum(curr_render_src_error, axis=[1, 2, 3])
        curr_render_src_error = curr_render_src_error / (curr_render_option_sum + 1e-6)
        curr_render_src_error = tf.reduce_mean(curr_render_src_error)

        gpmm_pixel_loss += curr_render_src_error

    return gpmm_pixel_loss, list_render_loss_error

def compute_3dmm_id_cos_loss(pred_batch_list, gt_batch_list):
    if isinstance(pred_batch_list, list) == False:
        pred_batch_list = [pred_batch_list]
    if isinstance(gt_batch_list, list) == False:
        gt_batch_list = [gt_batch_list]

    gpmm_id_loss = 0.0
    list_simi_norm = []
    for i in range(len(pred_batch_list)):
        pred = pred_batch_list[i]
        gt = gt_batch_list[i]

        simi = tf.reduce_sum(tf.multiply(pred, gt), axis=1)  # bs, 199
        #x_norm = tf_render.sqrt(tf_render.reduce_sum(tf_render.square(pred) + 1e-6, axis=1))
        #y_norm = tf_render.sqrt(tf_render.reduce_sum(tf_render.square(gt) + 1e-6, axis=1))
        #loss = loss / (x_norm*y_norm + 1e-6)
        #simi = tf.Print(simi, [simi], message="simi", summarize=4)
        simi_norm = (simi + 1.0) / 2.0
        list_simi_norm.append(simi_norm)

        loss = -simi + 1.0
        loss = tf.reduce_mean(loss)
        gpmm_id_loss += loss

    return gpmm_id_loss, list_simi_norm

# MGC losses
def combine_flag_sgl_mul_loss(loss_batch_list, flag_sgl_mul_curr, flag_batch_norm=True):
    loss = tf.constant(0.0)
    for i in range(len(loss_batch_list)):
        curr_proj_error = loss_batch_list[i]
        curr_proj_error = tf.expand_dims(curr_proj_error, -1)
        # curr_proj_error = tf.Print(curr_proj_error, [curr_proj_error.shape, self.flag_sgl_mul.shape], message='curr_proj_error', summarize=16)
        curr_proj_error_batch = curr_proj_error * flag_sgl_mul_curr  # bs
        #tf.print(curr_proj_error_batch,[tf.reduce_mean(curr_proj_error_batch), flag_sgl_mul_curr],message='flag_curr_proj_error_batch', summarize=12)
        # curr_proj_error_batch = tf.Print(curr_proj_error_batch, [curr_proj_error_batch], message='curr_proj_error_batch', summarize=16)
        # self.pixel_loss += tf.reduce_sum(curr_proj_error_batch) / (tf.reduce_sum(self.flag_sgl_mul) + 1e-6)
        if flag_batch_norm:
            loss += tf.reduce_mean(curr_proj_error_batch)
        else:
            loss += tf.reduce_sum(curr_proj_error_batch) / (tf.reduce_sum(flag_sgl_mul_curr) + 1e-6)
    return loss

def compute_pixel_eul_loss_list(pred_batch, pred_mask_batch, pred_gpmmmask_batch, gt_batch):
    render_loss = 0.0
    list_render_loss_batch = []
    list_render_loss_visual_batch = []
    for i in range(len(pred_batch)):
        pred = pred_batch[i]
        warp_mask = pred_mask_batch[i]
        render_mask = pred_gpmmmask_batch[i]
        gt = gt_batch[i]
        curr_render_error, curr_render_src_error_visual = compute_pixel_eul_loss(pred, warp_mask, render_mask, gt)

        list_render_loss_batch.append(curr_render_error)
        list_render_loss_visual_batch.append(curr_render_src_error_visual)
    return list_render_loss_batch, list_render_loss_visual_batch

def compute_pixel_eul_loss(pred_batch, pred_mask_batch, pred_gpmmmask_batch, gt_batch):
    # eul
    curr_render_error = pred_batch - gt_batch

    curr_render_src_error = tf.reduce_sum(tf.square(curr_render_error), 3)

    curr_render_src_error = tf.sqrt(curr_render_src_error + 1e-6)
    curr_render_src_error = tf.expand_dims(curr_render_src_error, -1)

    curr_render_src_error = curr_render_src_error * pred_mask_batch
    curr_render_src_error = curr_render_src_error * pred_gpmmmask_batch

    curr_render_src_error_visual = pixel_error_heatmap(curr_render_src_error)

    curr_render_option_sum = pred_mask_batch * pred_gpmmmask_batch
    curr_render_option_sum = tf.reduce_sum(curr_render_option_sum, axis=[1, 2, 3])

    curr_render_src_error = tf.reduce_sum(curr_render_src_error, axis=[1, 2, 3])
    curr_render_src_error = curr_render_src_error / (curr_render_option_sum + 1e-6)

    # curr_render_src_error = tf.Print(curr_render_src_error, [tf.reduce_sum(curr_render_src_error), tf.reduce_sum(curr_render_option_sum)],
    #                            message='error')

    return curr_render_src_error, curr_render_src_error_visual

def compute_ssim_loss_list(list_x, list_y, list_mask):
    list_ssim_error = []
    for i in range(len(list_x)):
        x = list_x[i]
        y = list_y[i]

        ssim_error = compute_ssim_loss(x,y) * list_mask[i]
        ssim_error = tf.reduce_mean(ssim_error, axis=[1, 2, 3])

        list_ssim_error.append(ssim_error)
    return list_ssim_error

def compute_ssim_loss(x, y):
    # reference https://github.com/tensorflow/models/tree/master/research/vid2depth/model.py
    """Computes a differentiable structured image similarity measure."""
    c1 = 0.01 ** 2
    c2 = 0.03 ** 2
    mu_x = slim.avg_pool2d(x, 3, 1, 'VALID')
    mu_y = slim.avg_pool2d(y, 3, 1, 'VALID')
    sigma_x = slim.avg_pool2d(x ** 2, 3, 1, 'VALID') - mu_x ** 2
    sigma_y = slim.avg_pool2d(y ** 2, 3, 1, 'VALID') - mu_y ** 2
    sigma_xy = slim.avg_pool2d(x * y, 3, 1, 'VALID') - mu_x * mu_y
    ssim_n = (2 * mu_x * mu_y + c1) * (2 * sigma_xy + c2)
    ssim_d = (mu_x ** 2 + mu_y ** 2 + c1) * (sigma_x + sigma_y + c2)
    ssim = ssim_n / ssim_d
    return tf.clip_by_value((1 - ssim) / 2, 0, 1)

def compute_depthmap_l1_loss_list(list_pred_batch, list_pred_mask_batch, list_gt_batch):
    list_curr_viewSyn_depth_error = []
    list_curr_viewSyn_pixel_depth_visual = []
    for i in range(len(list_pred_batch)):
        curr_render_src_error, curr_render_src_error_visual = \
            compute_depthmap_l1_loss(list_pred_batch[i], list_pred_mask_batch[i], list_pred_mask_batch[i], list_gt_batch[i])
        #curr_render_src_error = tf.Print(curr_render_src_error, [tf.reduce_mean(curr_render_src_error)],message='curr_render_src_error')
        list_curr_viewSyn_depth_error.append(curr_render_src_error)
        list_curr_viewSyn_pixel_depth_visual.append(curr_render_src_error_visual)
    return list_curr_viewSyn_depth_error, list_curr_viewSyn_pixel_depth_visual

def compute_depthmap_l1_loss(pred_batch, pred_mask_batch, pred_gpmmmask_batch, gt_batch):
    batch_size = pred_batch.shape[0]
    # l1
    curr_render_error = pred_batch - gt_batch

    curr_render_src_error = tf.abs(curr_render_error)

    curr_render_src_error = curr_render_src_error * pred_mask_batch
    curr_render_src_error = curr_render_src_error * pred_gpmmmask_batch


    error_max = tf.reduce_max(tf.reshape(curr_render_src_error, [batch_size, -1]), axis=1)
    curr_render_src_error_norm = tf.divide(curr_render_src_error, tf.reshape(error_max, [batch_size, 1, 1, 1]) + 1e-6)

    curr_render_src_error_visual = pixel_error_heatmap(curr_render_src_error_norm)

    # curr_render_option_sum = pred_mask_batch * pred_gpmmmask_batch
    # curr_render_option_sum = tf.reduce_sum(curr_render_option_sum, axis=[1, 2, 3])
    #
    # curr_render_src_error = tf.reduce_sum(curr_render_src_error, axis=[1, 2, 3])
    # curr_render_src_error = curr_render_src_error / (curr_render_option_sum + 1e-6)

    curr_render_src_error = tf.reduce_mean(curr_render_src_error, axis=[1, 2, 3])

    return curr_render_src_error, curr_render_src_error_visual

def compute_match_loss_list(list_points, pred_depth, list_pose, intrinsics):
    list_epiLoss_batch = []
    list_reprojLoss_batch = []
    mgc_epi_lines = []
    mgc_epi_distances = []
    for i in range(len(list_points)-1):
        dist_p2l_aver, reproj_error, epi_lines, dist_p2l = \
            compute_match_loss(list_points[0], list_points[1+i], pred_depth, list_pose[i], intrinsics)
        list_epiLoss_batch.append(dist_p2l_aver)
        list_reprojLoss_batch.append(reproj_error)
        mgc_epi_lines.append(epi_lines)
        mgc_epi_distances.append(dist_p2l)
    return list_epiLoss_batch, list_reprojLoss_batch, mgc_epi_lines, mgc_epi_distances

def compute_match_loss(points1, points2, pred_depth, pose, intrinsics):
    batch_size = points1.shape[0]
    match_num = tf.shape(points1)[1]

    ones = tf.ones([batch_size, match_num, 1])
    points1 = tf.concat([points1, ones], axis=2) # bs, num, 3
    points2 = tf.concat([points2, ones], axis=2)

    # compute fundamental matrix loss
    fmat = fundamental_matrix_from_rt(pose, intrinsics)
    fmat = tf.expand_dims(fmat, axis=1)
    fmat_tiles = tf.tile(fmat, [1, match_num, 1, 1])

    list_epi_lines = []
    list_dist_p2l = []
    for i in range(batch_size):
        epi_lines = tf.matmul(fmat_tiles[i], tf.expand_dims(points1, axis=3)[i])
        dist_p2l = tf.abs(tf.matmul(tf.transpose(epi_lines, perm=[0, 2, 1]), tf.expand_dims(points2, axis=3)[i]))
        list_epi_lines.append(epi_lines)
        list_dist_p2l.append(dist_p2l)
    epi_lines = tf.stack(list_epi_lines)
    dist_p2l = tf.stack(list_dist_p2l)

    a = tf.slice(epi_lines, [0, 0, 0, 0], [-1, -1, 1, -1])
    b = tf.slice(epi_lines, [0, 0, 1, 0], [-1, -1, 1, -1])
    dist_div = tf.sqrt(a * a + b * b) + 1e-6
    dist_p2l = (dist_p2l / dist_div)

    dist_p2l_aver = tf.reduce_mean(dist_p2l, axis=[1, 2, 3])
    #dist_p2l_aver = tf.Print(dist_p2l_aver, [tf.shape(dist_p2l_aver), dist_p2l_aver], message="dist_p2l_aver", summarize=2 * 16)

    # compute projection loss
    reproj_error = reprojection_error(points1, points2, pred_depth, pose, intrinsics)
    #reproj_error = tf.Print(reproj_error, [tf.shape(reproj_error), reproj_error], message="reproj_error", summarize=2 * 16)


    return dist_p2l_aver, reproj_error, epi_lines, dist_p2l