from __future__ import division
from __future__ import print_function

#
import os
import time

from src_common.common.format_helper import *
from src_common.common.visual_helper import *
# data
from src_common.data import find_dataloader_using_name
# geometry
from src_common.geometry.gpmm.bfm09_tf_uv import BFM_TF
# common
from src_common.common.parse_encoder import parse_coeff_list
from src_common.geometry.camera_distribute.camera_utils import *
from src_common.geometry.face_align.align_facenet import get_facenet_align_lm
from .deep_3dmm_decoder import *
# multiview
from src_common.geometry.geo_utils import projective_inverse_warp
from .decoder_multiView import *
# tf
#
# loss
from .deep_3dmm_loss import *

class MGC_TRAIN(object):
    def __init__(self, opt):
        self.opt = opt
        # 3dmm
        self.h_lrgp = BFM_TF(opt.path_gpmm, opt.gpmm_rank, opt.gpmm_exp_rank, opt.batch_size, full=1)


    def build_train_graph_dataLoader(self):
        opt = self.opt
        DataLoader = find_dataloader_using_name(opt.dataset_loader)
        #getattr(sys.modules[__name__], self.data_loader_name)
        data_loader = DataLoader(opt.dataset_dir,
                            opt.batch_size,
                            opt.img_height,
                            opt.img_width,
                            opt.num_source,
                            match_num=opt.match_num,
                            flag_data_aug=opt.flag_data_aug,
                            flag_shuffle=opt.flag_shuffle)

        with tf.name_scope("data_loading"):
            batch_sample = data_loader.load_train_batch()
            # give additional batch_size info since the input is undetermined placeholder
            batch_image_concat, batch_skin_concat, batch_flag_sgl_mul, batch_intrinsic, batch_intrinsic, batch_matches = \
                batch_sample.get_next()

            def process_skin(list_skin):
                list_skin_prop = []
                for skin in list_skin:
                    skin = tf.image.convert_image_dtype(skin, dtype=tf.float32)
                    skin = tf.image.rgb_to_grayscale(skin)

                    tgt_skin_add = skin > 0.5
                    tgt_skin_add = tf.cast(tgt_skin_add , tf.float32)

                    tgt_skin_store = 1.0 - tgt_skin_add

                    tgt_skin_pro = tgt_skin_add + skin * tgt_skin_store

                    list_skin_prop.append(tgt_skin_pro)
                return list_skin_prop

            #
            self.tgt_image = batch_image_concat[:, :, :, :3]
            self.src_image_stack = batch_image_concat[:, :, :, 3:]

            self.tgt_image.set_shape([opt.batch_size, opt.img_height, opt.img_width, 3])  # [bs, 128, 416, 3]
            self.src_image_stack.set_shape([opt.batch_size, opt.img_height, opt.img_width, 3 * opt.num_source]) # [bs, 128, 416, 6]

            self.tgt_image = preprocess_image(self.tgt_image)
            self.src_image_stack = preprocess_image(self.src_image_stack)

            self.list_tar_image = [self.tgt_image]
            self.list_src_image = [self.src_image_stack[:, :, :, i * 3:(i + 1) * 3] for i in range(opt.num_source)]
            self.list_image = self.list_tar_image + self.list_src_image

            #
            self.tgt_skin = batch_skin_concat[:, :, :, :3]
            self.tgt_skin = [self.tgt_skin]
            self.list_tar_skin = process_skin(self.tgt_skin)

            self.src_skin = batch_skin_concat[:, :, :, 3:]
            self.src_skin.set_shape([opt.batch_size, opt.img_height, opt.img_width, 3*opt.num_source])
            self.src_skin = [self.src_skin[:, :, :, i*3:(i+1)*3] for i in range(opt.num_source)]
            self.list_src_skin = process_skin(self.src_skin)
            self.list_skin = self.list_tar_skin + self.list_src_skin

            #
            self.flag_sgl_mul = tf.reshape(batch_flag_sgl_mul, [opt.batch_size])  # [bs, 1]
            self.flag_sgl_mul = tf.cast(self.flag_sgl_mul, dtype=tf.float32) # [0, 1, 2]


            self.matches = batch_matches
            self.matches.set_shape([opt.batch_size, (opt.num_source+1), opt.match_num, 2])

            self.lm2d_weight = np.ones(68, dtype=float)
            self.lm2d_weight[28 - 1:36] = opt.lm_detail_weight
            self.lm2d_weight[61 - 1:] = opt.lm_detail_weight
            self.lm2d_weight = tf.constant(self.lm2d_weight, dtype=tf.float32)

            self.list_lm2d_gt_tar = [self.matches[:, 0, :, :]]
            self.list_lm2d_gt_src = [self.matches[:, i, :, :] for i in range(1, self.matches.shape[1])]
            self.list_lm2d_gt = self.list_lm2d_gt_tar + self.list_lm2d_gt_src

        return data_loader, batch_sample


    def set_constant_node(self):
        opt = self.opt
        """
        ************************************    data load   ************************************
        """
        self.global_step = tf.Variable(0, name='global_step', trainable=False)

        self.batch_size = self.opt.batch_size

        # camera
        defined_pose_main = tf.constant([0.000000, 0.000000, 3.141593, 0.17440447, 9.1053238, 4994.3359], shape=[1, 6])
        self.intrinsics_single = tf.constant([4700.000000, 0., 112.000000, 0., 4700.000000, 112.000000, 0., 0., 1.], shape=[1, 3, 3])
        self.intrinsics_single = tf.tile(self.intrinsics_single, [self.batch_size, 1, 1])
        self.gpmm_frustrum = build_train_graph_3dmm_frustrum(self.intrinsics_single)
        self.defined_pose_main = tf.tile(defined_pose_main, [self.batch_size, 1])

        # identity loss
        defined_lm_facenet_align = get_facenet_align_lm(opt.img_height)
        self.defined_lm_facenet_align = tf.tile(defined_lm_facenet_align, [self.batch_size, 1, 1])

    """
    Train
    """
    def build_train_graph(self, list_coeffALL=None):
        '''[summary]
        build training graph
        Returns:
            data loader and batch sample for train() to initialize
            undefined placeholders
        '''
        opt = self.opt

        """
        ************************************    setting   **********************************************
        """
        self.set_constant_node()
        self.total_loss = tf.constant(0.0)

        # ********************************************  Network
        if list_coeffALL is None:
            list_coeffALL = pred_encoder_coeff_light(self.opt, self.defined_pose_main, self.list_image, is_training=True)

        # ********************************************  Common flow
        dict_loss_common, dict_intermedate_common = \
            self.build_decoderCommon(list_coeffALL, self.list_image, self.list_skin, self.list_lm2d_gt, self.flag_sgl_mul)
        self.dict_inter_comm = dict_intermedate_common

        # ********************************************  Intermediate result for(print, visual, tensorboard)
        # weighted loss for each view
        self.gpmm_regular_shape_loss = dict_loss_common['reg_shape_loss']
        self.gpmm_regular_color_loss = dict_loss_common['reg_color_loss']
        self.gpmm_lm_loss = dict_loss_common['lm2d_loss']
        self.gpmm_pixel_loss = dict_loss_common['render_loss']
        self.gpmm_id_loss = dict_loss_common['id_loss']

        # visual landmark on the rendered images/shade/render loss error map
        #self.gpmm_pose_tar, self.gpmm_pose_src = parse_seq(dict_intermedate_common['pred_6dof_pose'])
        self.lm2d_tar, self.lm2d_src = parse_seq(dict_intermedate_common['pred_lm2d'])
        self.gpmm_render_tar, self.gpmm_render_src = parse_seq(dict_intermedate_common['3dmm_render'])
        self.gpmm_render_mask_tar, self.gpmm_render_mask_src = parse_seq(dict_intermedate_common['3dmm_render_mask'])
        self.gpmm_render_shade_tar, self.gpmm_render_shade_src = parse_seq(dict_intermedate_common['3dmm_render_shade'])
        self.gpmm_render_tri_ids_tar, self.gpmm_render_tri_ids_src = parse_seq(dict_intermedate_common['3dmm_render_tri_id'])
        self.list_render_loss_error_tar, self.list_render_loss_error_src = parse_seq(dict_intermedate_common['3dmm_render_loss_heat'])

        # visual identity facenet input
        self.gpmm_render_tar_align, self.gpmm_render_src_align = parse_seq(dict_intermedate_common['id_render_align'])
        self.tar_image_align, self.src_image_align = parse_seq(dict_intermedate_common['id_image_align'])

        # visual depthmap
        self.tar_depths, self.lr_depths = parse_seq(dict_intermedate_common['3dmm_depthmap'])

        self.gpmm_consist_pixel_tar = self.list_lm2d_gt_tar
        self.gpmm_consist_pixel_src = self.list_lm2d_gt_src

        self.common_loss =  dict_loss_common['loss_common']
        self.total_loss += self.common_loss

        # ********************************************  Multi-view flow
        dict_loss_mgc, dict_inter_mgc = \
            self.build_decoderMGC(self.flag_sgl_mul, self.list_image, self.list_lm2d_gt, dict_intermedate_common)
        self.dict_inter_mgc = dict_inter_mgc

        # loss
        self.ga_loss = dict_loss_mgc['loss_mgc']
        self.ssim_loss = dict_loss_mgc['ssim_loss']
        self.pixel_loss = dict_loss_mgc['pixel_loss']
        self.epipolar_loss = dict_loss_mgc['epi_loss']
        self.depth_loss = dict_loss_mgc['depth_loss']

        self.total_loss += opt.MULTIVIEW_weight * dict_loss_mgc['loss_mgc']

        # ********************************************  Training op
        self.build_train_graph_train_op()


    def build_decoderCommon(self, list_coeff_all, list_image, list_skin=None, list_lm2d_gt=None, flag_sgl_mul=None):
        """
        A common mapping function from images to intermediate result
        :param gpmm_frustrum:
        :param list_image:
            1.can be single image or multi images
            2.shape: [bs, h, w, c]
        :param list_lm2d_gt:
        :return:
        """
        dict_loss_common = dict()
        dict_intermedate_common = dict()
        """
        ************************************    Coefficients (clean)    *********************************
        """
        list_gpmm, list_gpmm_color, list_gpmm_exp, list_gpmm_pose, list_gpmm_light = \
            parse_coeff_list(self.opt, list_coeff_all, self.defined_pose_main)

        dict_intermedate_common['pred_coeff_shape'] = list_gpmm
        dict_intermedate_common['pred_coeff_color'] = list_gpmm_color
        dict_intermedate_common['pred_coeff_exp'] = list_gpmm_exp
        dict_intermedate_common['pred_coeff_light'] = list_gpmm_light

        dict_intermedate_common['pred_6dof_pose'] = list_gpmm_pose

        """
        ************************************    Decoder    **********************************************
        """
        # bfm
        list_gpmm_vertex, list_gpmm_vertexNormal, list_gpmm_vertexColor, list_gpmm_vertexShade, list_gpmm_vertexColorOri = \
            decoder_colorMesh(self.h_lrgp, list_gpmm, list_gpmm_color, list_gpmm_exp, list_gpmm_light, flag_sgl_mul)

        dict_intermedate_common['gpmm_vertex'] = list_gpmm_vertex
        dict_intermedate_common['gpmm_vertexNormal'] = list_gpmm_vertexNormal
        dict_intermedate_common['gpmm_vertexColor'] = list_gpmm_vertexColor
        # cam
        list_gpmm_ext, list_gpmm_proj, list_gpmm_mv, list_gpmm_eye = \
            build_train_graph_3dmm_camera(self.intrinsics_single, list_gpmm_pose)
        dict_intermedate_common['pred_cam_mv'] = list_gpmm_mv
        dict_intermedate_common['pred_cam_eye'] = list_gpmm_eye
        """
        ************************************    Landmark (clean)    *************************************
        """
        # loss:lm
        list_lm2d = decoder_lm(self.h_lrgp, list_gpmm_vertex, list_gpmm_proj)  # bs, ver_num, xy

        dict_intermedate_common['pred_lm2d'] = list_lm2d

        """
        ************************************    Render     **********************************************
        """
        list_gpmm_render, list_gpmm_render_mask, list_gpmm_render_tri_ids = decoder_renderColorMesh(
            # gpmm_vertexColor: (0, Nan)
            self.opt, self.h_lrgp, list_gpmm_vertex, list_gpmm_vertexNormal, list_gpmm_vertexColor,
            self.gpmm_frustrum, list_gpmm_mv, list_gpmm_eye, fore= self.opt.flag_fore, tone=False
        )
        list_gpmm_render = gpmm_face_replace(list_image, list_gpmm_render, list_gpmm_render_mask)

        """
        ************************************    Visualization or Testing    *****************************
        """
        # render visual
        list_gpmm_render_shade, _, _ = decoder_renderColorMesh(  # gpmm_vertexShade: (0, Nan)
            self.opt, self.h_lrgp, list_gpmm_vertex, list_gpmm_vertexNormal, list_gpmm_vertexShade,
            self.gpmm_frustrum, list_gpmm_mv, list_gpmm_eye, fore= self.opt.flag_fore, tone=False
        )
        dict_intermedate_common['3dmm_render_shade'] = list_gpmm_render_shade
        # # main 3 view
        # gpmm_main_ext, gpmm_main_proj, gpmm_main_mv, gpmm_main_eye = \
        #     build_train_graph_3dmm_camera(self.intrinsics_single, self.defined_pose_main)
        # gpmm_render_tar_main, _, _ = decoder_renderColorMesh(
        #     opt, self.h_lrgp, gpmm_vertex, gpmm_vertexNormal, gpmm_vertexColorOri, gpmm_frustrum,
        #     gpmm_main_mv, gpmm_main_eye)

        """
        Weighted Loss
        """
        if list_lm2d_gt is not None:
            # loss:reg
            gpmm_regular_shape_loss = compute_3dmm_regular_l2_loss(list_gpmm)
            gpmm_regular_shape_loss += 0.8 * compute_3dmm_regular_l2_loss(list_gpmm_exp)
            gpmm_regular_color_loss = 0.0017 * compute_3dmm_regular_l2_loss(list_gpmm_color)

            gpmm_lm_loss = compute_lm_eul_square_loss(list_lm2d, list_lm2d_gt, self.lm2d_weight)  # clean

            gpmm_pixel_loss, list_render_loss_error = \
                compute_3dmm_render_eul_masknorm_skin_loss(list_gpmm_render, list_gpmm_render_mask, list_skin,
                                                           list_image)
            dict_intermedate_common['3dmm_render'] = list_gpmm_render
            dict_intermedate_common['3dmm_render_mask'] = list_gpmm_render_mask
            dict_intermedate_common['3dmm_render_tri_id'] = list_gpmm_render_tri_ids
            dict_intermedate_common['3dmm_render_loss_heat'] = list_render_loss_error

            """
            ************************************    Identity     ********************************************
            """

            list_gpmm_render_id, list_image_id, gpmm_render_align, image_align = \
                decoder_similar(self.opt, self.defined_lm_facenet_align, list_gpmm_render, list_image, list_lm2d,
                                list_lm2d_gt)
            gpmm_id_loss, _ = compute_3dmm_id_cos_loss(list_image_id, list_gpmm_render_id)
            dict_intermedate_common['id_render'] = list_gpmm_render_id
            dict_intermedate_common['id_image'] = list_image_id
            dict_intermedate_common['id_render_align'] = gpmm_render_align
            dict_intermedate_common['id_image_align'] = image_align

            """
            ************************************    Depthmap    *********************************************
            """
            list_gpmm_depthmap, _ \
                = decoder_depth(self.opt, self.h_lrgp, list_gpmm_vertex, self.gpmm_frustrum, list_gpmm_ext,
                                list_gpmm_mv,
                                list_gpmm_eye)
            dict_intermedate_common['3dmm_depthmap'] = list_gpmm_depthmap
            # dict_intermedate_common['3dmm_depthmap_mask'] = list_gpmm_depthmap_mask
            # dict_intermedate_common['3dmm_depthmap_min'] = list_gpmm_depthmap_min
            # dict_intermedate_common['3dmm_depthmap_max'] = list_gpmm_depthmap_max

            gpmm_regular_shape_loss = gpmm_regular_shape_loss / (self.opt.num_source + 1)
            gpmm_regular_color_loss = gpmm_regular_color_loss / (self.opt.num_source + 1)
            gpmm_lm_loss = gpmm_lm_loss / (self.opt.num_source + 1)
            gpmm_pixel_loss = gpmm_pixel_loss / (self.opt.num_source + 1)
            gpmm_id_loss = gpmm_id_loss / (self.opt.num_source + 1)
            dict_loss_common['reg_shape_loss'] = gpmm_regular_shape_loss
            dict_loss_common['reg_color_loss'] = gpmm_regular_color_loss
            dict_loss_common['lm2d_loss'] = gpmm_lm_loss
            dict_loss_common['render_loss'] = gpmm_pixel_loss
            dict_loss_common['id_loss'] = gpmm_id_loss

            loss_common = tf.constant(0.0)

            loss_common += self.opt.gpmm_regular_shape_loss_weight * gpmm_regular_shape_loss
            loss_common += self.opt.gpmm_regular_color_loss_weight * gpmm_regular_color_loss

            if self.opt.gpmm_lm_loss_weight > 0:
                loss_common += self.opt.gpmm_lm_loss_weight * gpmm_lm_loss
            if self.opt.gpmm_pixel_loss_weight > 0:
                loss_common += self.opt.gpmm_pixel_loss_weight * gpmm_pixel_loss
            if self.opt.gpmm_id_loss_weight:
                loss_common += self.opt.gpmm_id_loss_weight * gpmm_id_loss

            dict_loss_common['loss_common'] = loss_common

        return dict_loss_common, dict_intermedate_common


    def build_decoderMGC(self, flag_sgl_mul, list_image, list_lm2d_gt, dict_inter_common):
        """
        :param list_image:
        :param list_lm2d_gt:
        :param dict_intermedate_common:
        :return:
        """

        # input
        #relative pose from target to source
        list_rel_poses = decoder_warppose(self.opt, dict_inter_common['pred_6dof_pose'])

        #
        list_covisible_map = decoder_covisible_map(self.opt, self.h_lrgp, self.gpmm_frustrum, dict_inter_common)

        # inter
        list_tarTile_image = []
        list_tarTile_depth = []

        list_viewSyn_image = []
        list_viewSyn_depth = []
        list_viewSyn_mask = []
        list_viewSyn_image_masked = []
        list_render_mask = []
        list_viewSyn_ssim_mask = []

        # output
        dict_loss_common = dict()
        dict_intermedate_common = dict()

        # Start loop all source view
        curr_tar_image = list_image[0]
        curr_tar_depths = dict_inter_common['3dmm_depthmap'][0]
        for i in range(self.opt.num_source):
            list_tarTile_image.append(curr_tar_image)
            list_tarTile_depth.append(curr_tar_depths)
            # Inverse warp the source image to the target image frame
            with tf.name_scope("warp"):
                curr_src_image = list_image[1 + i]  # careful of [tar, src_all]
                curr_source_depth = dict_inter_common['3dmm_depthmap'][1 + i]
                warp_pose = list_rel_poses[i]
                # view synthetic
                # curr_src_image = tf.Print(curr_src_image, [tf.reduce_mean(curr_src_image)], message='curr_src_image')
                # curr_tar_depths = tf.Print(curr_tar_depths, [tf.reduce_mean(curr_tar_depths)], message='curr_tar_depths')
                # curr_source_depth = tf.Print(curr_source_depth, [tf.reduce_mean(curr_source_depth)], message='curr_source_depth')

                curr_viewSyn_image, curr_viewSyn_depth, viewSyn_mask = projective_inverse_warp(
                    curr_src_image, tf.squeeze(curr_tar_depths, axis=-1), curr_source_depth,
                    warp_pose, self.intrinsics_single[:, :, :], is_vec=True)
                #
                # curr_viewSyn_image = tf.Print(curr_viewSyn_image,
                #                            [tf.reduce_sum(curr_viewSyn_image), tf.reduce_sum(curr_tar_depths-852)], message='warp')

                #
                list_viewSyn_image.append(curr_viewSyn_image)
                list_viewSyn_depth.append(curr_viewSyn_depth)

                # covisible map
                view_syn_mask = viewSyn_mask * list_covisible_map[i]
                #view_syn_mask = tf.Print(view_syn_mask, [tf.reduce_sum(view_syn_mask)], message='view_syn_mask')

                # cut bg
                if 0:
                    depthRender_min = dict_inter_common['3dmm_depthmap_min'][1 + i]
                    depthRender_max = dict_inter_common['3dmm_depthmap_max'][1 + i]
                    l_one = tf.ones_like(curr_viewSyn_depth)
                    l_zero = tf.zeros_like(curr_viewSyn_depth)
                    depthValid_mask = tf.where(tf.greater(viewSyn_depth, depthRender_max), x=l_zero, y=l_one)
                    #view_syn_mask = view_syn_mask * depthValid_mask

                # mask dict_intermedate_common['3dmm_render_mask']
                list_viewSyn_mask.append(view_syn_mask)
                list_render_mask.append(dict_inter_common['3dmm_render_mask'][0])

                curr_viewSyn_image_mask = curr_viewSyn_image * view_syn_mask  # (0, 1)
                list_viewSyn_image_masked.append(curr_viewSyn_image_mask)

                # 1.pixel
                ssim_mask = slim.avg_pool2d(view_syn_mask, 3, 1, 'VALID')  # TODO: Right SSIM
                list_viewSyn_ssim_mask.append(ssim_mask)

                # 2.depth

                # 3.epipolar
        dict_intermedate_common['list_viewSyn_image'] = list_viewSyn_image
        dict_intermedate_common['list_viewSyn_mask'] = list_viewSyn_mask
        dict_intermedate_common['list_viewSyn_image_masked'] = list_viewSyn_image_masked


        # 1. pixel loss
        # photo loss
        list_curr_viewSyn_pixel_error, list_curr_viewSyn_pixel_error_visual = \
            compute_pixel_eul_loss_list(list_viewSyn_image_masked, list_viewSyn_mask, list_render_mask, list_tarTile_image)

        flag_sgl_mul_curr = flag_sgl_mul
        flag_sgl_mul_curr = tf.clip_by_value(flag_sgl_mul_curr, 0.0, 1.0)
        pixel_loss = combine_flag_sgl_mul_loss(list_curr_viewSyn_pixel_error, flag_sgl_mul_curr)

        dict_intermedate_common['list_curr_viewSyn_pixel_error_visual'] = list_curr_viewSyn_pixel_error_visual

        # ssim loss
        list_curr_viewSyn_ssim_error = compute_ssim_loss_list(list_viewSyn_image, list_tarTile_image, list_viewSyn_ssim_mask)

        flag_sgl_mul_curr = flag_sgl_mul
        flag_sgl_mul_curr = tf.clip_by_value(flag_sgl_mul_curr, 0.0, 1.0)
        ssim_loss = combine_flag_sgl_mul_loss(list_curr_viewSyn_ssim_error, flag_sgl_mul_curr)

        # 2. depth loss
        """
        depth: range(0, NAN+)
        proj_mask: range(0, 1)
        """
        list_viewSyn_depth_alinged = decoder_align_depthMap(self.opt, list_tarTile_depth, list_viewSyn_depth, list_viewSyn_mask)

        list_curr_viewSyn_depth_error, list_curr_viewSyn_depth_visual = \
            compute_depthmap_l1_loss_list(list_viewSyn_depth_alinged, list_viewSyn_mask, list_tarTile_depth) # TODO: bug!!!

        flag_sgl_mul_curr = flag_sgl_mul
        flag_sgl_mul_curr = tf.clip_by_value(flag_sgl_mul_curr, 0.0, 1.0)
        depth_loss = combine_flag_sgl_mul_loss(list_curr_viewSyn_depth_error, flag_sgl_mul_curr, flag_batch_norm=False)

        dict_intermedate_common['list_viewSyn_depth_alinged'] = list_viewSyn_depth_alinged
        dict_intermedate_common['list_curr_viewSyn_depth_visual'] = list_curr_viewSyn_depth_visual

        # 3. Eipipolar loss (fundamental matrix)
        list_epiLoss_batch, list_reprojLoss_batch, mgc_epi_lines, mgc_epi_distances = compute_match_loss_list(
            list_lm2d_gt, dict_inter_common['3dmm_depthmap'][0], list_rel_poses, self.intrinsics_single
        )
        flag_sgl_mul_curr = flag_sgl_mul - 1
        flag_sgl_mul_curr = tf.clip_by_value(flag_sgl_mul_curr, 0.0, 1.0)
        epi_loss = combine_flag_sgl_mul_loss(list_epiLoss_batch, flag_sgl_mul_curr, flag_batch_norm=False)


        dict_intermedate_common['mgc_epi_lines'] = mgc_epi_lines
        dict_intermedate_common['mgc_epi_distances'] = mgc_epi_distances

        """
        Weighted Loss
        """
        loss_multiView = tf.constant(0.0)
        if self.opt.photom_weight > 0:
            loss_multiView += self.opt.photom_weight * pixel_loss
            dict_loss_common['pixel_loss'] = pixel_loss
        else:
            dict_loss_common['pixel_loss'] = tf.constant(0.0)

        if self.opt.ssim_weight > 0:
            loss_multiView += self.opt.ssim_weight * ssim_loss
            dict_loss_common['ssim_loss'] = ssim_loss
        else:
            dict_loss_common['ssim_loss'] = tf.constant(0.0)

        if self.opt.epipolar_weight > 0:
            loss_multiView += self.opt.epipolar_weight * epi_loss
            dict_loss_common['epi_loss'] = epi_loss
        else:
            dict_loss_common['epi_loss'] = tf.constant(0.0)

        if self.opt.depth_weight > 0:
            loss_multiView += self.opt.depth_weight * depth_loss
            dict_loss_common['depth_loss'] = depth_loss
        else:
            dict_loss_common['depth_loss'] = tf.constant(0.0)

        dict_loss_common['loss_mgc'] = loss_multiView

        # inter
        dict_intermedate_common['list_rel_poses'] = list_rel_poses

        return dict_loss_common, dict_intermedate_common


    def build_train_graph_train_op(self):
        opt = self.opt

        with tf.name_scope("train_op"):
            #print('Global variables:', tf.global_variables())
            train_vars = [var for var in tf.trainable_variables()]
            #print('Optimized variables:', train_vars)

            #print("Global variables number: %d" % (len(tf.global_variables())))
            print("Optimized variables number: %d" % (len(train_vars)))
            """
            Clean
            """
            train_vars = [(var) for var in train_vars if var.name.find('InceptionResnetV1') == -1]
            #print("Optimized variables number(After clean forward var): %d" % (len(train_vars)))

            update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
            with tf.control_dependencies(update_ops):
                optim0 = tf.train.AdamOptimizer(1.0 * opt.learning_rate, opt.beta1) # face+pose: no constrain
                self.train_op = []

                if len(train_vars) != 0:
                    #self.total_loss = tf.Print(self.total_loss, [self.total_loss], message='self.total_loss', summarize=4)
                    self.grads_and_vars = optim0.compute_gradients(self.total_loss, var_list=train_vars)

                    for grad, var in self.grads_and_vars:
                        if grad is None:
                            print("Optimized variables grad is None: ", var)
                    self.grads_and_vars = [(grad, var) for grad, var in self.grads_and_vars if grad is not None]
                    sum_grads = [tf.reduce_sum(grad) for grad, var in self.grads_and_vars]
                    self.total_grad = tf.reduce_sum(sum_grads)
                    self.train_op.append(optim0.apply_gradients(self.grads_and_vars))
                self.incr_global_step = tf.assign(self.global_step, self.global_step + 1)


    def collect_summaries(self):
        opt = self.opt

        # scalar
        tf.summary.scalar("total_loss", self.total_loss)
        tf.summary.scalar("common_loss", self.common_loss)
        tf.summary.scalar("ga_loss", self.ga_loss)
        # common
        if opt.gpmm_regular_shape_loss_weight > 0:
            tf.summary.scalar("gpmm_regular_shape_loss", self.gpmm_regular_shape_loss)
        if opt.gpmm_regular_color_loss_weight > 0:
            tf.summary.scalar("gpmm_regular_color_loss", self.gpmm_regular_color_loss)
        if opt.gpmm_lm_loss_weight > 0:
            tf.summary.scalar("gpmm_lm_loss", self.gpmm_lm_loss)
        if opt.gpmm_pixel_loss_weight > 0:
            tf.summary.scalar("gpmm_pixel_loss", self.gpmm_pixel_loss)
        if opt.gpmm_id_loss_weight:
            tf.summary.scalar("gpmm_id_loss", self.gpmm_id_loss)
        # multi-view
        if opt.ssim_weight > 0:
            tf.summary.scalar("ssim_loss", self.ssim_loss)
        if opt.photom_weight > 0:
            tf.summary.scalar("pixel_loss", self.pixel_loss)
        if opt.epipolar_weight > 0:
            tf.summary.scalar("epipolar_loss", self.epipolar_loss)
        if opt.depth_weight > 0:
            tf.summary.scalar("depth_loss", self.depth_loss)

        if 1:

            """
            image
            """
            self.list_img_tar = deprocess_image_series(self.list_tar_image)
            self.list_img_src = deprocess_image_series(self.list_src_image)

            """
            image + landmark
            """
            list_img_lmDraw_tar = draw_landmark_image(self.list_tar_image, self.lm2d_tar, opt.img_height, opt.img_width, color=1)
            list_img_lmDraw_tar = draw_landmark_image(list_img_lmDraw_tar, self.list_lm2d_gt_tar, opt.img_height, opt.img_width, color=2)

            list_img_lmDraw_src = draw_landmark_image(self.list_src_image, self.lm2d_src, opt.img_height, opt.img_width, color=1)
            list_img_lmDraw_src = draw_landmark_image(list_img_lmDraw_src, self.list_lm2d_gt_src, opt.img_height, opt.img_width, color=2)

            """
            render main pose
            """
            #tf.summary.image('gpmm_render_tar_main', deprocess_image(self.gpmm_render_tar_main[0]))

            """
            render derivatives
            """
            # shade
            list_shade_tar = deprocess_image_series(self.gpmm_render_shade_tar)
            list_shade_src = deprocess_image_series(self.gpmm_render_shade_src)

            # mask
            list_render_mask_tar = deprocess_gary_image_series(self.gpmm_render_mask_tar)
            list_render_mask_src = deprocess_gary_image_series(self.gpmm_render_mask_src)

            # skin
            list_skin_tar = deprocess_gary_image_series(self.list_tar_skin)
            list_skin_src = deprocess_gary_image_series(self.list_src_skin)

            """
            render
            """
            # tar render and ori image
            self.list_render_image_tar = deprocess_image_series(self.gpmm_render_tar)
            self.list_render_image_src = deprocess_image_series(self.gpmm_render_src)

            list_render_loss_tar = deprocess_image_series(self.list_render_loss_error_tar)
            list_render_loss_src = deprocess_image_series(self.list_render_loss_error_src)

            # Render
            # 1
            show_img_imgLM_tar = concate_image_series(self.list_img_tar, list_img_lmDraw_tar, axis=1)
            show_img_imgLM_src = concate_image_series(self.list_img_src, list_img_lmDraw_src, axis=1)

            # 2
            show_light_mask_tar = concate_image_series(list_skin_tar, list_shade_tar, axis=1)
            show_light_mask_src = concate_image_series(list_skin_src, list_shade_src, axis=1)

            # fusion
            render_12_tar = concate_image_series(show_img_imgLM_tar, self.list_render_image_tar, axis=1)
            render_23_tar = concate_image_series(show_light_mask_tar, list_render_loss_tar, axis=1)
            render_123_tar = concate_image_series(render_12_tar, render_23_tar, axis=2)

            render_12_src = concate_image_series(show_img_imgLM_src, self.list_render_image_src, axis=1)
            render_23_src = concate_image_series(show_light_mask_src, list_render_loss_src, axis=1)
            render_123_src = concate_image_series(render_12_src, render_23_src, axis=2)

            self.show_gpmm_render_all = concate_semi_image_series(render_123_tar, render_123_src)
            tf.summary.image('gpmm_render_all', self.show_gpmm_render_all)

            """
            epipolar:
            image + consistance
            """
            self.list_img_lmConsistDraw_tar = draw_landmark_image(self.list_tar_image, self.gpmm_consist_pixel_tar, opt, color=1)
            self.list_img_lmConsistDraw_src = draw_landmark_image(self.list_src_image, self.gpmm_consist_pixel_src, opt, color=1)

        # epipolar
        list_img_lmConsistDraw = draw_landmark_image(self.list_image, self.list_lm2d_gt, opt.img_height, opt.img_width, color=1)

        # photematric
        list_geo_proj_img_src = deprocess_image_series(self.dict_inter_mgc['list_viewSyn_image'])
        list_geo_proj_img_fore_src = deprocess_image_series(self.dict_inter_mgc['list_viewSyn_image_masked'])

        list_geo_proj_mask_src = deprocess_gary_image_series(self.dict_inter_mgc['list_viewSyn_mask'])
        list_geo_proj_img_error_src = deprocess_image_series(self.dict_inter_mgc['list_curr_viewSyn_pixel_error_visual'])

        # 1
        show_geo_proj_img_tar = concate_image_series(self.list_tar_image, self.list_tar_image, axis=1)
        show_geo_proj_img_tar = deprocess_image_series(show_geo_proj_img_tar)
        show_geo_proj_img_src = concate_image_series(list_geo_proj_img_src, list_geo_proj_img_fore_src, axis=1)

        show_geo_proj_img_me_tar = concate_image_series(self.list_tar_image, self.list_tar_image, axis=1)
        show_geo_proj_img_me_tar = deprocess_image_series(show_geo_proj_img_me_tar)
        show_geo_proj_img_me_src = concate_image_series(list_geo_proj_mask_src, list_geo_proj_img_error_src, axis=1)

        # 2
        show_geo_proj_tar = concate_image_series(show_geo_proj_img_tar, show_geo_proj_img_me_tar, axis=1)
        show_geo_proj_src = concate_image_series(show_geo_proj_img_src, show_geo_proj_img_me_src, axis=1)

        # 3
        show_geo_epi_tar = concate_image_series(list_img_lmConsistDraw[0:1], show_geo_proj_tar, axis=1)
        show_geo_epi_src = concate_image_series(list_img_lmConsistDraw[1:], show_geo_proj_src, axis=1)

        # fusion
        self.show_proj_all = insert_semi_image_series(show_geo_epi_tar, show_geo_epi_src)

        tf.summary.image("show_warp_proj_all", self.show_proj_all)


    def train_pre(self, opt):
        self.opt = opt

        """
        1.continue training
        2.pretrain model
        """
        restore_vars = tf.global_variables()
        self.restorer = tf.train.Saver(restore_vars, max_to_keep=None)
        self.saver = tf.train.Saver(tf.global_variables(), max_to_keep=None)

        # pretrain model
        if opt.ckpt_face_pretrain is not None:
            face_variables_to_restore = []


            face_variables_to_restore_all = slim.get_model_variables("resnet_v1_50")
            for var in face_variables_to_restore_all:
                if var.op.name.find('logits') != -1 or var.op.name.find('predictions') != -1:
                    pass
                elif var.op.name.find('block1_final') != -1:
                    pass
                else:
                    face_variables_to_restore.append(var)
            print("Face network pretrain, number: %d" % (len(face_variables_to_restore)))
            self.face_restorer = slim.assign_from_checkpoint_fn(opt.ckpt_face_pretrain, face_variables_to_restore, True)

        if opt.ckpt_face_id_pretrain is not None:
            # 1
            # face_variables_to_restore = slim.get_model_variables("InceptionResnetV1")
            # print("ID network pretrain, number: %d" % (len(face_variables_to_restore)))
            # self.face_id_restorer = slim.assign_from_checkpoint_fn(opt.ckpt_face_id_pretrain, face_variables_to_restore, True)

            # 2
            face_variables_to_restore = tf.model_variables("InceptionResnetV1")
            print("Identity variables number: %d" % (len(face_variables_to_restore)))
            #saver = tf_render.train.Saver([var for var in test_var])
            self.face_id_restorer = tf.train.Saver(face_variables_to_restore)


    def train(self, opt):
        # FLAGS
        assert opt.num_source == opt.seq_length - 1
        """
        Build Graph
        """
        # all the data directly stored in the self.Graph
        data_loader, batch_sample = self.build_train_graph_dataLoader()
        #with tf.device('/cpu:0'):
        self.build_train_graph()

        #
        self.collect_summaries()

        #
        with tf.name_scope("parameter_count"):
            parameter_count = \
                tf.reduce_sum([tf.reduce_prod(tf.shape(v)) for v in tf.trainable_variables()])

        # model
        self.train_pre(opt)

        """
        Start Training
        """
        # Initialize variables
        sv = tf.train.Supervisor(logdir=opt.checkpoint_dir,
                                 save_summaries_secs=0,
                                 saver=None)
        config = tf.ConfigProto(allow_soft_placement=True)
        config.gpu_options.allocator_type = 'BFC'  # A "Best-fit with coalescing" algorithm, simplified from a version of dlmalloc.
        #config.gpu_options.per_process_gpu_memory_fraction = 0.8
        config.gpu_options.allow_growth = True
        with sv.managed_session(config=config) as sess:
            print("Parameter count =", sess.run(parameter_count))

            """
            Functional Define
            """
            # continue train
            if opt.continue_train:
                if opt.init_ckpt_file is None:
                    checkpoint = tf.train.latest_checkpoint(opt.checkpoint_dir)
                else:
                    checkpoint = opt.init_ckpt_file
                print("Resume training from previous checkpoint: %s" % checkpoint)
                self.restorer.restore(sess, checkpoint)
                #
                dic_ckpt, name_ckpt = os.path.split(checkpoint)
                gs = name_ckpt.split('-')[1].split('.')[0]
                #
                # self.global_step = tf.Variable(0, name='global_step', trainable=False)
                step_start = int(gs) + 1
            else:
                # pretrain model
                if opt.ckpt_face_pretrain is not None:
                    self.face_restorer(sess)
                step_start = 0 + 1

            if opt.ckpt_face_id_pretrain is not None:
                self.face_id_restorer.restore(sess, opt.ckpt_face_id_pretrain)

            # init global
            #sess.run(tf_render.global_variables_initializer())

            """
            Loop Start
            """
            start_time = time.time()
            # """
            # Data init
            # """
            sess.graph.finalize()
            for step in range(step_start, opt.max_steps+1):
                """
                Data init
                """
                if step == 1 or (opt.dataset_name_list == 'train' and step % self.steps_per_epoch == 0) or (opt.continue_train and step == step_start):
                    global_all_file_list = data_loader.format_file_list(opt.dataset_dir, opt.dataset_name_list)
                    self.steps_per_epoch = data_loader.steps_per_epoch  # Step count
                    data_loader.init_data_pipeline(sess, batch_sample, global_all_file_list)
                    print("Update dataloader list: (step %d in all %d)" % (step, self.steps_per_epoch))

                """
                Define fetch
                """
                fetches = {
                    "total_loss": self.total_loss,
                    #"total_grad": self.total_grad,
                    "train": self.train_op,
                    "global_step": self.global_step,
                    "incr_global_step": self.incr_global_step
                }
                if step % opt.summary_freq == 0:
                    fetches["ga_loss"] = self.ga_loss
                    fetches["pixel_loss"] = self.pixel_loss
                    fetches["ssim_loss"] = self.ssim_loss
                    fetches["depth_loss"] = self.depth_loss
                    fetches["epipolar_loss"] = self.epipolar_loss

                    fetches["gpmm_pixel_loss"] = self.gpmm_pixel_loss
                    fetches["gpmm_lm_loss"] = self.gpmm_lm_loss
                    fetches["gpmm_id_loss"] = self.gpmm_id_loss
                    fetches["gpmm_reg_shape_loss"] = self.gpmm_regular_shape_loss
                    fetches["gpmm_reg_color_loss"] = self.gpmm_regular_color_loss

                    fetches["summary"] = sv.summary_op

                """
                *********************************************   Start Trainning   *********************************************
                """
                results = sess.run(fetches)
                gs = results["global_step"]

                if step % opt.summary_freq == 0:
                    sv.summary_writer.add_summary(results["summary"], gs)
                    train_epoch = math.ceil(gs / self.steps_per_epoch)
                    train_step = gs - (train_epoch - 1) * self.steps_per_epoch
                    print("Epoch %2d: %5d/%5d (time: %4.4f), Step %d:"
                          % (train_epoch, train_step, gs, (time.time() - start_time) / opt.summary_freq, step))

                    print("total: [%.4f]" % (results["total_loss"]))

                    print("ga/pixel/ssim/depth/epipolar loss: [%.4f/%.4f/%.4f/%.4f/%.4f]" % (
                        results["ga_loss"], results["pixel_loss"], results["ssim_loss"], results["depth_loss"], results["epipolar_loss"]))

                    print("(weight)ga/pixel/ssim/depth/epipolar loss: [%.4f/%.4f/%.4f/%.4f/%.4f]" % (
                        results["ga_loss"] * opt.MULTIVIEW_weight,
                        results["pixel_loss"] * (1-opt.ssim_weight),
                        results["ssim_loss"] * opt.ssim_weight,
                        results["depth_loss"] * opt.depth_weight,
                        results["epipolar_loss"] * opt.epipolar_weight)
                    )

                    # 3dmm loss
                    print("mm_pixel/mm_lm/mm_id/mm_reg_s/mm_reg_c loss: [%.4f/%.4f/%.4f/%.4f/%.4f]" % (
                        results["gpmm_pixel_loss"], results["gpmm_lm_loss"], results["gpmm_id_loss"],
                        results["gpmm_reg_shape_loss"], results["gpmm_reg_color_loss"]))

                    print("(weight)mm_pixel/mm_lm/mm_id/mm_reg_s/mm_reg_c loss: [%.4f/%.4f/%.4f/%.4f/%.4f]\n" % (
                        results["gpmm_pixel_loss"] * opt.gpmm_pixel_loss_weight,
                        results["gpmm_lm_loss"] * opt.gpmm_lm_loss_weight,
                        results["gpmm_id_loss"] * opt.gpmm_id_loss_weight,
                        results["gpmm_reg_shape_loss"] * opt.gpmm_regular_shape_loss_weight,
                        results["gpmm_reg_color_loss"] * opt.gpmm_regular_color_loss_weight))
                    start_time = time.time()
                """
                Save model
                """
                if gs % opt.save_freq == 0 and step >= opt.min_steps:
                    self.save(sess, opt.checkpoint_dir, gs)


    def save(self, sess, checkpoint_dir, step):
        model_name = 'model'
        print(" [*] Saving checkpoint step %d to %s..." % (step, checkpoint_dir))
        self.saver.save(sess, os.path.join(checkpoint_dir, model_name), global_step=step)

    """
    Test
    """
    def set_constant_test(self):
        # cam
        self.set_constant_node()

        # multi pose
        defined_pose_left = tf.constant([-0.000000, -0.392699, -3.141593, 37.504993, 9.1053238, 4994.3359], shape=[1, 6])
        self.defined_pose_left = tf.tile(defined_pose_left, multiples=[self.batch_size, 1])
        defined_pose_right = tf.constant([-0.000000, 0.392699, -3.141593, -37.341232, 9.1053238, 4994.3359], shape=[1, 6])
        self.defined_pose_right = tf.tile(defined_pose_right, multiples=[self.batch_size, 1])

        self.define_pose = tf.stack([self.defined_pose_left, self.defined_pose_main, self.defined_pose_right], axis=0)

        # print color
        gpmm_vertexColor_gary = tf.constant([0.7529, 0.7529, 0.7529], shape=[1, 1, 3])
        #gpmm_vertexColor_gary = tf.constant([0.5, 0.5, 0.5], shape=[1, 1, 3])
        gpmm_vertexColor_gary = tf.tile(gpmm_vertexColor_gary, [self.batch_size, self.h_lrgp.h_curr.point3d_mean_np.shape[0], 1])
        self.list_vertexColor_gary = [gpmm_vertexColor_gary]


    def build_test_graph(self, opt, img_height, img_width, batch_size=1):
        self.opt = opt

        self.img_height = img_height
        self.img_width = img_width

        self.batch_size = batch_size
        self.rank = self.opt.gpmm_rank

        # start
        input_uint8 = tf.placeholder(tf.uint8, [self.batch_size, self.img_height, self.img_width, 3], name='pl_input')
        input_float = preprocess_image(input_uint8)
        self.list_input_float = [input_float]

        # setting
        self.set_constant_test()

        # single view
        list_coeffALL = pred_encoder_coeff_light(self.opt, self.defined_pose_main, self.list_input_float, is_training=False)

        dict_loss_common, dict_intermedate_common = \
            self.build_decoderCommon(list_coeffALL, self.list_input_float)
        self.dict_inter_comm = dict_intermedate_common
        self.dict_loss_common = dict_loss_common

        # multi-level
        self.list_vertex, self.list_vertexNormal, self.list_vertexColor, self.list_vertexShade, self.list_vertexColorOri = \
            decoder_colorMesh_test(self.h_lrgp, self.dict_inter_comm, exp=True)

        # visual
        if opt.flag_visual:
            self.build_testVisual_graph()


    def build_testVisual_graph(self):
        opt = self.opt
        self.gpmm_render_mask = []

        self.overlay_255 = []
        self.overlayTex_255 = []
        self.overlayLight_255 = []

        self.overlayGeo_255 = []
        self.overlayMain_255 = []
        self.overlayTexMain_255 = []
        self.overlayLightMain_255 = []

        self.overlayGeoMain_255 = []
        self.apper_mulPose_255 = []

        for v in range(len(self.list_vertex)):
            """
            0. single visual: overlay(color + texture + geometry + illumination)
            """
            color_overlay_single = [self.list_vertexColor[0],
                                    self.list_vertexColorOri[0],
                                    self.list_vertexShade[0]]
            overlay_single = []
            for i in range(len(color_overlay_single)):
                # render
                texture_color = color_overlay_single[i]

                gpmm_render, gpmm_render_mask_v, _ = decoder_renderColorMesh(
                    opt, self.h_lrgp, self.list_vertex[0], self.list_vertexNormal[0], texture_color,
                    self.gpmm_frustrum, self.dict_inter_comm['pred_cam_mv'][v], self.dict_inter_comm['pred_cam_eye'][v], fore=opt.flag_fore, tone=False
                )

                gpmm_render = gpmm_face_replace(self.list_input_float[v], gpmm_render, gpmm_render_mask_v)
                gpmm_render = tf.clip_by_value(gpmm_render, 0.0, 1.0)
                #gpmm_render = tf.Print(gpmm_render, [gpmm_render], message='gpmm_render')
                #
                gpmm_render_visual = tf.image.convert_image_dtype(gpmm_render[0], dtype=tf.uint8)
                overlay_single.append(gpmm_render_visual)
                #
            self.gpmm_render_mask.append(gpmm_render_mask_v[0])

            self.overlay_255.append(overlay_single[0])
            self.overlayTex_255.append(overlay_single[1])
            self.overlayLight_255.append(overlay_single[2])


            # single visual geo
            overlayGeo, _, _ = decoder_renderColorMesh_gary(
                opt, self.h_lrgp, self.list_vertex[0], self.list_vertexNormal[0], self.list_vertexColor_gary,
                self.gpmm_frustrum, self.dict_inter_comm['pred_cam_mv'][v], self.dict_inter_comm['pred_cam_eye'][v], fore=opt.flag_fore, tone=False, background=-1
            )
            overlayGeo = gpmm_face_replace(self.list_input_float[v], overlayGeo, self.gpmm_render_mask[v])
            overlayGeo = tf.clip_by_value(overlayGeo, 0.0, 1.0)
            overlayGeo_255 = tf.image.convert_image_dtype(overlayGeo[0], dtype=tf.uint8)
            self.overlayGeo_255.append(overlayGeo_255)

            """
            1. single visual: main(color + texture + geometry + illumination)
            """
            if v == 0:
                gpmm_tar_extMain, gpmm_tar_projMain, gpmm_tar_mvMain, gpmm_tar_eyeMain = \
                    build_train_graph_3dmm_camera(self.intrinsics_single, self.define_pose[1])

                overlay_single = []
                for i in range(len(color_overlay_single)):
                    # render
                    texture_color = color_overlay_single[i]

                    gpmm_render, gpmm_render_mask_v, _ = decoder_renderColorMesh(
                        opt, self.h_lrgp, self.list_vertex[0], self.list_vertexNormal[0], texture_color,
                        self.gpmm_frustrum, gpmm_tar_mvMain, gpmm_tar_eyeMain, fore=opt.flag_fore, tone=False
                    )

                    gpmm_render = tf.clip_by_value(gpmm_render, 0.0, 1.0)

                    gpmm_render_visual = tf.image.convert_image_dtype(gpmm_render[0], dtype=tf.uint8)
                    overlay_single.append(gpmm_render_visual)

                self.overlayMain_255.append(overlay_single[0])
                self.overlayTexMain_255.append(overlay_single[1])
                self.overlayLightMain_255.append(overlay_single[2])


                #
                overlayGeo, _, _ = decoder_renderColorMesh_gary(
                    opt, self.h_lrgp, self.list_vertex[0], self.list_vertexNormal[0], self.list_vertexColor_gary,
                    self.gpmm_frustrum, gpmm_tar_mvMain, gpmm_tar_eyeMain, fore=opt.flag_fore, tone=False, background=-1
                )
                #overlayGeo = gpmm_face_replace(self.input_float, overlayGeo, self.gpmm_render_mask)
                overlayGeo = tf.clip_by_value(overlayGeo, 0.0, 1.0)
                overlayGeoMain_255 = tf.image.convert_image_dtype(overlayGeo[0], dtype=tf.uint8)
                self.overlayGeoMain_255.append(overlayGeoMain_255)

                """
                2. multi-poses visual: 3 random pose
                """
                for i in range(self.define_pose.shape[0]):
                    pose = self.define_pose[i]
                    #pose = tf.tile(pose, multiples=[self.batch_size, 1])

                    gpmm_tar_ext, gpmm_tar_proj, gpmm_tar_mv, gpmm_tar_eye = \
                        build_train_graph_3dmm_camera(self.intrinsics_single, pose)

                    # render
                    gpmm_render, gpmm_render_mask, _ = decoder_renderColorMesh(
                        opt, self.h_lrgp, self.list_vertex[0], self.list_vertexNormal[0], self.list_vertexColor[0],
                        self.gpmm_frustrum, gpmm_tar_mv, gpmm_tar_eye, fore=opt.flag_fore, tone=False
                    )
                    gpmm_render = tf.clip_by_value(gpmm_render, 0.0, 1.0)

                    if i == 0:
                        apper_mulPose_255 = tf.image.convert_image_dtype(gpmm_render[0], dtype=tf.uint8)  # bs, y, x
                    else:
                        apper_mulPose_255 = tf.concat([apper_mulPose_255, tf.image.convert_image_dtype(gpmm_render[0], dtype=tf.uint8)], axis=2)  # bs, y, x
                self.apper_mulPose_255.append(apper_mulPose_255)


    def inference(self, sess, inputs):
        fetches = {}

        # Eval
        # 0. vertex
        fetches['vertex_shape'] = self.list_vertex

        # 1. color
        fetches['vertex_color'] = self.list_vertexColor
        fetches['vertex_color_ori'] = self.list_vertexColorOri

        # Visual
        if self.opt.flag_visual:
            fetches['gpmm_render_mask'] = self.gpmm_render_mask

            fetches['overlay_255'] = self.overlay_255
            fetches['overlayTex_255'] = self.overlayTex_255
            fetches['overlayLight_255'] = self.overlayLight_255
            fetches['overlayGeo_255'] = self.overlayGeo_255

            fetches['overlayMain_255'] = self.overlayMain_255
            fetches['overlayTexMain_255'] = self.overlayTexMain_255
            fetches['overlayLightMain_255'] = self.overlayLightMain_255
            fetches['overlayGeoMain_255'] = self.overlayGeoMain_255

            fetches['apper_mulPose_255'] = self.apper_mulPose_255

        # lm2d, pose
        fetches['lm2d'] = self.dict_inter_comm['pred_lm2d']
        fetches['gpmm_pose'] = self.dict_inter_comm['pred_6dof_pose']
        fetches['gpmm_intrinsic'] = self.intrinsics_single

        #
        results = sess.run(fetches, feed_dict={'pl_input:0':inputs})

        return results
