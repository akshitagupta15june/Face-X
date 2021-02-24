#!/usr/bin/env python
# encoding: utf-8
'''
@author: Jiaxiang Shang
@license: (C) Copyright 2013-2017, Node Supply Chain Manager Corporation Limited.
@contact: jiaxiang.shang@gmail.com
@time: 3/27/20 3:19 PM
@desc:
'''
import tensorflow as tf

def parse_coeff_list(opt, coeff_all_list, defined_pose_main):
    list_coeff_shape = []
    list_coeff_color = []
    list_coeff_exp = []
    list_coeff_pose = []
    list_coeff_sh = []

    for i in range(len(coeff_all_list)):
        coeff_all = coeff_all_list[i]
        pred_3dmm_shape, pred_3dmm_color, pred_3dmm_exp, pred_pose_render, pred_sh = parse_coeff(opt, coeff_all, defined_pose_main)
        list_coeff_shape.append(pred_3dmm_shape)
        list_coeff_color.append(pred_3dmm_color)
        list_coeff_exp.append(pred_3dmm_exp)

        #pred_pose_render = tf.Print(pred_pose_render, [pred_pose_render], summarize=16, message='pred_pose_render')
        list_coeff_pose.append(pred_pose_render)
        list_coeff_sh.append(pred_sh)

    return list_coeff_shape, list_coeff_color, list_coeff_exp, list_coeff_pose, list_coeff_sh

def parse_coeff(opt, coeff_all, defined_pose_main):
    #
    pred_3dmm_shape = coeff_all[:,  0                                          : opt.gpmm_rank]
    pred_3dmm_color = coeff_all[:,  opt.gpmm_rank                              : 2 * opt.gpmm_rank]
    pred_3dmm_exp = coeff_all[:,    2 * opt.gpmm_rank                          : 2 * opt.gpmm_rank + opt.gpmm_exp_rank]

    #
    pred_pose_render = coeff_all[:, 2 * opt.gpmm_rank + opt.gpmm_exp_rank      : 2 * opt.gpmm_rank + opt.gpmm_exp_rank + 6]
    pred_pose_render = pred_pose_render + defined_pose_main

    #
    pred_sh = coeff_all[:,          2 * opt.gpmm_rank + opt.gpmm_exp_rank + 6  : 2 * opt.gpmm_rank + opt.gpmm_exp_rank + 6 + 27]

    return pred_3dmm_shape, pred_3dmm_color, pred_3dmm_exp, pred_pose_render, pred_sh