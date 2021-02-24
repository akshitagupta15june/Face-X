from __future__ import division

import os
# python lib
import random
import sys

import numpy as np
# tf_render
import tensorflow as tf

#self
_curr_path = os.path.abspath(__file__) # /home/..../face
_cur_dir = os.path.dirname(_curr_path) # ./
_tf_dir = os.path.dirname(_cur_dir) # ./
_deep_learning_dir = os.path.dirname(_tf_dir) # ../
print(_deep_learning_dir)
sys.path.append(_deep_learning_dir) # /home/..../pytorch3d

from src_tfGraph.build_graph import MGC_TRAIN

#
flags = tf.app.flags

# data
flags.DEFINE_string("dataset_dir", "", "Dataset directory")
flags.DEFINE_string("dataset_loader", "", "data_loader_semi_unsupervised_skin")
flags.DEFINE_string("dataset_name_list", "train", "train train_debug")
flags.DEFINE_boolean("flag_shuffle", True, "source images (seq_length-1)")
flags.DEFINE_string("checkpoint_dir", "../default_checkpoints/", "Directory name to save the checkpoints")

# continue training
flags.DEFINE_boolean("continue_train", False, "Continue training from previous checkpoint")
flags.DEFINE_string("init_ckpt_file", None, "Specific checkpoint file to initialize from")

flags.DEFINE_boolean("flag_data_aug", False, "The size of of a sample batch")
flags.DEFINE_integer("batch_size", 1, "The size of of a sample batch")
flags.DEFINE_integer("img_height", 224, "Image height")
flags.DEFINE_integer("img_width", 224, "Image width")
flags.DEFINE_integer("seq_length", 3, "Sequence length for each example")
flags.DEFINE_integer("num_source", 2, "source images (seq_length-1)")

# save
flags.DEFINE_integer("min_steps", 200000, "Maximum number of training iterations")
flags.DEFINE_integer("max_steps", 200000, "Maximum number of training iterations")
flags.DEFINE_integer("max_d", 64, "Maximum depth step when training.")
flags.DEFINE_integer("summary_freq", 1, "Logging every log_freq iterations")
flags.DEFINE_integer("save_freq", 50000, "Save the model every save_freq iterations (overwrites the previous latest model)")

# opt
flags.DEFINE_float("learning_rate", 0.0001, "Learning rate of for adam")
flags.DEFINE_float("beta1", 0.9, "Momentum term of adam or decay rate for RMSProp")

# loss
flags.DEFINE_float("MULTIVIEW_weight", 0.1, "Weight for smoothness")

flags.DEFINE_float("photom_weight", 0.15, "Weight for SSIM loss")
flags.DEFINE_float("ssim_weight", 0.85, "Weight for SSIM loss")
flags.DEFINE_float("depth_weight", 0.1, "Weight for depth loss")
flags.DEFINE_float("epipolar_weight", 0.0, "Weight for epipolar_weight loss")

flags.DEFINE_float("gpmm_lm_loss_weight", 0.0, "")
flags.DEFINE_float("lm_detail_weight", 1.0, "Depth minimum")

flags.DEFINE_float("gpmm_pixel_loss_weight", 0.0, "")
flags.DEFINE_float("gpmm_id_loss_weight", 0.0, "")
flags.DEFINE_float("gpmm_regular_shape_loss_weight", 1.0, "3DMM coeffient rank")
flags.DEFINE_float("gpmm_regular_color_loss_weight", 1.0, "3DMM coeffient rank")

# aug
flags.DEFINE_integer("match_num", 0, "Train with epipolar matches")

flags.DEFINE_boolean("is_read_pose", False, "Train with pre-computed pose")
flags.DEFINE_boolean("is_read_gpmm", False, "Train with pre-computed pose")
flags.DEFINE_boolean("disable_log", False, "Disable image log in tensorboard to accelerate training")

# gpmm
flags.DEFINE_string("ckpt_face_pretrain", None, "Dataset directory")
flags.DEFINE_string("ckpt_face_id_pretrain", None, "Dataset directory")
flags.DEFINE_string("path_gpmm", "/home/jshang/SHANG_Data/ThirdLib/BFM2009/bfm09_trim_exp_uv_presplit.h5", "Dataset directory")

flags.DEFINE_integer("flag_fore", 1, "")
flags.DEFINE_integer("gpmm_rank", 80, "3DMM coeffient rank")
flags.DEFINE_integer("gpmm_exp_rank", 64, "3DMM coeffient rank")

#
flags.DEFINE_float("depth_min", 0.0, "Depth minimum")
flags.DEFINE_float("depth_max", 7500.0, "Depth minimum")

FLAGS = flags.FLAGS

"""
CUDA_VISIBLE_DEVICES=${gpu} python train_unsupervise.py --dataset_name_list train \
--dataset_loader data_loader_semi_unsupervised_skin \
--dataset_dir ./data/eccv2020_MGCNet_data \
--checkpoint_dir ./logs_release_2020.07.23/0_local \
--learning_rate 0.0001 --MULTIVIEW_weight 1.0 \
--photom_weight 0.15 --ssim_weight 0.0 --epipolar_weight 0.00 --depth_weight 0.0001 \
--gpmm_lm_loss_weight 0.001 --gpmm_pixel_loss_weight 1.9 --gpmm_id_loss_weight 0.2 \
--gpmm_regular_shape_loss_weight 0.0001 --gpmm_regular_color_loss_weight 0.0003 \
--flag_fore 1 \
--batch_size 2 --img_height 224 --img_width 224 --num_scales 1 \
--min_steps 2000 --max_steps 20001 --save_freq 20000 --summary_freq 100 \
--seq_length 3 --num_source 2 --match_num 68 \
--net resnet --net_id facenet \
--ckpt_face_pretrain ./pretrain/resnet_v1_50_2016_08_28/resnet_v1_50.ckpt \
--ckpt_face_id_pretrain ./pretrain/facenet_vgg2/model-20180402-114759.ckpt-275 \
--path_gpmm /home/jshang/SHANG_Data/ThirdLib/BFM2009/bfm09_trim_exp_uv_presplit.h5 \
--lm_detail_weight 5.0
"""

def main(_):
    # static random and shuffle
    seed = 8964
    tf.set_random_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    # print and store all flags
    print('**************** Arguments ******************')
    for key in FLAGS.__flags.keys():
        print('{}: {}'.format(key, getattr(FLAGS, key)))
    print('**************** Arguments ******************')

    if not os.path.exists(FLAGS.checkpoint_dir):
        os.makedirs(FLAGS.checkpoint_dir)
    path_arg_log = os.path.join(FLAGS.checkpoint_dir, "flag.txt")
    with open(path_arg_log, 'w') as f:
        for key in FLAGS.__flags.keys():
            v = '{} : {}'.format(key, getattr(FLAGS, key))
            f.write(v)
            f.write('\n')

    #
    system = MGC_TRAIN(FLAGS)
    system.train(FLAGS)

if __name__ == '__main__':
    tf.app.run()
