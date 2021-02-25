#!/usr/bin/env python
# encoding: utf-8
'''
@author: Jiaxiang Shang
@license: (C) Copyright 2013-2017, Node Supply Chain Manager Corporation Limited.
@contact: jiaxiang.shang@gmail.com
@time: 3/25/20 12:47 PM
@desc:
'''
import tensorflow as tf

#
def unpack_image_sequence(image_seq, img_height, img_width, num_source):
    if len(image_seq.shape) == 2:
        image_seq = tf.expand_dims(image_seq, -1)
    channel = image_seq.shape[2]

    # Assuming the center image is the target frame
    tgt_start_idx = int(img_width * (num_source // 2))
    tgt_image = tf.slice(image_seq,
                         [0, tgt_start_idx, 0],
                         [-1, img_width, -1])
    # Source frames before the target frame
    src_image_1 = tf.slice(image_seq,
                           [0, 0, 0],
                           [-1, int(img_width * (num_source // 2)), -1])
    # Source frames after the target frame
    src_image_2 = tf.slice(image_seq,
                           [0, int(tgt_start_idx + img_width), 0],
                           [-1, int(img_width * (num_source // 2)), -1])
    src_image_seq = tf.concat([src_image_1, src_image_2], axis=1)
    # Stack source frames along the color channels (i.e. [H, W, N*3])
    src_image_stack = tf.concat([tf.slice(src_image_seq,
                                          [0, i * img_width, 0],
                                          [-1, img_width, -1])
                                 for i in range(num_source)], axis=2)
    src_image_stack.set_shape([img_height, img_width, num_source * channel])
    tgt_image.set_shape([img_height, img_width, channel])
    return tgt_image, src_image_stack

def data_augmentation_mul(im, intrinsics, out_h, out_w, matches=None):
    out_h = tf.cast(out_h, dtype=tf.int32)
    out_w = tf.cast(out_w, dtype=tf.int32)

    # Random scaling
    def random_scaling(im, intrinsics, matches):
        # print(tf_render.unstack(tf_render.shape(im)))
        # print(im.get_shape().as_list())
        _, in_h, in_w, _ = tf.unstack(tf.shape(im))
        in_h = tf.cast(in_h, dtype=tf.float32)
        in_w = tf.cast(in_w, dtype=tf.float32)
        scaling = tf.random_uniform([2], 1.0, 1.2)
        x_scaling = scaling[0]
        y_scaling = scaling[0]

        out_h = tf.cast(in_h * y_scaling, dtype=tf.int32)
        out_w = tf.cast(in_w * x_scaling, dtype=tf.int32)

        im = tf.image.resize_area(im, [out_h, out_w])

        list_intrinsics = []
        for i in range(intrinsics.shape[1]): # bs, num_src+1, 3, 3
            fx = intrinsics[:, i, 0, 0] * x_scaling
            fy = intrinsics[:, i, 1, 1] * y_scaling
            cx = intrinsics[:, i, 0, 2] * x_scaling
            cy = intrinsics[:, i, 1, 2] * y_scaling
            intrinsics_new = make_intrinsics_matrix(fx, fy, cx, cy)
            list_intrinsics.append(intrinsics_new)
        intrinsics = tf.stack(list_intrinsics, axis=1)

        if matches is None:
            return im, intrinsics, None
        else:
            x = matches[:, :, :, 0] * x_scaling
            y = matches[:, :, :, 1] * y_scaling
            matches = tf.stack([x, y], axis=3)  # bs, tar, num, axis
            return im, intrinsics, matches

    # Random cropping
    def random_cropping(im, intrinsics, out_h, out_w, matches):
        # batch_size, in_h, in_w, _ = im.get_shape().as_list()
        batch_size, in_h, in_w, _ = tf.unstack(tf.shape(im))
        offset_y = tf.random_uniform([1], 0, in_h - out_h + 1, dtype=tf.int32)[0]
        offset_x = offset_y
        im = tf.image.crop_to_bounding_box(
            im, offset_y, offset_x, out_h, out_w)

        list_intrinsics = []
        for i in range(intrinsics.shape[1]): # bs, num_src+1, 3, 3
            fx = intrinsics[:, i, 0, 0]
            fy = intrinsics[:, i, 1, 1]
            cx = intrinsics[:, i, 0, 2] - tf.cast(offset_x, dtype=tf.float32)
            cy = intrinsics[:, i, 1, 2] - tf.cast(offset_y, dtype=tf.float32)
            intrinsics_new = make_intrinsics_matrix(fx, fy, cx, cy)
            list_intrinsics.append(intrinsics_new)
        intrinsics = tf.stack(list_intrinsics, axis=1)

        if matches is None:
            return im, intrinsics, None
        else:
            x = matches[:, :, :, 0] - tf.cast(offset_x, dtype=tf.float32)
            y = matches[:, :, :, 1] - tf.cast(offset_y, dtype=tf.float32)
            matches = tf.stack([x, y], axis=3)  # bs, tar, num, axis
            return im, intrinsics, matches

    batch_size, in_h, in_w, _ = tf.unstack(tf.shape(im))
    im, intrinsics, matches = random_scaling(im, intrinsics, matches)
    im, intrinsics, matches = random_cropping(im, intrinsics, out_h, out_w, matches)
    # im, intrinsics, matches = random_scaling(im, intrinsics, matches, in_h, in_w)
    im = tf.cast(im, dtype=tf.uint8)

    if matches is None:
        return im, intrinsics, None
    else:
        return im, intrinsics, matches

#
def unpack_image_batch_list(image_seq, img_height, img_width, num_source):
    tar_list = []
    src_list = []
    for i in range(image_seq.shape[0]):
        tgt_image, src_image_stack = unpack_image_sequence(image_seq[i], img_height, img_width, num_source)
        tar_list.append(tgt_image)
        src_list.append(src_image_stack)
    tgt_image_b = tf.stack(tar_list)
    src_image_stack_b = tf.stack(src_list)

    list_tar_image = [tgt_image_b]
    list_src_image = [src_image_stack_b[:, :, :, i * 3:(i + 1) * 3] for i in range(num_source)]
    list_image = list_tar_image + list_src_image

    return list_image

# np
def unpack_image_np(image_seq, img_height, img_width, num_source):

    tgt_start_idx = int(img_width * (num_source // 2))

    tgt_image = image_seq[:, tgt_start_idx:tgt_start_idx+img_width, :]
    src_image_1 = image_seq[:, 0:int(img_width * (num_source // 2)), :]
    src_image_2 = image_seq[:, tgt_start_idx+img_width:tgt_start_idx+img_width+int(img_width * (num_source // 2)), :]

    return src_image_1, tgt_image, src_image_2, [tgt_image, src_image_1, src_image_2]