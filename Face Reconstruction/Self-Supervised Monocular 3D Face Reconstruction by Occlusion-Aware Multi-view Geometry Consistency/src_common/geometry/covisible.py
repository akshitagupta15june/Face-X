#!/usr/bin/env python
# encoding: utf-8
'''
@author: Jiaxiang Shang
@license: (C) Copyright 2013-2017, Node Supply Chain Manager Corporation Limited.
@contact: jiaxiang.shang@gmail.com
@time: 3/16/20 1:28 PM
@desc:
'''
import tensorflow as tf

def mm_covisible_tri(h_lrgp, tri_ids_tar, tri_ids_src):
    batch_size = h_lrgp.batch_size

    tri_ids_tar = tf.reshape(tri_ids_tar, [-1])
    ver_ids_tar = tf.gather(h_lrgp.h_fore.mesh_tri, tri_ids_tar)  # vertex idx
    ver_ids_tar = tf.reshape(ver_ids_tar, [batch_size, -1])

    tri_ids_src = tf.reshape(tri_ids_src, [batch_size, -1])
    ver_ids_src = tf.gather(h_lrgp.h_fore.mesh_tri, tri_ids_src)
    ver_ids_src = tf.reshape(ver_ids_src, [batch_size, -1])

    ver_ids_consistency = tf.sets.set_intersection(ver_ids_tar, ver_ids_src, False)
    ver_ids_consistency = tf.sparse_tensor_to_dense(ver_ids_consistency, validate_indices=False)  # bs, h*w*3

    tri_consistency = []
    for j in range(batch_size):
        # find adjacent triangle for robust
        tri_ids_consistency_b = tf.gather(h_lrgp.h_fore.mesh_vertex_refer_face_pad, ver_ids_consistency[j])  # num, 8
        tri_ids_consistency_b = tf.reshape(tri_ids_consistency_b, [-1])
        tri_consistency_b = tf.gather(h_lrgp.h_fore.mesh_tri, tri_ids_consistency_b)  # vertex idx
        # [4w 3]
        # tri_consistency_b = tf.Print(tri_consistency_b, [tf.shape(tri_consistency_b)], message='tri_consistency')
        tri_consistency.append(tri_consistency_b)
    tri_consistency = tf.stack(tri_consistency, axis=0)

    return tri_consistency