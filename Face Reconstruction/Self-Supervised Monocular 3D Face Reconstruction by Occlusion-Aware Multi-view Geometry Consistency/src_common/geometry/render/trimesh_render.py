
# system
from __future__ import print_function

# python lib
import trimesh
from copy import deepcopy
import numpy as np

# tf_render
import tensorflow as tf

# self
from tfmatchd.face.common.format_helper import batch_size_extract

from tfmatchd.face.geometry.camera.rotation import RotationMtxBatch, ExtMtxBatch

#
from tf_mesh_renderer.mesh_renderer.mesh_renderer import tone_mapper
from tfmatchd.face.geometry.render.api_tf_mesh_render import mesh_renderer_camera, mesh_depthmap_camera

def Render_Trimesh_feed(vertices, triangles, normals, diffuse_colors,
                        mtx_cam, mtx_perspect_frustrum, cam_position, image_width, image_height):
    """
    :param trimesh:
    :param mtx_cam:
    :param mtx_perspect_frustrum:
    :param cam_position:
    :param image_width:
    :param image_height:
    :return:
        A 4-D float32 tensor of shape [batch_size, image_height, image_width, 4]
        containing the lit RGBA color values for each image at each pixel. RGB
        colors are the intensity values before tonemapping and can be in the range
        [0, infinity]. Clipping to the range [0,1] with tf_render.clip_by_value is likely
        reasonable for both viewing and training most scenes. More complex scenes
        with multiple lights should tone map color values for display only. One
        simple tonemapping approach is to rescale color values as x/(1+x); gamma
        compression is another common techinque. Alpha values are zero for
        background pixels and near one for mesh pixels.
    """
    batch_size = batch_size_extract(vertices, normals, diffuse_colors,
                                    mtx_cam, mtx_perspect_frustrum, cam_position)

    light_positions = tf.constant([[0.0, 0.0, 1000.0, -1000.0, 0.0, 1000.0, 1000.0, 0.0, 1000.0]], shape=[1, 3, 3])
    light_positions = tf.tile(light_positions, [batch_size, 1, 1])
    light_intensities = tf.constant([[0.50, 0.50, 0.50]], shape=[1, 3, 3])
    light_intensities = tf.tile(light_intensities, [batch_size, 1, 1])

    # light_positions = tf.constant([[0.0, 0.0, 2000.0]], shape=[1, 1, 3])
    # light_intensities = tf.constant([[0.5, 0.5, 0.5]], shape=[1, 1, 3])
    #light_intensities = tf.constant([[1.0, 1.0, 1.0]], shape=[1, 1, 3])
    #ambient_color = tf.constant([[1.0, 1.0, 1.0]])
    # Batch

    #print(batch_size, image_width, image_height)

    if vertices.shape[0] != batch_size:
        vertices = tf.tile(vertices, [batch_size, 1, 1])
        normals = tf.tile(normals, [batch_size, 1, 1])
        diffuse_colors = tf.tile(diffuse_colors, [batch_size, 1, 1])

    if mtx_perspect_frustrum.shape[0] != batch_size:
        mtx_perspect_frustrum = tf.tile(mtx_perspect_frustrum, [batch_size, 1, 1])

    # if ambient_color.shape[0] != batch_size:
    #     ambient_color = tf.tile(ambient_color, [batch_size, 1])

    renders, pixel_mask = mesh_renderer_camera(
        vertices, triangles, normals, diffuse_colors,
        mtx_cam, mtx_perspect_frustrum, cam_position,
        light_positions, light_intensities, image_width, image_height#, ambient_color=ambient_color
    )
    renders = tf.clip_by_value(renders, 0.0, 1.0)
    # tonemapped_renders = tf.concat(
    #     [
    #         tone_mapper(renders[:, :, :, 0:3], 0.7),
    #         renders[:, :, :, 3:4]
    #     ],
    #     axis=3)
    # return tonemapped_renders
    return renders

def RenderDepthmap_Trimesh_feed(vertices, triangles, mtx_ext, mtx_cam, mtx_perspect_frustrum,
                                image_width, image_height):
    """
    :param trimesh:
    :param mtx_cam:
    :param mtx_perspect_frustrum:
    :param cam_position:
    :param image_width:
    :param image_height:
    :return:
        A 4-D float32 tensor of shape [batch_size, image_height, image_width, 4]
        containing the lit RGBA color values for each image at each pixel. RGB
        colors are the intensity values before tonemapping and can be in the range
        [0, infinity]. Clipping to the range [0,1] with tf_render.clip_by_value is likely
        reasonable for both viewing and training most scenes. More complex scenes
        with multiple lights should tone map color values for display only. One
        simple tonemapping approach is to rescale color values as x/(1+x); gamma
        compression is another common techinque. Alpha values are zero for
        background pixels and near one for mesh pixels.
    """
    # Batch
    batch_size = batch_size_extract(vertices, mtx_cam, mtx_perspect_frustrum)
    #print(batch_size, image_width, image_height)

    if vertices.shape[0] != batch_size:
        vertices = tf.tile(vertices, [batch_size, 1, 1])

    if mtx_perspect_frustrum.shape[0] != batch_size:
        mtx_perspect_frustrum = tf.tile(mtx_perspect_frustrum, [batch_size, 1, 1])

    renders, pixel_mask = mesh_depthmap_camera(
        vertices, triangles, mtx_ext, mtx_cam, mtx_perspect_frustrum, image_width, image_height
    )
    renders = tf.expand_dims(renders, -1) # * pixel_mask
    #renders = tf.clip_by_value(renders, 0.0, 100000.0)
    # tonemapped_renders = tf.concat(
    #     [
    #         tone_mapper(renders[:, :, :, 0:3], 0.7),
    #         renders[:, :, :, 3:4]
    #     ],
    #     axis=3)
    # return tonemapped_renders
    return renders, pixel_mask


def Render_Trimesh(trimesh, mtx_cam, mtx_perspect_frustrum, cam_position,
                   light_positions, light_intensities, image_width, image_height, ambient_color=None):
    """

    :param trimesh:
    :param mtx_cam:
    :param mtx_perspect_frustrum:
    :param cam_position:
    :param light_positions:
    :param light_intensities:
    :param image_width:
    :param image_height:
    :param ambient_color:
    :return:
        A 4-D float32 tensor of shape [batch_size, image_height, image_width, 4]
        containing the lit RGBA color values for each image at each pixel. RGB
        colors are the intensity values before tonemapping and can be in the range
        [0, infinity]. Clipping to the range [0,1] with tf_render.clip_by_value is likely
        reasonable for both viewing and training most scenes. More complex scenes
        with multiple lights should tone map color values for display only. One
        simple tonemapping approach is to rescale color values as x/(1+x); gamma
        compression is another common techinque. Alpha values are zero for
        background pixels and near one for mesh pixels.
    """
    vertices = tf.constant(np.array(trimesh.vertices), dtype=tf.float32)
    vertices = tf.reshape(vertices, [1, -1, 3])
    triangles = tf.constant(np.array(trimesh.faces), dtype=tf.int32)
    triangles = tf.reshape(triangles, [-1, 3])
    #normals = tf_render.nn.l2_normalize(vertices, dim=2)
    normals = tf.constant(np.array(trimesh.vertex_normals), dtype=tf.float32)
    normals = tf.reshape(normals, [1, -1, 3])
    diffuse_colors = tf.constant(np.array(trimesh.visual.vertex_colors[:, 0:3])/255.0, dtype=tf.float32)
    diffuse_colors = tf.reshape(diffuse_colors, [1, -1, 3])

    # Batch
    batch_size = batch_size_extract(vertices, normals, diffuse_colors,
                                    mtx_cam, mtx_perspect_frustrum, cam_position,
                                    light_positions, light_intensities)

    if vertices.shape[0] != batch_size:
        vertices = tf.tile(vertices, [batch_size, 1, 1])
        normals = tf.tile(normals, [batch_size, 1, 1])
        diffuse_colors = tf.tile(diffuse_colors, [batch_size, 1, 1])

    if mtx_perspect_frustrum.shape[0] != batch_size:
        mtx_perspect_frustrum = tf.tile(mtx_perspect_frustrum, [batch_size, 1, 1])

    if ambient_color.shape[0] != batch_size:
        ambient_color = tf.tile(ambient_color, [batch_size, 1])

    renders, pixel_mask = mesh_renderer_camera(
        vertices, triangles, normals, diffuse_colors,
        mtx_cam, mtx_perspect_frustrum, cam_position,
        light_positions, light_intensities, image_width, image_height, ambient_color=ambient_color
    )
    tonemapped_renders = tf.concat(
        [
            tone_mapper(renders[:, :, :, 0:3], 0.7),
            renders[:, :, :, 3:4]
        ],
        axis=3)
    return tonemapped_renders


def RotationMtx_Trimesh(mesh_tri, idx_nose, camera_centre_distance):
    mesh_c = tf.constant(mesh_tri.vertices[idx_nose], dtype=tf.float32)  # mm

    cam_front_eye = tf.Variable(
        [[mesh_c[0], mesh_c[1], mesh_c[2] + camera_centre_distance]], dtype=tf.float32
    )
    cam_front_center = tf.Variable(
        [[mesh_c[0], mesh_c[1], mesh_c[2]]], dtype=tf.float32
    )
    cam_front_up = tf.Variable(
        [[0.0, 1.0, 0.0]], dtype=tf.float32
    )
    location = tf.stack([cam_front_eye, cam_front_center, cam_front_up], axis=1)

    h_ext = ExtMtxBatch.create_location_batch(location)

    mesh_c_batch = tf.expand_dims(mesh_c, 0)
    return h_ext, mesh_c_batch