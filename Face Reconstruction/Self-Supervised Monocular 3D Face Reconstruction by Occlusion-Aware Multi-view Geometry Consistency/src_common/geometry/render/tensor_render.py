
#
import tensorflow as tf

#
from .api_tf_mesh_render import mesh_depthmap_camera, mesh_renderer_camera, mesh_renderer_camera_light

def gpmm_render_image(opt, vertex, tri, vertex_normal, vertex_color, mtx_perspect_frustrum, mtx_model_view, cam_position):
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

    if len(tri.shape) == 2: # common render for bfm09
        render_image, render_image_mask, render_tri_ids = \
            mesh_renderer_camera_light(vertex, tri, vertex_normal, vertex_color, mtx_model_view,
                                       mtx_perspect_frustrum, cam_position, opt.img_width, opt.img_height)
        tonemapped_renders = tf.clip_by_value(render_image, 0.0, 100000.0)
    else: # convisiable mask render have diff tri for sample in batch
        list_tonemapped_renders = []
        list_render_image_mask = []
        list_render_tri_ids = []
        for i in range(tri.shape[0]):  # bs
            render_image, render_image_mask, render_tri_ids = \
                mesh_renderer_camera_light(
                    vertex[i:i + 1, :, :], tri[i], vertex_normal[i:i + 1, :, :], vertex_color[i:i + 1, :, :],
                    mtx_model_view[i:i + 1, :, :], mtx_perspect_frustrum[i:i + 1, :, :], cam_position[i:i + 1, :],
                    opt.img_width, opt.img_height)
            tonemapped_renders = tf.clip_by_value(render_image, 0.0, 100000.0)

            list_tonemapped_renders.append(tonemapped_renders)
            list_render_image_mask.append(render_image_mask)
            list_render_tri_ids.append(render_tri_ids)

        tonemapped_renders = tf.concat(list_tonemapped_renders, axis=0)
        render_image_mask = tf.concat(list_render_image_mask, axis=0)
        render_tri_ids = tf.concat(list_render_tri_ids, axis=0)

    return tonemapped_renders[:, :, :, 0:3], render_image_mask, render_tri_ids


def gpmm_render_image_garyLight(opt, vertex, tri, vertex_normal, vertex_color, mtx_perspect_frustrum, mtx_model_view, cam_position, background=10.999):
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
    light_positions = tf.constant([[0.0, 0.0, 1000.0]], shape=[1, 1, 3])
    light_positions = tf.tile(light_positions, [opt.batch_size, 1, 1])
    light_intensities = tf.constant([[1.0, 1.0, 1.0]], shape=[1, 1, 3])
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
    return tonemapped_renders[:, :, :, 0:3], render_image_mask, render_image_mask


def gpmm_generate_depthmap(opt, mesh, tri, mtx_perspect_frustrum, mtx_ext, mtx_model_view, cam_position, background=99999999):
    depthmap, depthmap_mask, depth_min, depth_max = mesh_depthmap_camera(mesh, tri, mtx_ext, mtx_model_view, mtx_perspect_frustrum,
                                                   opt.img_width, opt.img_height, background=background)

    #depthmap = depthmap * tf.squeeze(depthmap_mask, axis=-1)
    #depthmap = tf.clip_by_value(depthmap, opt.depth_min, opt.depth_max)
    depthmap = tf.expand_dims(depthmap, axis=-1)

    return depthmap, depthmap_mask, depth_min, depth_max
