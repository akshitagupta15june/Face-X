
# system
from __future__ import print_function

# python lib
import math
from copy import deepcopy
import numpy as np

# tf_render
import tensorflow as tf

# self
from thirdParty.tf_mesh_renderer.mesh_renderer.mesh_renderer import phong_shader, tone_mapper
from thirdParty.tf_mesh_renderer.mesh_renderer.rasterize_triangles import rasterize_triangles

# perspective
def mesh_renderer_camera_light(vertices, triangles, normals, diffuse_colors,
                  mtx_camera, mtx_perspective_frustrum, camera_position,
                  image_width, image_height):
  """Renders an input scene using phong shading, and returns an output image.

  Args:
    vertices: 3-D float32 tensor with shape [batch_size, vertex_count, 3]. Each
        triplet is an xyz position in world space.
    triangles: 2-D int32 tensor with shape [triangle_count, 3]. Each triplet
        should contain vertex indices describing a triangle such that the
        triangle's normal points toward the viewer if the forward order of the
        triplet defines a clockwise winding of the vertices. Gradients with
        respect to this tensor are not available.
    normals: 3-D float32 tensor with shape [batch_size, vertex_count, 3]. Each
        triplet is the xyz vertex normal for its corresponding vertex. Each
        vector is assumed to be already normalized.
    diffuse_colors: 3-D float32 tensor with shape [batch_size,
        vertex_count, 3]. The RGB diffuse reflection in the range [0,1] for
        each vertex.

    mtx_camera: 3-D tensor with shape [batch_size, 4, 4] or 2-D tensor with
        shape [4, 4] specifying the camera model view matrix
    mtx_perspective_frustrum: 3-D tensor with shape [batch_size, 4, 4] or 2-D tensor with
        shape [4, 4] specifying the perspective and frustrum matrix
    camera_position: 2-D tensor with shape [batch_size, 3] or 1-D tensor with
        shape [3] specifying the XYZ world space camera position.

    light_intensities: a 3-D tensor with shape [batch_size, light_count, 3]. The
        RGB intensity values for each light. Intensities may be above one.
    image_width: int specifying desired output image width in pixels.
    image_height: int specifying desired output image height in pixels.

  Returns:
    A 4-D float32 tensor of shape [batch_size, image_height, image_width, 4]
    containing the lit RGBA color values for each image at each pixel. RGB
    colors are the intensity values before tonemapping and can be in the range
    [0, infinity]. Clipping to the range [0,1] with tf_render.clip_by_value is likely
    reasonable for both viewing and training most scenes. More complex scenes
    with multiple lights should tone map color values for display only. One
    simple tonemapping approach is to rescale color values as x/(1+x); gamma
    compression is another common techinque. Alpha values are zero for
    background pixels and near one for mesh pixels.
  Raises:
    ValueError: An invalid argument to the method is detected.
  """
  if len(vertices.shape) != 3:
    raise ValueError('Vertices must have shape [batch_size, vertex_count, 3].')
  batch_size = vertices.shape[0].value
  if len(normals.shape) != 3:
    raise ValueError('Normals must have shape [batch_size, vertex_count, 3].')

  if len(diffuse_colors.shape) != 3:
    raise ValueError(
        'vertex_diffuse_colors must have shape [batch_size, vertex_count, 3].')

  if camera_position.get_shape().as_list() == [3]:
    camera_position = tf.tile(
        tf.expand_dims(camera_position, axis=0), [batch_size, 1])
  elif camera_position.get_shape().as_list() != [batch_size, 3]:
    raise ValueError('Camera_position must have shape [batch_size, 3]')

  # TODO: Debug Shape
  if mtx_camera.get_shape().as_list() == [4, 4]:
      mtx_camera = tf.tile(
        tf.expand_dims(mtx_camera, axis=0), [batch_size, 1, 1])
  elif mtx_camera.get_shape().as_list() != [batch_size, 4, 4]:
    raise ValueError('Camera_lookat must have shape [batch_size, 4, 4]')

  if mtx_perspective_frustrum.get_shape().as_list() == [4, 4]:
      mtx_camera = tf.tile(
        tf.expand_dims(mtx_perspective_frustrum, axis=0), [batch_size, 1])
  elif mtx_camera.get_shape().as_list() != [batch_size, 4, 4]:
    raise ValueError('Camera_lookat must have shape [batch_size, 4, 4]')


  vertex_attributes = tf.concat([normals, vertices, diffuse_colors], axis=2)

  clip_space_transforms = tf.matmul(mtx_perspective_frustrum, mtx_camera, name="mtx_clip_space_transforms_batch")

  pixel_attributes, alpha, tri_ids = rasterize_triangles(
      vertices, vertex_attributes, triangles, clip_space_transforms,
      image_width, image_height, [-1] * vertex_attributes.shape[2].value)

  # Extract the interpolated vertex attributes from the pixel buffer and
  # supply them to the shader:
  #pixel_normals = tf.nn.l2_normalize(pixel_attributes[:, :, :, 0:3], dim=3)
  #pixel_positions = pixel_attributes[:, :, :, 3:6]
  diffuse_colors = pixel_attributes[:, :, :, 6:9]
  diffuse_colors = tf.reverse(diffuse_colors, axis=[1])

  #return renders, pixel_mask
  pixel_mask = alpha > 0.5
  pixel_mask = tf.cast(pixel_mask, dtype=tf.float32)
  pixel_mask = tf.reverse(pixel_mask, axis=[1])

  #
  tri_ids = tf.expand_dims(tri_ids, -1)

  return diffuse_colors, pixel_mask, tri_ids


def mesh_renderer_camera(vertices, triangles, normals, diffuse_colors,
                  mtx_camera, mtx_perspective_frustrum, camera_position,
                  light_positions, light_intensities, image_width, image_height,
                  specular_colors=None, shininess_coefficients=None, ambient_color=None, background=-1
                  ):
  """Renders an input scene using phong shading, and returns an output image.

  Args:
    vertices: 3-D float32 tensor with shape [batch_size, vertex_count, 3]. Each
        triplet is an xyz position in world space.
    triangles: 2-D int32 tensor with shape [triangle_count, 3]. Each triplet
        should contain vertex indices describing a triangle such that the
        triangle's normal points toward the viewer if the forward order of the
        triplet defines a clockwise winding of the vertices. Gradients with
        respect to this tensor are not available.
    normals: 3-D float32 tensor with shape [batch_size, vertex_count, 3]. Each
        triplet is the xyz vertex normal for its corresponding vertex. Each
        vector is assumed to be already normalized.
    diffuse_colors: 3-D float32 tensor with shape [batch_size,
        vertex_count, 3]. The RGB diffuse reflection in the range [0,1] for
        each vertex.

    mtx_camera: 3-D tensor with shape [batch_size, 4, 4] or 2-D tensor with
        shape [4, 4] specifying the camera model view matrix
    mtx_perspective_frustrum: 3-D tensor with shape [batch_size, 4, 4] or 2-D tensor with
        shape [4, 4] specifying the perspective and frustrum matrix
    camera_position: 2-D tensor with shape [batch_size, 3] or 1-D tensor with
        shape [3] specifying the XYZ world space camera position.

    light_positions: a 3-D tensor with shape [batch_size, light_count, 3]. The
        XYZ position of each light in the scene. In the same coordinate space as
        pixel_positions.
    light_intensities: a 3-D tensor with shape [batch_size, light_count, 3]. The
        RGB intensity values for each light. Intensities may be above one.
    image_width: int specifying desired output image width in pixels.
    image_height: int specifying desired output image height in pixels.

    specular_colors: 3-D float32 tensor with shape [batch_size,
        vertex_count, 3]. The RGB specular reflection in the range [0, 1] for
        each vertex.  If supplied, specular reflections will be computed, and
        both specular_colors and shininess_coefficients are expected.
    shininess_coefficients: a 0D-2D float32 tensor with maximum shape
       [batch_size, vertex_count]. The phong shininess coefficient of each
       vertex. A 0D tensor or float gives a constant shininess coefficient
       across all batches and images. A 1D tensor must have shape [batch_size],
       and a single shininess coefficient per image is used.
    ambient_color: a 2D tensor with shape [batch_size, 3]. The RGB ambient
        color, which is added to each pixel in the scene. If None, it is
        assumed to be black.


  Returns:
    A 4-D float32 tensor of shape [batch_size, image_height, image_width, 4]
    containing the lit RGBA color values for each image at each pixel. RGB
    colors are the intensity values before tonemapping and can be in the range
    [0, infinity]. Clipping to the range [0,1] with tf_render.clip_by_value is likely
    reasonable for both viewing and training most scenes. More complex scenes
    with multiple lights should tone map color values for display only. One
    simple tonemapping approach is to rescale color values as x/(1+x); gamma
    compression is another common techinque. Alpha values are zero for
    background pixels and near one for mesh pixels.
  Raises:
    ValueError: An invalid argument to the method is detected.
  """
  if len(vertices.shape) != 3:
    raise ValueError('Vertices must have shape [batch_size, vertex_count, 3].')
  batch_size = vertices.shape[0].value
  if len(normals.shape) != 3:
    raise ValueError('Normals must have shape [batch_size, vertex_count, 3].')
  if len(light_positions.shape) != 3:
    raise ValueError(
        'Light_positions must have shape [batch_size, light_count, 3].')
  if len(light_intensities.shape) != 3:
    raise ValueError(
        'Light_intensities must have shape [batch_size, light_count, 3].')
  if len(diffuse_colors.shape) != 3:
    raise ValueError(
        'vertex_diffuse_colors must have shape [batch_size, vertex_count, 3].')
  if (ambient_color is not None and
      ambient_color.get_shape().as_list() != [batch_size, 3]):
    raise ValueError('Ambient_color must have shape [batch_size, 3].')
  if camera_position.get_shape().as_list() == [3]:
    camera_position = tf.tile(
        tf.expand_dims(camera_position, axis=0), [batch_size, 1])
  elif camera_position.get_shape().as_list() != [batch_size, 3]:
    raise ValueError('Camera_position must have shape [batch_size, 3]')

  # TODO: Debug Shape
  if mtx_camera.get_shape().as_list() == [4, 4]:
      mtx_camera = tf.tile(
        tf.expand_dims(mtx_camera, axis=0), [batch_size, 1, 1])
  elif mtx_camera.get_shape().as_list() != [batch_size, 4, 4]:
    raise ValueError('Camera_lookat must have shape [batch_size, 4, 4]')

  if mtx_perspective_frustrum.get_shape().as_list() == [4, 4]:
      mtx_camera = tf.tile(
        tf.expand_dims(mtx_perspective_frustrum, axis=0), [batch_size, 1])
  elif mtx_camera.get_shape().as_list() != [batch_size, 4, 4]:
    raise ValueError('Camera_lookat must have shape [batch_size, 4, 4]')

  if specular_colors is not None and shininess_coefficients is None:
    raise ValueError(
        'Specular colors were supplied without shininess coefficients.')
  if shininess_coefficients is not None and specular_colors is None:
    raise ValueError(
        'Shininess coefficients were supplied without specular colors.')
  if specular_colors is not None:
    # Since a 0-D float32 tensor is accepted, also accept a float.
    if isinstance(shininess_coefficients, float):
      shininess_coefficients = tf.constant(
          shininess_coefficients, dtype=tf.float32)
    if len(specular_colors.shape) != 3:
      raise ValueError('The specular colors must have shape [batch_size, '
                       'vertex_count, 3].')
    if len(shininess_coefficients.shape) > 2:
      raise ValueError('The shininess coefficients must have shape at most'
                       '[batch_size, vertex_count].')
    # If we don't have per-vertex coefficients, we can just reshape the
    # input shininess to broadcast later, rather than interpolating an
    # additional vertex attribute:
    if len(shininess_coefficients.shape) < 2:
      vertex_attributes = tf.concat(
          [normals, vertices, diffuse_colors, specular_colors], axis=2)
    else:
      vertex_attributes = tf.concat(
          [
              normals, vertices, diffuse_colors, specular_colors,
              tf.expand_dims(shininess_coefficients, axis=2)
          ],
          axis=2)
  else:
    vertex_attributes = tf.concat([normals, vertices, diffuse_colors], axis=2)

  # camera_matrices = camera_utils.look_at(camera_position, camera_lookat,
  #                                        camera_up)
  #
  # perspective_transforms = camera_utils.perspective(image_width / image_height,
  #                                                   fov_y, near_clip, far_clip)

  clip_space_transforms = tf.matmul(mtx_perspective_frustrum, mtx_camera, name="mtx_clip_space_transforms_batch")

  pixel_attributes, alpha, tri_ids = rasterize_triangles(
      vertices, vertex_attributes, triangles, clip_space_transforms,
      image_width, image_height, [background] * vertex_attributes.shape[2].value)

  # Extract the interpolated vertex attributes from the pixel buffer and
  # supply them to the shader:
  pixel_normals = tf.nn.l2_normalize(pixel_attributes[:, :, :, 0:3], dim=3)
  pixel_positions = pixel_attributes[:, :, :, 3:6]
  diffuse_colors = pixel_attributes[:, :, :, 6:9]
  if specular_colors is not None:
    specular_colors = pixel_attributes[:, :, :, 9:12]
    # Retrieve the interpolated shininess coefficients if necessary, or just
    # reshape our input for broadcasting:
    if len(shininess_coefficients.shape) == 2:
      shininess_coefficients = pixel_attributes[:, :, :, 12]
    else:
      shininess_coefficients = tf.reshape(shininess_coefficients, [-1, 1, 1])

  pixel_mask = tf.cast(tf.reduce_any(diffuse_colors >= 0, axis=3), tf.float32)

  renders = phong_shader(
      normals=pixel_normals,
      alphas=pixel_mask,
      pixel_positions=pixel_positions,
      light_positions=light_positions,
      light_intensities=light_intensities,
      diffuse_colors=diffuse_colors,
      camera_position=camera_position if specular_colors is not None else None,
      specular_colors=specular_colors,
      shininess_coefficients=shininess_coefficients,
      ambient_color=ambient_color)

  #return renders, pixel_mask
  pixel_mask = alpha > 0.5
  pixel_mask = tf.cast(pixel_mask, dtype=tf.float32)
  pixel_mask = tf.reverse(pixel_mask, axis=[1])

  return renders, pixel_mask


def mesh_depthmap_camera(vertices, triangles, mtx_ext,
                  mtx_camera, mtx_perspective_frustrum,
                  image_width, image_height
                  ):
  """Renders an input scene using phong shading, and returns an output image.

  Args:
    vertices: 3-D float32 tensor with shape [batch_size, vertex_count, 3]. Each
        triplet is an xyz position in world space.
    triangles: 2-D int32 tensor with shape [triangle_count, 3]. Each triplet
        should contain vertex indices describing a triangle such that the
        triangle's normal points toward the viewer if the forward order of the
        triplet defines a clockwise winding of the vertices. Gradients with
        respect to this tensor are not available.
    normals: 3-D float32 tensor with shape [batch_size, vertex_count, 3]. Each
        triplet is the xyz vertex normal for its corresponding vertex. Each
        vector is assumed to be already normalized.

    mtx_camera: 3-D tensor with shape [batch_size, 4, 4] or 2-D tensor with
        shape [4, 4] specifying the camera model view matrix
    mtx_perspective_frustrum: 3-D tensor with shape [batch_size, 4, 4] or 2-D tensor with
        shape [4, 4] specifying the perspective and frustrum matrix
    camera_position: 2-D tensor with shape [batch_size, 3] or 1-D tensor with
        shape [3] specifying the XYZ world space camera position.

    image_width: int specifying desired output image width in pixels.
    image_height: int specifying desired output image height in pixels.

  Returns:
    A 4-D float32 tensor of shape [batch_size, image_height, image_width, 4]
    containing the lit RGBA color values for each image at each pixel. RGB
    colors are the intensity values before tonemapping and can be in the range
    [0, infinity]. Clipping to the range [0,1] with tf_render.clip_by_value is likely
    reasonable for both viewing and training most scenes. More complex scenes
    with multiple lights should tone map color values for display only. One
    simple tonemapping approach is to rescale color values as x/(1+x); gamma
    compression is another common techinque. Alpha values are zero for
    background pixels and near one for mesh pixels.
  Raises:
    ValueError: An invalid argument to the method is detected.
  """
  if len(vertices.shape) != 3:
    raise ValueError('Vertices must have shape [batch_size, vertex_count, 3].')
  batch_size = vertices.shape[0].value

  # TODO: Debug Shape
  if mtx_camera.get_shape().as_list() == [4, 4]:
      mtx_camera = tf.tile(
        tf.expand_dims(mtx_camera, axis=0), [batch_size, 1, 1])
  elif mtx_camera.get_shape().as_list() != [batch_size, 4, 4]:
    raise ValueError('Camera_lookat must have shape [batch_size, 4, 4]')

  if mtx_perspective_frustrum.get_shape().as_list() == [4, 4]:
      mtx_camera = tf.tile(
        tf.expand_dims(mtx_perspective_frustrum, axis=0), [batch_size, 1])
  elif mtx_camera.get_shape().as_list() != [batch_size, 4, 4]:
    raise ValueError('Camera_lookat must have shape [batch_size, 4, 4]')


  # vertex attribute of depthmap is only z
  vertex_attributes = vertices
  #vertex_attributes = tf_render.expand_dims(vertex_attributes, -1)
  # camera_matrices = camera_utils.look_at(camera_position, camera_lookat,
  #                                        camera_up)
  #
  # perspective_transforms = camera_utils.perspective(image_width / image_height,
  #                                                   fov_y, near_clip, far_clip)

  clip_space_transforms = tf.matmul(mtx_perspective_frustrum, mtx_camera, name="mtx_clip_space_transforms_batch")

  pixel_attributes, alpha, _ = rasterize_triangles(
      vertices, vertex_attributes, triangles, clip_space_transforms,
      image_width, image_height, [99999999] * vertex_attributes.shape[2].value)

  # Extract the interpolated vertex attributes from the pixel buffer and
  # supply them to the shader:
  filler_homo = tf.ones(shape=[pixel_attributes.shape[0], pixel_attributes.shape[1], pixel_attributes.shape[2], 1])
  pixel_attributes = tf.concat([pixel_attributes, filler_homo], axis=3)
  pixel_attributes = tf.reshape(pixel_attributes, shape=[batch_size, -1, 4])
  pixel_attributes = tf.transpose(pixel_attributes, perm=[0, 2, 1])

  pixel_attributes = tf.matmul(mtx_ext, pixel_attributes)
  pixel_attributes = tf.transpose(pixel_attributes, perm=[0, 2, 1])
  pixel_attributes = tf.reshape(pixel_attributes, shape=[batch_size, image_height, image_width, 4])
  depth_map = pixel_attributes[:, :, :, 2]

  pixel_mask = alpha > 0.5
  pixel_mask = tf.cast(pixel_mask, dtype=tf.float32)

  depth_map = tf.reverse(depth_map, axis=[1])
  pixel_mask = tf.reverse(pixel_mask, axis=[1])

  return depth_map, pixel_mask

# ortho
def mesh_rendererOrtho_camera(vertices, triangles, normals, diffuse_colors,
                  mtx_camera, mtx_perspective_frustrum, light_positions, light_intensities,
                  image_width, image_height, ambient_color=None, background=-1
                  ):
  """Renders an input scene using phong shading, and returns an output image.

  Args:
    vertices: 3-D float32 tensor with shape [batch_size, vertex_count, 3]. Each
        triplet is an xyz position in world space.
    triangles: 2-D int32 tensor with shape [triangle_count, 3]. Each triplet
        should contain vertex indices describing a triangle such that the
        triangle's normal points toward the viewer if the forward order of the
        triplet defines a clockwise winding of the vertices. Gradients with
        respect to this tensor are not available.
    normals: 3-D float32 tensor with shape [batch_size, vertex_count, 3]. Each
        triplet is the xyz vertex normal for its corresponding vertex. Each
        vector is assumed to be already normalized.
    diffuse_colors: 3-D float32 tensor with shape [batch_size,
        vertex_count, 3]. The RGB diffuse reflection in the range [0,1] for
        each vertex.

    mtx_camera: 3-D tensor with shape [batch_size, 4, 4] or 2-D tensor with
        shape [4, 4] specifying the camera model view matrix
    mtx_perspective_frustrum: 3-D tensor with shape [batch_size, 4, 4] or 2-D tensor with
        shape [4, 4] specifying the perspective and frustrum matrix
    camera_position: 2-D tensor with shape [batch_size, 3] or 1-D tensor with
        shape [3] specifying the XYZ world space camera position.

    light_positions: a 3-D tensor with shape [batch_size, light_count, 3]. The
        XYZ position of each light in the scene. In the same coordinate space as
        pixel_positions.
    light_intensities: a 3-D tensor with shape [batch_size, light_count, 3]. The
        RGB intensity values for each light. Intensities may be above one.
    image_width: int specifying desired output image width in pixels.
    image_height: int specifying desired output image height in pixels.

    specular_colors: 3-D float32 tensor with shape [batch_size,
        vertex_count, 3]. The RGB specular reflection in the range [0, 1] for
        each vertex.  If supplied, specular reflections will be computed, and
        both specular_colors and shininess_coefficients are expected.
    shininess_coefficients: a 0D-2D float32 tensor with maximum shape
       [batch_size, vertex_count]. The phong shininess coefficient of each
       vertex. A 0D tensor or float gives a constant shininess coefficient
       across all batches and images. A 1D tensor must have shape [batch_size],
       and a single shininess coefficient per image is used.
    ambient_color: a 2D tensor with shape [batch_size, 3]. The RGB ambient
        color, which is added to each pixel in the scene. If None, it is
        assumed to be black.


  Returns:
    A 4-D float32 tensor of shape [batch_size, image_height, image_width, 4]
    containing the lit RGBA color values for each image at each pixel. RGB
    colors are the intensity values before tonemapping and can be in the range
    [0, infinity]. Clipping to the range [0,1] with tf_render.clip_by_value is likely
    reasonable for both viewing and training most scenes. More complex scenes
    with multiple lights should tone map color values for display only. One
    simple tonemapping approach is to rescale color values as x/(1+x); gamma
    compression is another common techinque. Alpha values are zero for
    background pixels and near one for mesh pixels.
  Raises:
    ValueError: An invalid argument to the method is detected.
  """
  if len(vertices.shape) != 3:
    raise ValueError('Vertices must have shape [batch_size, vertex_count, 3].')
  batch_size = vertices.shape[0].value
  if len(normals.shape) != 3:
    raise ValueError('Normals must have shape [batch_size, vertex_count, 3].')
  if len(light_positions.shape) != 3:
    raise ValueError(
        'Light_positions must have shape [batch_size, light_count, 3].')
  if len(light_intensities.shape) != 3:
    raise ValueError(
        'Light_intensities must have shape [batch_size, light_count, 3].')
  if len(diffuse_colors.shape) != 3:
    raise ValueError(
        'vertex_diffuse_colors must have shape [batch_size, vertex_count, 3].')
  if (ambient_color is not None and
      ambient_color.get_shape().as_list() != [batch_size, 3]):
    raise ValueError('Ambient_color must have shape [batch_size, 3].')

  # TODO: Debug Shape
  if mtx_camera.get_shape().as_list() == [4, 4]:
      mtx_camera = tf.tile(
        tf.expand_dims(mtx_camera, axis=0), [batch_size, 1, 1])
  elif mtx_camera.get_shape().as_list() != [batch_size, 4, 4]:
    raise ValueError('Camera_lookat must have shape [batch_size, 4, 4]')

  if mtx_perspective_frustrum.get_shape().as_list() == [4, 4]:
      mtx_camera = tf.tile(
        tf.expand_dims(mtx_perspective_frustrum, axis=0), [batch_size, 1])
  elif mtx_camera.get_shape().as_list() != [batch_size, 4, 4]:
    raise ValueError('Camera_lookat must have shape [batch_size, 4, 4]')


  vertex_attributes = tf.concat([normals, vertices, diffuse_colors], axis=2)

  clip_space_transforms = tf.matmul(mtx_perspective_frustrum, mtx_camera, name="mtx_clip_space_transforms_batch")

  pixel_attributes, alpha, tri_ids = rasterize_triangles(
      vertices, vertex_attributes, triangles, clip_space_transforms,
      image_width, image_height, [background] * vertex_attributes.shape[2].value)

  # Extract the interpolated vertex attributes from the pixel buffer and
  # supply them to the shader:
  pixel_normals = tf.nn.l2_normalize(pixel_attributes[:, :, :, 0:3], dim=3)
  pixel_positions = pixel_attributes[:, :, :, 3:6]
  diffuse_colors = pixel_attributes[:, :, :, 6:9]

  pixel_mask = tf.cast(tf.reduce_any(diffuse_colors >= 0, axis=3), tf.float32)

  renders = phong_shader(
      normals=pixel_normals,
      alphas=pixel_mask,
      pixel_positions=pixel_positions,
      light_positions=light_positions,
      light_intensities=light_intensities,
      diffuse_colors=diffuse_colors,
      camera_position=None,
      specular_colors=None,
      shininess_coefficients=None,
      ambient_color=ambient_color)

  #return renders, pixel_mask
  pixel_mask = alpha > 0.5
  pixel_mask = tf.cast(pixel_mask, dtype=tf.float32)
  pixel_mask = tf.reverse(pixel_mask, axis=[1])

  return renders, pixel_mask


def mesh_depthmapOrtho_camera(vertices, triangles,
                              mtx_ext, mtx_perspective_frustrum, image_width, image_height
                  ):
  """Renders an input scene using phong shading, and returns an output image.

  Args:
    vertices: 3-D float32 tensor with shape [batch_size, vertex_count, 3]. Each
        triplet is an xyz position in world space.
    triangles: 2-D int32 tensor with shape [triangle_count, 3]. Each triplet
        should contain vertex indices describing a triangle such that the
        triangle's normal points toward the viewer if the forward order of the
        triplet defines a clockwise winding of the vertices. Gradients with
        respect to this tensor are not available.
    normals: 3-D float32 tensor with shape [batch_size, vertex_count, 3]. Each
        triplet is the xyz vertex normal for its corresponding vertex. Each
        vector is assumed to be already normalized.

    mtx_camera: 3-D tensor with shape [batch_size, 4, 4] or 2-D tensor with
        shape [4, 4] specifying the camera model view matrix
    mtx_perspective_frustrum: 3-D tensor with shape [batch_size, 4, 4] or 2-D tensor with
        shape [4, 4] specifying the perspective and frustrum matrix
    camera_position: 2-D tensor with shape [batch_size, 3] or 1-D tensor with
        shape [3] specifying the XYZ world space camera position.

    image_width: int specifying desired output image width in pixels.
    image_height: int specifying desired output image height in pixels.

  Returns:
    A 4-D float32 tensor of shape [batch_size, image_height, image_width, 4]
    containing the lit RGBA color values for each image at each pixel. RGB
    colors are the intensity values before tonemapping and can be in the range
    [0, infinity]. Clipping to the range [0,1] with tf_render.clip_by_value is likely
    reasonable for both viewing and training most scenes. More complex scenes
    with multiple lights should tone map color values for display only. One
    simple tonemapping approach is to rescale color values as x/(1+x); gamma
    compression is another common techinque. Alpha values are zero for
    background pixels and near one for mesh pixels.
  Raises:
    ValueError: An invalid argument to the method is detected.
  """
  if len(vertices.shape) != 3:
    raise ValueError('Vertices must have shape [batch_size, vertex_count, 3].')
  batch_size = vertices.shape[0].value

  # TODO: Debug Shape
  if mtx_ext.get_shape().as_list() == [4, 4]:
      mtx_ext = tf.tile(
        tf.expand_dims(mtx_ext, axis=0), [batch_size, 1, 1])
  elif mtx_ext.get_shape().as_list() != [batch_size, 4, 4]:
    raise ValueError('Camera_lookat must have shape [batch_size, 4, 4]')

  if mtx_perspective_frustrum.get_shape().as_list() == [4, 4]:
      mtx_perspective_frustrum = tf.tile(
        tf.expand_dims(mtx_perspective_frustrum, axis=0), [batch_size, 1])
  elif mtx_perspective_frustrum.get_shape().as_list() != [batch_size, 4, 4]:
    raise ValueError('Camera_lookat must have shape [batch_size, 4, 4]')


  # vertex attribute of depthmap is only z
  vertex_attributes = vertices
  #vertex_attributes = tf_render.expand_dims(vertex_attributes, -1)
  # camera_matrices = camera_utils.look_at(camera_position, camera_lookat,
  #                                        camera_up)
  #
  # perspective_transforms = camera_utils.perspective(image_width / image_height,
  #                                                   fov_y, near_clip, far_clip)

  clip_space_transforms = tf.matmul(mtx_perspective_frustrum, mtx_ext, name="mtx_clip_space_transforms_batch")

  pixel_attributes, alpha, _ = rasterize_triangles(
      vertices, vertex_attributes, triangles, clip_space_transforms,
      image_width, image_height, [99999999] * vertex_attributes.shape[2].value)

  # Extract the interpolated vertex attributes from the pixel buffer and
  # supply them to the shader:
  filler_homo = tf.ones(shape=[pixel_attributes.shape[0], pixel_attributes.shape[1], pixel_attributes.shape[2], 1])
  pixel_attributes = tf.concat([pixel_attributes, filler_homo], axis=3)
  pixel_attributes = tf.reshape(pixel_attributes, shape=[batch_size, -1, 4])
  pixel_attributes = tf.transpose(pixel_attributes, perm=[0, 2, 1])

  pixel_attributes = tf.matmul(mtx_ext, pixel_attributes)
  pixel_attributes = tf.transpose(pixel_attributes, perm=[0, 2, 1])
  pixel_attributes = tf.reshape(pixel_attributes, shape=[batch_size, image_height, image_width, 4])
  depth_map = pixel_attributes[:, :, :, 2]

  pixel_mask = alpha > 0.5
  pixel_mask = tf.cast(pixel_mask, dtype=tf.float32)

  depth_map = tf.reverse(depth_map, axis=[1])
  pixel_mask = tf.reverse(pixel_mask, axis=[1])

  return depth_map, pixel_mask