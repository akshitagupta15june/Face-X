import logging

import tensorflow as tf

from base_model import BaseModel

logger = logging.getLogger('x')


class Model(BaseModel):
  """
    Mesh Convolutional Autoencoder which uses the Chebyshev approximation.
  """

  def __init__(self, *args, **kwargs):
    super(Model, self).__init__(*args, **kwargs)
    logger.info('Using Normal Model...')

  def mesh_generator(self, image_emb, pca_color, reuse=False):
    with tf.variable_scope('mesh_generator', reuse=reuse):
      decode_color = self.mesh_decoder(image_emb, reuse=reuse)
      refine_color = self.mesh_refiner(pca_color, reuse=reuse)
      with tf.variable_scope('mesh_concat'):
        concat = tf.concat([decode_color, refine_color], axis=-1)
        outputs = self.chebyshev5(concat, self.laplacians[0], 3, 6)
      outputs = tf.nn.tanh(outputs)
      return outputs

  def mesh_decoder(self, image_emb, reuse=False):
    if self.wide:
      F = [32, 32, 64, 128, 256]
    else:
      F = [32, 16, 16, 16, 16]
    with tf.variable_scope('mesh_decoder', reuse=reuse):
      with tf.variable_scope('fc'):
        x = self.fc(image_emb, self.pool_size[-1] * F[0])  # N x MF
      x = tf.reshape(x,
                     [self.batch_size, self.pool_size[-1], F[0]])  # N x M x F

      for i in range(4):
        with tf.variable_scope('upconv{}'.format(i + 1)):
          with tf.name_scope('unpooling'):
            x = self.unpool(x, self.upsamp_trans[-i - 1])
          with tf.name_scope('filter'):
            x = self.chebyshev5(x, self.laplacians[-i - 2], F[i + 1], 6)
          with tf.name_scope('bias_relu'):
            x = self.brelu(x)

      with tf.name_scope('outputs'):
        x = self.chebyshev5(x, self.laplacians[0], 3, 6)
        x = self.brelu(x)
        # outputs = tf.nn.tanh(x)

    return x

  # def mesh_refiner(self, pca_color, reuse=False):
  #   if self.wide:
  #     F = [16, 32, 64, 128]
  #   else:
  #     F = [16, 32, 32, 16]
  #   with tf.variable_scope('mesh_refiner', reuse=reuse):
  #     x = pca_color
  #     for i in range(4):
  #       with tf.variable_scope('graph_conv{}'.format(i + 1)):
  #         with tf.name_scope('filter'):
  #           x = self.chebyshev5(x, self.laplacians[0], F[i], 6)
  #         with tf.name_scope('bias_relu'):
  #           x = self.brelu(x)

  #     with tf.name_scope('outputs'):
  #       x = self.chebyshev5(x, self.laplacians[0], 3, 6)
  #       x = self.brelu(x)
  #       # outputs = tf.nn.tanh(x)

  #   return x

  # def mesh_refiner(self, pca_color, reuse=False):
  #   if self.wide:
  #     F = [16, 32, 64, 128]
  #   else:
  #     F = [16, 32, 64]
  #   with tf.variable_scope('mesh_refiner', reuse=reuse):
  #     x = pca_color
  #     with tf.variable_scope('graph_conv0'):
  #       with tf.name_scope('filter'):
  #         x = self.chebyshev5(x, self.laplacians[0], F[0], 6)
  #     for i in range(3):
  #       with tf.variable_scope('graph_conv{}'.format(i + 1)):
  #         with tf.name_scope('pooling'):
  #           x = self.unpool(x, self.downsamp_trans[i])
  #         with tf.name_scope('filter'):
  #           x = self.chebyshev5(x, self.laplacians[i + 1], F[i], 6)
  #         with tf.name_scope('bias_relu'):
  #           x = self.brelu(x)
  #     for i in range(3):
  #       with tf.variable_scope('graph_conv{}'.format(i + 4)):
  #         with tf.name_scope('unpooling'):
  #           x = self.unpool(x, self.upsamp_trans[-i - 2])
  #         with tf.name_scope('filter'):
  #           x = self.chebyshev5(x, self.laplacians[-i - 3], F[-i - 1], 6)
  #         with tf.name_scope('bias_relu'):
  #           x = self.brelu(x)

  #     with tf.name_scope('outputs'):
  #       x = self.chebyshev5(x, self.laplacians[0], 3, 6)
  #       x = self.brelu(x)
  #       # outputs = tf.nn.tanh(x)

  #   return x

  def mesh_refiner(self, pca_color, reuse=False):
    if self.wide:
      F = [16, 32, 64, 128]
    else:
      F = [16, 32, 64, 128]
    with tf.variable_scope('mesh_refiner', reuse=reuse):
      x = pca_color
      with tf.variable_scope('graph_conv0'):
        with tf.name_scope('filter'):
          x = self.chebyshev5(x, self.laplacians[0], F[0], 6)
      layer_enc = []
      for i in range(4):
        with tf.variable_scope('graph_conv{}'.format(i + 1)):
          with tf.name_scope('pooling'):
            x = self.unpool(x, self.downsamp_trans[i])
          with tf.name_scope('filter'):
            x = self.chebyshev5(x, self.laplacians[i + 1], F[i], 6)
          with tf.name_scope('bias_relu'):
            x = self.brelu(x)
          layer_enc.append(x)

      x = tf.reshape(x, [self.batch_size, self.pool_size[-1] * F[-1]])  # N x MF
      with tf.variable_scope('fc'):
        x = self.fc(x, int(self.z_dim))  # N x M0
      with tf.variable_scope('fc2'):
        x = self.fc(x, self.pool_size[-1] * F[-1])  # N x MF
      x = tf.reshape(x,
                     [self.batch_size, self.pool_size[-1], F[-1]])  # N x M x F

      for i in range(4):
        with tf.variable_scope('graph_conv{}'.format(i + 5)):
          with tf.name_scope('unpooling'):
            x = self.unpool(x, self.upsamp_trans[-i - 1])
          #TODO: with skip or not
          if i < 2:
            x = tf.concat([x, layer_enc[-i - 2]], axis=-1)
          with tf.name_scope('filter'):
            x = self.chebyshev5(x, self.laplacians[-i - 2], F[-i - 1], 6)
          with tf.name_scope('bias_relu'):
            x = self.brelu(x)

      with tf.name_scope('outputs'):
        x = self.chebyshev5(x, self.laplacians[0], 3, 6)
        x = self.brelu(x)
        # outputs = tf.nn.tanh(x)

    return x

  def image_disc(self, inputs, t=True, reuse=False):
    with tf.variable_scope('image_disc', reuse=reuse):
      x = inputs
      x = self.conv2d(x, 16, 1, 1, is_training=t, name='conv1_1')
      # x = self.conv2d(x, 32, 3, 1, is_training=t, name='conv1_2')
      x = tf.nn.max_pool(x, [1, 2, 2, 1], [1, 2, 2, 1], 'SAME')
      x = self.conv2d(x, 32, 3, 1, is_training=t, name='conv2_1')
      # x = self.conv2d(x, 64, 3, 1, is_training=t, name='conv2_2')
      x = tf.nn.max_pool(x, [1, 2, 2, 1], [1, 2, 2, 1], 'SAME')
      x = self.conv2d(x, 64, 3, 1, is_training=t, name='conv3_1')
      # x = self.conv2d(x, 128, 3, 1, is_training=t, name='conv3_2')
      x = tf.nn.max_pool(x, [1, 2, 2, 1], [1, 2, 2, 1], 'SAME')
      x = self.conv2d(x, 128, 3, 1, is_training=t, name='conv4_1')
      # x = self.conv2d(x, 256, 3, 1, is_training=t, name='conv4_2')
      x = tf.nn.max_pool(x, [1, 2, 2, 1], [1, 2, 2, 1], 'SAME')
      x = self.conv2d(x, 256, 3, 1, is_training=t, name='conv5_1')
      # x = self.conv2d(x, 512, 3, 1, is_training=t, name='conv5_2')
      x = tf.nn.max_pool(x, [1, 2, 2, 1], [1, 2, 2, 1], 'SAME')
      x = self.conv2d(x, 512, 3, 1, is_training=t, name='conv6_1')
      x = self.conv2d(x, 1, 7, 1, 'VALID', False, False, t, 'outputs')

      return tf.squeeze(x, axis=[1, 2])
