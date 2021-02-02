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
    logger.info('Using ResNet Model...')

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
      F = [32, 64, 128, 256]
    else:
      F = [32, 16, 16, 16]
    with tf.variable_scope('mesh_decoder', reuse=reuse):
      with tf.variable_scope('fc'):
        layer1 = self.fc(image_emb, self.pool_size[-1] * F[0])  # N x MF
      layer1 = tf.reshape(
          layer1, [self.batch_size, self.pool_size[-1], F[0]])  # N x M x F

      with tf.variable_scope('resblock1'):
        with tf.name_scope('unpooling'):
          layer2 = self.unpool(layer1, self.upsamp_trans[-1])
        layer2 = self.cheb_res_block(layer2, self.laplacians[-2], F[1],
                                     self.c_k)
      with tf.variable_scope('resblock2'):
        # layer3 = tf.nn.dropout(layer2, 1 - self.drop_rate)
        with tf.name_scope('unpooling'):
          layer3 = self.unpool(layer2, self.upsamp_trans[-2])
        layer3 = self.cheb_res_block(layer3, self.laplacians[-3], F[2],
                                     self.c_k)
      with tf.variable_scope('resblock3'):
        # layer4 = tf.nn.dropout(layer3, 1 - self.drop_rate)
        with tf.name_scope('unpooling'):
          layer4 = self.unpool(layer3, self.upsamp_trans[-3])
        layer4 = self.cheb_res_block(layer4, self.laplacians[-4], F[3],
                                     self.c_k)
      with tf.variable_scope('resblock4'):
        # layer5 = tf.nn.dropout(layer4, 1 - self.drop_rate)
        with tf.name_scope('unpooling'):
          layer5 = self.unpool(layer4, self.upsamp_trans[-4])
        outputs = self.cheb_res_block(layer5, self.laplacians[-5], 3, self.c_k)
        #  relu=False)
      # outputs = tf.nn.tanh(outputs)
    return outputs

  def mesh_refiner(self, pca_color, reuse=False):
    if self.wide:
      F = [16, 32, 64, 128]
    else:
      F = [16, 32, 32, 16]
    with tf.variable_scope('mesh_refiner', reuse=reuse):
      with tf.variable_scope('resblock1'):
        layer1 = self.cheb_res_block(pca_color, self.laplacians[0], F[0],
                                     self.c_k)
      with tf.variable_scope('resblock2'):
        with tf.name_scope('pooling'):
          layer2 = self.pool(layer1, self.downsamp_trans[0])
        layer2 = self.cheb_res_block(layer2, self.laplacians[1], F[1], self.c_k)
      with tf.variable_scope('resblock3'):
        # layer3 = tf.nn.dropout(layer2, 1 - self.drop_rate)
        layer3 = self.cheb_res_block(layer2, self.laplacians[1], F[2], self.c_k)
      with tf.variable_scope('resblock4'):
        # layer4 = tf.nn.dropout(layer3, 1 - self.drop_rate)
        with tf.name_scope('unpooling'):
          layer4 = self.unpool(layer3, self.upsamp_trans[0])
        layer4 = self.cheb_res_block(layer4, self.laplacians[0], F[3], self.c_k)
      with tf.variable_scope('resblock5'):
        # layer5 = tf.nn.dropout(layer4, 1 - self.drop_rate)
        outputs = self.cheb_res_block(layer4, self.laplacians[0], 3, self.c_k)
        #  relu=False)
      # outputs = tf.nn.tanh(outputs)
    return outputs

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
