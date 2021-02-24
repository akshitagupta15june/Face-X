# Copyright 2016 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Contains definitions for the original form of Residual Networks.
The 'v1' residual networks (ResNets) implemented in this module were proposed
by:
[1] Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun
    Deep Residual Learning for Image Recognition. arXiv:1512.03385
Other variants were introduced in:
[2] Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun
    Identity Mappings in Deep Residual Networks. arXiv: 1603.05027
The networks defined in this module utilize the bottleneck building block of
[1] with projection shortcuts only for increasing depths. They employ batch
normalization *after* every weight layer. This is the architecture used by
MSRA in the Imagenet and MSCOCO 2016 competition models ResNet-101 and
ResNet-152. See [2; Fig. 1a] for a comparison between the current 'v1'
architecture and the alternative 'v2' architecture of [2] which uses batch
normalization *before* every weight layer in the so-called full pre-activation
units.
Typical use:
   from tensorflow.contrib.slim.nets import resnet_v1
ResNet-101 for image classification into 1000 classes:
   # inputs has shape [batch, 224, 224, 3]
   with slim.arg_scope(resnet_v1.resnet_arg_scope()):
      net, end_points = resnet_v1.resnet_v1_101(inputs, 1000, is_training=False)
ResNet-101 for semantic segmentation into 21 classes:
   # inputs has shape [batch, 513, 513, 3]
   with slim.arg_scope(resnet_v1.resnet_arg_scope()):
      net, end_points = resnet_v1.resnet_v1_101(inputs,
                                                21,
                                                is_training=False,
                                                global_pool=False,
                                                output_stride=16)
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# import os, sys
# #self
# _curr_path = os.path.abspath(__file__) # /home/..../face
# _cur_dir = os.path.dirname(_curr_path) # ./
# print(_cur_dir)
# sys.path.append(_cur_dir) # /home/..../pytorch3d

#
import tensorflow as tf
from tensorflow.contrib import layers
from tensorflow.contrib.slim.nets import resnet_v1

def encoder_resnet50(images, num_classes, is_training=True, reuse=None):

    """Predict prediction tensors from inputs tensor.

    Outputs of this function can be passed to loss or postprocess functions.

    Args:
        preprocessed_inputs: A float32 tensor with shape [batch_size,
            height, width, num_channels] representing a batch of images.

    Returns:
        prediction_dict: A dictionary holding prediction tensors to be
            passed to the Loss or Postprocess functions.
    """
    net, endpoints = resnet_v1.resnet_v1_50(
        images,
        num_classes=num_classes,
        is_training=is_training,
        reuse = reuse)
    net = tf.squeeze(net, axis=[1, 2])
    return net, endpoints
