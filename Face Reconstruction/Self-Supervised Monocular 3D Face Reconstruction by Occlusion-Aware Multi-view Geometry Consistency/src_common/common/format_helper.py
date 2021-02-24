
# system
from __future__ import print_function

# python lib
import numpy as np

# tf_render
import tensorflow as tf

# self

def parse_seq(list_seq):
    return list_seq[0:1], list_seq[1:]

def parse_gpu_list(gpu_list):
    return gpu_list.split(',')

def batch_size_extract(*object): # TODO: More robust
    """
    :param object: np, tensor, scalar
    :return:
    """
    batch_size = None
    for inst in object:
        if inst is not None:
            if (isinstance(inst, tf.Tensor) or isinstance(inst, np.ndarray)) and len(inst.shape) > 1:
                batch_size = max(inst.shape[0], batch_size)
            else:
                batch_size = max(1, batch_size)
    return batch_size

