# system
from __future__ import print_function
import os
import sys

# python lib
import numpy as np

def vertex_y_max(trimesh):
    vertex = np.array(trimesh.vertices)
    vertex_y = list(vertex[:, 1])
    y_idx = vertex_y.index(max(vertex_y))
    return vertex[y_idx]

def vertex_y_min(trimesh):
    vertex = np.array(trimesh.vertices)
    vertex_y = list(vertex[:, 1])
    y_idx = vertex_y.index(min(vertex_y))
    return vertex[y_idx]
