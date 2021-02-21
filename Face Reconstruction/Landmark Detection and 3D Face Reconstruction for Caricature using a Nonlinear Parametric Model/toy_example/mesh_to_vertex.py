# To obtain vertices of 3D faces ".obj"

import openmesh as om
import numpy as np

mesh = om.read_trimesh("mean_face.obj")
vertex = np.zeros((3,6144), dtype=float)
for i in range(6144):
    vertex[:, i] = mesh.points()[i]
np.save("mean_face.npy", vertex)
