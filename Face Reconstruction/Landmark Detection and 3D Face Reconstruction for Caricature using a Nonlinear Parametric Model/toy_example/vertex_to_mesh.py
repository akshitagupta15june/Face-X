# To recover 3D faces from vertices '.npy'

import openmesh as om
import numpy as np

mesh = om.read_trimesh("mean_face.obj")
vertex = np.load("1.npy")
for i in range(6144):
    mesh.points()[i] = vertex[:, i]
om.write_mesh("1.obj", mesh)
