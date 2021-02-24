
# system
from __future__ import print_function

import os
import sys

#
import tensorflow as tf
# third party
import trimesh

# self
_curr_path = os.path.abspath(__file__) # /home/..../face
_cur_dir = os.path.dirname(_curr_path) # ./
_tf_dir = os.path.dirname(_cur_dir) # ./
_tool_data_dir = os.path.dirname(_tf_dir) # ../
_deep_learning_dir = os.path.dirname(_tool_data_dir) # ../
print(_deep_learning_dir)
sys.path.append(_deep_learning_dir) # /home/..../pytorch3d

from .HDF5IO import *
from .trimesh_util import *


class BFM_singleTopo():
    #
    def __init__(self, path_gpmm, name, batch_size, rank=80, gpmm_exp_rank=64, mode_light=False, CVRT_MICRON_MM = 1000.0):
        self.path_gpmm = path_gpmm
        self.name = name
        self.hdf5io = HDF5IO(path_gpmm, mode='r')

        self.batch_size = batch_size
        self.rank = rank
        self.gpmm_exp_rank = gpmm_exp_rank
        self.CVRT_MICRON_MM = CVRT_MICRON_MM

        self._read_hdf5()
        self._generate_tensor(mode_light)

    def _read_hdf5(self):
        CVRT_MICRON_MM = self.CVRT_MICRON_MM
        name = self.name
        """
        Shape
        Origin Unit of measurement is micron
        We convert to mm
        pcaVar is eignvalue rather Var
        """
        hdf5io_shape = HDF5IO(self.path_gpmm, self.hdf5io.handler_file['shape' + name])
        self.hdf5io_pt_model = HDF5IO(self.path_gpmm, hdf5io_shape.handler_file['model'])
        self.hdf5io_pt_representer = HDF5IO(self.path_gpmm, hdf5io_shape.handler_file['representer'])

        pt_mean = self.hdf5io_pt_model.GetValue('mean').value
        pt_mean = np.reshape(pt_mean, [-1])
        self.pt_mean_np = pt_mean / CVRT_MICRON_MM

        pt_pcaBasis = self.hdf5io_pt_model.GetValue('pcaBasis').value
        self.pt_pcaBasis_np = pt_pcaBasis / CVRT_MICRON_MM

        pt_pcaVariance = self.hdf5io_pt_model.GetValue('pcaVariance').value
        pt_pcaVariance = np.reshape(pt_pcaVariance, [-1])
        self.pt_pcaVariance_np = np.square(pt_pcaVariance)

        self.point3d_mean_np = np.reshape(self.pt_mean_np, [-1, 3])

        """
        Vertex color
        Origin Unit of measurement is uint
        We convert to float
        pcaVar is eignvalue rather Var
        """
        hdf5io_color = HDF5IO(self.path_gpmm, self.hdf5io.handler_file['color' + name])
        self.hdf5io_rgb_model = HDF5IO(self.path_gpmm, hdf5io_color.handler_file['model'])

        rgb_mean = self.hdf5io_rgb_model.GetValue('mean').value
        rgb_mean = np.reshape(rgb_mean, [-1])
        self.rgb_mean_np = rgb_mean / 255.0

        rgb_pcaBasis = self.hdf5io_rgb_model.GetValue('pcaBasis').value
        self.rgb_pcaBasis_np = rgb_pcaBasis / 255.0

        rgb_pcaVariance = self.hdf5io_rgb_model.GetValue('pcaVariance').value
        rgb_pcaVariance = np.reshape(rgb_pcaVariance, [-1])
        self.rgb_pcaVariance_np = np.square(rgb_pcaVariance)

        self.rgb3d_mean_np = np.reshape(self.rgb_mean_np, [-1, 3])

        uv = self.hdf5io_rgb_model.GetValue('uv').value
        self.uv_np = np.reshape(uv, [-1, 2])

        if 0:
            import cv2
            texMU_fore_point = self.rgb3d_mean_np
            image = np.zeros(shape=[224, 224, 3])
            for i in range(len(texMU_fore_point)):
                color = texMU_fore_point[i] * 255.0
                uv = self.uv_np[i]
                u = int(uv[0] * 223)
                v = int((1 - uv[1]) * 223)
                image[v, u, :] = color
            image = np.asarray(image, dtype=np.uint8)
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            cv2.imshow("Image Debug", image)
            k = cv2.waitKey(0) & 0xFF
            if k == 27:
                cv2.destroyAllWindows()

        list_uvImageIndex = []
        for i in range(len(self.rgb3d_mean_np)):
            uv = self.uv_np[i]
            u = int(uv[0] * 224)-1
            v = int((1 - uv[1]) * 224)-1
            if u < 0:
                u = 0
            if v < 0:
                v = 0

            idx = v * 224 + u
            list_uvImageIndex.append(idx)
        self.uvIdx_np = np.array(list_uvImageIndex)

        """
        Expression
        Origin Unit of measurement is micron
        We convert to mm
        pcaVar is eignvalue rather Var
        """
        hdf5io_exp = HDF5IO(self.path_gpmm, self.hdf5io.handler_file['expression' + name])
        self.hdf5io_exp_model = HDF5IO(self.path_gpmm, hdf5io_exp.handler_file['model'])

        # self.exp_mean = self.hdf5io_exp_model.GetValue('mean').value
        exp_pcaBasis = self.hdf5io_exp_model.GetValue('pcaBasis').value
        self.exp_pcaBasis_np = exp_pcaBasis / CVRT_MICRON_MM

        exp_pcaVariance = self.hdf5io_exp_model.GetValue('pcaVariance').value
        exp_pcaVariance = np.reshape(exp_pcaVariance, [-1])
        self.exp_pcaVariance_np = np.square(exp_pcaVariance)
        # self.exp3d_mean_np = np.reshape(self.exp_mean, [-1, 3])

        """
        Tri
        Index from 1
        """
        mesh_tri_reference = self.hdf5io_pt_representer.GetValue('tri').value
        mesh_tri_reference = mesh_tri_reference - 1
        self.mesh_tri_np = np.reshape(mesh_tri_reference.astype(np.int32), [-1, 3])

        # Here depend on how to generate
        # Here depend on how to generate
        # Here depend on how to generate
        if 'idx_sub' in self.hdf5io_pt_representer.GetMainKeys():
            idx_subTopo = self.hdf5io_pt_representer.GetValue('idx_sub').value
            idx_subTopo = idx_subTopo # Here depend on how to generate
            self.idx_subTopo_np = np.reshape(idx_subTopo.astype(np.int32), [-1])
            self.idx_subTopo = tf.constant(self.idx_subTopo_np, dtype=tf.int32)

        self.nplist_v_ring_f_flat_np = self.hdf5io_pt_representer.GetValue('vertex_ring_face_flat').value
        # used in tensor
        self.nplist_ver_ref_face_num = self.hdf5io_pt_representer.GetValue('vertex_ring_face_num').value
        self.nplist_v_ring_f, self.nplist_v_ring_f_index = \
            self._get_v_ring_f(self.nplist_v_ring_f_flat_np, self.nplist_ver_ref_face_num)
        """
        lm idx
        """
        if 'idx_lm68' in self.hdf5io_pt_representer.GetMainKeys():
            idx_lm68_np = self.hdf5io_pt_representer.GetValue('idx_lm68').value
            idx_lm68_np = np.reshape(idx_lm68_np, [-1])
            self.idx_lm68_np = idx_lm68_np.astype(dtype=np.int32)

    def _generate_tensor(self, mode_light=False):
        rank = self.rank
        gpmm_exp_rank = self.gpmm_exp_rank
        if mode_light:
            pass
        else:
            """
            Vertex
            """
            self.pt_mean = tf.constant(self.pt_mean_np, dtype=tf.float32)
            self.pt_pcaBasis = tf.constant(self.pt_pcaBasis_np[:, :rank], dtype=tf.float32)
            self.pt_pcaVariance = tf.constant(self.pt_pcaVariance_np[:rank], dtype=tf.float32)

            """
            Vertex color
            """
            self.rgb_mean = tf.constant(self.rgb_mean_np, dtype=tf.float32)
            self.rgb_pcaBasis = tf.constant(self.rgb_pcaBasis_np[:, :rank], dtype=tf.float32)
            self.rgb_pcaVariance = tf.constant(self.rgb_pcaVariance_np[:rank], dtype=tf.float32)
            self.uv = tf.constant(self.uv_np[:rank], dtype=tf.float32)

            """
            Expression
            """
            self.exp_pcaBasis = tf.constant(self.exp_pcaBasis_np[:, :gpmm_exp_rank], dtype=tf.float32)
            self.exp_pcaVariance = tf.constant(self.exp_pcaVariance_np[:gpmm_exp_rank], dtype=tf.float32)

        """
        Generate normal presplit
        """
        self.nplist_v_ring_f_flat = [item for sublist in self.nplist_v_ring_f for item in sublist]

        max_padding = max(self.nplist_v_ring_f, key=len)
        max_padding = len(max_padding)
        self.nplist_v_ring_f_index_pad = []
        for sublist in self.nplist_v_ring_f:
            def trp(l, n):
                return np.concatenate([l[:n], [l[-1]] * (n - len(l))])
            sublist_pad = trp(sublist, max_padding)
            self.nplist_v_ring_f_index_pad.append(sublist_pad)
        self.nplist_v_ring_f_index_pad = np.array(self.nplist_v_ring_f_index_pad, dtype=np.int32)
        self.nplist_v_ring_f_index_flat = [item for sublist in self.nplist_v_ring_f_index for item in sublist]

        self.mesh_vertex_refer_face = tf.constant(self.nplist_v_ring_f_flat, dtype=tf.int32) # vertex_num*[2/3...8]
        self.mesh_vertex_refer_face_pad = tf.constant(self.nplist_v_ring_f_index_pad, dtype=tf.int32)  # vertex_num, max_padding
        self.mesh_vertex_refer_face_index = tf.constant(self.nplist_v_ring_f_index_flat, dtype=tf.int32) # vertex_num*[2/3...8]
        self.mesh_vertex_refer_face_num = tf.constant(self.nplist_ver_ref_face_num, dtype=tf.float32)  # vertex_num

        # tri
        self.mesh_tri = tf.constant(self.mesh_tri_np, dtype=tf.int32)

        # uv
        self.uvIdx = tf.constant(self.uvIdx_np, dtype=tf.int32)

    def _get_v_ring_f(self, v_ring_f_flat, v_ring_f_num):
        list_v_ring_f = []
        list_v_ring_f_index = []
        idx_start = 0
        for i in range(len(v_ring_f_num)):
            vf_num = v_ring_f_num[i]
            v_ring_f = v_ring_f_flat[idx_start:idx_start+vf_num]
            list_v_ring_f.append(v_ring_f)
            v_ring_f_index = np.zeros([len(v_ring_f)], dtype=np.int32) + i
            list_v_ring_f_index.append(v_ring_f_index)
            idx_start = idx_start+vf_num
        return np.array(list_v_ring_f), np.array(list_v_ring_f_index)

    def instance(self, coeff_batch, coeff_exp_batch=None):
        """
        :param coeff_batch: shape=[bs, 80]
        :param coeff_exp_batch: shape=[bs, 64]
        :return:
        """

        """
        Vertex
        """
        coeff_var_batch = coeff_batch * tf.sqrt(self.pt_pcaVariance)
        coeff_var_batch = tf.transpose(coeff_var_batch)
        #coeff_var_batch = tf.expand_dims(coeff_var_batch, -1)

        mesh_diff = tf.matmul(self.pt_pcaBasis, coeff_var_batch)
        #mesh_diff = tf.squeeze(mesh_diff, axis=-1)
        mesh_diff = tf.transpose(mesh_diff) # shape=[bs, 80]

        """
        Exp
        """
        if coeff_exp_batch is not None:
            coeff_var_batch = coeff_exp_batch * tf.sqrt(self.exp_pcaVariance)
            coeff_var_batch = tf.transpose(coeff_var_batch)
            #coeff_var_batch = tf.expand_dims(coeff_var_batch, -1)

            exp_diff = tf.matmul(self.exp_pcaBasis, coeff_var_batch)
            #exp_diff = tf.squeeze(exp_diff, axis=-1)
            exp_diff = tf.transpose(exp_diff)

            mesh = self.pt_mean + mesh_diff + exp_diff
        else:
            mesh = self.pt_mean + mesh_diff

        mesh = tf.reshape(mesh, [self.batch_size, -1, 3])
        return mesh

    def instance_color(self, coeff_batch):

        coeff_var_batch = coeff_batch * tf.sqrt(self.rgb_pcaVariance)
        coeff_var_batch = tf.transpose(coeff_var_batch)

        #coeff_var_batch = tf.Print(coeff_var_batch, [coeff_batch, coeff_var_batch], summarize=256, message='instance_color')
        #coeff_var_batch = tf.expand_dims(coeff_var_batch, -1)

        mesh_diff = tf.matmul(self.rgb_pcaBasis, coeff_var_batch)
        mesh_diff = tf.transpose(mesh_diff) # shape=[bs, 80]
        #mesh_diff = tf.squeeze(mesh_diff, axis=-1)

        mesh = self.rgb_mean + mesh_diff
        #mesh = tf.Print(mesh, [self.rgb_mean[:10], mesh_diff[:10]], summarize=256, message='mesh_color')

        mesh = tf.clip_by_value(mesh, 0.0, 1.0)

        mesh = tf.reshape(mesh, [self.batch_size, -1, 3])
        return mesh

    # np only
    def get_mesh_mean(self):
        pt_mean_3d = self.pt_mean_np.reshape(-1, 3)
        rgb_mean_3d = self.rgb_mean_np.reshape(-1, 3)

        mesh_mean = trimesh.Trimesh(
            pt_mean_3d,
            self.mesh_tri_np,
            vertex_colors=rgb_mean_3d,
            process = False
        )
        return mesh_mean


class BFM_TF():
    #
    def __init__(self, path_gpmm, rank=80, gpmm_exp_rank=64, batch_size=1, full=False):

        # 0. Read HDF5 IO
        self.path_gpmm = path_gpmm
        self.rank = rank
        self.gpmm_exp_rank = gpmm_exp_rank
        self.batch_size = batch_size

        """
        Read origin model, np only
        """
        self.h_curr = self._get_origin_model()
        if full:
            self.h_full = self._get_full_model()
        self.h_fore = self._get_fore_model()

        """
        Tri
        """
        # self.mesh_idx_fore = tf.constant(self.mesh_idx_fore_np, dtype=tf.int32) # 27660
        # self.mesh_tri_reference_fore = tf.constant(self.mesh_tri_reference_fore_np, dtype=tf.int32) # 54681, 3

        """
        LM
        """
        self.idx_lm68 = tf.constant(self.h_curr.idx_lm68_np, dtype=tf.int32)

    def _get_origin_model(self):
        return BFM_singleTopo(self.path_gpmm, name='', batch_size=self.batch_size)

    def _get_full_model(self):
        return BFM_singleTopo(self.path_gpmm, name='_full', batch_size=self.batch_size, mode_light=False)

    def _get_fore_model(self):
        return BFM_singleTopo(self.path_gpmm, name='_fore', batch_size=self.batch_size, mode_light=True)

    def instance(self, coeff_batch, coeff_exp_batch=None):
        return self.h_curr.instance(coeff_batch, coeff_exp_batch)

    def instance_color(self, coeff_batch):
        return self.h_curr.instance_color(coeff_batch)

    def instance_full(self, coeff_batch, coeff_exp_batch=None):
        return self.h_full.instance(coeff_batch, coeff_exp_batch)

    def instance_color_full(self, coeff_batch):
        return self.h_full.instance_color(coeff_batch)

    def get_lm3d_instance_vertex(self, lm_idx, points_tensor_batch):
        lm3d_batch = tf.gather(points_tensor_batch, lm_idx, axis=1)
        return lm3d_batch

    def get_lm3d_mean(self):
        pt_mean = tf.reshape(self.pt_mean, [-1, 3])
        lm3d_mean = tf.gather(pt_mean, self.idx_lm68)
        return lm3d_mean

    # np only
    def get_mesh_mean(self, mode_str):
        if mode_str == 'curr':
            return self.h_curr.get_mesh_mean()
        elif mode_str == 'fore':
            return self.h_fore.get_mesh_mean()
        elif mode_str == 'full':
            return self.h_full.get_mesh_mean()
        else:
            return None

    def get_mesh_fore_mean(self):
        pt_mean_3d = self.pt_mean_np.reshape(-1, 3)
        rgb_mean_3d = self.rgb_mean_np.reshape(-1, 3)

        mesh_mean = trimesh.Trimesh(
            pt_mean_3d[self.mesh_idx_fore_np],
            self.mesh_tri_reference_fore_np,
            vertex_colors=rgb_mean_3d[self.mesh_idx_fore_np],
            process=False
        )
        return mesh_mean

    def get_lm3d(self, vertices, idx_lm68_np=None):
        if idx_lm68_np is None:
            idx_lm = self.idx_lm68_np
        else:
            idx_lm = idx_lm68_np
        return vertices[idx_lm]

    def get_random_vertex_color_batch(self):
        coeff_shape_batch = []
        coeff_exp_batch = []
        for i in range(self.batch_size):
            coeff_shape = tf.random.normal(shape=[self.rank], mean=0, stddev=tf.sqrt(3.0))
            coeff_shape_batch.append(coeff_shape)
            exp_shape = tf.random.normal(shape=[self.gpmm_exp_rank], mean=0, stddev=tf.sqrt(3.0))
            coeff_exp_batch.append(exp_shape)
        coeff_shape_batch = tf.stack(coeff_shape_batch)
        coeff_exp_batch = tf.stack(coeff_exp_batch)
        points_tensor_batch = self.instance(coeff_shape_batch, coeff_exp_batch)

        coeff_color_batch = []
        for i in range(self.batch_size):
            coeff_color = tf.random.normal(shape=[self.rank], mean=0, stddev=tf.sqrt(3.0))
            coeff_color_batch.append(coeff_color)
        coeff_color_batch = tf.stack(coeff_color_batch)
        points_color_tensor_batch = self.instance_color(coeff_color_batch)

        # mesh_tri_shape_list = []
        # for i in range(batch):
        #     points_np = points_tensor_batch[i]
        #     tri_np = tf.transpose(self.mesh_tri_reference)
        #     points_color_np = tf.uint8(points_color_tensor_batch[i]*255)
        #
        #
        #     mesh_tri_shape = trimesh.Trimesh(
        #         points_np,
        #         tri_np,
        #         vertex_colors=points_color_np,
        #         process=False
        #     )
        #     #mesh_tri_shape.show()
        #     #mesh_tri_shape.export("/home/jx.ply")
        #     mesh_tri_shape_list.append(mesh_tri_shape)

        return points_tensor_batch, points_color_tensor_batch, coeff_shape_batch, coeff_color_batch


if __name__ == '__main__':
    path_gpmm = '/home/jshang/SHANG_Data/ThirdLib/BFM2009/bfm09_trim_exp_uv_presplit.h5'
    h_lrgp = BFM_TF(path_gpmm, 80, 2, full=True)
    tri = h_lrgp.get_mesh_mean('curr')
    #tri.show()
    tri.export("/home/jshang/SHANG_Data/ThirdLib/BFM2009/bfm09_mean.ply")
    tri = h_lrgp.get_mesh_mean('fore')
    tri.export("/home/jshang/SHANG_Data/ThirdLib/BFM2009/bfm09_mean_fore.ply")
    tri = h_lrgp.get_mesh_mean('full')
    tri.export("/home/jshang/SHANG_Data/ThirdLib/BFM2009/bfm09_mean_full.ply")

    """
    build graph
    """
    ver, ver_color, _, _ = h_lrgp.get_random_vertex_color_batch()
    ver_color = tf.cast(ver_color*255.0, dtype=tf.uint8)

    lm3d_mean = h_lrgp.get_lm3d_mean()
    lm3d_mean = tf.expand_dims(lm3d_mean, 0)
    print(lm3d_mean)


    # test normal
    from tfmatchd.face.geometry.lighting import vertex_normals_pre_split_fixtopo
    vertexNormal = vertex_normals_pre_split_fixtopo(
        ver, h_lrgp.mesh_tri_reference, h_lrgp.mesh_vertex_refer_face,
        h_lrgp.mesh_vertex_refer_face_index, h_lrgp.mesh_vertex_refer_face_num
    )

    """
    run
    """
    sv = tf.train.Supervisor()
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    with sv.managed_session(config=config) as sess:
        fetches = {
            "ver": ver,
            "ver_color": ver_color,
            "vertexNormal": vertexNormal,
            "lm3d_mean":lm3d_mean
        }
        """
        *********************************************   Start Trainning   *********************************************
        """
        results = sess.run(fetches)

        ver = results["ver"]
        ver_color = results["ver_color"]
        vertexNormal_np = results["vertexNormal"]
        lm3d_mean_np = results["lm3d_mean"]
        print(lm3d_mean_np)

    # # normal test
    # for i in range(len(vertexNormal_np)):
    #     ver_trimesh = trimesh.Trimesh(
    #         ver[i],
    #         h_lrgp.mesh_tri_reference_np,
    #         vertex_colors=ver_color[i],
    #         process=False
    #     )
    #     vn_trimesh = ver_trimesh.vertex_normals
    #     vn_tf = vertexNormal_np[i]
    #     print(vn_trimesh[180:190])
    #     print(vn_tf[180:190])
    #
    #     error = abs(vn_trimesh - vn_tf)
    #     error = np.sum(error)
    #     print(error)
