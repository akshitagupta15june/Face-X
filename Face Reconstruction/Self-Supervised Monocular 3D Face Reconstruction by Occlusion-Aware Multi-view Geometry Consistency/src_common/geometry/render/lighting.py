
import tensorflow as tf
from tensorflow.python.framework import ops


def _repeat_1d(tensor, count):

    assert tensor.get_shape().ndims == 1
    return tf.reshape(tf.tile(tensor[:, tf.newaxis], tf.convert_to_tensor([1, count])), [-1])


def _prepare_vertices_and_faces(vertices, faces):

    if isinstance(vertices, tf.Tensor) == False:
        vertices = tf.convert_to_tensor(vertices, name='vertices')
        faces = tf.convert_to_tensor(faces, name='faces')

    if faces.dtype is not tf.int32:
        assert faces.dtype is tf.int64
        faces = tf.cast(faces, tf.int32)

    return vertices, faces


def _get_face_normals(vertices, faces):

    vertices_ndim = vertices.get_shape().ndims
    v_trans_axis = [vertices_ndim - 2] + list(range(vertices_ndim - 2)) + [vertices_ndim - 1]
    vertices_by_index = tf.transpose(vertices, v_trans_axis)  # indexed by vertex-index, *, x/y/z
    vertices_by_face = tf.gather(vertices_by_index, faces)  # indexed by face-index, vertex-in-face, *, x/y/z
    normals_by_face = tf.cross(vertices_by_face[:, 1] - vertices_by_face[:, 0], vertices_by_face[:, 2] - vertices_by_face[:, 0])  # indexed by face-index, *, x/y/z
    normals_by_face /= (tf.norm(normals_by_face, axis=-1, keepdims=True) + 1.e-12)  # ditto
    return normals_by_face, vertices_by_index


def vertex_normals(vertices, faces, name=None):
    """Computes vertex normals for the given meshes.

    This function takes a batch of meshes with common topology, and calculates vertex normals for each.

    Args:
        vertices: a `Tensor` of shape [*, vertex count, 3] or [*, vertex count, 4], where * represents arbitrarily
            many leading (batch) dimensions.
        faces: an int32 `Tensor` of shape [face count, 3]; each value is an index into the first dimension of `vertices`, and
            each row defines one triangle.
        name: an optional name for the operation

    Returns:
        a `Tensor` of shape [*, vertex count, 3], which for each vertex, gives the (normalised) average of the normals of
        all faces that include that vertex
    """

    # This computes vertex normals, as the average of the normals of the faces each vertex is part of
    # vertices is indexed by *, vertex-index, x/y/z[/w]
    # faces is indexed by face-index, vertex-in-face
    # result is indexed by *, vertex-index, x/y/z

    with ops.name_scope(name, 'VertexNormals', [vertices, faces]) as scope:

        vertices, faces = _prepare_vertices_and_faces(vertices, faces)
        vertices = vertices[..., :3]  # drop the w-coordinate if present

        vertices_ndim = vertices.get_shape().ndims
        normals_by_face, vertices_by_index = _get_face_normals(vertices, faces)  # normals_by_face is indexed by face-index, *, x/y/z

        face_count = tf.shape(faces)[0]
        vbi_shape = tf.shape(vertices_by_index)
        N_extra = tf.reduce_prod(vbi_shape[1:-1])  # this is the number of 'elements' in the * dimensions

        assert vertices_ndim in {2, 3}  # ** keep it simple for now; in the general case we need a flattened outer product of ranges
        if vertices_ndim == 2:
            extra_indices = []
        else:
            extra_indices = [tf.tile(_repeat_1d(tf.range(N_extra), 3), [face_count * 3])]

        sparse_index = tf.cast(
                tf.stack(
                    [  # each element of this stack is repeated a number of times matching the things after, then tiled a number of times matching the things before, so that each has the same length
                        _repeat_1d(tf.range(face_count, dtype=tf.int32), N_extra * 9),
                        _repeat_1d(tf.reshape(faces, [-1]), N_extra * 3)
                    ] + extra_indices + [
                        tf.tile(tf.constant([0, 1, 2], dtype=tf.int32), tf.convert_to_tensor([face_count * N_extra * 3]))
                    ], axis=1
                ),
                tf.int64
            )
        sparse_value = tf.reshape(tf.tile(normals_by_face[:, tf.newaxis, ...], [1, 3] + [1] * (vertices_ndim - 1)), [-1])
        sparse_dense_shape = tf.cast(tf.concat([[face_count], vbi_shape], axis=0), tf.int64)
        normals_by_face_and_vertex = tf.SparseTensor(
            indices=sparse_index,
            values=sparse_value,
            dense_shape=sparse_dense_shape
        )  # indexed by face-index, vertex-index, *, x/y/z

        summed_normals_by_vertex = tf.sparse_reduce_sum(normals_by_face_and_vertex, axis=0)  # indexed by vertex-index, *, x/y/z
        # summed_normals_by_vertex = tf_render.Print(summed_normals_by_vertex, [summed_normals_by_vertex.shape],
        #                                     message='summed_normals_by_vertex', summarize=16)
        renormalised_normals_by_vertex = summed_normals_by_vertex / (tf.norm(summed_normals_by_vertex, axis=-1, keep_dims=True) + 1.e-12)  # ditto

        result = tf.transpose(renormalised_normals_by_vertex, range(1, vertices_ndim - 1) + [0, vertices_ndim - 1])
        result.set_shape(vertices.get_shape())
        return result


def _static_map_fn(f, elements):
    assert elements.get_shape()[0].value is not None
    return tf.stack([f(elements[index]) for index in xrange(int(elements.get_shape()[0]))])


def vertex_normals_pre_split_fixtopo(vertices, faces, ver_ref_face, ver_ref_face_index, ver_ref_face_num, name=None):
    """
    :param vertices: batch size, vertex-index, x/y/z[/w]
    :param faces: face-index, vertex-in-face, tf_render.int32
    :param ver_ref_face: vertex-index*flat
    :param ver_ref_face_index: vertex-index*flat
    :param ver_ref_face_num: vertex-index
    :param name:
    :return:
    """
    """Computes vertex normals for the given pre-split meshes.

    This function is identical to `vertex_normals`, except that it assumes each vertex is used by just one face, which
    allows a more efficient implementation.
    """

    # This is identical to vertex_normals, but assumes each vertex appears in exactly one face, e.g. due to having been
    # processed by split_vertices_by_face
    # vertices is indexed by
    # faces is indexed by
    # result is indexed by *
    with ops.name_scope(name, 'VertexNormalsPreSplit', [vertices, faces]) as scope:
        vertices_num = int(vertices.get_shape()[1])
        vertices, faces = _prepare_vertices_and_faces(vertices, faces)
        normals_by_face, _ = _get_face_normals(vertices, faces) # indexed by face-index, batch_size, x/y/z
        normals_by_face = tf.transpose(normals_by_face, perm=[1, 0, 2])

        ver_ref_face_num_tile = tf.tile(tf.expand_dims(ver_ref_face_num, -1), multiples=[1, 3])

        list_normals_by_ver = []
        for b in range(vertices.shape[0]):
            normals_by_face_b = normals_by_face[b]
            normals_by_vertex_flat_b = tf.gather(normals_by_face_b, ver_ref_face)

            nv = tf.scatter_add(
                tf.Variable(tf.zeros(shape=[vertices_num, 3]), trainable=False),
                ver_ref_face_index,
                normals_by_vertex_flat_b
            )

            nv = nv / (ver_ref_face_num_tile + 1e-6)
            nv = nv / (tf.norm(nv, axis=-1, keep_dims=True) + 1e-12) # ditto

            list_normals_by_ver.append(nv)

        normals_by_vertex = tf.stack(list_normals_by_ver)
        return normals_by_vertex


def vertex_normals_pre_split(vertices, faces, name=None, static=False):
    """Computes vertex normals for the given pre-split meshes.

    This function is identical to `vertex_normals`, except that it assumes each vertex is used by just one face, which
    allows a more efficient implementation.
    """

    # This is identical to vertex_normals, but assumes each vertex appears in exactly one face, e.g. due to having been
    # processed by split_vertices_by_face
    # vertices is indexed by *, vertex-index, x/y/z[/w]
    # faces is indexed by face-index, vertex-in-face
    # result is indexed by *, vertex-index, x/y/z

    with ops.name_scope(name, 'VertexNormalsPreSplit', [vertices, faces]) as scope:

        vertices, faces = _prepare_vertices_and_faces(vertices, faces)
        vertices = vertices[..., :3]  # drop the w-coordinate if present
        face_count = int(faces.get_shape()[0]) if static else tf.shape(faces)[0]

        normals_by_face, _ = _get_face_normals(vertices, faces)  # indexed by face-index, *, x/y/z
        normals_by_face_flat = tf.reshape(
            tf.transpose(normals_by_face, range(1, normals_by_face.get_shape().ndims - 1) + [0, normals_by_face.get_shape().ndims - 1]),
            [-1, face_count, 3]
        )  # indexed by prod(*), face-index, x/y/z

        normals_by_vertex_flat = (_static_map_fn if static else tf.map_fn)(
            lambda normals_for_iib: tf.scatter_nd(
                indices=tf.reshape(faces, [-1, 1]),
                updates=tf.reshape(tf.tile(normals_for_iib[:, tf.newaxis, :], [1, 3, 1]), [-1, 3]),
                shape=tf.shape(vertices)[-2:]
            ), normals_by_face_flat
        )
        normals_by_vertex = tf.reshape(normals_by_vertex_flat, tf.shape(vertices))

        return normals_by_vertex


def split_vertices_by_face(vertices, faces, name=None):
    """Returns a new mesh where each vertex is used by exactly one face.

    This function takes a batch of meshes with common topology as input, and also returns a batch of meshes
    with common topology. The resulting meshes have the same geometry, but each vertex is used by exactly
    one face.

    Args:
        vertices: a `Tensor` of shape [*, vertex count, 3] or [*, vertex count, 4], where * represents arbitrarily
            many leading (batch) dimensions.
        faces: an int32 `Tensor` of shape [face count, 3]; each value is an index into the first dimension of `vertices`, and
            each row defines one triangle.

    Returns:
        a tuple of two tensors `new_vertices, new_faces`, where `new_vertices` has shape [*, V, 3] or [*,  V, 4], where
        V is the new vertex count after splitting, and `new_faces` has shape [F, 3] where F is the new face count after
        splitting.
    """

    # This returns an equivalent mesh, with vertices duplicated such that there is exactly one vertex per face it is used in
    # vertices is indexed by *, vertex-index, x/y/z[/w]
    # faces is indexed by face-index, vertex-in-face
    # Ditto for results

    with ops.name_scope(name, 'SplitVerticesByFace', [vertices, faces]) as scope:

        vertices, faces = _prepare_vertices_and_faces(vertices, faces)

        vertices_shape = tf.shape(vertices)
        face_count = tf.shape(faces)[0]

        flat_vertices = tf.reshape(vertices, [-1, vertices_shape[-2], vertices_shape[-1]])
        new_flat_vertices = tf.map_fn(lambda vertices_for_iib: tf.gather(vertices_for_iib, faces), flat_vertices)
        new_vertices = tf.reshape(new_flat_vertices, tf.concat([vertices_shape[:-2], [face_count * 3, vertices_shape[-1]]], axis=0))

        new_faces = tf.reshape(tf.range(face_count * 3), [-1, 3])

        static_face_count = faces.get_shape().dims[0] if faces.get_shape().dims is not None else None
        static_new_vertex_count = static_face_count * 3 if static_face_count is not None else None
        if vertices.get_shape().dims is not None:
            new_vertices.set_shape(vertices.get_shape().dims[:-2] + [static_new_vertex_count] + vertices.get_shape().dims[-1:])
        new_faces.set_shape([static_face_count, 3])

        return new_vertices, new_faces