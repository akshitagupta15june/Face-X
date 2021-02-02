#pylint: disable=len-as-condition
import heapq
import math
import os
from random import choice

import numpy as np
import scipy.sparse as sp

import utils
from lib.mesh_io import write_obj


def vertex_quadrics(vertice, triangle):
  """Computes a quadric for each vertex in the Mesh.

    Returns:
       v_quadrics: an (N x 4 x 4) array, where N is # vertices.
    """

  # Allocate quadrics
  v_quadrics = np.zeros((len(vertice), 4, 4))

  # For each face...
  for _, tri in enumerate(triangle):
    # Compute normalized plane equation for that face
    vert_idxs = tri
    verts = np.hstack((vertice[vert_idxs], np.array([1, 1, 1]).reshape(-1, 1)))
    _, _, v = np.linalg.svd(verts)
    eq = v[-1, :].reshape(-1, 1)
    eq = eq / (np.linalg.norm(eq[0:3]))

    # Add the outer product of the plane equation to the
    # quadrics of the vertices for this face
    for k in range(3):
      v_quadrics[tri[k], :, :] += np.outer(eq, eq)

  return v_quadrics


def setup_deformation_transfer(src_vert, src_tri, tgt_vert):
  rows = np.zeros(3 * tgt_vert.shape[0])
  cols = np.zeros(3 * tgt_vert.shape[0])
  coeffs_v = np.zeros(3 * tgt_vert.shape[0])
  # coeffs_n = np.zeros(3 * tgt_vert.shape[0])

  # nearest_faces, nearest_parts, nearest_vertices = source.compute_aabb_tree(
  # ).nearest(tgt_vert, True)
  nearest_faces, nearest_parts, nearest_vertices = utils.aabbtree_compute_nearest(
      src_vert, src_tri, tgt_vert, True)
  nearest_faces = nearest_faces.ravel().astype(np.int64)
  nearest_parts = nearest_parts.ravel().astype(np.int64)
  nearest_vertices = nearest_vertices.ravel()

  for i in range(tgt_vert.shape[0]):
    # Closest triangle index
    f_id = nearest_faces[i]
    # Closest triangle vertex ids
    nearest_f = src_tri[f_id]

    # Closest surface point
    nearest_v = nearest_vertices[3 * i:3 * i + 3]
    # Distance vector to the closest surface point
    # dist_vec = tgt_vert[i] - nearest_v

    rows[3 * i:3 * i + 3] = i * np.ones(3)
    cols[3 * i:3 * i + 3] = nearest_f

    n_id = nearest_parts[i]
    if n_id == 0:
      # Closest surface point in triangle
      A = np.vstack((src_vert[nearest_f])).T
      coeffs_v[3 * i:3 * i + 3] = np.linalg.lstsq(A, nearest_v, rcond=None)[0]
    elif 0 < n_id <= 3:
      # Closest surface point on edge
      A = np.vstack(
          (src_vert[nearest_f[n_id - 1]], src_vert[nearest_f[n_id % 3]])).T
      tmp_coeffs = np.linalg.lstsq(A, tgt_vert[i], rcond=None)[0]
      coeffs_v[3 * i + n_id - 1] = tmp_coeffs[0]
      coeffs_v[3 * i + n_id % 3] = tmp_coeffs[1]
    else:
      # Closest surface point a vertex
      coeffs_v[3 * i + n_id - 4] = 1.0

    # if use_normals:
    #   A = np.vstack((vn[nearest_f])).T
    #   coeffs_n[3 * i:3 * i + 3] = np.linalg.lstsq(A, dist_vec)[0]

  #coeffs = np.hstack((coeffs_v, coeffs_n))
  #rows = np.hstack((rows, rows))
  #cols = np.hstack((cols, source.v.shape[0] + cols))
  matrix = sp.csc_matrix((coeffs_v, (rows, cols)),
                         shape=(tgt_vert.shape[0], src_vert.shape[0]))
  return matrix


def qslim_decimator_transformer(vertice,
                                triangle,
                                factor=None,
                                n_verts_desired=None):
  """Return a simplified version of this mesh.

    A Qslim-style approach is used here.

    :param factor: fraction of the original vertices to retain
    :param n_verts_desired: number of the original vertices to retain
    :returns: new_faces: An Fx3 array of faces, mtx: Transformation matrix
    """

  if factor is None and n_verts_desired is None:
    raise Exception('Need either factor or n_verts_desired.')

  if n_verts_desired is None:
    n_verts_desired = math.ceil(len(vertice) * factor) * 1.0

  Qv = vertex_quadrics(vertice, triangle)

  # fill out a sparse matrix indicating vertex-vertex adjacency
  # from psbody.mesh.topology.connectivity import get_vertices_per_edge
  vert_adj = utils.get_vertices_per_edge(vertice, triangle)
  # vert_adj = sp.lil_matrix((len(vertice), len(vertice)))
  # for f_idx in range(len(triangle)):
  #     vert_adj[triangle[f_idx], triangle[f_idx]] = 1

  vert_adj = sp.csc_matrix(
      (vert_adj[:, 0] * 0 + 1, (vert_adj[:, 0], vert_adj[:, 1])),
      shape=(len(vertice), len(vertice)))
  vert_adj = vert_adj + vert_adj.T
  vert_adj = vert_adj.tocoo()

  def collapse_cost(Qv, r, c, v):
    Qsum = Qv[r, :, :] + Qv[c, :, :]
    p1 = np.vstack((v[r].reshape(-1, 1), np.array([1]).reshape(-1, 1)))
    p2 = np.vstack((v[c].reshape(-1, 1), np.array([1]).reshape(-1, 1)))

    destroy_c_cost = p1.T.dot(Qsum).dot(p1)
    destroy_r_cost = p2.T.dot(Qsum).dot(p2)
    result = {
        'destroy_c_cost': destroy_c_cost,
        'destroy_r_cost': destroy_r_cost,
        'collapse_cost': min([destroy_c_cost, destroy_r_cost]),
        'Qsum': Qsum
    }
    return result

  # construct a queue of edges with costs
  queue = []
  for k in range(vert_adj.nnz):
    r = vert_adj.row[k]
    c = vert_adj.col[k]

    if r > c:
      continue

    cost = collapse_cost(Qv, r, c, vertice)['collapse_cost']
    heapq.heappush(queue, (cost, (r, c)))

  # decimate
  collapse_list = []
  nverts_total = len(vertice)
  faces = triangle.copy()
  while nverts_total > n_verts_desired:
    e = heapq.heappop(queue)
    r = e[1][0]
    c = e[1][1]
    if r == c:
      continue

    cost = collapse_cost(Qv, r, c, vertice)
    if cost['collapse_cost'] > e[0]:
      heapq.heappush(queue, (cost['collapse_cost'], e[1]))
      # print 'found outdated cost, %.2f < %.2f' % (e[0], cost['collapse_cost'])
      continue
    else:

      # update old vert idxs to new one,
      # in queue and in face list
      if cost['destroy_c_cost'] < cost['destroy_r_cost']:
        to_destroy = c
        to_keep = r
      else:
        to_destroy = r
        to_keep = c

      collapse_list.append([to_keep, to_destroy])

      # in our face array, replace "to_destroy" vertidx with "to_keep" vertidx
      np.place(faces, faces == to_destroy, to_keep)

      # same for queue
      which1 = [
          idx for idx in range(len(queue)) if queue[idx][1][0] == to_destroy
      ]
      which2 = [
          idx for idx in range(len(queue)) if queue[idx][1][1] == to_destroy
      ]
      for k in which1:
        queue[k] = (queue[k][0], (to_keep, queue[k][1][1]))
      for k in which2:
        queue[k] = (queue[k][0], (queue[k][1][0], to_keep))

      Qv[r, :, :] = cost['Qsum']
      Qv[c, :, :] = cost['Qsum']

      a = faces[:, 0] == faces[:, 1]
      b = faces[:, 1] == faces[:, 2]
      c = faces[:, 2] == faces[:, 0]

      # remove degenerate faces
      def logical_or3(x, y, z):
        return np.logical_or(x, np.logical_or(y, z))

      faces_to_keep = np.logical_not(logical_or3(a, b, c))
      faces = faces[faces_to_keep, :].copy()

    nverts_total = (len(np.unique(faces.flatten())))

  new_faces, mtx = _get_sparse_transform(faces, len(vertice))
  return new_faces, mtx


def _get_sparse_transform(faces, num_original_verts):
  verts_left = np.unique(faces.flatten())
  IS = np.arange(len(verts_left))
  JS = verts_left
  data = np.ones(len(JS))

  mp = np.arange(0, np.max(faces.flatten()) + 1)
  mp[JS] = IS
  new_faces = mp[faces.copy().flatten()].reshape((-1, 3))

  ij = np.vstack((IS.flatten(), JS.flatten()))
  mtx = sp.csc_matrix((data, ij), shape=(len(verts_left), num_original_verts))

  return (new_faces, mtx)


def generate_transform_matrices(name, refer_vertices, refer_triangles, factors):
  """Generates len(factors) meshes, each of them is scaled by factors[i] and
       computes the transformations between them.

    Returns:
       M: a set of meshes downsampled from mesh by a factor specified in factors.
       A: Adjacency matrix for each of the meshes
       D: Downsampling transforms between each of the meshes
       U: Upsampling transforms between each of the meshes
    """

  factors = [1.0 / x for x in factors]
  # M, A, D, U = [], [], [], []
  # V, T, A, D, U = [], [], [], [], []
  vertices = []
  triangles = []
  adjacencies = []
  downsamp_trans = []
  upsamp_trans = []
  adjacencies.append(
      utils.get_vert_connectivity(refer_vertices, refer_triangles))
  # M.append(mesh)
  vertices.append(refer_vertices)
  triangles.append(refer_triangles)

  for factor in factors:
    ds_triangle, ds_transform = qslim_decimator_transformer(vertices[-1],
                                                            triangles[-1],
                                                            factor=factor)
    downsamp_trans.append(ds_transform)
    # new_mesh_v = ds_D.dot(M[-1].v)
    ds_vertice = ds_transform.dot(vertices[-1])
    # new_mesh = Mesh(v=new_mesh_v, f=ds_f)
    # M.append(new_mesh)
    vertices.append(ds_vertice)
    triangles.append(ds_triangle)
    adjacencies.append(utils.get_vert_connectivity(ds_vertice, ds_triangle))
    # U.append(setup_deformation_transfer(M[-1], M[-2]))
    upsamp_trans.append(
        setup_deformation_transfer(vertices[-1], triangles[-1], vertices[-2]))

  for i, (vertice, triangle) in enumerate(zip(vertices, triangles)):
    write_obj(
        os.path.join('data', 'reference', name, 'reference{}.obj'.format(i)),
        vertice, triangle)

  return adjacencies, downsamp_trans, upsamp_trans


def generate_spirals(
    step_sizes,
    M,
    Adj,
    Trigs,
    reference_points,
    dilation=None,
    random=False,
    #  meshpackage='mpi-mesh',
    counter_clockwise=True,
    nb_stds=2):
  Adj_spirals = []
  for i, _ in enumerate(Adj):
    mesh_vertices = M[i]['vertices']

    spiral = get_spirals(
        mesh_vertices,
        Adj[i],
        Trigs[i],
        reference_points[i],
        n_steps=step_sizes[i],
        #  padding='zero',
        counter_clockwise=counter_clockwise,
        random=random)
    Adj_spirals.append(spiral)
    print('spiral generation for hierarchy %d (%d vertices) finished' %
          (i, len(Adj_spirals[-1])))

  ## Dilated convolution
  if dilation:
    for i, _ in enumerate(dilation):
      dil = dilation[i]
      dil_spirals = []
      for j, _ in enumerate(Adj_spirals[i]):
        s = Adj_spirals[i][j][:1] + Adj_spirals[i][j][1::dil]
        dil_spirals.append(s)
      Adj_spirals[i] = dil_spirals

  # Calculate the lengths of spirals
  #   Use mean + 2 * std_dev, to capture 97% of data
  L = []
  for i, _ in enumerate(Adj_spirals):
    L.append([])
    for j, _ in enumerate(Adj_spirals[i]):
      L[i].append(len(Adj_spirals[i][j]))
    L[i] = np.array(L[i])
  spiral_sizes = []
  for i, _ in enumerate(L):
    sz = L[i].mean() + nb_stds * L[i].std()
    spiral_sizes.append(int(sz))
    print('spiral sizes for hierarchy %d:  %d' % (i, spiral_sizes[-1]))

  # 1) fill with -1 (index to the dummy vertex, i.e the zero padding) the spirals with length smaller than the chosen one
  # 2) Truncate larger spirals
  spirals_np = []
  for i, _ in enumerate(spiral_sizes):  #len(Adj_spirals)):
    S = np.zeros((1, len(Adj_spirals[i]) + 1, spiral_sizes[i])) - 1
    for j, _ in enumerate(Adj_spirals[i]):
      S[0, j, :len(Adj_spirals[i][j])] = Adj_spirals[i][j][:spiral_sizes[i]]
    #spirals_np.append(np.repeat(S,args['batch_size'],axis=0))
    spirals_np.append(S)

  return spirals_np, spiral_sizes, Adj_spirals


def get_spirals(
    mesh,
    adj,
    trig,
    reference_points,
    n_steps=1,
    # padding='zero',
    counter_clockwise=True,
    random=False):
  spirals = []

  if not random:
    heat_path = None
    dist = None
    for reference_point in reference_points:
      heat_path, dist = single_source_shortest_path(mesh, adj, reference_point,
                                                    dist, heat_path)
    heat_source = reference_points

  for i in range(mesh.shape[0]):
    seen = set()
    seen.add(i)
    trig_central = list(trig[i])
    A = adj[i]
    spiral = [i]

    # 1) Frist degree of freedom - choose starting pooint:
    if not random:
      if i in heat_source:  # choose closest neighbor
        shortest_dist = np.inf
        init_vert = None
        for neighbor in A:
          d = np.sum(np.square(mesh[i] - mesh[neighbor]))
          if d < shortest_dist:
            shortest_dist = d
            init_vert = neighbor

      else:  #   on the shortest path to the reference point
        init_vert = heat_path[i]
    else:
      # choose starting point:
      #   random for first ring
      init_vert = choice(A)

    # first ring
    if init_vert is not None:
      ring = [init_vert]
      seen.add(init_vert)
    else:
      ring = []
    while len(trig_central) > 0 and init_vert is not None:
      cur_v = ring[-1]
      cur_t = [t for t in trig_central if t in trig[cur_v]]
      if len(ring) == 1:
        orientation_0 = (cur_t[0][0] == i and cur_t[0][1] == cur_v)\
                     or (cur_t[0][1] == i and cur_t[0][2] == cur_v)\
                     or (cur_t[0][2] == i and cur_t[0][0] == cur_v)
        if not counter_clockwise:
          orientation_0 = not orientation_0

        # 2) Second degree of freedom - 2nd point/orientation ambiguity
        if len(cur_t) >= 2:
          # Choose the triangle that will direct the spiral counter-clockwise
          if orientation_0:
            # Third point in the triangle - next vertex in the spiral
            third = [p for p in cur_t[0] if p != i and p != cur_v][0]
            trig_central.remove(cur_t[0])
          else:
            third = [p for p in cur_t[1] if p != i and p != cur_v][0]
            trig_central.remove(cur_t[1])
          ring.append(third)
          seen.add(third)
        # 3) Stop if the spiral hits the boundary in the first point
        elif len(cur_t) == 1:
          break
      else:
        # 4) Unique ordering for the rest of the points (3rd onwards)
        if len(cur_t) >= 1:
          # Third point in the triangle - next vertex in the spiral
          third = [p for p in cur_t[0] if p != cur_v and p != i][0]
          # Don't append the spiral if the vertex has been visited already
          # (happens when the first ring is completed and the spiral returns to the central vertex)
          if third not in seen:
            ring.append(third)
            seen.add(third)
          trig_central.remove(cur_t[0])
      # 4) Stop when the spiral hits the boundary (the already visited triangle is no longer in the list): First half of the spiral
        elif len(cur_t) == 0:
          break

    rev_i = len(ring)
    if init_vert is not None:
      v = init_vert

      if orientation_0 and len(ring) == 1:
        reverse_order = False
      else:
        reverse_order = True
    need_padding = False

    # 5) If on the boundary: restart from the initial vertex towards the other direction,
    # but put the vertices in reverse order: Second half of the spiral
    # One exception if the starting point is on the boundary +  2nd point towards the desired direction
    while len(trig_central) > 0 and init_vert is not None:
      cur_t = [t for t in trig_central if t in trig[v]]
      if len(cur_t) != 1:
        break
      else:
        need_padding = True

      third = [p for p in cur_t[0] if p != v and p != i][0]
      trig_central.remove(cur_t[0])
      if third not in seen:
        ring.insert(rev_i, third)
        seen.add(third)
        if not reverse_order:
          rev_i = len(ring)
        v = third

    # Add a dummy vertex between the first half of the spiral and the second half - similar to zero padding in a 2d grid
    if need_padding:
      ring.insert(rev_i, -1)
      """
            ring_copy = list(ring[1:])
            rev_i = rev_i - 1
            for z in range(len(ring_copy)-2):
                if padding == 'zero':
                    ring.insert(rev_i,-1) # -1 is our sink node
                elif padding == 'mirror':
                    ring.insert(rev_i,ring_copy[rev_i-z-1])
            """
    spiral += ring

    # Next rings:
    for _ in range(n_steps - 1):
      next_ring = set([])
      next_trigs = set([])
      if len(ring) == 0:
        break
      base_triangle = None
      init_vert = None

      # Find next hop neighbors
      for w in ring:
        if w != -1:
          for u in adj[w]:
            if u not in seen:
              next_ring.add(u)

      # Find triangles that contain two outer ring nodes. That way one can folllow the spiral ordering in the same way
      # as done in the first ring: by simply discarding the already visited triangles+nodes.
      for u in next_ring:
        for tr in trig[u]:
          if len([x for x in tr if x in seen]) == 1:
            next_trigs.add(tr)
          elif ring[0] in tr and ring[-1] in tr:
            base_triangle = tr
      # Normal case: starting point in the second ring ->
      # the 3rd point in the triangle that connects the 1st and the last point in the 1st ring with the 2nd ring
      if base_triangle is not None:
        init_vert = [x for x in base_triangle if x != ring[0] and x != ring[-1]]
        # Make sure that the the initial point is appropriate for starting the spiral,
        # i.e it is connected to at least one of the next candidate vertices
        if len(list(next_trigs.intersection(set(trig[init_vert[0]])))) == 0:
          init_vert = None

      # If no such triangle exists (one of the vertices is dummy,
      # or both the first and the last vertex take part in a specific type of boundary)
      # or the init vertex is not connected with the rest of the ring -->
      # Find the relative point in the the triangle that connects the 1st point with the 2nd, or the 2nd with the 3rd
      # and so on and so forth. Note: This is a slight abuse of the spiral topology
      if init_vert is None:
        for r in range(len(ring) - 1):
          if ring[r] != -1 and ring[r + 1] != -1:
            tr = [t for t in trig[ring[r]] if t in trig[ring[r + 1]]]
            for t in tr:
              init_vert = [v for v in t if v not in seen]
              # make sure that the next vertex is appropriate to start the spiral ordering in the next ring
              if len(init_vert) > 0 and len(
                  list(next_trigs.intersection(set(trig[init_vert[0]])))) > 0:
                break
              else:
                init_vert = []
            if len(init_vert) > 0 and len(
                list(next_trigs.intersection(set(trig[init_vert[0]])))) > 0:
              break
            else:
              init_vert = []

      # The rest of the procedure is the same as the first ring
      if init_vert is None:
        init_vert = []
      if len(init_vert) > 0:
        init_vert = init_vert[0]
        ring = [init_vert]
        seen.add(init_vert)
      else:
        init_vert = None
        ring = []

      # if i == 57:
      #     import pdb;pdb.set_trace()
      while len(next_trigs) > 0 and init_vert is not None:
        cur_v = ring[-1]
        cur_t = list(next_trigs.intersection(set(trig[cur_v])))

        if len(ring) == 1:
          try:
            orientation_0 = (cur_t[0][0] in seen and cur_t[0][1] == cur_v) \
                            or (cur_t[0][1] in seen and cur_t[0][2] == cur_v) \
                            or (cur_t[0][2] in seen and cur_t[0][0] == cur_v)
          except:
            import pdb
            pdb.set_trace()
          if not counter_clockwise:
            orientation_0 = not orientation_0

          # 1) orientation ambiguity for the next ring
          if len(cur_t) >= 2:
            # Choose the triangle that will direct the spiral counter-clockwise
            if orientation_0:
              # Third point in the triangle - next vertex in the spiral
              third = [p for p in cur_t[0] if p not in seen and p != cur_v][0]
              next_trigs.remove(cur_t[0])
            else:
              third = [p for p in cur_t[1] if p not in seen and p != cur_v][0]
              next_trigs.remove(cur_t[1])
            ring.append(third)
            seen.add(third)
          # 2) Stop if the spiral hits the boundary in the first point
          elif len(cur_t) == 1:
            break
        else:
          # 3) Unique ordering for the rest of the points
          if len(cur_t) >= 1:
            third = [p for p in cur_t[0] if p != v and p not in seen]
            next_trigs.remove(cur_t[0])
            if len(third) > 0:
              third = third[0]
              if third not in seen:
                ring.append(third)
                seen.add(third)
            else:
              break
          # 4) Stop when the spiral hits the boundary
          # (the already visited triangle is no longer in the list): First half of the spiral
          elif len(cur_t) == 0:
            break

      rev_i = len(ring)
      if init_vert is not None:
        v = init_vert

        if orientation_0 and len(ring) == 1:
          reverse_order = False
        else:
          reverse_order = True

      need_padding = False

      while len(next_trigs) > 0 and init_vert is not None:
        cur_t = [t for t in next_trigs if t in trig[v]]
        if len(cur_t) != 1:
          break
        else:
          need_padding = True

        third = [p for p in cur_t[0] if p != v and p not in seen]
        next_trigs.remove(cur_t[0])
        if len(third) > 0:
          third = third[0]
          if third not in seen:
            ring.insert(rev_i, third)
            seen.add(third)
          if not reverse_order:
            rev_i = len(ring)
          v = third

      if need_padding:
        ring.insert(rev_i, -1)
        """
                ring_copy = list(ring[1:])
                rev_i = rev_i - 1
                for z in range(len(ring_copy)-2):
                    if padding == 'zero':
                        ring.insert(rev_i,-1) # -1 is our sink node
                    elif padding == 'mirror':
                        ring.insert(rev_i,ring_copy[rev_i-z-1])
                """

      spiral += ring

    spirals.append(spiral)
  return spirals


def distance(v, w):
  return np.sqrt(np.sum(np.square(v - w)))


def single_source_shortest_path(V, E, source, dist=None, prev=None):
  if dist is None:
    dist = [None for i in range(len(V))]
    prev = [None for i in range(len(V))]
  q = []
  seen = set()
  heapq.heappush(q, (0, source, None))
  while len(q) > 0 and len(seen) < len(V):
    d_, v, p = heapq.heappop(q)
    if v in seen:
      continue
    seen.add(v)
    prev[v] = p
    dist[v] = d_
    for w in E[v]:
      if w in seen:
        continue
      dw = d_ + distance(V[v], V[w])
      heapq.heappush(q, (dw, w, v))

  return prev, dist
