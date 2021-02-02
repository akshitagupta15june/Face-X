import collections
import os

import numpy as np
from PIL import Image


def read_obj_with_group(filename, swapyz=False):
  vertices = []
  colors = []
  normals = []
  texcoords = []
  faces = []
  normal_idxs = []
  texcoord_idxs = []
  materials = []
  mtl = None
  group = [[], [], []]

  material = None
  for line in open(filename, "r"):
    if line.startswith('#'):
      continue
    values = line.split()
    if not values:
      continue
    if values[0] == 'g':
      group[0].append(len(vertices))
      group[1].append(len(faces))
      g_name = ''
      for n in values[1:]:
        g_name += n + ' '
      group[2].append(g_name[:-1])
    elif values[0] == 'v':
      v = [float(x) for x in values[1:4]]
      if swapyz:
        v = v[0], v[2], v[1]
      vertices.append(v)
      if len(values) > 4:
        c = [float(x) for x in values[4:7]]
        colors.append(c)
    elif values[0] == 'vn':
      v = [float(x) for x in values[1:4]]
      if swapyz:
        v = v[0], v[2], v[1]
      normals.append(v)
    elif values[0] == 'vt':
      v = [float(x) for x in values[1:3]]
      texcoords.append(v)
    elif values[0] in ('usemtl', 'usemat'):
      material = values[1]
    elif values[0] == 'mtllib':
      mtl = [os.path.split(filename)[0], values[1]]
    elif values[0] == 'f':
      face = []
      texcoord = []
      normal = []
      for v in values[1:]:
        w = v.split('/')
        face.append(int(w[0]))
        # if len(w) >= 2 and len(w[1]) > 0:
        if len(w) >= 2 and w[1]:
          texcoord.append(int(w[1]))
        else:
          texcoord.append(0)
        # if len(w) >= 3 and len(w[2]) > 0:
        if len(w) >= 3 and w[2]:
          normal.append(int(w[2]))
        else:
          normal.append(0)
      # faces.append((face, normal, texcoord, material))
      faces.append(face)
      normal_idxs.append(normal)
      texcoord_idxs.append(texcoord)
      materials.append(material)

  dicts = {
      'groups': group,
      'vertices': vertices,
      'colors': colors,
      'faces': faces,
      'normals': normals,
      'normal_idxs': normal_idxs,
      'texcoords': texcoords,
      'texcoord_idxs': texcoord_idxs,
      'materials': materials,
      'mtl': mtl
  }
  return dicts


def read_obj(filename, uv_filename=None):
  """
    Parse raw OBJ text into vertices, vertex normals,
    vertex colors, and vertex textures.
  """

  # get text as bytes or string blob
  with open(filename, 'r') as f:
    text = f.read()

    try:
      text = text.decode('utf-8')
    except:
      pass

    text = '\n{}\n'.format(text.strip().replace('\r\n', '\n'))
    # extract vertices from raw text
    v, vn, vt, vc = _parse_vertices(text=text)

    face_tuples = _preprocess_faces(text=text)

    # geometry = {}
    while face_tuples:
      # consume the next chunk of text
      _, _, chunk = face_tuples.pop()
      face_lines = [i.split('\n', 1)[0] for i in chunk.split('\nf ')[1:]]
      joined = ' '.join(face_lines).replace('/', ' ')

      array = np.fromstring(joined, sep=' ', dtype=np.int64) - 1

      columns = len(face_lines[0].strip().replace('/', ' ').split())

      if len(array) == (columns * len(face_lines)):
        faces, faces_tex, faces_norm = _parse_faces_vectorized(
            array=array, columns=columns, sample_line=face_lines[0])
      else:
        faces, faces_tex, faces_norm = _parse_faces_fallback(face_lines)

    if uv_filename is not None:
      uv_image = Image.open(uv_filename)
      vc = uv_to_color(vt, uv_image)
      vc = vc[:, :3]

  dicts = {
      'vertices': v,
      'colors': vc,
      'faces': faces,
      'normals': vn,
      'normal_idxs': faces_norm,
      'texcoords': vt,
      'texcoord_idxs': faces_tex,
  }
  return dicts


def _parse_vertices(text):
  starts = {k: text.find('\n{} '.format(k)) for k in ['v', 'vt', 'vn']}

  # no valid values so exit early
  if not any(v >= 0 for v in starts.values()):
    return None, None, None, None

  # find the last position of each valid value
  ends = {
      k: text.find('\n',
                   text.rfind('\n{} '.format(k)) + 2 + len(k))
      for k, v in starts.items()
      if v >= 0
  }

  # take the first and last position of any vertex property
  start = min(s for s in starts.values() if s >= 0)
  end = max(e for e in ends.values() if e >= 0)
  # get the chunk of test that contains vertex data
  chunk = text[start:end].replace('+e', 'e').replace('-e', 'e')

  # get the clean-ish data from the file as python lists
  data = {
      k: [i.split('\n', 1)[0] for i in chunk.split('\n{} '.format(k))[1:]
         ] for k, v in starts.items() if v >= 0
  }

  # count the number of data values per row on a sample row
  per_row = {k: len(v[1].split()) for k, v in data.items()}

  # convert data values into numpy arrays
  result = collections.defaultdict(lambda: None)
  for k, value in data.items():
    # use joining and fromstring to get as numpy array
    array = np.fromstring(' '.join(value), sep=' ', dtype=np.float64)
    # what should our shape be
    shape = (len(value), per_row[k])
    # check shape of flat data
    if len(array) == np.product(shape):
      # we have a nice 2D array
      result[k] = array.reshape(shape)
    else:
      # try to recover with a slightly more expensive loop
      count = per_row[k]
      try:
        # try to get result through reshaping
        result[k] = np.fromstring(' '.join(i.split()[:count] for i in value),
                                  sep=' ',
                                  dtype=np.float64).reshape(shape)
      except BaseException:
        pass

  # vertices
  v = result['v']
  # vertex colors are stored next to vertices
  vc = None
  if v is not None and v.shape[1] >= 6:
    # vertex colors are stored after vertices
    v, vc = v[:, :3], v[:, 3:6]
  elif v is not None and v.shape[1] > 3:
    # we got a lot of something unknowable
    v = v[:, :3]

  # vertex texture or None
  vt = result['vt']
  if vt is not None:
    # sometimes UV coordinates come in as UVW
    vt = vt[:, :2]
  # vertex normals or None
  vn = result['vn']

  return v, vn, vt, vc


def _preprocess_faces(text, split_object=False):
  # Pre-Process Face Text
  # Rather than looking at each line in a loop we're
  # going to split lines by directives which indicate
  # a new mesh, specifically 'usemtl' and 'o' keys
  # search for materials, objects, faces, or groups
  starters = ['\nusemtl ', '\no ', '\nf ', '\ng ', '\ns ']
  f_start = len(text)
  # first index of material, object, face, group, or smoother
  for st in starters:
    search = text.find(st, 0, f_start)
    # if not contained find will return -1
    if search < 0:
      continue
    # subtract the length of the key from the position
    # to make sure it's included in the slice of text
    if search < f_start:
      f_start = search
  # index in blob of the newline after the last face
  f_end = text.find('\n', text.rfind('\nf ') + 3)
  # get the chunk of the file that has face information
  if f_end >= 0:
    # clip to the newline after the last face
    f_chunk = text[f_start:f_end]
  else:
    # no newline after last face
    f_chunk = text[f_start:]

  # start with undefined objects and material
  current_object = None
  current_material = None
  # where we're going to store result tuples
  # containing (material, object, face lines)
  face_tuples = []

  # two things cause new meshes to be created: objects and materials
  # first divide faces into groups split by material and objects
  # face chunks using different materials will be treated
  # as different meshes
  for m_chunk in f_chunk.split('\nusemtl '):
    # if empty continue
    # if len(m_chunk) == 0:
    #   continue
    if not m_chunk:
      continue

    # find the first newline in the chunk
    # everything before it will be the usemtl direction
    new_line = m_chunk.find('\n')
    # if the file contained no materials it will start with a newline
    if new_line == 0:
      current_material = None
    else:
      # remove internal double spaces because why wouldn't that be OK
      current_material = ' '.join(m_chunk[:new_line].strip().split())

    # material chunk contains multiple objects
    if split_object:
      o_split = m_chunk.split('\no ')
    else:
      o_split = [m_chunk]
    if len(o_split) > 1:
      for o_chunk in o_split:
        # set the object label
        current_object = o_chunk[:o_chunk.find('\n')].strip()
        # find the first face in the chunk
        f_idx = o_chunk.find('\nf ')
        # if we have any faces append it to our search tuple
        if f_idx >= 0:
          face_tuples.append(
              (current_material, current_object, o_chunk[f_idx:]))
    else:
      # if there are any faces in this chunk add them
      f_idx = m_chunk.find('\nf ')
      if f_idx >= 0:
        face_tuples.append((current_material, current_object, m_chunk[f_idx:]))
  return face_tuples


def _parse_faces_vectorized(array, columns, sample_line):
  """
    Parse loaded homogeneous (tri/quad) face data in a
    vectorized manner.
  """
  # reshape to columns
  array = array.reshape((-1, columns))
  # how many elements are in the first line of faces
  # i.e '13/1/13 14/1/14 2/1/2 1/2/1' is 4
  group_count = len(sample_line.strip().split())
  # how many elements are there for each vertex reference
  # i.e. '12/1/13' is 3
  per_ref = int(columns / group_count)
  # create an index mask we can use to slice vertex references
  index = np.arange(group_count) * per_ref
  # slice the faces out of the blob array
  faces = array[:, index]

  # or do something more general
  faces_tex, faces_norm = None, None
  if columns == 6:
    # if we have two values per vertex the second
    # one is index of texture coordinate (`vt`)
    # count how many delimiters are in the first face line
    # to see if our second value is texture or normals
    count = sample_line.count('/')
    if count == columns:
      # case where each face line looks like:
      # ' 75//139 76//141 77//141'
      # which is vertex/nothing/normal
      faces_norm = array[:, index + 1]
    elif count == int(columns / 2):
      # case where each face line looks like:
      # '75/139 76/141 77/141'
      # which is vertex/texture
      faces_tex = array[:, index + 1]
    # else:
    #   log.warning('face lines are weird: {}'.format(sample_line))
  elif columns == 9:
    # if we have three values per vertex
    # second value is always texture
    faces_tex = array[:, index + 1]
    # third value is reference to vertex normal (`vn`)
    faces_norm = array[:, index + 2]
  return faces, faces_tex, faces_norm


def _parse_faces_fallback(lines):
  """
    Use a slow but more flexible looping method to process
    face lines as a fallback option to faster vectorized methods.
  """

  # collect vertex, texture, and vertex normal indexes
  v, vt, vn = [], [], []

  # loop through every line starting with a face
  for line in lines:
    # remove leading newlines then
    # take first bit before newline then split by whitespace
    split = line.strip().split('\n')[0].split()
    # split into: ['76/558/76', '498/265/498', '456/267/456']
    if len(split) == 4:
      # triangulate quad face
      split = [split[0], split[1], split[2], split[2], split[3], split[0]]
    elif len(split) != 3:
      # log.warning('face has {} elements! skipping!'.format(len(split)))
      continue

    # f is like: '76/558/76'
    for f in split:
      # vertex, vertex texture, vertex normal
      split = f.split('/')
      # we always have a vertex reference
      v.append(int(split[0]))

      # faster to try/except than check in loop
      try:
        vt.append(int(split[1]))
      except BaseException:
        pass
      try:
        # vertex normal is the third index
        vn.append(int(split[2]))
      except BaseException:
        pass

  # shape into triangles and switch to 0-indexed
  faces = np.array(v, dtype=np.int64).reshape((-1, 3)) - 1
  faces_tex, normals = None, None
  if len(vt) == len(v):
    faces_tex = np.array(vt, dtype=np.int64).reshape((-1, 3)) - 1
  if len(vn) == len(v):
    normals = np.array(vn, dtype=np.int64).reshape((-1, 3)) - 1

  return faces, faces_tex, normals


def uv_to_color(uv, image):
  """
  Get the color in a texture image.

  Parameters
  -------------
  uv : (n, 2) float
    UV coordinates on texture image
  image : PIL.Image
    Texture image

  Returns
  ----------
  colors : (n, 4) float
    RGBA color at each of the UV coordinates
  """
  if image is None or uv is None:
    return None

  # UV coordinates should be (n, 2) float
  uv = np.asanyarray(uv, dtype=np.float64)

  # get texture image pixel positions of UV coordinates
  x = (uv[:, 0] * (image.width - 1))
  y = ((1 - uv[:, 1]) * (image.height - 1))

  # convert to int and wrap to image
  # size in the manner of GL_REPEAT
  x = x.round().astype(np.int64) % image.width
  y = y.round().astype(np.int64) % image.height

  # access colors from pixel locations
  # make sure image is RGBA before getting values
  colors = np.asanyarray(image.convert('RGBA'))[y, x]

  # conversion to RGBA should have corrected shape
  assert colors.ndim == 2 and colors.shape[1] == 4

  return colors


def to_float(colors):
  """
    Convert integer colors to 0.0 - 1.0 floating point colors

    Parameters
    -------------
    colors : (n, d) int
      Integer colors

    Returns
    -------------
    as_float : (n, d) float
      Float colors 0.0 - 1.0
    """

  # colors as numpy array
  colors = np.asanyarray(colors)
  if colors.dtype.kind == 'f':
    return colors
  elif colors.dtype.kind in 'iu':
    # integer value for opaque alpha given our datatype
    opaque = np.iinfo(colors.dtype).max
    return colors.astype(np.float64) / opaque
  else:
    raise ValueError('only works on int or float colors!')


def array_to_string(array,
                    col_delim=' ',
                    row_delim='\n',
                    digits=8,
                    value_format='{}'):
  """
    Convert a 1 or 2D array into a string with a specified number
    of digits and delimiter. The reason this exists is that the
    basic numpy array to string conversions are surprisingly bad.

    Parameters
    ------------
    array : (n,) or (n, d) float or int
       Data to be converted
       If shape is (n,) only column delimiter will be used
    col_delim : str
      What string should separate values in a column
    row_delim : str
      What string should separate values in a row
    digits : int
      How many digits should floating point numbers include
    value_format : str
       Format string for each value or sequence of values
       If multiple values per value_format it must divide
       into array evenly.

    Returns
    ----------
    formatted : str
       String representation of original array
    """
  # convert inputs to correct types
  array = np.asanyarray(array)
  digits = int(digits)
  row_delim = str(row_delim)
  col_delim = str(col_delim)
  value_format = str(value_format)

  # abort for non- flat arrays
  # if len(array.shape) > 2:
  #   raise ValueError('conversion only works on 1D/2D arrays not %s!',
  #                    str(array.shape))

  # allow a value to be repeated in a value format
  repeats = value_format.count('{}')

  if array.dtype.kind == 'i':
    # integer types don't need a specified precision
    format_str = value_format + col_delim
  elif array.dtype.kind == 'f':
    # add the digits formatting to floats
    format_str = value_format.replace('{}',
                                      '{:.' + str(digits) + 'f}') + col_delim
  # else:
  #   raise (ValueError('dtype %s not convertible!', array.dtype.name))

  # length of extra delimiters at the end
  end_junk = len(col_delim)
  # if we have a 2D array add a row delimiter
  if len(array.shape) == 2:
    format_str *= array.shape[1]
    # cut off the last column delimiter and add a row delimiter
    format_str = format_str[:-len(col_delim)] + row_delim
    end_junk = len(row_delim)

  # expand format string to whole array
  format_str *= len(array)

  # if an array is repeated in the value format
  # do the shaping here so we don't need to specify indexes
  shaped = np.tile(array.reshape((-1, 1)), (1, repeats)).reshape(-1)

  # run the format operation and remove the extra delimiters
  formatted = format_str.format(*shaped)[:-end_junk]

  return formatted


def read_obj_bak(filename, swapyz=False):
  vertices = []
  colors = []
  normals = []
  texcoords = []
  faces = []
  normal_idxs = []
  texcoord_idxs = []
  materials = []
  mtl = None

  material = None
  for line in open(filename, "r"):
    if line.startswith('#'):
      continue
    values = line.split()
    if not values:
      continue
    if values[0] == 'v':
      v = [float(x) for x in values[1:4]]
      if swapyz:
        v = v[0], v[2], v[1]
      vertices.append(v)
      if len(values) > 4:
        c = [float(x) for x in values[4:7]]
        colors.append(c)
    elif values[0] == 'vn':
      v = [float(x) for x in values[1:4]]
      if swapyz:
        v = v[0], v[2], v[1]
      normals.append(v)
    elif values[0] == 'vt':
      v = [float(x) for x in values[1:3]]
      texcoords.append(v)
    elif values[0] in ('usemtl', 'usemat'):
      material = values[1]
    elif values[0] == 'mtllib':
      mtl = [os.path.split(filename)[0], values[1]]
    elif values[0] == 'f':
      face = []
      texcoord = []
      normal = []
      for v in values[1:]:
        w = v.split('/')
        face.append(int(w[0]))
        # if len(w) >= 2 and len(w[1]) > 0:
        if len(w) >= 2 and w[1]:
          texcoord.append(int(w[1]))
        else:
          texcoord.append(0)
        # if len(w) >= 3 and len(w[2]) > 0:
        if len(w) >= 3 and w[2]:
          normal.append(int(w[2]))
        else:
          normal.append(0)
      # faces.append((face, normal, texcoord, material))
      faces.append(face)
      normal_idxs.append(normal)
      texcoord_idxs.append(texcoord)
      materials.append(material)

  dicts = {
      'vertices': vertices,
      'colors': colors,
      'faces': faces,
      'normals': normals,
      'normal_idxs': normal_idxs,
      'texcoords': texcoords,
      'texcoord_idxs': texcoord_idxs,
      'materials': materials,
      'mtl': mtl
  }
  return dicts


def write_obj(obj_name, vertices, triangles=None, colors=None, normals=None):
  ''' Save 3D face model with texture represented by colors.
    Args:
        obj_name: str
        vertices: shape = (nver, 3)
        triangles: shape = (ntri, 3)
        colors: shape = (nver, 3)
  '''
  try:
    if not os.path.isdir(os.path.split(obj_name)[0]):
      os.makedirs(os.path.split(obj_name)[0])
  except FileNotFoundError:
    pass

  if triangles is not None:
    triangles = triangles.copy()
    triangles += 1  # meshlab start with 1

  if obj_name.split('.')[-1] != 'obj':
    obj_name = obj_name + '.obj'

  with open(obj_name, 'w') as f:
    if colors is None:
      for v in vertices:
        s = 'v {} {} {}\n'.format(v[0], v[1], v[2])
        f.write(s)
    else:
      for v, c in zip(vertices, colors):
        s = 'v {} {} {} {} {} {}\n'.format(v[0], v[1], v[2], int(c[0]),
                                           int(c[1]), int(c[2]))
        f.write(s)

    if normals is not None:
      for vn in normals:
        s = 'vn {} {} {}\n'.format(vn[0], vn[1], vn[2])
        f.write(s)

    if triangles is not None:
      for t in triangles:
        s = 'f {} {} {}\n'.format(t[0], t[1], t[2])
        f.write(s)
