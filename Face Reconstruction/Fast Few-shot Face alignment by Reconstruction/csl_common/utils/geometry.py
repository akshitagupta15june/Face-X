import numpy as np
import math


def bboxRelOverlap(bbox1, bbox2):
    '''
    Compute ratio of intersection of bounding boxes over ratio of their union.

    @param[in]    bbox1   - [center_x, center_y, bbox_wid/2, bbox_ht/2]
    @param[in]    bbox2   - [center_x, center_y, bbox_wid/2, bbox_ht/2]

    @return    overlap - intersection(bbox1,bbox2) / union(bbox1,bbox2)
    '''
    bbox1_area = 4 * bbox1[2] * bbox1[3]
    bbox2_area = 4 * bbox2[2] * bbox2[3]
    intersect_x_edge = min(bbox1[0] + bbox1[2], bbox2[0] + bbox2[2]) - max(bbox1[0] - bbox1[2], bbox2[0] - bbox2[2])
    intersect_y_edge = min(bbox1[1] + bbox1[3], bbox2[1] + bbox2[3]) - max(bbox1[1] - bbox1[3], bbox2[1] - bbox2[3])
    if intersect_x_edge <= 0 or intersect_y_edge <= 0:
        return 0
    intersection_area = intersect_x_edge * intersect_y_edge
    union_area = bbox1_area + bbox2_area - intersection_area
    return intersection_area / float(union_area)


def bboxRelOverlap2(bbox1, bbox2):
    '''
    Compute ratio of intersection of bounding boxes over ratio of their union.

    @param[in]    bbox1   - [x1,y1,x2,y2]
    @param[in]    bbox2   - [x1,y1,x2,y2]

    @return    overlap - intersection(bbox1,bbox2) / union(bbox1,bbox2)
    '''
    bbox1_area = (bbox1[2] - bbox1[0]) * (bbox1[3] - bbox1[1])
    bbox2_area = (bbox2[2] - bbox2[0]) * (bbox2[3] - bbox2[1])
    intersect_x_edge = min(bbox1[2], bbox2[2]) - max(bbox1[0], bbox2[0])
    intersect_y_edge = min(bbox1[3], bbox2[3]) - max(bbox1[1], bbox2[1])
    if intersect_x_edge <= 0 or intersect_y_edge <= 0:
        return 0
    intersection_area = intersect_x_edge * intersect_y_edge
    union_area = bbox1_area + bbox2_area - intersection_area
    return intersection_area / float(union_area)


def bboxIntersection(bbox1, bbox2, typeBB):
    if typeBB == 1:
        bb1 = convertBB1to2raw(bbox1)
        bb2 = convertBB1to2raw(bbox2)
    else:
        bb1 = np.array(bbox1)
        bb2 = np.array(bbox2)
    bbIntersect = np.hstack((np.maximum(bb1[:2], bb2[:2]), np.minimum(bb1[2:4], bb2[2:4])))
    if typeBB == 1:
        bbIntersect = convertBB2to1(bbIntersect)
    return bbIntersect


def bboxUnion(bbox1, bbox2, typeBB):
    if typeBB == 1:
        bb1 = convertBB1to2raw(bbox1)
        bb2 = convertBB1to2raw(bbox2)
    else:
        bb1 = np.array(bbox1)
        bb2 = np.array(bbox2)
    bbUnion = np.hstack((np.minimum(bb1[:2], bb2[:2]), np.maximum(bb1[2:4], bb2[2:4])))
    if typeBB == 1:
        bbUnion = convertBB2to1(bbUnion)
    return bbUnion


def enlargeBB(bb, enlargX, enlargY, typeBB):
    '''
    Enlarge the bounding box of a certain percentage on X and Y.

    @param[in] bb - input bbox
    @param[in] percX - enlargement factor on X (between 0 and 1)
    @param[in] percY - enlargement factor on Y (between 0 and 1)
    @param[in] typeBB - type of the bounding box.

    @return bbEnlarged - enlarged bbox (same type as the input)
    '''
    if typeBB == 2:
        bb = convertBB2to1(bb)

    bbEnlarged = [bb[0], bb[1], bb[2] + bb[2] * enlargX * 2, bb[3] + bb[3] * enlargY * 2]

    if typeBB == 2:
        bbEnlarged = convertBB1to2raw(bbEnlarged)

    return bbEnlarged


def scaleBB(bb, fx, fy, typeBB):
    if typeBB == 2:
        bb = convertBB2to1(bb)

    bbEnlarged = [bb[0], bb[1], bb[2] * fx, bb[3] * fy]

    if typeBB == 2:
        bbEnlarged = convertBB1to2raw(bbEnlarged)

    return bbEnlarged


def convertBB1to2raw(bb):
    '''
    convert from [mid_x,mid_y,w/2,h/2]
    to [x1,y1,x2,y2]
    This version does no clipping and no casting to 'int'.
    '''
    if isinstance(bb, np.ndarray):
        bounding_box = bb.copy()
    else:
        bounding_box = np.array(bb)

    bounding_box[:2] -= bounding_box[2:4]
    bounding_box[2:4] = 2 * bounding_box[2:4] + bounding_box[:2]
    return bounding_box


def convertBB1to2(bb, clip=True):
    '''
    convert from [mid_x,mid_y,w/2,h/2]
    to [x1,y1,x2,y2],clipping if required
    '''
    if isinstance(bb, np.ndarray):
        bounding_box = bb.copy()
    else:
        bounding_box = np.array(bb)

        #
    # bounding_box = bounding_box.round()#prevent problems with aspect ratio change
    bounding_box[:2] -= bounding_box[2:4]
    bounding_box[2:4] = 2 * bounding_box[2:4] + bounding_box[:2]
    # bounding_box = np.array([round(b) for b in bounding_box]).astype('int32')
    bounding_box = np.round(bounding_box).astype('int32')
    if clip:
        bounding_box = np.clip(bounding_box, 0, 10000)
    return bounding_box


def convertBB2to1(bb):
    '''
    convert from[x1,y1,x2,y2]
    to [mid_x,mid_y,w/2,h/2]
    '''
    if isinstance(bb, np.ndarray):
        bounding_box = bb.astype(np.float32)
    else:
        bounding_box = np.array(bb, np.float32)
    bounding_box[2:] = (bounding_box[2:] - bounding_box[:2]) * 0.5
    bounding_box[:2] = bounding_box[:2] + bounding_box[2:]
    return bounding_box


def extend_bbox(bbox, dl=0, dt=0, dr=0, db=0):
    '''
    Move bounding box sides by fractions of width/height. Positive values enlarge bbox for all sided.
    e.g. Enlarge height bei 10 percent by moving top:
    extend_bbox(bbox, dt=0.1) -> top_new = top - 0.1 * height
    '''
    l, t, r, b = bbox

    if t > b:
        t, b = b, t
    if l > r:
        l, r = r, l
    h = b - t
    w = r - l
    assert h >= 0
    assert w >= 0

    t_new, b_new = int(t - dt * h), int(b + db * h)
    l_new, r_new = int(l - dl * w), int(r + dr * w)

    return np.array([l_new, t_new, r_new, b_new])


def get_diagonal(h):
    return math.ceil(h * 2**0.5)