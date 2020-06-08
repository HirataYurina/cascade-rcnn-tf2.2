import tensorflow as tf


def trim_zeros(boxes, name=None):
    '''Often boxes are represented with matrices of shape [N, 4] and
    are padded with zeros. This removes zero boxes.
    
    Args
    ---
        boxes: [N, 4] matrix of boxes.
        non_zeros: [N] a 1D boolean mask identifying the rows to keep
    '''
    non_zeros = tf.cast(tf.reduce_sum(tf.abs(boxes), axis=1), tf.bool)
    boxes = tf.boolean_mask(boxes, non_zeros, name=name)
    return boxes, non_zeros


def parse_image_meta(meta):
    '''Parses a tensor that contains image attributes to its components.
    
    Args
    ---
        meta: [..., 11]

    Returns
    ---
        a dict of the parsed tensors.
    '''
    meta = meta.numpy()
    ori_shape = meta[..., 0:3]
    img_shape = meta[..., 3:6]
    pad_shape = meta[..., 6:9]
    scale = meta[..., 9]  
    flip = meta[..., 10]
    return {
        'ori_shape': ori_shape,
        'img_shape': img_shape,
        'pad_shape': pad_shape,
        'scale': scale,
        'flip': flip
    }


def calc_batch_padded_shape(meta):
    '''
    Args
    ---
        meta: [batch_size, 11]
    
    Returns
    ---
        nd.ndarray. Tuple of (height, width)
    '''
    return tf.cast(tf.reduce_max(meta[:, 6:8], axis=0), tf.int32).numpy()


def calc_img_shapes(meta):
    '''
    Args
    ---
        meta: [..., 11]
    
    Returns
    ---
        nd.ndarray. [..., (height, width)]
    '''
    return tf.cast(meta[..., 3:5], tf.int32).numpy()


def calc_pad_shapes(meta):
    '''
    Args
    ---
        meta: [..., 11]
    
    Returns
    ---
        nd.ndarray. [..., (height, width)]
    '''
    return tf.cast(meta[..., 6:8], tf.int32).numpy()


# TODO 测试这个方法
def cal_iou(anchors, gt_boxes):

    # 转化数据类型
    anchors = tf.cast(anchors, dtype=tf.float32)
    gt_boxes = tf.cast(gt_boxes, dtype=tf.float32)

    anchors = tf.expand_dims(anchors, axis=1)
    gt_boxes = tf.expand_dims(gt_boxes, axis=0)
    anchors_left = anchors[..., :2]
    anchors_right = anchors[..., 2:]
    gt_left = gt_boxes[..., :2]
    gt_right = gt_boxes[..., 2:]

    anchors_wh = anchors_right - anchors_left
    gt_wh = gt_right - gt_left

    anchors_area = anchors_wh[..., 0] * anchors_wh[..., 1]
    gt_area = gt_wh[..., 0] * gt_wh[..., 1]

    inter_min = tf.maximum(anchors_left, gt_left)
    inter_max = tf.maximum(anchors_right, gt_right)
    inter_wh = inter_max - inter_min

    # 不存在overlap
    inter_w = tf.maximum(0.0, inter_wh[..., 0])
    inter_h = tf.maximum(0.0, inter_wh[..., 1])
    inter_area = inter_w * inter_h
    iou = inter_area / (anchors_area + gt_area - inter_area)

    return iou


# TODO 测试这个方法
def box2offset(boxes, gt_boxes, target_mean, target_std):
    # boxes:(num_boxes, 4)
    # gt_boxes:(num_gt_boxes, 4)
    # num_boxes = num_gt_boxes

    # 进行数据类型转换
    boxes, gt_boxes = tf.cast(boxes, dtype=tf.float32), tf.cast(gt_boxes, dtype=tf.float32)
    target_mean, target_std = tf.constant(target_mean, dtype=tf.float32), tf.constant(target_std, dtype=tf.float32)

    center_boxes = (boxes[:, :2] + boxes[:, 2:]) / 2
    center_gt_boxes = (gt_boxes[:, :2] + gt_boxes[:, 2:]) / 2
    wh_boxes = boxes[:, 2:] - boxes[:, :2]
    wh_gt_boxes = gt_boxes[:, 2:] - boxes[:, :2]

    # (num_boxes, 2)
    xy_offset = (center_gt_boxes - center_boxes) / wh_boxes
    wh_offset = tf.math.log(wh_gt_boxes / wh_boxes)

    # (num_boxes, 4)
    offset = tf.concat([xy_offset, wh_offset], axis=-1)
    offset = (offset - target_mean) / target_std

    return offset


def offset2box(anchors, box_pred, target_mean, target_std):

    target_mean = tf.constant(target_mean, tf.float32)
    target_std = tf.constant(target_std, tf.float32)

    box_pred = (box_pred * target_std) + target_mean

    anchors_center = (anchors[..., :2] + anchors[..., 2:]) / 2
    anchors_wh = anchors[..., 2:] - anchors[..., :2]

    box_xy = box_pred[..., :2] * anchors_wh + anchors_center
    box_wh = tf.math.exp(box_pred[..., 2:]) * anchors_wh

    box_left = box_xy - box_wh / 2
    box_right = box_xy + box_wh / 2
    box_pred = tf.concat([box_left, box_right], axis=-1)

    return box_pred






