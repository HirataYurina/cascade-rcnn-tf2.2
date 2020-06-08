# -*- coding:utf-8 -*-
# author:平手友梨奈ii
# e-mail:1353593259@qq.com
# datetime:1993/12/01
# filename:anchor_assign.py
# software: PyCharm

import tensorflow as tf
from utils.misc import cal_iou
from utils.misc import box2offset


def trim_zeros(boxes, name=None):
    """Often boxes are represented with matrices of shape [N, 4] and
    are padded with zeros. This removes zero boxes.

    return
    ---
        boxes: [N, 4] matrix of boxes.
        non_zeros: [N] a 1D boolean mask identifying the rows to keep
    """
    non_zeros = tf.cast(tf.reduce_sum(tf.abs(boxes), axis=1), tf.bool)
    boxes = tf.boolean_mask(boxes, non_zeros, name=name)
    return boxes, non_zeros


class AnchorAssign(object):

    def __init__(self, pos_iou=0.7, neg_iou=0.3, pos_fraction=0.5, num_sample=256,
                 target_means=(0., 0., 0., 0.),
                 target_stds=(0.1, 0.1, 0.2, 0.2)
                 ):
        # ------------------------------------------------------ #
        # if iou > pos_iou: positive target
        # max(iou) = positive
        # if iou < neg_iou: negative target
        # if neg_iou < iou < pos_iou: ignore target
        # ------------------------------------------------------------------------ #
        # pos_fraction: mini-batch loss的时候，为了正负样本的平衡，共sample
        # 256个anchor box，正负样本比例为1:1
        # ------------------------------------------------------------------------ #
        # 进行transform:box2offset
        # ------------------------------------------------------------------------ #
        self.pos_iou = pos_iou
        self.neg_iou = neg_iou
        self.pos_fraction = pos_fraction
        self.num_sample = num_sample
        self.mean = target_means
        self.std = target_stds

    def anchor_assign(self, gt_boxes, anchors, valid_anchors):
        """
        进行gt_boxes的分配

        --------------------
        :param gt_boxes: (batch, num_gt_boxes, (x1, y1, x2, y2))
        :param anchors: (num_anchors, (x1, y1, x2, y2))
        :param valid_anchors: (batch, num_anchors)

        --------------------
        :return:
        """
        batch_anchor_labels = []
        batch_class_weights = []
        batch_offset = []
        batch_offset_weights = []

        batch_size = tf.shape(gt_boxes)[0]
        for i in range(batch_size):
            anchors_labels, offset, class_weights, offset_weights = self.anchor_assign_single(gt_boxes[i],
                                                                                              anchors,
                                                                                              valid_anchors[i])
            batch_anchor_labels.append(anchors_labels)
            batch_class_weights.append(class_weights)
            batch_offset_weights.append(offset_weights)
            batch_offset.append(offset)

        # (batch, num_anchors)
        batch_anchor_labels = tf.stack(batch_anchor_labels)
        # (batch, num_anchors, 4)
        batch_offset = tf.stack(batch_offset)
        # (batch,)
        batch_class_weights = tf.stack(batch_class_weights)
        batch_offset_weights = tf.stack(batch_offset_weights)

        # TODO 输出系数矩阵，在计算loss的时候需要使用
        return batch_anchor_labels, batch_offset, batch_class_weights, batch_offset_weights

    def anchor_assign_single(self, gt_boxes, anchors, valid_anchors):
        # gt_boxes in image(have padded) coordinate

        gt_boxes, _ = trim_zeros(gt_boxes)

        # 计算iou
        anchor_shape = tf.shape(anchors)
        # (num_anchors, num_gt)
        ious = cal_iou(anchors, gt_boxes)
        # (num_anchors,)
        anchor_labels = -tf.ones(shape=(anchor_shape[0],), dtype=tf.int32)

        anchor_iou_max = tf.reduce_max(ious, axis=-1)

        # 1.negative_anchors
        anchor_labels = tf.where(anchor_iou_max < self.neg_iou, tf.zeros(anchor_shape[0], dtype=tf.int32),
                                 anchor_labels)
        # 2.positive_anchors
        anchor_labels = tf.where(anchor_iou_max >= self.pos_iou, tf.ones(anchor_shape[0], dtype=tf.int32),
                                 anchor_labels)
        # 3.如果anchor_iou_max均小于0.7，选取每个gt_box重合度最大的anchor为positive
        # (num_gt,)
        gt_iou_max = tf.argmax(ious, axis=0)
        anchor_labels = tf.tensor_scatter_nd_update(anchor_labels, tf.reshape(gt_iou_max, (-1, 1)),
                                                    tf.ones(tf.shape(gt_iou_max), dtype=tf.int32))
        # 4.filter invalid
        anchor_labels = tf.where(valid_anchors == 1, anchor_labels, -tf.ones(anchor_shape[0], dtype=tf.int32))

        # start sample 256 positive and negative anchors
        num_positive = tf.shape(tf.where(anchor_labels == 1))[0]
        index_positive = tf.where(anchor_labels == 1)
        num_negative = tf.shape(tf.where(anchor_labels == 0))[0]
        index_negative = tf.where(anchor_labels == 0)

        # sample num_class*pos_fraction positive
        number1 = num_positive - tf.cast(self.num_sample * self.pos_fraction, tf.int32)
        if number1 > 0:
            # (number1, 1)
            ids = tf.random.shuffle(index_positive)[:number1]
            anchor_labels = tf.tensor_scatter_nd_update(anchor_labels, ids,
                                                        -tf.ones(shape=(number1,), dtype=tf.int32))

        # sample num_sample - num_positive negative
        num_positive = tf.shape(tf.where(anchor_labels == 1))[0]
        number2 = num_negative - (self.num_sample - num_positive)
        if number2 > 0:
            ids = tf.random.shuffle(index_negative)[:number2]
            anchor_labels = tf.tensor_scatter_nd_update(anchor_labels, ids,
                                                        -tf.ones((number2,), dtype=tf.int32))

        # ---------------------------------------------------------------------- #
        # faster rcnn:
        #   positive anchor is responsible for the best gt which has the best iou
        # ---------------------------------------------------------------------- #
        anchor_iou_argmax = tf.argmax(ious, axis=-1)
        # (num_anchors, 4)
        gt_boxes = tf.gather(gt_boxes, anchor_iou_argmax)
        # 计算offset
        offset = box2offset(anchors, gt_boxes, self.mean, self.std)

        # 计算loss函数中的系数
        num_total = tf.shape(tf.where(anchor_labels >= 0), out_type=tf.float32)[0]
        class_weights = tf.zeros(anchor_shape[0], dtype=tf.float32)
        offset_weights = tf.zeros(anchor_shape[0], dtype=tf.float32)

        if num_total > 0:
            class_weights = tf.where(anchor_labels >= 0, tf.ones(anchor_shape[0], dtype=tf.float32) / num_total,
                                     class_weights)

            offset_weights = tf.where(anchor_shape > 0, tf.ones(anchor_shape[0], dtype=tf.float32) / num_total,
                                      offset_weights)
        offset_weights = tf.tile(tf.reshape(offset_weights, shape=(-1, 1)), multiples=[1, 4])

        return anchor_labels, offset, class_weights, offset_weights
