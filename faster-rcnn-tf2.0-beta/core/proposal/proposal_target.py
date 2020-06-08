# -*- coding:utf-8 -*-
# author:平手友梨奈ii
# e-mail:1353593259@qq.com
# datetime:1993/12/01
# filename:proposal_target.py
# software: PyCharm

# import tensorflow as tf
from utils.misc import *


class ProposalTarget(object):
    """
        proposal target
        if training:
            使用此方法构造y_true
    """

    def __init__(self, target_mean=(0.0, 0.0, 0.0, 0.0), target_std=(0.1, 0.1, 0.2, 0.2),
                 num_classifier=256, pos_fraction=0.25,
                 pos_iou=0.5, neg_iou=0.5,
                 num_classes=81):

        # param:
        #   target_mean & target_std: used in box2offset
        #   num_classifier & pos_fraction: used in sampling proposals
        #   pos_iou & neg_iou: used in assign proposals to positive or negative targets
        #   num_classes: 在最后构建offset_true的使用
        #   offset_true:(num_proposals,num_classes,4)
        self.target_mean = target_mean
        self.targrt_std = target_std
        self.num_classifier = num_classifier
        self.pos_fraction = pos_fraction
        self.pos_iou = pos_iou
        self.neg_iou = neg_iou
        self.num_classes = num_classes

    def get_proposal_target_single(self, proposals, gt_boxes, gt_class_id, img_shape):
        """
            param:
            ------------------------
            proposals: (num_proposals, (batch_ind, x1, y1, x2, y2)) [have normalized]
            gt_boxes: (num_gt, 4)
            gt_class_id: (num_gt,)
            img_shape: (h, w)

            return:
            ------------------------
            rois:
            class_true:
            class_true_weights:
            offset_true:
            offset_true_weights:
        """

        h, w = img_shape

        # 1.trim zero
        proposals, _ = trim_zeros(proposals[:, 1:])
        gt_boxes, gt_boxes_none_zero = trim_zeros(gt_boxes)
        gt_class_id = tf.boolean_mask(gt_class_id, gt_boxes_none_zero)

        # 2.normalize gt_boxes
        gt_boxes = gt_boxes / tf.constant([w, h, w, h], dtype=tf.float32)

        # 3.assign positive or negative
        # (num_proposals, num_gt_boxes)
        ious = cal_iou(proposals, gt_boxes)
        ious_argmax = tf.argmax(ious, axis=-1)
        ious_max = tf.reduce_max(ious, axis=-1)

        # (num_pos, 1)
        pos_indices = tf.where(ious_max >= self.pos_iou)
        neg_indices = tf.where(ious_max < self.neg_iou)
        num_positive = tf.shape(pos_indices)[0]
        num_nagative = tf.shape(neg_indices)[0]

        # 4.sample positive
        num_pos_sample = tf.cast(self.num_classifier * self.pos_fraction, dtype=tf.int32)
        if num_positive > num_pos_sample:
            pos_indices = tf.random.shuffle(pos_indices)[:num_pos_sample]

        num_positive = tf.shape(pos_indices)[0]

        # 5.sample negative
        num_neg_sample = tf.cast((1 - self.pos_fraction) / self.pos_fraction * num_positive, dtype=tf.int32)
        if num_nagative > num_neg_sample:
            neg_indices = tf.random.shuffle(neg_indices)[:num_pos_sample]
        num_nagative = tf.shape(neg_indices)[0]

        # 6.gather rois
        rois_pos = tf.gather(proposals, pos_indices[:, 0])
        rois_neg = tf.gather(proposals, neg_indices[:, 0])

        # 7.make class_true
        rois_pos_iou_max = tf.gather(ious_argmax, pos_indices)
        # positive sample
        rois_pos_classid = tf.gather(gt_class_id, rois_pos_iou_max)
        # negative sample
        rois_pos_classid = tf.pad(rois_pos_classid, [(0, num_nagative)], constant_values=0)
        # roi不足self.num_classifier，则pad
        pad_num = tf.maximum(self.num_classifier - (num_pos_sample + num_pos_sample), 0)
        class_true = tf.pad(rois_pos_classid, [(0, pad_num)], constant_values=-1)

        # 8.make rois
        rois = tf.concat([rois_pos, rois_neg], axis=0)
        rois = tf.pad(rois, [(0, pad_num), (0, 0)])

        # 9.make offset_true
        pos_to_gt = tf.gather(gt_boxes, rois_pos_iou_max)
        pos_offset = box2offset(rois_pos, pos_to_gt, target_mean=self.target_mean, target_std=self.targrt_std)
        offset_true = tf.pad(pos_offset, [(0, num_nagative), (0, 0)], constant_values=0)

        # 10.make weights
        class_true_weights = tf.zeros(shape=(self.num_classifier,), dtype=tf.float32)
        offset_true_weights = tf.zeros(shape=(self.num_classifier,), dtype=tf.float32)
        if num_nagative + num_positive > 0:
            class_true_weights = tf.where(class_true >= 0,
                                          tf.ones(shape=(self.num_classifier,),
                                                  dtype=tf.float32) / (num_positive + num_nagative),
                                          class_true_weights)
            offset_true_weights = tf.where(class_true > 0,
                                           tf.ones(shape=(self.num_classifier,),
                                                   dtype=tf.float32) / (num_nagative + num_positive),
                                           offset_true_weights)
        # 11.构建(num_rois, num_class, 4)
        offset_true_organized = tf.zeros(shape=(self.num_classifier, self.num_classes, 4), dtype=tf.float32)
        offset_true_weights_organized = tf.zeros(shape=(self.num_classifier, self.num_classes), dtype=tf.float32)
        offset_index = tf.stack([tf.range(self.num_classifier), class_true], axis=1)
        offset_true_organized = tf.tensor_scatter_nd_update(offset_true_organized, offset_index, offset_true)
        offset_true_weights_organized = tf.tensor_scatter_nd_update(offset_true_weights_organized,
                                                                    offset_index, offset_true_weights)

        return rois, class_true, class_true_weights, offset_true_organized, offset_true_weights_organized

    def get_proposal_target(self, batch_proposals, batch_gt_boxes, batch_gt_class_id, batch_img_metas):
        img_shapes = calc_pad_shapes(batch_img_metas)
        batch_size = tf.shape(batch_img_metas)[0]
        batch_proposals = tf.reshape(batch_proposals, (batch_size, -1, 5))

        rois_ = []
        class_true_ = []
        class_true_weights_ = []
        offset_true_organized_ = []
        offset_true_weights_organized_ = []

        for i in range(batch_size):
            rois, class_true, class_true_weights, offset_true_organized, offset_true_weights_organized = \
                self.get_proposal_target_single(batch_proposals[i], batch_gt_boxes[i],
                                                batch_gt_class_id[i], img_shapes[i])
            rois_.append(rois)
            class_true_.append(class_true)
            class_true_weights_.append(class_true_weights)
            offset_true_organized_.append(offset_true_organized)
            offset_true_weights_organized_.append(offset_true_weights_organized)

        rois = tf.concat(rois_, axis=0)
        class_true = tf.concat(class_true_, axis=0)
        class_true_weights = tf.concat(class_true_weights_, axis=0)
        offset_true_organized = tf.concat(offset_true_organized_, axis=0)
        offset_true_weights_organized = tf.concat(offset_true_weights_organized_, axis=0)

        return rois, class_true, class_true_weights, offset_true_organized, offset_true_weights_organized
