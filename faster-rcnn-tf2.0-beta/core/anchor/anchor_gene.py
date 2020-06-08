# -*- coding:utf-8 -*-
# author:平手友梨奈ii
# e-mail:1353593259@qq.com
# datetime:1993/12/01
# filename:anchor_gene.py
# software: PyCharm

import tensorflow as tf
from utils.misc import *


class AnchorGenerator(object):

    def __init__(self, scales=(32, 64, 128, 256, 512),
                 ratios=(0.5, 1, 2),
                 feature_strides=(4, 8, 16, 32, 64)):
        self.scales = scales
        self.ratios = ratios
        self.feature_strides = feature_strides

    def _generate_level_anchors(self, feature_shape, level):
        """
        generate the anchors of level
        ------------------------------
        :param feature_shape:(h, w)
        :param level:level of feature pyramid
        :return:anchors:[num_anchors, x1, y1, x2, y2]
        """
        scale = tf.cast(self.scales[level], tf.float32)
        # (3,)
        ratios = tf.cast(self.ratios, tf.float32)
        scale, ratios = tf.meshgrid(scale, ratios)
        scale = tf.reshape(scale, [-1])
        ratios = tf.reshape(ratios, [-1])

        # anchors的长宽
        # (3,)
        anchor_h = scale / tf.sqrt(ratios)
        anchor_w = scale * tf.sqrt(ratios)

        feature_stride = self.feature_strides[level]
        feature_stride = tf.cast(feature_stride, tf.float32)

        height, width = feature_shape
        y_range = (tf.cast(tf.range(height), tf.float32) + 0.5) * feature_stride
        x_range = (tf.cast(tf.range(width), tf.float32) + 0.5) * feature_stride
        # (h, w) type=float32
        x_center, y_center = tf.meshgrid(x_range, y_range)

        # 构建(h*w, num_anchors)
        box_w, box_centerx = tf.meshgrid(anchor_w, x_center)
        box_h, box_centery = tf.meshgrid(anchor_h, y_center)

        # 构建(h*w, num_anchors, 2)
        # 构建(h*w, num_anchors, 2) 最后2维：x, y
        # 构建(h*w, num_anchors, 2) 最后2维：w, h
        # 使用stack函数要注意数据堆叠顺序
        box_center_xy = tf.stack([box_centerx, box_centery], axis=-1)
        box_wh = tf.stack([box_w, box_h], axis=-1)

        # 转化为corner坐标
        box_left = box_center_xy - 0.5 * box_wh
        box_right = box_center_xy + 0.5 * box_wh

        # 构建(h*w, num_anchors, 4)
        anchors = tf.concat([box_left, box_right], axis=-1)
        # (h*w*num_anchors, 4)
        anchors = tf.reshape(anchors, shape=(-1, 4))

        return anchors

    def valid_anchor(self, anchors, image_shape):
        # anchors:(num_anchors, 4) in image coordinate
        # feature:(h, w)
        h, w = image_shape

        valid_anchors = tf.ones(anchors.shape[0], dtype=tf.int32)
        invalid_anchors = tf.zeros(anchors.shape[0], dtype=tf.int32)
        y_center = (anchors[:, 1] + anchors[:, 3]) / 2
        x_center = (anchors[:, 2] + anchors[:, 0]) / 2

        valid_anchors = tf.where(y_center <= h, valid_anchors, invalid_anchors)
        valid_anchors = tf.where(x_center <= w, valid_anchors, invalid_anchors)

        return valid_anchors

    def generate_anchors(self, imgs_meta):
        # faster rcnn为了统一输入尺寸，对batch进行了pad
        # (max_h, max_w)
        pad_shape = calc_batch_padded_shape(imgs_meta)
        # (num_strides, 2)
        features_shape = [[pad_shape[0] // stride, pad_shape[1] // stride] for stride in self.feature_strides]
        # [(num_anchors_1, 4), (num_anchors_2, 4)...]
        anchors = [self._generate_level_anchors(feature_shape, level)
                   for level, feature_shape in enumerate(features_shape)]

        # (num_anchors, 4)
        anchors = tf.concat(anchors, axis=0)

        # valid_anchors
        imgs_shape = calc_img_shapes(imgs_meta)

        # [(num_anchors, 4), (num_anchors, 4)...]
        valid_anchors = [self.valid_anchor(anchors, img_shape) for img_shape in imgs_shape]
        valid_anchors = tf.stack(valid_anchors, axis=0)

        return anchors, valid_anchors
