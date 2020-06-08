# -*- coding:utf-8 -*-
# author:平手友梨奈ii
# e-mail:1353593259@qq.com
# datetime:1993/12/01
# filename:roi_align.py
# software: PyCharm

import tensorflow as tf
import tensorflow.keras as keras
from utils.misc import calc_pad_shapes


class ROIAlign(keras.layers.Layer):

    def __init__(self, pool_size):
        super(ROIAlign, self).__init__()
        self.pool_size = pool_size

    def call(self, inputs, **kwargs):
        # rois align
        rois, feature_map_list, img_metas = inputs
        # (batch*num_rois, 1)
        x1, y1, x2, y2 = tf.split(rois[:, 1:], 4, axis=1)
        w = tf.maximum(0, x2 - x1)
        h = tf.maximum(0, y2 - y1)
        area = w * h

        pad_shapes = calc_pad_shapes(img_metas)
        area_img = tf.cast(pad_shapes[..., 0] * pad_shapes[..., 1], tf.float32)
        area_img = area_img[0]

        area = tf.squeeze(area, axis=1)

        # area are normalized
        # 1.cal levels
        area = area * area_img
        levels = tf.math.log(tf.math.sqrt(area) / 224.0) / tf.math.log(2.0)
        levels = tf.minimum(5, tf.maximum(2, 4 + tf.cast(tf.round(levels), tf.float32)))

        # 2.crop img with pool size
        pooled_rois = []
        pooled_rois_order = []
        for i, level in enumerate(range(2, 6)):
            # i:index of feature map list
            feature_map = feature_map_list[i]
            # (num,)
            level_index = tf.where(levels == level)[:, 0]

            pooled_rois_order.append(level_index)

            rois_of_level = tf.gather(rois, level_index)
            # 将index转换为int32
            rois_level_indices = tf.cast(rois_of_level[..., 0], tf.int32)
            # (num_rois_level, pool_size, pool_size, channel)
            pooled_rois.append(tf.image.crop_and_resize(feature_map, rois_of_level,
                                                        rois_level_indices, self.pool_size))

        # (num_rois, h, w, c)
        pooled_rois = tf.concat(pooled_rois, axis=0)
        # (num_rois,)
        pooled_rois_order = tf.concat(pooled_rois_order, axis=0)
        # 3.重新排序pooled rois，使得顺序和rois一样
        order_sort = tf.argsort(pooled_rois_order)
        pooled_rois = tf.gather(pooled_rois, order_sort)

        return pooled_rois
