# -*- coding:utf-8 -*-
# author:平手友梨奈ii
# e-mail:1353593259@qq.com
# datetime:1993/12/01
# filename:losses.py
# software: PyCharm

import tensorflow as tf
import tensorflow.keras as keras


# smooth_l1


class RPNClassLoss(object):

    def __init__(self):
        # sparse categorical cross entropy
        # 这边y_true是稀疏编码
        self.loss = keras.losses.SparseCategoricalCrossentropy(from_logits=True,
                                                          reduction=keras.losses.Reduction.NONE)

    def __call__(self, anchor_lables, class_logit, class_weights):
        index = tf.where(tf.not_equal(anchor_lables, -1))
        rpn_logit = tf.gather_nd(class_logit, index)
        rpn_labels = tf.gather_nd(anchor_lables, index)
        class_weights = tf.gather_nd(class_weights, index)

        # (num_p&n, )
        losses = self.loss(rpn_labels, rpn_logit)
        losses = tf.multiply(losses, class_weights)
        losses = tf.reduce_sum(losses)

        return losses


def rpn_box_loss(offset, box_pred, offset_weights, sigma=1):
    # smooth l1 loss

    sigma_square = tf.cast(tf.square(sigma), tf.float32)

    delta = tf.abs(box_pred - offset)

    loss = tf.where(tf.greater_equal(delta, 1.0 / sigma_square), delta - 0.5 / sigma_square,
                    0.5 * tf.square(delta) * sigma_square)

    loss = tf.multiply(loss, offset_weights)

    loss = tf.reduce_sum(loss)

    return loss
