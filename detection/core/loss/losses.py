# -*- coding:utf-8 -*-
# author:栗山未来ii
# e-mail:1353593259@qq.com
# datetime:1993/12/01
# filename:aaa.py
# software: PyCharm

import tensorflow as tf
import numpy as np

layers = tf.keras.layers
losses = tf.keras.losses


class SmoothL1Loss(layers.Layer):
    def __init__(self, rho=1):
        super(SmoothL1Loss, self).__init__()
        self._rho = rho

    def __call__(self, y_true, y_pred, sample_weight=None):
        loss = tf.abs(y_true - y_pred)
        loss = tf.where(loss > self._rho, loss - 0.5 * self._rho,
                        (0.5 / self._rho) * tf.square(loss))

        if sample_weight is not None:
            loss = tf.multiply(loss, sample_weight)

        return loss


class RPNClassLoss(layers.Layer):
    def __init__(self):
        super(RPNClassLoss, self).__init__()
        self.sparse_categorical_crossentropy = \
            losses.SparseCategoricalCrossentropy(from_logits=True,
                                                 reduction=losses.Reduction.NONE)

    def __call__(self, rpn_labels, rpn_class_logits, rpn_label_weights):
        # Filtering if label == -1
        indices = tf.where(tf.not_equal(rpn_labels, -1))
        rpn_labels = tf.gather_nd(rpn_labels, indices)
        rpn_label_weights = tf.gather_nd(rpn_label_weights, indices)
        rpn_class_logits = tf.gather_nd(rpn_class_logits, indices)

        # Calculate loss
        loss = self.sparse_categorical_crossentropy(y_true=rpn_labels,
                                                    y_pred=rpn_class_logits,
                                                    sample_weight=rpn_label_weights)
        loss = tf.reduce_sum(loss)
        return loss


class RPNBBoxLoss(layers.Layer):
    def __init__(self):
        super(RPNBBoxLoss, self).__init__()
        self.smooth_l1_loss = SmoothL1Loss()

    def __call__(self, rpn_delta_targets, rpn_deltas, rpn_delta_weights):
        loss = self.smooth_l1_loss(y_true=rpn_delta_targets,
                                   y_pred=rpn_deltas,
                                   sample_weight=rpn_delta_weights)
        loss = tf.reduce_sum(loss)
        return loss


class RCNNClassLoss(layers.Layer):
    def __init__(self):
        super(RCNNClassLoss, self).__init__()
        self.sparse_categorical_crossentropy = \
            losses.SparseCategoricalCrossentropy(from_logits=True,
                                                 reduction=losses.Reduction.NONE)

    def __call__(self, rcnn_labels, rcnn_class_logits, rcnn_label_weights):
        # Filtering if label == -1
        indices = tf.where(tf.not_equal(rcnn_labels, -1))
        rcnn_labels = tf.gather_nd(rcnn_labels, indices)
        rcnn_label_weights = tf.gather_nd(rcnn_label_weights, indices)
        rcnn_class_logits = tf.gather_nd(rcnn_class_logits, indices)

        # Calculate loss
        loss = self.sparse_categorical_crossentropy(y_true=rcnn_labels,
                                                    y_pred=rcnn_class_logits,
                                                    sample_weight=rcnn_label_weights)
        loss = tf.reduce_sum(loss)
        return loss


class RCNNBBoxLoss(layers.Layer):
    def __init__(self):
        super(RCNNBBoxLoss, self).__init__()
        self.smooth_l1_loss = SmoothL1Loss()

    def __call__(self, rcnn_delta_targets, rcnn_deltas, rcnn_delta_weights):
        loss = self.smooth_l1_loss(y_true=rcnn_delta_targets,
                                   y_pred=rcnn_deltas,
                                   sample_weight=rcnn_delta_weights)
        loss = tf.reduce_sum(loss)
        return loss


class GHMCLoss(layers.Layer):
    """
    TODO: debug these codes
    Use GHM Loss in one stage object detection.
    Unit Region Approximation:
        devide gradients into num_bins bins.
    """
    def __init__(self, batch, num_anchors, num_classes, bins=10):
        super(GHMCLoss, self).__init__()
        # devide gradients into self.bins bins
        self.bins = bins
        self.batch = batch
        self.num_anchors = num_anchors
        self.num_classes = num_classes

        self.packages = tf.range(self.bins + 1, dtype=tf.float32) / self.bins
        self.packages = tf.tensor_scatter_nd_add(self.packages, [[self.bins]], [1e-6])

    def __call__(self, class_labels, class_logits, label_weights):
        """Calculate GHM CLoss

        Args:
            class_labels:  [int]     (batch, h, w, num_anchors * num_classes)
            class_logits:  [float32] (batch, h, w, num_anchors * num_classes)
            label_weights: [float32] (batch, h, w, num_anchors * num_classes)

        Returns:
            class_losses

        """
        class_probs = tf.sigmoid(class_logits)
        probs_norm = tf.where(tf.equal(class_labels, 1), 1 - class_probs, class_probs)
        total_num = tf.maximum(tf.reduce_sum(label_weights), 1.0)

        valid = label_weights > 0
        beta = tf.zeros_like(probs_norm)
        # the counter of valid bins
        n = 0

        for i in range(self.bins):
            ids = (probs_norm >= self.packages[i]) & (probs_norm < self.packages[i+1]) & valid
            # bool to int
            ids = tf.cast(ids, dtype=tf.float32)
            num_ids = tf.reduce_sum(ids)
            if num_ids > 0:
                ids_non_zero = tf.where(ids)
                beta = tf.tensor_scatter_nd_update(beta, ids_non_zero,
                                                   tf.ones(shape=(tf.shape(ids_non_zero)[0],), dtype=tf.float32)
                                                   * total_num / num_ids)
                n += 1

        # calculate loss
        if n > 0:
            beta = beta / n
        class_losses = tf.nn.sigmoid_cross_entropy_with_logits(class_labels, class_logits) * beta
        class_losses = tf.reduce_sum(class_losses)

        return class_losses / total_num


class GHMRLoss(layers.Layer):
    def __init__(self, batch, num_anchors, miu, bins=10):
        super(GHMRLoss, self).__init__()
        # devide gradients into self.bins bins
        self.bins = bins
        self.batch = batch
        self.num_anchors = num_anchors
        self.miu = miu

        self.packages = tf.range(self.bins + 1, dtype=tf.float32) / self.bins
        self.packages = tf.tensor_scatter_nd_add(self.packages, [[self.bins]], [1e-6])

    def __call__(self, delta_targets, deltas, delta_weights):
        """GHM RLoss
        TODO: debug these codes

        Args:
            delta_targets: (batch, h, w, num_anchors * 4)
            deltas:        (batch, h, w, num_anchors * 4)
            delta_weights: (batch, h, w, num_anchors * 4)

        Returns:
            delta_losses
        """
        # calculate delta loss by ASL1
        ASL1 = tf.sqrt(tf.square(deltas - delta_targets) + self.miu * self.miu) - self.miu
        gradients = (deltas - delta_targets) / tf.sqrt(tf.square(deltas - delta_targets) + tf.square(self.miu))
        gradients = tf.abs(gradients)

        total_num = tf.maximum(tf.reduce_sum(delta_weights), 1.0)

        valid = delta_weights > 0
        beta = tf.zeros_like(gradients)
        # the counter of valid bins
        n = 0

        for i in range(self.bins):
            ids = (gradients >= self.packages[i]) & (gradients < self.packages[i + 1]) & valid
            # bool to int
            ids = tf.cast(ids, dtype=tf.float32)
            num_ids = tf.reduce_sum(ids)
            if num_ids > 0:
                ids_non_zero = tf.where(ids)
                beta = tf.tensor_scatter_nd_update(beta, ids_non_zero,
                                                   tf.ones(shape=(tf.shape(ids_non_zero)[0],), dtype=tf.float32)
                                                   * total_num / num_ids)
                n += 1

        if n > 0:
            beta = beta / n
        box_losses = ASL1 * beta

        return box_losses / total_num


if __name__ == '__main__':
    # debug GHM Closs
    ghm_c_loss = GHMCLoss(batch=8, num_anchors=3, num_classes=10)
    class_labels = np.random.randint(2, size=(8, 7, 7, 3 * 10)).astype(np.float32)
    class_logits = np.random.randn(8, 7, 7, 3 * 10).astype(np.float32)
    class_weights = np.random.randint(2, size=(8, 7, 7, 30)).astype(np.float32)
    class_losses = ghm_c_loss(class_labels=class_labels, class_logits=class_logits, label_weights=class_weights)

    # debug GHM RLoss
    ghm_r_loss = GHMRLoss(batch=8, num_anchors=3, miu=0.02)
    delta_targets = np.random.randn(8, 7, 7, 3 * 4).astype(np.float32)
    deltas = np.random.randn(8, 7, 7, 3 * 4).astype(np.float32)
    delta_weights = np.random.randint(2, size=(8, 7, 7, 3 * 4)).astype(np.float32)
    box_losses = ghm_r_loss(delta_targets, deltas, delta_weights )
