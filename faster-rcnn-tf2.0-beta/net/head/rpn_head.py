# -*- coding:utf-8 -*-
# author:平手友梨奈ii
# e-mail:1353593259@qq.com
# datetime:1993/12/01
# filename:rpn_head.py
# software: PyCharm

import tensorflow as tf
import tensorflow.keras as keras
import tensorflow.keras.layers as layers
from core.anchor.anchor_gene import AnchorGenerator
from core.anchor.anchor_assign import AnchorAssign
from core.loss.losses import RPNClassLoss
from core.loss.losses import rpn_box_loss
from utils.misc import offset2box, calc_pad_shapes


class RPNHead(keras.Model):

    def __init__(self, anchor_scales=(32, 64, 128, 256, 512), anchor_ratios=(0.5, 1, 2),
                 anchor_feature_strides=(4, 8, 16, 32, 64), proposal_counts=2000, nms_threshold=0.7,
                 num_rpn_deltas=256, positive_fraction=0.5, posi_iou_thres=0.7, neg_iou_thres=0.3,
                 target_means=(0.0, 0.0, 0.0, 0.0), target_stds=(0.1, 0.1, 0.2, 0.2)):
        super(RPNHead, self).__init__()

        # fpn多层级输出的公共特征提取层
        self.rpn_share = layers.Conv2D(512, 3, padding='same', kernel_initializer='he_normal', name='rpn_share')
        self.rpn_class_raw = layers.Conv2D(len(anchor_ratios) * 2, 1,
                                           kernel_initializer='he_normal', name='rpn_class_raw')
        self.rpn_box_pred = layers.Conv2D(len(anchor_ratios) * 4, 1,
                                          kernel_initializer='he_normal', name='rpn_box_pred')

        # anchor generator
        self.anchor_generator = AnchorGenerator(scales=anchor_scales, ratios=anchor_ratios,
                                                feature_strides=anchor_feature_strides)

        # anchor assign
        self.anchor_assign = AnchorAssign(posi_iou_thres, neg_iou_thres, positive_fraction, num_rpn_deltas,
                                          target_means, target_stds)

        # get proposal
        self.target_mean = target_means
        self.target_std = target_stds
        self.proposal_counts = proposal_counts
        self.nms_threshold = nms_threshold

    def call(self, inputs, training=None, mask=None):

        # [[class_logiy, class_prob, box_pred], ...]
        layers_outputs = []

        for feature in inputs:
            share_feat = self.rpn_share(feature)
            share_feat = layers.ReLU()(share_feat)

            # (bacth, h, w, anchors, 2)
            class_logit = self.rpn_class_raw(share_feat)
            class_logit_shape = tf.shape(class_logit)
            # (batch, h*w*anchors, 2)
            # positive or negative
            # class_logit = layers.Reshape(target_shape=(class_logit_shape[0], -1, 2))(class_logit)
            # dont use layer.Reshape because -1 in target shape is not working,
            # layers.Reshape([n, -1, 2])  # shape(n, None, 2)
            class_logit = tf.reshape(class_logit, shape=(class_logit_shape[0], -1, 2))
            class_prob = layers.Softmax()(class_logit)

            # (batch, h, w, anchors, 4)
            box_pred = self.rpn_box_pred(share_feat)
            # (batch, h*w*anchors, 4)
            # box_pred = layers.Reshape(target_shape=(class_logit_shape[0], -1, 2))(box_pred)
            box_pred = tf.reshape(box_pred, shape=(class_logit_shape[0], -1, 4))
            layers_outputs.append([class_logit, class_prob, box_pred])

        # 将feature pyramid的不同尺寸特征的预测结果进行合并
        layers_out = list(zip(*layers_outputs))
        layers_out = [tf.concat(layer_out, axis=1) for layer_out in layers_out]

        class_logit, class_prob, box_pred = layers_out

        return class_logit, class_prob, box_pred

    def loss(self, class_logit, box_pred, gt_boxes, img_meta):
        # 计算rpn loss
        anchors, valid_anchors = self.anchor_generator.generate_anchors(img_meta)

        batch_anchor_labels, batch_offset, batch_class_weights, batch_offset_weights = \
            self.anchor_assign.anchor_assign(gt_boxes, anchors, valid_anchors)

        rpn_class_loss = RPNClassLoss()
        rpn_class_loss = rpn_class_loss(batch_anchor_labels, class_logit, batch_class_weights)

        rpn_offset_loss = rpn_box_loss(batch_offset, box_pred, batch_offset_weights)

        return rpn_class_loss, rpn_offset_loss

    def get_proposal_single(self,
                            rpn_prob,
                            rpn_offset,
                            anchors,
                            valid_anchors,
                            img_shape,
                            batch_index,
                            with_prob):
        # 1.选出6000个score最高的proposal
        # 2.进行nms 选出2000个proposal

        h, w = img_shape

        # 1.pick valid anchors
        valid_anchors_bool = tf.cast(valid_anchors, tf.bool)
        rpn_prob = tf.boolean_mask(rpn_prob, valid_anchors_bool)
        rpn_offset = tf.boolean_mask(rpn_offset, valid_anchors_bool)
        anchors = tf.boolean_mask(anchors, valid_anchors_bool)

        # 2.get 6000 proposal
        pre_nms_limit = min(6000, tf.shape(rpn_prob)[0])
        top_6000 = tf.nn.top_k(rpn_prob, pre_nms_limit, sorted=True)
        top_6000_index = top_6000.indices
        rpn_prob = tf.gather(rpn_prob, top_6000_index)
        rpn_offset = tf.gather(rpn_offset, top_6000_index)
        anchors = tf.gather(anchors, top_6000_index)

        # 3.offset2box
        rpn_boxes = offset2box(anchors, rpn_offset, self.target_mean, self.target_std)

        # 4.refine rpn_boxes
        rpn_boxes_x1x2 = rpn_boxes[..., 0::2]
        rpn_boxes_y1y2 = rpn_boxes[..., 1::2]
        rpn_boxes_x1x2 = tf.clip_by_value(rpn_boxes_x1x2, 0, w)
        rpn_boxes_y1y2 = tf.clip_by_value(rpn_boxes_y1y2, 0, h)
        rpn_boxes_x1 = rpn_boxes_x1x2[..., 0]
        rpn_boxes_x2 = rpn_boxes_x1x2[..., 1]
        rpn_boxes_y1 = rpn_boxes_y1y2[..., 0]
        rpn_boxes_y2 = rpn_boxes_y1y2[..., 1]
        rpn_boxes = tf.stack([rpn_boxes_x1, rpn_boxes_y1, rpn_boxes_x2, rpn_boxes_y2], axis=-1)

        # 5.进行归一化
        rpn_boxes = rpn_boxes / tf.constant([w, h, w, h], dtype=tf.float32)

        # 6.进行nms nms_limit = self.proposal_count
        index = tf.image.non_max_suppression(rpn_boxes, rpn_prob, self.proposal_counts, self.nms_threshold)
        proposal_boxes = tf.gather(rpn_boxes, index)

        if with_prob:
            # (num_proposals, 1)
            rpn_prob = tf.expand_dims(tf.gather(rpn_prob, index), axis=-1)
            # (num_proposals, (x1, y1, x2, y2, prob))
            proposal_boxes = tf.concat([proposal_boxes, rpn_prob], axis=-1)

        # -------------- #
        # 7.pad
        # -------------- #
        num_pad = self.proposal_counts - tf.shape(proposal_boxes)[0]
        proposal_boxes = tf.pad(proposal_boxes, [[0, num_pad], [0, 0]])

        # 8.add batch_index
        # (num_proposals, (batch_index, ...))
        batch_index = tf.tile(tf.reshape(batch_index, shape=(-1, 1)), [tf.shape(proposal_boxes)[0], 1])
        proposal_boxes = tf.concat([batch_index, proposal_boxes], axis=-1)

        return proposal_boxes

    def get_proposals(self, img_meta, batch_rpn_prob, batch_rpn_offset, with_prob=False):
        batch_rpn_prob = batch_rpn_prob[..., 1]

        pad_shapes = calc_pad_shapes(img_meta)

        anchors, valid_anchors = self.anchor_generator.generate_anchors(img_meta)

        proposals = [self.get_proposal_single(batch_rpn_prob[i], batch_rpn_offset[i], anchors,
                                              valid_anchors[i], pad_shapes[i], i, with_prob)
                     for i in range(tf.shape(batch_rpn_prob)[0])]
        proposals = tf.concat(proposals, axis=0)

        return tf.stop_gradient(proposals)
