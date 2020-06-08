# -*- coding:utf-8 -*-
# author:平手友梨奈ii
# e-mail:1353593259@qq.com
# datetime:1993/12/01
# filename:classifier_head.py
# software: PyCharm

import tensorflow as tf
import tensorflow.keras as keras
import tensorflow.keras.layers as layers
from utils.misc import offset2box
from utils.misc import calc_pad_shapes
from core.loss.losses import RPNClassLoss, rpn_box_loss


class ClassifierHead(keras.Model):
    
    def __init__(self, num_classes,
                 pool_size=(7, 7),
                 target_means=(0., 0., 0., 0.),
                 target_stds=(0.1, 0.1, 0.2, 0.2),
                 min_confidence=0.05,
                 nms_threshold=0.5,
                 max_instances=100):
        """
                                          / dense class units=num_classes
            pooled_rois —— conv1 —— conv2
                                          \ dense offset units=num_classes*4

        """
        super(ClassifierHead, self).__init__()

        self.num_classes = num_classes
        self.pool_size = pool_size
        self.target_means = target_means
        self.target_stds = target_stds
        self.min_confidence = min_confidence
        self.nms_threshold = nms_threshold
        self.max_instances = max_instances

        self.class_conv1 = layers.Conv2D(1024, self.pool_size, name='rcnn_class_conv1')
        self.class_bn1 = layers.BatchNormalization(name='rcnn_class_bn1')

        self.class_conv2 = layers.Conv2D(1024, 1, name='rcnn_class_conv2')
        self.class_bn2 = layers.BatchNormalization(name='rcnn_class_bn2')

        # classifier
        self.classifier = layers.Dense(num_classes, name='rcnn_class_logits')
        self.offset_fc = layers.Dense(num_classes * 4, name='rcnn_bbox_fc')

        self.relu = layers.ReLU()
        self.softmax = layers.Softmax()

    def call(self, inputs, training=None, mask=None):

        # inputs
        # (num_rois, pool_size, pool_size, c)
        y = self.class_conv1(inputs)
        y = self.class_bn1(y)
        y = self.relu(y)

        y = self.class_conv2(y)
        y = self.class_bn2(y)
        y = self.relu(y)

        y = tf.squeeze(y, axis=[1, 2])

        class_logit = self.classifier(y)
        class_prob = self.softmax(class_logit)

        offset_output = self.offset_fc(y)
        # reshape   (batch*num_rois, num_classes, 4)
        offset_output = tf.reshape(offset_output, (-1, self.num_classes, 4))

        return class_logit, class_prob, offset_output

    def get_object_single(self, prob, offset, rois, img_shape):

        num_rois = tf.shape(prob)[0]
        h, w = img_shape

        class_id = tf.argmax(prob, axis=1)

        max_class_ind = tf.stack([tf.range(num_rois), class_id], axis=1)

        # 将offset中置信度最大的预测值取出
        # offset2box
        # 进行校正(0<x<w)
        offset = tf.gather(offset, max_class_ind)
        rois = offset2box(rois, offset, self.target_means, self.target_stds)
        rois = rois * tf.constant([w, h, w, h], dtype=tf.float32)
        # TODO 把roi校正封装成一个方法
        x1, y1, x2, y2 = tf.split(rois, 4, axis=1)
        x1 = tf.clip_by_value(x1, 0.0, w * 1.0)
        y1 = tf.clip_by_value(y1, 0.0, h * 1.0)
        x2 = tf.clip_by_value(x2, 0.0, w * 1.0)
        y2 = tf.clip_by_value(y2, 0.0, h * 1.0)
        rois = tf.concat([x1, y1, x2, y2], axis=1)

        # (num_rois, )
        max_prob = tf.gather_nd(prob, max_class_ind)
        # 1.filter negative samples
        positives_ind = tf.where(class_id > 0)[:, 0]
        # 2.filter low confidence samples
        high_conf_ind = tf.where(max_prob > self.min_confidence)[:, 0]

        positives_ind = tf.expand_dims(positives_ind, axis=0)
        high_conf_ind = tf.expand_dims(high_conf_ind, axis=0)

        roi_ind = tf.sets.intersection(positives_ind, high_conf_ind).values

        # 3.对每个类别进行nms
        keep_rois = tf.gather(rois, roi_ind)
        keep_class_id = tf.gather(class_id, roi_ind)
        keep_max_prob = tf.gather(max_prob, roi_ind)
        keep_unique_id = tf.unique(keep_class_id)

        final_keep = []

        for id in keep_unique_id:
            # 取出属于这个类的roi
            roi_belong_id_ind = tf.where(keep_class_id == id)[:, 0]
            roi_belong_id = tf.gather(keep_rois, roi_belong_id_ind)
            roi_score = tf.gather(keep_max_prob, roi_belong_id_ind)
            keep = tf.image.non_max_suppression(roi_belong_id, roi_score, self.max_instances, self.nms_threshold)

            # 添加index
            keep = tf.gather(roi_belong_id_ind, keep)
            final_keep.append(keep)

        final_keep = tf.concat(final_keep, axis=0)

        # 4. keep top max_instancesdetections
        num_keep = tf.shape(final_keep)[0]
        num_keep = tf.minimum(num_keep, self.max_instances)
        # 排序
        final_score = tf.gather(keep_max_prob, final_keep)
        indices = tf.nn.top_k(final_score, num_keep).indices

        final_keep = tf.gather(final_keep, indices)

        # 5.打包置信度、类别、offset
        # (num_keep, 4)
        final_rois = tf.gather(keep_rois, final_keep)
        # (num_keep, )
        final_id = tf.gather(keep_class_id, final_keep)
        # (num_keep, )
        final_confi = tf.gather(keep_max_prob, final_keep)

        return [final_rois, final_id, final_confi]

    def get_object(self, batch_prob, batch_offset, batch_rois, img_metas):
        """
        通过nms得到的最终预测结果
        param:
        --------------------------------------
        batch_prob: (batch*num_rois, num_classes)
        batch_offset:   (batch*num_rois, num_classes, 4)
        batch_rois: (batch*num_rois, 5)
        img_metas:(batch_size, 11)
        """

        batch_size = tf.shape(img_metas)[0]
        batch_prob = tf.reshape(batch_prob, (batch_size, -1, self.num_classes))
        batch_offset = tf.reshape(batch_offset, (batch_size, -1, self.num_classes, 4))
        batch_rois = tf.reshape(batch_rois, (batch_size, -1, 5))[..., 1:]
        pad_shapes = calc_pad_shapes(img_metas)

        detect_results = [self.get_object_single(batch_prob[i], batch_offset[i],
                                                 batch_rois[i], pad_shapes[i]) for i in range(batch_size)]

        return detect_results

    def loss(self, class_logit, offset_output, class_true, class_true_weights,
             offset_true, offset_true_weights):
        """
        计算loss：
            class_loss(positive+negative参与计算） + offset_loss(只有positive参与计算）
        class_logit:(batch*num_rois, num_classes)
        offset_output:(batch*num_rois, num_class, 4)
        class_true:(batch*num_rois, )
        class_true_weights:(batch*num_rois, )
        offset_true:(batch*num_rois, num_class, 4)
        offset_true_weights:(batch*num_rois, num_class, 4)
        """
        class_loss = RPNClassLoss()
        class_loss = class_loss(class_true, class_logit, class_true_weights)

        offset_loss = rpn_box_loss(offset_true, offset_output, offset_true_weights)

        return class_loss + offset_loss
