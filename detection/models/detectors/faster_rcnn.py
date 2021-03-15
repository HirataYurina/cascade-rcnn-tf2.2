# -*- coding:utf-8 -*-
# author:栗山未来ii
# e-mail:1353593259@qq.com
# datetime:1993/12/01
# filename:aaa.py
# software: PyCharm

import tensorflow as tf

from detection.models.backbones import resnet
from detection.models.necks import fpn
from detection.models.rpn_heads import rpn_head
from detection.models.bbox_heads import bbox_head
from detection.models.roi_extractors import roi_align
from detection.models.detectors.test_mixins import RPNTestMixin, BBoxTestMixin

from detection.core.bbox import bbox_target


class FasterRCNN(tf.keras.Model, RPNTestMixin, BBoxTestMixin):
    def __init__(self, num_classes, **kwags):
        super(FasterRCNN, self).__init__(**kwags)

        self.NUM_CLASSES = num_classes

        # RPN configuration
        # Anchor attributes
        self.ANCHOR_SCALES = (32, 64, 128, 256, 512)
        self.ANCHOR_RATIOS = (0.5, 1, 2)
        self.ANCHOR_FEATURE_STRIDES = (4, 8, 16, 32, 64)

        # Bounding box refinement mean and standard deviation
        self.RPN_TARGET_MEANS = (0., 0., 0., 0.)
        self.RPN_TARGET_STDS = (0.1, 0.1, 0.2, 0.2)

        # RPN training configuration
        self.PRN_BATCH_SIZE = 256
        self.RPN_POS_FRAC = 0.5
        self.RPN_POS_IOU_THR = 0.7
        self.RPN_NEG_IOU_THR = 0.3

        # ROIs kept configuration
        self.PRN_PROPOSAL_COUNT = 2000
        self.PRN_NMS_THRESHOLD = 0.7

        # RCNN configuration
        # Bounding box refinement mean and standard deviation
        self.RCNN_TARGET_MEANS = (0., 0., 0., 0.)
        self.RCNN_TARGET_STDS = (0.1, 0.1, 0.2, 0.2)

        # ROI Feat Size
        self.POOL_SIZE = (7, 7)

        # RCNN training configuration
        self.RCNN_BATCH_SIZE = 256
        self.RCNN_POS_FRAC = 0.25
        self.RCNN_POS_IOU_THR = 0.5
        self.RCNN_NEG_IOU_THR = 0.5

        # Boxes kept configuration
        self.RCNN_MIN_CONFIDENCE = 0.05
        self.RCNN_NMS_THRESHOLD = 0.5
        self.RCNN_MAX_INSTANCES = 100

        # Target Generator for the second stage.
        self.bbox_target = bbox_target.ProposalTarget(
            target_means=self.RCNN_TARGET_MEANS,
            target_stds=self.RPN_TARGET_STDS,
            num_rcnn_deltas=self.RCNN_BATCH_SIZE,
            positive_fraction=self.RCNN_POS_FRAC,
            pos_iou_thr=self.RCNN_POS_IOU_THR,
            neg_iou_thr=self.RCNN_NEG_IOU_THR,
            num_classes=self.NUM_CLASSES)

        # Modules
        self.backbone = resnet.ResNet(
            depth=101,
            name='res_net')

        self.neck = fpn.FPN(
            name='fpn')

        self.rpn_head = rpn_head.RPNHead(
            anchor_scales=self.ANCHOR_SCALES,
            anchor_ratios=self.ANCHOR_RATIOS,
            anchor_feature_strides=self.ANCHOR_FEATURE_STRIDES,
            proposal_count=self.PRN_PROPOSAL_COUNT,
            nms_threshold=self.PRN_NMS_THRESHOLD,
            target_means=self.RPN_TARGET_MEANS,
            target_stds=self.RPN_TARGET_STDS,
            num_rpn_deltas=self.PRN_BATCH_SIZE,
            positive_fraction=self.RPN_POS_FRAC,
            pos_iou_thr=self.RPN_POS_IOU_THR,
            neg_iou_thr=self.RPN_NEG_IOU_THR,
            name='rpn_head')

        self.roi_align = roi_align.PyramidROIAlign(
            pool_shape=self.POOL_SIZE,
            name='pyramid_roi_align')

        self.bbox_head = bbox_head.BBoxHead(
            num_classes=self.NUM_CLASSES,
            pool_size=self.POOL_SIZE,
            target_means=self.RCNN_TARGET_MEANS,
            target_stds=self.RCNN_TARGET_STDS,
            min_confidence=self.RCNN_MIN_CONFIDENCE,
            nms_threshold=self.RCNN_NMS_THRESHOLD,
            max_instances=self.RCNN_MAX_INSTANCES,
            name='b_box_head')

    def __call__(self, inputs, training=True):
        if training:  # training
            imgs, img_metas, gt_boxes, gt_class_ids = inputs
        else:  # inference
            imgs, img_metas = inputs

        C2, C3, C4, C5 = self.backbone(imgs,
                                       training=training)

        P2, P3, P4, P5, P6 = self.neck([C2, C3, C4, C5],
                                       training=training)

        rpn_feature_maps = [P2, P3, P4, P5, P6]
        rcnn_feature_maps = [P2, P3, P4, P5]

        rpn_class_logits, rpn_probs, rpn_deltas = self.rpn_head(
            rpn_feature_maps, training=training)

        proposals = self.rpn_head.get_proposals(
            rpn_probs, rpn_deltas, img_metas)

        # Cascade Rcnn has one RPN stage and three detection stages
        # The iou threshold are [0.5, 0.6, 0.7] for three detection stages
        if training:
            rois, rcnn_labels, rcnn_label_weights, rcnn_delta_targets, rcnn_delta_weights = \
                self.bbox_target.build_targets(
                    proposals, gt_boxes, gt_class_ids, img_metas)
        else:
            rois = proposals

        pooled_regions = self.roi_align(
            (rois, rcnn_feature_maps, img_metas), training=training)

        rcnn_class_logits, rcnn_probs, rcnn_deltas = \
            self.bbox_head(pooled_regions, training=training)

        if training:
            rpn_class_loss, rpn_bbox_loss = self.rpn_head.loss(
                rpn_class_logits, rpn_deltas,
                gt_boxes, gt_class_ids, img_metas)

            rcnn_class_loss, rcnn_bbox_loss = self.bbox_head.loss(
                rcnn_class_logits, rcnn_deltas,
                rcnn_labels, rcnn_label_weights, rcnn_delta_targets, rcnn_delta_weights)

            return [rpn_class_loss, rpn_bbox_loss,
                    rcnn_class_loss, rcnn_bbox_loss]
        else:
            detections_list = self.bbox_head.get_bboxes(
                rcnn_probs, rcnn_deltas, rois, img_metas)

            return detections_list
