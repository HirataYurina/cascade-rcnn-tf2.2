# -*- coding:utf-8 -*-
# author:平手友梨奈ii
# e-mail:1353593259@qq.com
# datetime:1993/12/01
# filename:cascade_rcnn.py
# software: PyCharm

import tensorflow as tf
from detection.models.backbones import resnet
from detection.models.necks import fpn
from detection.models.rpn_heads import rpn_head
from detection.models.bbox_heads import bbox_head
from detection.models.roi_extractors import roi_align
from detection.core.bbox import bbox_target
from detection.core.bbox import transforms
from detection.models.detectors.test_mixins import RPNTestMixin, BBoxTestMixin


class CascadeRCNN(tf.keras.Model, RPNTestMixin, BBoxTestMixin):
    '''Cascade RCNN
    Mismatching is excited in two stages detector like Faster-RCNN.
    What is mismatching?
    The proposals are fed into detection stage when training are different from predicting.
    The proposals are sampled by iou between proposals and gt.
    But we don't have gt when predicting. So, we feed all proposals into detection stage.
    And this leads distribution difference between training and predicting.

    '''
    def __init__(self, num_classes, **kwags):
        super(CascadeRCNN, self).__init__(**kwags)

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
        self.RCNN_POS_IOU_THR = [0.5, 0.6, 0.7]  # This can be smaller like [0.3, 0.4, 0.5] / [0.4, 0.5, 0.6]
        self.RCNN_NEG_IOU_THR = [0.5, 0.4, 0.3]

        # Boxes kept configuration
        self.RCNN_MIN_CONFIDENCE = 0.05
        self.RCNN_NMS_THRESHOLD = 0.5
        self.RCNN_MAX_INSTANCES = 100

        # Target Generator for the second stage.
        self.bbox_target1 = bbox_target.ProposalTarget(
            target_means=self.RCNN_TARGET_MEANS,
            target_stds=self.RPN_TARGET_STDS,
            num_rcnn_deltas=self.RCNN_BATCH_SIZE,
            positive_fraction=self.RCNN_POS_FRAC,
            pos_iou_thr=self.RCNN_POS_IOU_THR[0],
            neg_iou_thr=self.RCNN_NEG_IOU_THR[0],
            num_classes=self.NUM_CLASSES)

        self.bbox_target2 = bbox_target.ProposalTarget(
            target_means=self.RCNN_TARGET_MEANS,
            target_stds=self.RPN_TARGET_STDS,
            num_rcnn_deltas=self.RCNN_BATCH_SIZE,
            positive_fraction=self.RCNN_POS_FRAC,
            pos_iou_thr=self.RCNN_POS_IOU_THR[1],
            neg_iou_thr=self.RCNN_NEG_IOU_THR[1],
            num_classes=self.NUM_CLASSES)

        self.bbox_target3 = bbox_target.ProposalTarget(
            target_means=self.RCNN_TARGET_MEANS,
            target_stds=self.RPN_TARGET_STDS,
            num_rcnn_deltas=self.RCNN_BATCH_SIZE,
            positive_fraction=self.RCNN_POS_FRAC,
            pos_iou_thr=self.RCNN_POS_IOU_THR[2],
            neg_iou_thr=self.RCNN_NEG_IOU_THR[2],
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

        # first detection stage
        self.bbox_head1 = bbox_head.BBoxHead(num_classes=self.NUM_CLASSES,
                                             pool_size=self.POOL_SIZE,
                                             target_means=self.RCNN_TARGET_MEANS,
                                             target_stds=self.RCNN_TARGET_STDS,
                                             min_confidence=self.RCNN_MIN_CONFIDENCE,
                                             nms_threshold=self.RCNN_NMS_THRESHOLD,
                                             max_instances=self.RCNN_MAX_INSTANCES,
                                             name='b_box_head1')

        # second detection stage
        self.bbox_head2 = bbox_head.BBoxHead(num_classes=self.NUM_CLASSES,
                                             pool_size=self.POOL_SIZE,
                                             target_means=self.RCNN_TARGET_MEANS,
                                             target_stds=self.RCNN_TARGET_STDS,
                                             min_confidence=self.RCNN_MIN_CONFIDENCE,
                                             nms_threshold=self.RCNN_NMS_THRESHOLD,
                                             max_instances=self.RCNN_MAX_INSTANCES,
                                             name='b_box_head2')

        # third detection stage
        self.bbox_head3 = bbox_head.BBoxHead(num_classes=self.NUM_CLASSES,
                                             pool_size=self.POOL_SIZE,
                                             target_means=self.RCNN_TARGET_MEANS,
                                             target_stds=self.RCNN_TARGET_STDS,
                                             min_confidence=self.RCNN_MIN_CONFIDENCE,
                                             nms_threshold=self.RCNN_NMS_THRESHOLD,
                                             max_instances=self.RCNN_MAX_INSTANCES,
                                             name='b_box_head3')

    def __call__(self, inputs, training=True):
        if training:  # training
            imgs, img_metas, gt_boxes, gt_class_ids = inputs
        else:  # inference
            imgs, img_metas = inputs

        # How to train Faster-Rcnn
        # 1.get feature from backbone(resnet50)
        C2, C3, C4, C5 = self.backbone(imgs,
                                       training=training)

        # 2.feature pyramid
        P2, P3, P4, P5, P6 = self.neck([C2, C3, C4, C5],
                                       training=training)

        # The feature used in RPN
        rpn_feature_maps = [P2, P3, P4, P5, P6]
        # The feature used in ROI Pooling to get feature of ROI
        rcnn_feature_maps = [P2, P3, P4, P5]

        # 3.get prediction of RPN
        rpn_class_logits, rpn_probs, rpn_deltas = self.rpn_head(
            rpn_feature_maps, training=training)

        # 4.get proposals from prediction of RPN
        proposals = self.rpn_head.get_proposals(
            rpn_probs, rpn_deltas, img_metas)

        # ##########################################################################################################
        # TODO: you can add detection stage and use 4 detection stages.
        # Cascade Rcnn has one RPN stage and three detection stages
        # The iou threshold are [0.5, 0.6, 0.7] for three detection stages
        # The first detection stage
        if training:
            # 5.generate targets of proposals
            # This leads distribution difference between training and predicting.
            rois, rcnn_labels, rcnn_label_weights, rcnn_delta_targets, rcnn_delta_weights = \
                self.bbox_target1.build_targets(
                    proposals, gt_boxes, gt_class_ids, img_metas)
        else:
            rois = proposals

        # 6.use ROI Pooling to get feature of proposals
        pooled_regions = self.roi_align(
            (rois, rcnn_feature_maps, img_metas), training=training)

        # 7.get prediction of detection stage
        rcnn_class_logits, rcnn_probs, rcnn_deltas = \
            self.bbox_head1(pooled_regions, training=training)

        # The second detection stage
        proposals = transforms.delta2bbox(rois, rcnn_deltas, self.RCNN_TARGET_MEANS, self.RCNN_TARGET_STDS)
        if training:
            # sampling rois
            rois, rcnn_labels_2, rcnn_label_weights_2, rcnn_delta_targets_2, rcnn_delta_weights_2 = \
                self.bbox_target2.build_targets(proposals, gt_boxes, gt_class_ids, img_metas)
        else:
            rois = proposals
        pooled_regions = self.roi_align((rois, rcnn_feature_maps, img_metas), training=training)
        rcnn_class_logits_2, rcnn_probs_2, rcnn_deltas_2 = \
            self.bbox_head2(pooled_regions, training=training)

        # The third detection stage
        proposals = transforms.delta2bbox(rois, rcnn_deltas_2, self.RCNN_TARGET_MEANS, self.RCNN_TARGET_STDS)
        if training:
            # sampling rois
            rois, rcnn_labels_3, rcnn_label_weights_3, rcnn_delta_targets_3, rcnn_delta_weights_3 = \
                self.bbox_target3.build_targets(proposals, gt_boxes, gt_class_ids, img_metas)
        else:
            rois = proposals
        pooled_regions = self.roi_align((rois, rcnn_feature_maps, img_metas), training=training)
        rcnn_class_logits_3, rcnn_probs_3, rcnn_deltas_3 = \
            self.bbox_head3(pooled_regions, training=training)
        # ##########################################################################################################

        if training:
            # 8.get loss of RPN
            rpn_class_loss, rpn_bbox_loss = self.rpn_head.loss(
                rpn_class_logits, rpn_deltas,
                gt_boxes, gt_class_ids, img_metas)
            # 9.get loss of Fast R-cnn
            rcnn_class_loss, rcnn_bbox_loss = self.bbox_head1.loss(rcnn_class_logits,
                                                                   rcnn_deltas,
                                                                   rcnn_labels,
                                                                   rcnn_label_weights,
                                                                   rcnn_delta_targets,
                                                                   rcnn_delta_weights)
            rcnn_class_loss_2, rcnn_bbox_loss_2 = self.bbox_head2.loss(rcnn_class_logits_2,
                                                                       rcnn_deltas_2,
                                                                       rcnn_labels_2,
                                                                       rcnn_label_weights_2,
                                                                       rcnn_delta_targets_2,
                                                                       rcnn_delta_weights_2)
            rcnn_class_loss_3, rcnn_bbox_loss_3 = self.bbox_head3.loss(rcnn_class_logits_3,
                                                                       rcnn_deltas_3,
                                                                       rcnn_labels_3,
                                                                       rcnn_label_weights_3,
                                                                       rcnn_delta_targets_3,
                                                                       rcnn_delta_weights_3)

            return [rpn_class_loss, rpn_bbox_loss,
                    rcnn_class_loss, rcnn_bbox_loss,
                    rcnn_class_loss_2, rcnn_bbox_loss_2,
                    rcnn_class_loss_3, rcnn_bbox_loss_3]
        else:
            # TODO：get class_probs_average of three detection stages
            rcnn_class_logits_2_3, rcnn_probs_2_3, _ = self.bbox_head2(pooled_regions, training=training)
            rcnn_class_logits_1_3, rcnn_probs_1_3, _ = self.bbox_head1(pooled_regions, training=training)
            rcnn_class_logits_average = (rcnn_class_logits_3 + rcnn_class_logits_1_3 + rcnn_class_logits_2_3) / 3
            rcnn_class_probs_average = (rcnn_probs_3 + rcnn_probs_1_3 + rcnn_probs_2_3) / 3

            detections_list = self.bbox_head3.get_bboxes(rcnn_class_probs_average,
                                                         rcnn_deltas_3,
                                                         rois,
                                                         img_metas)

            return detections_list


if __name__ == '__main__':
    cascade_rcnn = CascadeRCNN(num_classes=81)
    a = tf.random.normal(shape=(10, 1))
    print(tf.multiply(a, a))
