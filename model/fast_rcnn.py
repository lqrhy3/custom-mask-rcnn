from typing import Tuple, List, Dict

import torch
from torch import nn
from torchvision.ops import RoIPool, box_iou

from utils.utils import ProposalMathcer, BalancedBatchSampler, boxes_to_anchor_transforms


Tensor = torch.tensor


class FastRCNN(nn.Module):
    def __init__(
            self,
            roi_output_size: int or Tuple[int, int],
            roi_spatial_scale: float,
            backbone_out_channels: int,
            nms_iou_pos_threshold: float,
            nms_iou_neg_threshold: float,
            image_size: Tuple[int, int],
            representation_dim: int,
            num_classes: int,
            fast_rcnn_batch_size: int
    ):
        super(FastRCNN, self).__init__()

        self.fast_rcnn_batch_size = fast_rcnn_batch_size
        self.proposal_matcher = ProposalMathcer(
            nms_iou_pos_threshold, nms_iou_neg_threshold, image_size
        )

        assert fast_rcnn_batch_size % 2 == 0
        self.batch_sampler = BalancedBatchSampler(fast_rcnn_batch_size)

        self.roi = RoIPool(roi_output_size, roi_spatial_scale)

        roi_box_size = self.roi.output_size[0]
        self.mlp = nn.Sequential(
            nn.Flatten(),
            nn.Linear(backbone_out_channels * roi_box_size ** 2, 1024),
            nn.ReLU(),
            nn.Linear(1024, representation_dim),
            nn.ReLU()
        )

        self.box_predictor = nn.Linear(representation_dim, 4 * num_classes)
        self.cls_predictor = nn.Linear(representation_dim, num_classes)

        self.box_criterion = nn.SmoothL1Loss(reduction='sum')
        self.cls_criterion = nn.CrossEntropyLoss(reduction='mean')

    def forward(
            self,
            backbone_features: Tensor,
            proposals: List[Tensor],
            gt_boxes: List[Tensor],
            gt_classes: List[Tensor]
    ) -> Tuple[List[Tensor], Dict[str, Tensor]]:

        rois = self.roi(backbone_features, proposals) # [K, C, h, w]
        features = self.mlp(rois)
        self.box_coder.decode(box_regression, proposals)
        pred_transforms = self.box_predictor(features)
        pred_logits = self.cls_predictor(features)

        loss = {}
        if self.training:
            matched_boxes, gt_labels = self.assign_targets_to_proposals(proposals, gt_boxes, gt_classes)
            gt_transforms = boxes_to_anchor_transforms(matched_boxes, proposals)
            batched_pred_transforms, batched_pred_logits, batched_gt_transforms, batched_gt_labels =\
                self._collate_fn(pred_transforms, pred_logits, gt_transforms, gt_labels)

            loss = self.compute_loss(
                batched_pred_transforms, batched_pred_logits, batched_gt_transforms, batched_gt_labels
            )

        return pred_transforms, pred_logits, loss

    def compute_loss(
            self,
            pred_transforms: Tensor,
            pred_logits: Tensor,
            gt_transforms: Tensor,
            gt_labels: Tensor
    ) -> Dict[str, Tensor]:

        cls_loss = self.cls_criterion(pred_logits, gt_labels)

        pred_transforms = pred_transforms.reshape(self.fast_rcnn_batch_size, pred_transforms.shape[-1] // 4, 4)
        pos_idxs = torch.where(gt_labels > 0)[0]
        pos_labels = gt_labels[pos_idxs]

        box_loss = self.box_criterion(pred_transforms[pos_idxs, pos_labels], gt_transforms[pos_idxs]) / gt_labels.numel()

        loss = box_loss + cls_loss

        return {'loss': loss, 'box_loss': box_loss.detach(), 'cls_loss': cls_loss.detach()}

    def assign_targets_to_proposals(
            self,
            proposals: List[Tensor],
            gt_boxes: List[Tensor],
            gt_classes: List[Tensor]
    ) -> Tuple[List[Tensor], List[Tensor]]:

        batch_sise = len(proposals)

        matched_boxes = []
        labels = []

        for i in range(batch_sise):
            proposals_i = proposals[i]
            gt_boxes_i, gt_classes_i = gt_boxes[i], gt_classes[i]

            matched_idxs_i = self.proposal_matcher(proposals_i, gt_boxes_i)
            clamped_matched_idxs_i = torch.clamp(matched_idxs_i, min=0)

            matched_boxes_i = gt_boxes_i[clamped_matched_idxs_i]
            classes_i = gt_classes_i[clamped_matched_idxs_i]

            neg_idxs = matched_idxs_i == self.proposal_matcher.NEG_CONST
            classes_i[neg_idxs] = 0

            discard_idxs = matched_idxs_i == self.proposal_matcher.DISCARD_CONST
            classes_i[discard_idxs] = -1

            matched_boxes.append(matched_boxes_i)
            labels.append(classes_i)

        return matched_boxes, labels

    def _collate_fn(self, pred_transforms, pred_logits, gt_transforms, gt_labels):
        batch_size = pred_transforms.shape[0]

        batched_pred_transforms = torch.empty((batch_size, self.fast_rcnn_batch_size, 4))
        batched_pred_logits = torch.empty((batch_size, self.fast_rcnn_batch_size))
        batched_gt_transforms = torch.empty((batch_size, self.fast_rcnn_batch_size, 4))
        batched_gt_labels = torch.empty((batch_size, self.fast_rcnn_batch_size))

        for i in range(batch_size):
            pred_transforms_i, pred_logits_i = pred_transforms[i], pred_logits[i]
            gt_transforms_i, gt_labels_i = gt_transforms[i], gt_labels[i]

            pos_idxs, neg_idxs = self.batch_sampler(gt_labels_i)

            batched_pred_transforms[i] = torch.cat([
                pred_transforms_i[pos_idxs], pred_transforms_i[neg_idxs]
            ], dim=0)

            batched_gt_transforms[i] = torch.cat([
                gt_transforms_i[pos_idxs], gt_transforms_i[neg_idxs]
            ], dim=0)

            batched_pred_logits[i] = torch.cat([
                pred_logits_i[pos_idxs], pred_logits_i[neg_idxs]
            ], dim=0)

            batched_gt_labels[i] = torch.cat([
                gt_labels_i[pos_idxs], gt_labels_i[neg_idxs]
            ], dim=0)

        return batched_pred_transforms, batched_pred_logits, batched_gt_transforms, batched_gt_labels

    def filter_predictions(self, pred_transforms, pred_logits, proposals):
        pred_scores = torch.softmax(pred_logits, dim=-1)
