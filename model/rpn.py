from typing import Tuple, List, Dict, Optional

import matplotlib.pyplot as plt
import torch
from torch import nn
from torchvision.ops import clip_boxes_to_image, nms, batched_nms, box_iou, box_convert, remove_small_boxes
from torchvision.utils import draw_bounding_boxes

from utils.utils import anchor_transforms_to_boxes, boxes_to_anchor_transforms, get_cross_boundary_box_idxs, ProposalMathcer, BalancedBatchSampler


Tensor = torch.tensor


class SlidingNetwork(nn.Module):
    def __init__(self, in_channels: int, num_anchors: int):
        super(SlidingNetwork, self).__init__()

        self.feature_extractor = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.ReLU()
        )

        self.box_regressor = nn.Conv2d(in_channels, 4 * num_anchors, kernel_size=(1, 1))
        self.objectness_scorer = nn.Conv2d(in_channels, num_anchors, kernel_size=(1, 1))

    def forward(self, backbone_features):
        features = self.feature_extractor(backbone_features)
        anchor_transforms = self.box_regressor(features)
        anchor_objectnesses = self.objectness_scorer(features)

        return anchor_transforms, anchor_objectnesses


class RPN(nn.Module):
    def __init__(
            self,
            backbone_out_channels: int,
            anchor_generator,
            image_size: Tuple[int, int],
            pre_nms_top_n: Optional[Dict[str, int]],
            proposal_min_size: Optional[float],
            objectness_prob_threshold: Optional[float],
            nms_iou_threshold: float,
            post_nms_top_n: Dict[str, int],
            nms_iou_pos_threshold: float,
            nms_iou_neg_threshold: float,
            rpn_batch_size: int
    ):
        super().__init__()

        self.anchor_generator = anchor_generator
        self.image_size = image_size
        self.pre_nms_top_n = pre_nms_top_n
        self.proposal_min_size = proposal_min_size
        self.objectness_prob_threshold = objectness_prob_threshold
        self.nms_iou_threshold = nms_iou_threshold
        self.post_nms_top_n = post_nms_top_n

        self.proposal_matcher = ProposalMathcer(
            nms_iou_pos_threshold, nms_iou_neg_threshold, image_size
        )

        assert rpn_batch_size % 2 == 0
        self.batch_sampler = BalancedBatchSampler(rpn_batch_size)

        self.sliding_network = SlidingNetwork(
            backbone_out_channels, self.anchor_generator.num_cell_anchors[0]
        )

        self.cls_criterion = nn.BCEWithLogitsLoss(reduction='mean')
        self.reg_criterion = nn.SmoothL1Loss(reduction='sum')

    def forward(self, backbone_features: Tensor, gt_boxes: list = None) -> Tuple[List[Tensor], Dict[str, Tensor]]:
        anchor_transforms, anchor_objectnesses = self.sliding_network(backbone_features)  # [N, 4 * num_anchors, H, W]
                                                                                          # [N, num_anchors, H, W]
        anchor_transforms = torch.reshape(anchor_transforms, (anchor_transforms.shape[0], 4, -1)).permute((0, 2, 1))

        anchors = self.anchor_generator(backbone_features)
        proposals = anchor_transforms_to_boxes(anchor_transforms.detach(), anchors)

        proposal_objectnesses = torch.reshape(anchor_objectnesses.detach().clone(), (anchor_transforms.shape[0], -1))

        proposals = self._filter_proposals(proposals, proposal_objectnesses)

        loss = {}
        if self.training:
            assert gt_boxes is not None

            matched_gt_boxes, labels = self.assign_targets_to_anchors(anchors, gt_boxes)
            gt_transforms = boxes_to_anchor_transforms(matched_gt_boxes, anchors)

            anchor_objectnesses = torch.reshape(anchor_objectnesses, (anchor_transforms.shape[0], -1))

            anchor_transforms_batched, gt_transforms_batched, objectnesses_bathed, labels_batched = self._collate_fn(
                anchor_transforms, gt_transforms, anchor_objectnesses, labels
            )
            loss = self.compute_loss(
                anchor_transforms_batched, gt_transforms_batched, objectnesses_bathed, labels_batched
            )

        return proposals, loss

    def _filter_proposals(self, proposals: Tensor, objectnesses: Tensor) -> List[Tensor]:
        """
        :param proposals: [N, K, 4]
        :param objectnesses: [N, K],
        """
        batch_size = proposals.shape[0]

        proposals = clip_boxes_to_image(proposals, self.image_size)

        objectnesses = objectnesses.detach()
        objectnesses_prob = torch.sigmoid(objectnesses)

        # select top N pre nms proposals based on objectness_prob
        if self.pre_nms_top_n is not None:
            pre_nms_top_n = min(self._pre_nms_top_n(), proposals.shape[1])
            _, top_n_idxs = torch.topk(objectnesses_prob, pre_nms_top_n, dim=1)

            proposals = proposals[torch.arange(batch_size)[:, None], top_n_idxs]
            objectnesses_prob = objectnesses_prob[torch.arange(batch_size)[:, None], top_n_idxs]

        filtered_proposals = []

        for i in range(batch_size):
            proposals_i, objectnesses_prob_i = proposals[i], objectnesses_prob[i]

            if self.proposal_min_size is not None:
                keep = remove_small_boxes(proposals_i, self.proposal_min_size)
                proposals_i = proposals_i[keep]
                objectnesses_prob_i = objectnesses_prob_i[keep]

            if self.objectness_prob_threshold is not None:
                keep = torch.where(objectnesses_prob_i >= self.objectness_prob_threshold)[0]
                proposals_i = proposals_i[keep]
                objectnesses_prob_i = objectnesses_prob_i[keep]

            keep = nms(proposals_i, objectnesses_prob_i, self.nms_iou_threshold)
            if self._post_nms_top_n() > 0:
                keep = keep[:self._post_nms_top_n()]

            proposals_i = proposals_i[keep]
            objectnesses_prob_i = objectnesses_prob_i[keep]

            filtered_proposals.append(proposals_i)
        return filtered_proposals

    def assign_targets_to_anchors(self, anchors: Tensor, gt_boxes: List[Tensor]) -> Tuple[Tensor, Tensor]:
        """
        :param anchors: Tensor[N, K, 4]
        :param gt_boxes: List[Tensor[G_i, 4], ...]
        """
        batch_size = anchors.shape[0]

        matched_gt_boxes = torch.empty_like(anchors)
        labels = torch.empty((batch_size, anchors.shape[1]))

        for i in range(batch_size):
            anchors_i = box_convert(anchors[i], 'cxcywh', 'xyxy')
            gt_boxes_i = gt_boxes[i]

            matches_i = self.proposal_matcher(anchors_i, gt_boxes_i)
            matched_gt_boxes_i = gt_boxes_i[matches_i.clamp(min=0)]

            labels_i = matches_i >= 0
            labels_i = labels_i.to(dtype=torch.float32)

            neg_idxs = matches_i == -1
            labels_i[neg_idxs] = 0.

            discard_idxs = matches_i == -2
            labels_i[discard_idxs] = -1.

            matched_gt_boxes[i] = matched_gt_boxes_i
            labels[i] = labels_i

        return matched_gt_boxes, labels

    def _collate_fn(self, anchor_transforms: Tensor, gt_transforms: Tensor, objectnesses: Tensor, labels: Tensor) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
        """
        :param anchor_transforms: [N, K, 4]
        :param gt_transforms: [N, K, 4]
        :param objectnesses: [N, K]
        :param labels: [N, K]
        """
        batch_size = anchor_transforms.shape[0]

        anchor_transforms_batched = torch.empty((batch_size, self.rpn_batch_size, 4))
        gt_transforms_batched = torch.empty((batch_size, self.rpn_batch_size, 4))
        objectnesses_batched = torch.empty((batch_size, self.rpn_batch_size))
        labels_batched = torch.empty((batch_size, self.rpn_batch_size))

        for i in range(batch_size):
            anchor_transforms_i = anchor_transforms[i]
            gt_transforms_i = gt_transforms[i]
            objectnesses_i = objectnesses[i]
            labels_i = labels[i]

            pos_idxs_i, neg_idxs_i = self.batch_sampler(labels_i)

            anchor_transforms_batched[i] = torch.cat(
                [anchor_transforms_i[pos_idxs_i], anchor_transforms_i[neg_idxs_i]], dim=0
            )

            gt_transforms_batched[i] = torch.cat(
                [gt_transforms_i[pos_idxs_i], gt_transforms_i[neg_idxs_i]], dim=0
            )

            objectnesses_batched[i] = torch.cat([objectnesses_i[pos_idxs_i], objectnesses_i[neg_idxs_i]], dim=0)
            labels_batched[i] = torch.cat([labels_i[pos_idxs_i], labels_i[neg_idxs_i]], dim=0)

        return anchor_transforms_batched, gt_transforms_batched, objectnesses_batched, labels_batched

    def compute_loss(self, anchor_transforms, gt_transforms, objectnesses, labels):
        cls_loss = self.cls_criterion(objectnesses, labels)

        pos_anchor_transforms = anchor_transforms[labels.nonzero()]
        pos_gt_transforms = gt_transforms[labels.nonzero()]
        reg_loss = self.reg_criterion(pos_anchor_transforms, pos_gt_transforms) / labels.numel()

        loss = cls_loss + reg_loss

        return {'loss': loss, 'cls_loss': cls_loss.detach(), 'reg_loss': reg_loss.detach()}

    def _pre_nms_top_n(self) -> int:
        if self.training:
            return self.pre_nms_top_n['training']
        return self.pre_nms_top_n['testing']

    def _post_nms_top_n(self) -> int:
        if self.training:
            return self.post_nms_top_n['training']
        return self.post_nms_top_n['testing']
