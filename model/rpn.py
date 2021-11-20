import torch
from torch import nn
from torchvision.ops import clip_boxes_to_image, batched_nms, box_iou

from utils.utils import anchor_transforms_to_boxes, get_cross_boundary_box_idxs


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
            backbone: nn.Module,
            backbone_out_features: int,
            anchor_generator,
            image_size,
            pre_nms_top_n: int,
            post_nms_top_n: int,
            nms_iou_threshold: float
    ):
        super().__init__()

        self.backbone = backbone
        self.anchor_generator = anchor_generator
        self.image_size = image_size
        self.pre_nms_top_n = pre_nms_top_n
        self.post_nms_top_n = post_nms_top_n
        self.nms_iou_threshold = nms_iou_threshold

        self.sliding_network = SlidingNetwork(
            backbone_out_features, len(self.anchors)
        )

    def forward(self, x, gt_boxes: list = None, gt_classes: list = None):
        backbone_features = self.backbone(x)['features']

        anchor_transforms, anchor_objectnesses = self.sliding_network(backbone_features)  # [N, 4 * num_anchors, H, W]
                                                                                          # [N, num_anchors, H, W]
        anchor_transforms = torch.reshape(anchor_transforms, (anchor_transforms.shape[0], 4, -1)).permute((0, 2, 1))

        anchors = self.anchor_generator(backbone_features)
        proposals = anchor_transforms_to_boxes(anchor_transforms, anchors)

        proposal_objectnesses = torch.reshape(anchor_objectnesses, (anchor_transforms.shape[0], -1))

        if self.training:
            assert gt_boxes is not None
            proposals, proposal_objectnesses = self._filter_proposals(proposals, proposal_objectnesses, gt_classes)
            proposals_batched, objectnesses_batched = self._collate_fn(
                proposals, proposal_objectnesses, gt_boxes
            )  # TODO: return positive and negative labels

            losses = ...  # compute losses
            return proposals_batched, objectnesses_batched, losses

        else:
            proposals, proposal_objectnesses = clip_boxes_to_image(proposals, self.image_size)
            proposals = self._filter_proposals(proposals, proposal_objectnesses, gt_classes)

            return backbone_features, proposals, proposal_objectnesses

    def _collate_fn(self, proposals, proposal_objectnesses, gt_boxes: list):
        """
        :param proposals: [N, post_nms_top_k, 4] zero padded
        :param proposal_objectnesses: [N, post_nms_top_k] zero padded
        :param gt_boxes: [Tensor[G_i, 4], ...]
        :return [N, 256, 4]
        """
        batch_size = proposals.shape[0]
        batched_proposals = torch.empty((batch_size, 256, 4))
        batched_objectnesses = torch.empty((batch_size, 256))

        for i in range(batch_size):
            proposals_i = proposals[i]
            proposal_objectnesses_i = proposal_objectnesses[i]
            gt_boxes_i = gt_boxes[i]

            ious = box_iou(proposals_i, gt_boxes_i)
            mask = torch.any(ious > 0.7, dim=1)
            mask = mask | torch.any(ious == torch.max(ious, dim=0)[0], dim=1)

            positive_idxs = mask.nonzero()
            negative_idxs = (~mask).nonzero()

            num_positive_samples = 128 if positive_idxs.shape[0] >= 128 else positive_idxs.shape[0]
            num_negative_samples = 128 if num_positive_samples == 128 else 256 - num_positive_samples

            positive_idxs = positive_idxs[torch.randperm(len(positive_idxs))[:num_positive_samples]]
            negative_idxs = negative_idxs[torch.randperm(len(negative_idxs))[:num_negative_samples]]

            positive_proposals_i = proposals_i[positive_idxs].squeeze(1)
            positive_objectnesses_i = proposal_objectnesses_i[positive_idxs].squeeze(1)
            negative_proposals_i = proposals_i[negative_idxs].squeeze(1)
            negative_objectnesses_i = proposal_objectnesses_i[negative_idxs].squeeze(1)

            batched_proposals[i] = torch.cat([positive_proposals_i, negative_proposals_i])
            batched_objectnesses[i] = torch.cat([positive_objectnesses_i, negative_objectnesses_i])

        return batched_proposals, batched_objectnesses

    def _filter_proposals(self, proposals, proposal_objectnesses, gt_classes):
        """
        :param proposals: [N, K, 4]
        :param proposal_objectnesses: [N, K],
        :param gt_classes: [Tensor[G_i,], ...]
        """
        batch_size = proposals.shape[0]
        proposal_objectnesses = proposal_objectnesses.detach()

        _, orders = torch.sort(proposal_objectnesses, dim=1)
        output_proposals = torch.zeros((batch_size, self.post_nms_top_n, 4), dtype=proposals.dtype)
        output_objectnesses = torch.zeros((batch_size, self.post_nms_top_n), dtype=proposal_objectnesses.dtype)

        for i in range(batch_size):
            proposals_i, proposal_objectnesses_i = proposals[i], proposal_objectnesses[i]
            orders_i = orders[i]
            gt_classes_i = gt_classes[i]

            if self.training:
                cross_boundary_box_idxs = get_cross_boundary_box_idxs(proposals_i, self.image_size)
                proposals_i = proposals_i[~cross_boundary_box_idxs]
                proposal_objectnesses_i = proposal_objectnesses_i[~cross_boundary_box_idxs]
                orders_i = orders_i[~cross_boundary_box_idxs]

            if self.pre_nms_top_n is not None and self.pre_nms_top_n > 0:
                orders_i = orders_i[:self.post_nms_top_n]

            proposals_i = proposals_i[orders_i]
            proposal_objectnesses_i = proposal_objectnesses_i[orders_i]

            keep_idx = batched_nms(proposals_i, proposal_objectnesses_i, gt_classes_i, self.nms_iou_threshold)
            if self.post_nms_top_n > 0:
                keep_idx = keep_idx[:self.post_nms_top_n]

            proposals_i = proposals_i[keep_idx]
            proposal_objectnesses_i = proposal_objectnesses_i[keep_idx]

            num_proposal = proposals_i.shape[0]
            output_proposals[i, :num_proposal, :] = proposals_i
            output_objectnesses[i, :num_proposal] = proposal_objectnesses_i

        return output_proposals, output_objectnesses
