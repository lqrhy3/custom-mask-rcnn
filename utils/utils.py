from typing import Tuple, List, Dict

import torch
from torchvision.ops.boxes import box_convert, box_iou

Tensor = torch.tensor


def anchor_transforms_to_boxes(anchor_transforms, anchors):
    """
    :param anchors: [N, K * num_anchors, 4]
    :param anchor_transforms: [N, K * num_anchors, 4]
    :return:
    """

    boxes = torch.empty_like(anchors)
    boxes[:, :, 0] = anchors[:, :, 2] * anchor_transforms[:, :, 0] + anchors[:, :, 0]
    boxes[:, :, 1] = anchors[:, :, 3] * anchor_transforms[:, :, 1] + anchors[:, :, 1]
    boxes[:, :, 2] = torch.exp(anchor_transforms[:, :, 2]) * anchors[:, :, 2]
    boxes[:, :, 3] = torch.exp(anchor_transforms[:, :, 3]) * anchors[:, :, 3]

    boxes = box_convert(boxes, 'cxcywh', out_fmt='xyxy')

    return boxes


def boxes_to_anchor_transforms(boxes: Tensor, anchors: Tensor) -> Tensor:
    """
    :param boxes: List[Tensor[K * num_anchors, 4], ...]
    :param anchors: Tensor[N, K * num_anchors, 4]
    """

    boxes = torch.stack(boxes, dim=0)
    boxes = box_convert(boxes, 'xyxy', out_fmt='cxcywh')

    tx = (boxes[:, :, 0] - anchors[:, :, 0]) / anchors[:, :, 2]
    ty = (boxes[:, :, 1] - anchors[:, :, 1]) / anchors[:, :, 3]

    tw = torch.log(boxes[:, :, 2] / anchors[:, :, 2])
    th = torch.log(boxes[:, :, 3] / anchors[:, :, 3])

    return torch.stack([tx, ty, tw, th], dim=-1)




def get_cross_boundary_box_idxs(boxes, image_size):
    """
    :param boxes: [K, 4]
    :param image_size: [2,]
    """
    mask = torch.zeros((boxes.shape[0],), dtype=torch.bool, device=boxes.device)
    mask = mask | (boxes[:, 0] < 0)
    mask = mask | (boxes[:, 0] > image_size[1])

    mask = mask | (boxes[:, 1] < 0)
    mask = mask | (boxes[:, 1] > image_size[0])

    mask = mask | (boxes[:, 2] < 0)
    mask = mask | (boxes[:, 2] > image_size[1])

    mask = mask | (boxes[:, 3] < 0)
    mask = mask | (boxes[:, 3] > image_size[0])

    idxs = mask.nonzero().squeeze(-1)
    return idxs


class ProposalMathcer:
    def __init__(self,
                 nms_iou_pos_threshold: float,
                 nms_iou_neg_threshold: float,
                 image_size: Tuple[int, int]):

        self.nms_iou_pos_threshold = nms_iou_pos_threshold
        self.nms_iou_neg_threshold = nms_iou_neg_threshold
        self.image_size = image_size

        self.NEG_CONST = -1
        self.DISCARD_CONST = -2

    def __call__(self, proposals, gt_boxes):
        """
        :param proposals: [K, 4]
        :param gt_boxes: Tensor[G_i, 4]
        """

        ious = box_iou(proposals, gt_boxes)  # K x G_i
        matched_vals, matches = ious.max(dim=1)

        below_threshold = matched_vals < self.nms_iou_neg_threshold
        between_thresholds = (matched_vals >= self.nms_iou_neg_threshold) &\
                             (matched_vals < self.nms_iou_pos_threshold)

        cross_boundary = get_cross_boundary_box_idxs(proposals, self.image_size)

        matches[below_threshold] = self.NEG_CONST
        matches[between_thresholds] = self.DISCARD_CONST
        matches[cross_boundary] = self.DISCARD_CONST

        return matches


class BalancedBatchSampler:
    def __init__(self, batch_size: int):
        self.batch_size = batch_size

    def __call__(self, labels):
        pos_idxs = torch.where(labels > 0)[0]
        neg_idxs = torch.where(labels == 0)[0]

        num_pos = self.batch_size // 2 if len(pos_idxs) >= self.batch_size // 2 else len(pos_idxs)
        num_neg = self.batch_size - num_pos
        pos_idxs = pos_idxs[torch.randperm(len(pos_idxs))[:num_pos]]
        neg_idxs = neg_idxs[torch.randperm(len(neg_idxs))[:num_neg]]

        return pos_idxs, neg_idxs
