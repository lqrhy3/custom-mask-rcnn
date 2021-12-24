from typing import Tuple, List, Dict

import torch
from torchvision.ops.boxes import box_convert, box_iou

Tensor = torch.Tensor


def transforms_to_boxes(transforms: Tensor, anchors: List[Tensor]) -> Tensor:
    """
    :param transforms: Tensor[L x 4 * num_classes] (ctx, cty, tw, th)
    :param anchors: List[Tensor[L_i, 4] x N] (cxcywh format)
    :return: (xyxy format)
    """
    boxes_per_image = [boxes_in_image.shape[0] for boxes_in_image in anchors]
    num_boxes = sum(boxes_per_image)
    concat_anchors = torch.cat(anchors, dim=0)

    transforms = transforms.reshape(num_boxes, -1)

    boxes = torch.empty_like(transforms)
    boxes[:, 0::4] = concat_anchors[:, 0].view(-1, 1) + concat_anchors[:, 2].view(-1, 1) * transforms[:, 0::4]
    boxes[:, 1::4] = concat_anchors[:, 1].view(-1, 1) + concat_anchors[:, 3].view(-1, 1) * transforms[:, 1::4]

    boxes[:, 2::4] = concat_anchors[:, 2].view(-1, 1) * torch.exp(transforms[:, 2::4])
    boxes[:, 3::4] = concat_anchors[:, 3].view(-1, 1) * torch.exp(transforms[:, 3::4])

    boxes = boxes.reshape(num_boxes, -1, 4)
    boxes = box_convert(boxes, 'cxcywh', out_fmt='xyxy')

    return boxes


def boxes_to_transforms(boxes: List[Tensor], anchors: List[Tensor]) -> List[Tensor]:
    """
    :param boxes: List[Tensor[L_i x 4], N] (xyxy format)
    :param anchors: List[Tensor[L_i x 4], N] (cxcywh format)
    :return (ctx, cty, tw, th)
    """

    boxes_per_image = [boxes_in_image.shape[0] for boxes_in_image in boxes]

    boxes = torch.cat(boxes, dim=0)
    boxes = box_convert(boxes, 'xyxy', out_fmt='cxcywh')
    anchors = torch.cat(anchors, dim=0)

    tx = (boxes[:, 0] - anchors[:, 0]) / anchors[:, 2]
    ty = (boxes[:, 1] - anchors[:, 1]) / anchors[:, 3]

    tw = torch.log(boxes[:, 2] / anchors[:, 2])
    th = torch.log(boxes[:, 3] / anchors[:, 3])

    transforms = torch.stack([tx, ty, tw, th], dim=-1)

    return transforms.split(boxes_per_image)


def get_cross_boundary_box_idxs(boxes: Tensor, image_size: Tuple[int, int]):
    """
    :param boxes: [K, 4] (format xyxy)
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
        :param proposals: [K, 4] (format xyxy)
        :param gt_boxes: Tensor[G_i, 4] (format xyxy)
        """

        ious = box_iou(proposals, gt_boxes)  # K x G_i
        matched_vals, matches = ious.max(dim=1)

        below_threshold = matched_vals < self.nms_iou_neg_threshold
        between_thresholds = (matched_vals >= self.nms_iou_neg_threshold) &\
                             (matched_vals < self.nms_iou_pos_threshold)

        cross_boundary = get_cross_boundary_box_idxs(proposals, self.image_size)

        matches[below_threshold] = self.NEG_CONST
        matches[between_thresholds] = self.DISCARD_CONST
        # matches[cross_boundary] = self.DISCARD_CONST

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
