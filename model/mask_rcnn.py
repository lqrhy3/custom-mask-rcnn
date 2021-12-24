from typing import Tuple, List, Dict, Optional
from utils.anchor_generator import DefaultAnchorGenerator as AnchorGenerator

import torch
from torch import nn

from .rpn import RPN
from .fast_rcnn import FastRCNN

Tensor = torch.Tensor


class MaskRCNN(nn.Module):
    def __init__(
            self,
            backbone: nn.Module,
            backbone_out_channels: int,
            backbone_stride: int,
            anchor_sizes: List[int],
            anchor_ratios: List[int or float],
            image_size: Tuple[int, int],
            pre_nms_top_n: Optional[Dict[str, int]],
            proposal_min_size: Optional[float],
            objectness_prob_threshold: Optional[float],
            nms_iou_threshold: float,
            post_nms_top_n: Dict[str, int],
            nms_iou_pos_threshold: float,
            nms_iou_neg_threshold: float,
            rpn_batch_size: int,
            roi_output_size: Tuple[int, int],
            roi_spatial_scale: float,
            representation_dim: int,
            num_classes: int,
            fast_rcnn_batch_size: int,
            detections_per_img: int,
            fast_post_nms_top_n: Dict[str, int],
            inference_score_thresh: float,
            inference_nms_thresh: float,
            fast_nms_iou_pos_threshold: float,
            fast_nms_iou_neg_threshold: float
    ):
        super(MaskRCNN, self).__init__()

        self.backbone = backbone
        anchor_generator = AnchorGenerator(
            sizes=anchor_sizes, aspect_ratios=anchor_ratios, strides=[backbone_stride] * backbone_out_channels)

        self.rpn = RPN(
            backbone_out_channels=backbone_out_channels,
            anchor_generator=anchor_generator,
            image_size=image_size,
            pre_nms_top_n=pre_nms_top_n,
            proposal_min_size=proposal_min_size,
            objectness_prob_threshold=objectness_prob_threshold,
            nms_iou_threshold=nms_iou_threshold,
            post_nms_top_n=post_nms_top_n,
            nms_iou_neg_threshold=nms_iou_neg_threshold,
            nms_iou_pos_threshold=nms_iou_pos_threshold,
            rpn_batch_size=rpn_batch_size
        )

        self.fast_rcnn = FastRCNN(
            mask_branch=True,
            roi_output_size=roi_output_size,
            roi_spatial_scale=roi_spatial_scale,
            backbone_out_channels=backbone_out_channels,
            nms_iou_pos_threshold=fast_nms_iou_pos_threshold,
            nms_iou_neg_threshold=fast_nms_iou_neg_threshold,
            image_size=image_size,
            representation_dim=representation_dim,
            num_classes=num_classes,
            fast_rcnn_batch_size=fast_rcnn_batch_size,
            inference_score_thresh=inference_score_thresh,
            inference_nms_thresh=inference_nms_thresh,
            post_nms_top_n=fast_post_nms_top_n,
            detections_per_img=detections_per_img,
        )

    def forward(self, images, gt_boxes=None, gt_classes=None):
        backbone_features = self.backbone(images)['0']
        proposals, rpn_loss = self.rpn(backbone_features, gt_boxes)

        fast_rcnn_output, fast_rcnn_loss = self.fast_rcnn(backbone_features, proposals, gt_boxes, gt_classes)

        loss = {}
        if self.training:
            loss = self.compute_loss(rpn_loss, fast_rcnn_loss)

        return fast_rcnn_output, loss

    def compute_loss(self, rpn_loss: Dict[str, Tensor], fast_rcnn_loss: Dict[str, Tensor]) -> Dict[str, Tensor]:
        loss = {'loss': rpn_loss['loss'] + fast_rcnn_loss['loss']}
        del rpn_loss['loss']
        del fast_rcnn_loss['loss']

        rpn_loss = {f'rpn_{key}': value for key, value in rpn_loss.items()}
        fast_rcnn_loss = {f'fast_{key}': value for key, value in fast_rcnn_loss.items()}

        loss.update(rpn_loss)
        loss.update(fast_rcnn_loss)

        return loss
