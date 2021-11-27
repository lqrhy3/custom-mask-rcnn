from typing import Tuple, List, Dict, Optional
from utils.anchor_generator import DefaultAnchorGenerator as AnchorGenerator

from torch import nn

from .rpn import RPN
from .fast_rcnn import FastRCNN


class FasterRCNN(nn.Module):
    def __init__(
            self,
            backbone: nn.Module,
            backbone_out_channels: int,
            backbone_stride: int,
            anchor_sizes: List[List[int]],
            anchor_ratios: List[List[int or float]],
            batch_size: int,
            image_size: Tuple[int, int],
            pre_nms_top_n: Optional[Dict[str, int]],
            proposal_min_size: Optional[float],
            objectness_prob_threshold: Optional[float],
            nms_iou_threshold: float,
            post_nms_top_n: Dict[str, int],
            nms_iou_pos_threshold: float,
            nms_iou_neg_threshold: float,
            rpn_batch_size: int,
            roi_output_size: int,
            roi_spatial_scale: float,
            representation_dim: int,
            num_classes: int
    ):
        super(FasterRCNN, self).__init__()

        self.backbone = backbone

        anchor_generator = AnchorGenerator(
            sizes=anchor_sizes, aspect_ratios=anchor_ratios, strides=[backbone_stride] * batch_size)

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
            roi_output_size, roi_spatial_scale, backbone_out_channels, representation_dim, num_classes
        )

    def forward(self, images, gt_boxes=None, gt_classes=None):
        backbone_features = self.backbone(images)['features']
        proposals, rpn_loss = self.rpn(backbone_features, gt_boxes)

        fast_rcnn_output = self.fast_rcnn(backbone_features, proposals, gt_boxes, gt_classes)
        return fast_rcnn_output
