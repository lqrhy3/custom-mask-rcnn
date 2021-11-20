from torch import nn

from .rpn import RPN
from .fast_rcnn import FastRCNN


class FasterRCNN(nn.Module):
    def __init__(
            self,
            backbone: nn.Module,
            backbone_out_channels: int,
            num_anchors: int,
            roi_output_size: int,
            roi_spatial_scale: float,
            representation_dim: int,
            num_classes: int
    ):
        super(FasterRCNN, self).__init__()

        self.rpn = RPN(
            backbone,
            backbone_out_channels,
            num_anchors
        )

        self.fast_rcnn = FastRCNN(
            roi_output_size, roi_spatial_scale, backbone_out_channels, representation_dim, num_classes
        )

    def forward(self, images):
        rpn_output = self.rpn(images)
        features, rois = rpn_output['features'], rpn_output['rois']  # features[N, 512, H, W, 7]; rois[N, ]

        fast_rcnn_output = self.fast_rcnn(rpn_output)
        return fast_rcnn_output
