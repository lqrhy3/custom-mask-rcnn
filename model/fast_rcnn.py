from typing import Tuple
from torch import nn
from torchvision.ops import RoIPool


class FastRCNN(nn.Module):
    def __init__(
            self,
            roi_output_size: int or Tuple[int, int],
            roi_spatial_scale: float,
            backbone_out_channels: int,
            representation_dim: int,
            num_classes: int
    ):
        super(FastRCNN, self).__init__()

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

    def forward(self, backbone_features, proposals, gt_boxes, gt_classes):
        rois = self.roi(backbone_features, proposals)
        features = self.mlp(rois)

        box_predictions = self.box_predictor(features)
        cls_predictions = self.cls_predictor(features)

        loss = {}
        if self.training:
            loss = self.compute_loss(box_predictions, cls_predictions, gt_boxes, gt_classes)

        return box_predictions, cls_predictions, loss

    def compute_loss(self, box_preds, cls_preds, gt_boxes, gt_classes):

        return {'loss': ...}