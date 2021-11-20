from typing import Tuple
from torch import nn
from torchvision.ops import RoIPool as RoIPoolBase, roi_pool


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

    def forward(self, input: dict):
        features, rois = self._unpack_input(input)
        rois_features = self.roi({'features': features, 'rois': rois})['roi_features']
        features = self.mlp(rois_features)

        box_predictions = self.box_predictor(features)
        cls_predictions = self.cls_predictor(features)

        return {'box_predictions': box_predictions, 'cls_predictions': cls_predictions}

    def _unpack_input(self, input: dict):
        return input['features'], input['rois']


class RoIPool(RoIPoolBase):
    def __init__(self, output_size: int or Tuple[int, int], spatial_scale: float):
        super(RoIPool, self).__init__(output_size, spatial_scale)

    def forward(self, input: dict) -> dict:
        features, boxes = self._unpack_input(input)
        rois = roi_pool(features, boxes, self.output_size, self.spatial_scale)
        return {'roi_features': rois}

    def _unpack_input(self, input: dict):
        return input['features'], input['rois']