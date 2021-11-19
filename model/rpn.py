import torch
from torch import nn

from utils import anchor_transforms_to_boxes

class SlidingNetwork(nn.Module):
    def __init__(self, in_channels: int, num_anchors: int):
        super(SlidingNetwork, self).__init__()

        self.feature_extractor = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.ReLU()
        )

        self.box_regressor = nn.Conv2d(in_channels, 4 * num_anchors, kernel_size=(1, 1))
        self.objectness_scorer = nn.Conv2d(in_channels, 2 * num_anchors, kernel_size=(1, 1))

    def forward(self, input: dict) -> dict:
        x = self._unpack_input(input)
        features = self.feature_extractor(x)
        anchor_transforms = self.box_regressor(features)
        anchor_objectnesses = self.objectness_scorer(features)

        return {
            'anchor_transforms': anchor_transforms,
            'anchor_objectnesses': anchor_objectnesses
        }

    def _unpack_input(self, input: dict):
        x = input['features']
        return x


class RPN(nn.Module):
    def __init__(
            self,
            backbone: nn.Module,
            backbone_out_features: int,
            num_anchors
    ):
        super().__init__()

        self.backbone = backbone
        self.sliding_network = SlidingNetwork(
            backbone_out_features, num_anchors
        )

    def forward(self, input: dict) -> dict:
        x = self._unpack_input(input)
        backbone_features = self.backbone(x)

        sliding_network_result = self.sliding_network(backbone_features)
        anchor_transforms = sliding_network_result['anchor_transforms'] # [N, 4 * num_anchors, H, W]
        anchor_transforms = torch.reshape(anchor_transforms, (anchor_transforms.shape[0], -1))

        if self.training:
            pass

        return sliding_network_result

    def _unpack_input(self, input: dict):
        x = input['x']
        return x

