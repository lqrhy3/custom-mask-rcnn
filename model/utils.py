import torch
from torchvision.ops.boxes import box_convert


def anchor_transforms_to_boxes(anchors, anchor_transforms):
    """
    :param anchors: [num_anchors, 4]
    :param anchor_transforms: [N, K * num_anchors, 4]
    :return:
    """
    k = anchor_transforms.shape[1] // anchors.shape[0]
    anchors = anchors.repeat(k, 1)  # [K * num_anchors, 4]

    boxes = torch.empty_like(anchor_transforms)
    boxes[:, :, 0] = anchors[:, 2] * anchor_transforms[:, :, 0] + anchors[:, 0]
    boxes[:, :, 1] = anchors[:, 3] * anchor_transforms[:, :, 1] + anchors[:, 1]
    boxes[:, :, 2] = torch.exp(anchor_transforms[:, :, 2]) * anchors[:, 2]
    boxes[:, :, 3] = torch.exp(anchor_transforms[:, :, 3]) * anchors[:, 3]

    boxes = box_convert(boxes, 'xywh', out_fmt='xyxy')

    return boxes


def drop_cross_boundary_boxes(boxes, image_size):
    pass
