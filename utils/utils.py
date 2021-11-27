import torch
from torchvision.ops.boxes import box_convert


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


def boxes_to_anchor_transforms(boxes, anchors):
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
