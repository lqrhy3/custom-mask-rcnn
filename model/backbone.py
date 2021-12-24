from torchvision import models
from torchvision.models.detection.backbone_utils import resnet_fpn_backbone
from torchvision.models._utils import IntermediateLayerGetter


def make_vgg_16_backbone(pretrained=False):
    vgg_16 = models.vgg16(pretrained)
    layer_to_return = 'features'

    return IntermediateLayerGetter(vgg_16, {layer_to_return: '0'})


def make_resnet_fpn_backbone(pretrained=False):
    resnet_fpn = resnet_fpn_backbone('resnet50', pretrained)

    return resnet_fpn
