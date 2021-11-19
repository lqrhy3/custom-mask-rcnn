from torchvision import models
from torchvision.models._utils import IntermediateLayerGetter


def make_vgg_16_backbone(pretrained=False):
    vgg_16 = models.vgg16(pretrained)
    layer_to_return = 'features'

    return IntermediateLayerGetter(vgg_16, {layer_to_return: 'features'})
