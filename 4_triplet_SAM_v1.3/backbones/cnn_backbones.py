import torch.nn as nn
from torchvision import models as models_2d
from torchvision.models import VGG16_Weights, ResNet18_Weights, ResNet34_Weights, ResNet50_Weights

class Identity(nn.Module):
    """Identity layer to replace last fully connected layer"""
    def forward(self, x):
        return x

def vgg_16(pretrained=True):
    model = models_2d.vgg16(weights=VGG16_Weights.IMAGENET1K_V1)
    # model = nn.Sequential(*list(model.features.children())[:24])   只取前4个block

    return model


################################################################################
# ResNet Family
################################################################################


def resnet_18(pretrained=True):
    model = models_2d.resnet18(weights=ResNet18_Weights.DEFAULT)
    feature_dims = model.fc.in_features
    model.fc = Identity()
    return model, feature_dims, 1024


def resnet_34(pretrained=True):
    model = models_2d.resnet34(weights=ResNet34_Weights.DEFAULT)
    feature_dims = model.fc.in_features
    model.fc = Identity()
    return model, feature_dims, 1024


def resnet_50(pretrained=True):
    model = models_2d.resnet50(weights=ResNet50_Weights.DEFAULT)
    feature_dims = model.fc.in_features
    model.fc = Identity()
    return model, feature_dims, 1024