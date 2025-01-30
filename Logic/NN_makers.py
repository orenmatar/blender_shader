import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models, transforms


def make_siamese_vgg(layers_to_take_and_size, final_emb=128, use_avg_pool=False):
    # Load pretrained VGG16
    layers, size = layers_to_take_and_size
    vgg = models.vgg16(weights=models.VGG16_Weights.IMAGENET1K_V1)

    original_weight = vgg.features[0].weight.data
    # Sum the weights across the 3 channels to create a 1-channel equivalent
    new_weight = original_weight.sum(dim=1, keepdim=True)  # Sum across 3 channels
    # Now replace the first conv layer with the new weight
    vgg.features[0] = nn.Conv2d(1, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    vgg.features[0].weight.data = new_weight  # Set the modified weights

    # Extract first k layers
    feature_extractor = nn.Sequential(*list(vgg.features.children())[:layers])

    # Add global average pooling
    pooling = nn.AdaptiveAvgPool2d((1, 1)) if use_avg_pool else nn.AdaptiveMaxPool2d((1, 1))

    # Add dense layers
    dense = nn.Sequential(
        nn.Flatten(),
        nn.Linear(size, 256),
        nn.ReLU(),
        nn.Linear(256, final_emb)  # Final embedding size
    )

    # Combine into a single model
    model = nn.Sequential(
        feature_extractor,
        pooling,
        dense
    )
    return model


def make_siamese_resnet(layers_to_take_and_size, final_emb=128, use_avg_pool=False):
    layers, size = layers_to_take_and_size
    resnet = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)  # Load pretrained ResNet

    # Modify first conv layer for grayscale (1-channel input)
    original_weight = resnet.conv1.weight.data
    new_weight = original_weight.sum(dim=1, keepdim=True)  # Convert 3-channel to 1-channel equivalent
    resnet.conv1 = nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
    resnet.conv1.weight.data = new_weight  # Set new grayscale weights

    # Extract first few layers (up to specified number of layers)
    feature_extractor = nn.Sequential(*list(resnet.children())[:layers])

    # Add global max pooling
    pooling = nn.AdaptiveAvgPool2d((1, 1)) if use_avg_pool else nn.AdaptiveMaxPool2d((1, 1))

    # Add dense layers
    dense = nn.Sequential(
        nn.Flatten(),
        nn.Linear(size, 256),
        nn.ReLU(),
        nn.Linear(256, final_emb)  # Final embedding size
    )

    # Combine into a single model
    model = nn.Sequential(
        feature_extractor,
        pooling,
        dense
    )

    return model


class L2pooling(nn.Module):
    def __init__(self, filter_size=5, stride=2, channels=None, pad_off=0):
        super(L2pooling, self).__init__()
        self.padding = (filter_size - 2 )//2
        self.stride = stride
        self.channels = channels
        a = np.hanning(filter_size)[1:-1]
        g = torch.Tensor(a[:,None]*a[None,:])
        g = g/torch.sum(g)
        self.register_buffer('filter', g[None,None,:,:].repeat((self.channels,1,1,1)))

    def forward(self, input):
        input = input**2
        out = F.conv2d(input, self.filter, stride=self.stride, padding=self.padding, groups=input.shape[1])
        return (out+1e-12).sqrt()


def make_siamese_dists(final_emb=128):
    # Load pretrained VGG16
    vgg = models.vgg16(weights=models.VGG16_Weights.IMAGENET1K_V1)

    original_weight = vgg.features[0].weight.data
    # Sum the weights across the 3 channels to create a 1-channel equivalent
    new_weight = original_weight.sum(dim=1, keepdim=True)  # Sum across 3 channels
    # Now replace the first conv layer with the new weight
    vgg.features[0] = nn.Conv2d(1, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    vgg.features[0].weight.data = new_weight  # Set the modified weights

    # Extract the VGG layers (without the classifier)
    vgg_layers = list(vgg.features.children())

    # Split the layers into stages (similar to DISTS)
    stage1 = nn.Sequential(*vgg_layers[:4])  # First 4 layers
    stage2 = nn.Sequential(*vgg_layers[4:9])  # Next 5 layers
    stage3 = nn.Sequential(*vgg_layers[9:16])  # Next 7 layers
    stage4 = nn.Sequential(*vgg_layers[16:23])  # Next 7 layers
    stage5 = nn.Sequential(*vgg_layers[23:])  # Last 7 layers

    # Apply L2 Pooling after each stage (like in DISTS model)
    stage1.add_module('L2pool', L2pooling(channels=64))  # L2 pooling after stage2 (64 channels)
    stage2.add_module('L2pool', L2pooling(channels=128))  # L2 pooling after stage3 (128 channels)
    stage3.add_module('L2pool', L2pooling(channels=256))  # L2 pooling after stage4 (256 channels)
    stage4.add_module('L2pool', L2pooling(channels=512))  # L2 pooling after stage5 (512 channels)

    # Define the complete model with the VGG backbone and L2 pooling
    model = nn.Sequential(
        stage1,
        stage2,
        stage3,
        stage4,
        nn.Flatten(),  # Flatten the tensor to pass it to a fully connected layer
        nn.Linear(2048, final_emb)  # Final embedding layer (512 channels from VGG)
    )

    return model