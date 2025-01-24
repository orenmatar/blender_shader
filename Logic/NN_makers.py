import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torchvision import models, transforms


def make_siamese_vgg(layers_to_take_and_size, final_emb=128):
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
    pooling = nn.AdaptiveMaxPool2d((1, 1))

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