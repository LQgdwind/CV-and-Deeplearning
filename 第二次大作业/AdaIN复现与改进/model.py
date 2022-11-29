import torch
import torch.nn as nn
import torchvision.models as models
from dataDisplaying import mu
from dataDisplaying import sigma

class AdaIN(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x, y):
        assert (x.size()[:2] == y.size()[:2])
        size = x.size()
        style_mean, style_std = mu(y), sigma(y)
        content_mean, content_std = mu(x), sigma(x)
        normalized_feat = (x - content_mean.expand(size)) / content_std.expand(size)
        return normalized_feat * style_std.expand(size) + style_mean.expand(size)


class Encoder(nn.Module):
    def __init__(self):
        super().__init__()
        layers = [nn.Conv2d(3, 3, 1)]
        vgg_features = models.vgg19(pretrained=True).features.children()
        for layer in vgg_features:
            layers.append(layer)
            if isinstance(layer, torch.nn.Conv2d):
                layer.padding_mode = 'reflect'

        self.net = nn.Sequential(*(layers[: 22]))

    def forward(self, x):
        p1, p2, p3, p4 = None, None, None, None
        for i, layer in enumerate(self.net):
            x = layer(x)
            if i == 2:
                p1 = x
            elif i == 7:
                p2 = x
            elif i == 12:
                p3 = x
            elif i == 21:
                p4 = x

        return p1, p2, p3, p4


class Decoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(512, 256, 3, padding=1, padding_mode='reflect'),
            nn.ReLU(),
            nn.Upsample(scale_factor=2),
            nn.Conv2d(256, 256, 3, padding=1, padding_mode='reflect'),
            nn.ReLU(),
            nn.Conv2d(256, 256, 3, padding=1, padding_mode='reflect'),
            nn.ReLU(),
            nn.Conv2d(256, 256, 3, padding=1, padding_mode='reflect'),
            nn.ReLU(),
            nn.Conv2d(256, 128, 3, padding=1, padding_mode='reflect'),
            nn.ReLU(),
            nn.Upsample(scale_factor=2),
            nn.Conv2d(128, 128, 3, padding=1, padding_mode='reflect'),
            nn.ReLU(),
            nn.Conv2d(128, 64, 3, padding=1, padding_mode='reflect'),
            nn.ReLU(),
            nn.Upsample(scale_factor=2),
            nn.Conv2d(64, 64, 3, padding=1, padding_mode='reflect'),
            nn.ReLU(),
            nn.Conv2d(64, 3, 3, padding=1, padding_mode='reflect'),
            nn.ReLU())

    def forward(self, x):
        return self.net(x)