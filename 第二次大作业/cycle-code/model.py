from __future__ import division
import torch.nn as nn

def weights_init(m):
    classname = m.__class__.__name__
    if hasattr(m, 'weight') and (classname.find('Conv') != -1 or classname.find('Linear') != -1):
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)


def model_summary(model):
    print('=================================================================')
    print(model)
    print('=================================================================')
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print('Total params: {:,}'.format(total_params))
    print('Trainable params: {:,}'.format(trainable_params))
    print('Non-trainable params: {:,}'.format(total_params - trainable_params))


class ConvBlock(nn.Module):
    def __init__(self, in_channels, filters, kernel_size=7, strides=1, padding=0, activation=nn.ReLU):
        super(ConvBlock, self).__init__()
        self.blocks = nn.Sequential(
            nn.Conv2d(in_channels, out_channels=filters, kernel_size=kernel_size, stride=strides, padding=padding),
            nn.InstanceNorm2d(num_features=filters),
            activation(inplace=True))

    def forward(self, input_tensor):
        x = self.blocks(input_tensor)
        return x


class DeConvBlock(nn.Module):
    def __init__(self, in_channels, filters, kernel_size=7, strides=1, padding=0, output_padding=1, activation=nn.ReLU):
        super(DeConvBlock, self).__init__()
        self.blocks = nn.Sequential(
            nn.ConvTranspose2d(in_channels, out_channels=filters, kernel_size=kernel_size,
                               stride=strides, padding=padding, output_padding=output_padding),
            nn.InstanceNorm2d(num_features=filters),
            activation(inplace=True))

    def forward(self, input_tensor):
        x = self.blocks(input_tensor)
        return x


class ResidualBlock(nn.Module):
    def __init__(self, in_channels, filters, kernel_size=3, strides=1, padding=0, activation=nn.ReLU):
        super(ResidualBlock, self).__init__()
        self.conv_blocks = nn.Sequential(
            nn.Conv2d(in_channels, out_channels=filters, kernel_size=kernel_size, stride=strides, padding=padding),
            nn.ReflectionPad2d(1),
            nn.InstanceNorm2d(num_features=filters),
            activation(inplace=True),
            nn.Conv2d(in_channels, out_channels=filters, kernel_size=kernel_size, stride=strides, padding=padding),
            nn.ReflectionPad2d(1),
            nn.InstanceNorm2d(num_features=filters))

    def forward(self, input_tensor):
        x = self.conv_blocks(input_tensor)
        x = x + input_tensor
        return x


class BaseModel(nn.Module):
    def __init__(self):
        super(BaseModel, self).__init__()
        self.model = None

    def forward(self, x):
        pass

    def summary(self):
        if self.model != None:
            print('=================================================================')
            print(self.model)
            print('=================================================================')
            total_params = sum(p.numel() for p in self.model.parameters())
            trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
            print('Total params: {:,}'.format(total_params))
            print('Trainable params: {:,}'.format(trainable_params))
            print('Non-trainable params: {:,}'.format(total_params - trainable_params))
        else:
            print('Model not created')


class ResNetGenerator(BaseModel):
    def __init__(self, input_channel=3, output_channel=3, filters=64, n_blocks=9):
        super(ResNetGenerator, self).__init__()
        layers = [
            nn.ReflectionPad2d(3),
            ConvBlock(in_channels=input_channel, filters=filters, kernel_size=7, strides=1, activation=nn.LeakyReLU),
            ConvBlock(in_channels=filters, filters=filters * 2, kernel_size=3, strides=2, padding=1,
                      activation=nn.LeakyReLU),
            ConvBlock(in_channels=filters * 2, filters=filters * 4, kernel_size=3, strides=2, padding=1,
                      activation=nn.LeakyReLU)
        ]
        for i in range(n_blocks):
            layers.append(ResidualBlock(in_channels=filters * 4, filters=filters * 4, kernel_size=3, strides=1,
                                        activation=nn.LeakyReLU))
        layers += [
            DeConvBlock(in_channels=filters * 4, filters=filters * 2, kernel_size=3, strides=2, padding=1,
                        output_padding=1, activation=nn.LeakyReLU),
            DeConvBlock(in_channels=filters * 2, filters=filters, kernel_size=3, strides=2, padding=1, output_padding=1,
                        activation=nn.LeakyReLU),
            nn.ReflectionPad2d(3),
            nn.Conv2d(in_channels=filters, out_channels=output_channel, kernel_size=7, stride=1, padding=0)
        ]
        layers += [nn.Tanh()]
        self.model = nn.Sequential(*layers)

    def forward(self, input_tensor):
        x = self.model(input_tensor)
        return x


class PatchGANDiscriminator(BaseModel):
    def __init__(self, input_channel, filters=64):
        super(PatchGANDiscriminator, self).__init__()
        layers = [
            nn.Conv2d(in_channels=input_channel, out_channels=filters, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(inplace=True),
            ConvBlock(in_channels=filters, filters=filters * 2, kernel_size=4, strides=2, padding=1,
                      activation=nn.LeakyReLU),
            ConvBlock(in_channels=filters * 2, filters=filters * 4, kernel_size=4, strides=2, padding=1,
                      activation=nn.LeakyReLU),
            ConvBlock(in_channels=filters * 4, filters=filters * 8, kernel_size=4, strides=1, padding=1,
                      activation=nn.LeakyReLU),
        ]
        layers += [nn.Conv2d(in_channels=filters * 8, out_channels=1, kernel_size=4, stride=1, padding=1)]
        self.model = nn.Sequential(*layers)

    def forward(self, input_tensor):
        x = self.model(input_tensor)
        return x