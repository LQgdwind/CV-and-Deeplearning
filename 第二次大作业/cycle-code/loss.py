from __future__ import division


import torch
from torch import device

from main import SOFT_REAL_LABEL_RANGE, SOFT_FAKE_LABEL_RANGE, LAMBDA
from model import ResNetGenerator, weights_init, PatchGANDiscriminator

def discriminator_loss(real_image, generated_image):
    real_loss = (real_image - torch.FloatTensor(real_image.size()).uniform_(SOFT_REAL_LABEL_RANGE[0],SOFT_REAL_LABEL_RANGE[1]).to(device)).pow(2).mean()
    fake_loss = (generated_image - torch.FloatTensor(generated_image.size()).uniform_(SOFT_FAKE_LABEL_RANGE[0],SOFT_FAKE_LABEL_RANGE[1]).to(device)).pow(2).mean()
    total_loss = real_loss + fake_loss
    return total_loss * 0.5

def generator_loss(generated_image):
    loss =(generated_image - torch.FloatTensor(generated_image.size()).uniform_(SOFT_REAL_LABEL_RANGE[0],SOFT_REAL_LABEL_RANGE[1]).to(device)).pow(2).mean()
    return loss

def cycle_consistency_loss(real_image, cycled_image):
    loss = (real_image - cycled_image).abs().mean()
    return loss * LAMBDA

def identity_loss(real_image, generated_image):
    loss = (real_image - generated_image).abs().mean()
    return loss * 0.5 * LAMBDA

def discriminator_loss_test(real_image, generated_image):
    mse = torch.nn.MSELoss()
    real_loss = mse(real_image, torch.FloatTensor(real_image.size()).uniform_(SOFT_REAL_LABEL_RANGE[0],
                                                                           SOFT_REAL_LABEL_RANGE[1]).to(device))
    fake_loss = mse(generated_image, torch.FloatTensor(generated_image.size()).uniform_(SOFT_FAKE_LABEL_RANGE[0],
                                                                           SOFT_FAKE_LABEL_RANGE[1]).to(device))
    total_loss = real_loss + fake_loss
    return total_loss * 0.5

def generator_loss_test(generated_image):
    mse = torch.nn.MSELoss()
    loss = mse(generated_image, torch.FloatTensor(generated_image.size()).uniform_(SOFT_REAL_LABEL_RANGE[0],
                                                                           SOFT_REAL_LABEL_RANGE[1]).to(device))
    return loss

def cycle_consistency_loss_test(real_image, cycled_image):
    mae = torch.nn.L1Loss()
    loss = mae(real_image, cycled_image).abs().mean()
    return loss * LAMBDA

def identity_loss_test(real_image, generated_image):
    mae = torch.nn.L1Loss()
    loss = mae(real_image, generated_image).abs().mean()
    return loss * 0.5 * LAMBDA