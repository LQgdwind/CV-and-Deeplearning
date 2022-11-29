from __future__ import division
import gc
import os
import time
import sys
import random
from tqdm.notebook import tqdm
import numpy as np
import matplotlib.pyplot as plt

from PIL import Image
from IPython.display import clear_output
from collections import namedtuple, deque

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler

import torch.utils.data as data

import torchvision.utils as vutils
import torchvision.transforms as transforms

from dataset import GeneratorDataset
from image_buffer import ImageBuffer
from loss import discriminator_loss, generator_loss, cycle_consistency_loss, identity_loss
from model import ResNetGenerator, weights_init, PatchGANDiscriminator
from util import save_test_images, save_training_checkpoint, lambda_rule

INPUT_SHAPE = 256
SCALE_WIDTH = 256
DATASET_PATH = "/content/cycle-data" # Dataset path
OUTPUT_PATH = '/content/drive/MyDrive/cycle/output'
checkpoint_path = '/content/drive/MyDrive/cycle/checkpoint'

USE_BUFFER = True # Use image buffer to train discriminator
REPLAY_PROB = 0.5 # The probability of using previous fake images to train discriminator
BUFFER_SIZE = 50 # The maximum size of image buffer
BATCH_SIZE = 1

EPOCHs = 200
CURRENT_EPOCH = 1 # Epoch start from
SAVE_EVERY_N_EPOCH = 25 # Save checkpoint at every n epoch4

DISCRIMINATOR_LOSS_WEIGHT = 0.5 # Discriminator loss will be multiplied by this weight
SOFT_FAKE_LABEL_RANGE = [0.0, 0.3] # The label of fake label will be generated within this range.
SOFT_REAL_LABEL_RANGE = [0.7, 1.2] # The label of real label will be generated within this range.
LR = 0.0002
LR_DECAY_EPOCH = 100
LAMBDA = 10 # loss weight for cycle consistency

ngpu = 1

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

G_XtoY_loss_writer = open('/content/drive/MyDrive/cycle/log/G_XtoY_loss.txt','a+')
G_YtoX_loss_writer = open('/content/drive/MyDrive/cycle/log/G_YtoX_loss.txt','a+')
Dx_loss_writer = open('/content/drive/MyDrive/cycle/log/Dx_loss.txt','a+')
Dy_loss_writer = open('/content/drive/MyDrive/cycle/log/Dy_loss.txt','a+')


def train():
    # 针对两种风格各自从测试集中选取一张图片，每10个epoch打印出风格迁移后的结果
    sample_X = next(iter_test_image_X)
    sample_Y = next(iter_test_image_Y)

    # 遍历训练集所需步数
    training_steps = min(len(train_data_X), len(train_data_Y))

    # 若存在checkpoint则恢复
    if os.path.isfile(os.path.join(checkpoint_path, 'training-checkpoint')):
        checkpoint = torch.load(os.path.join(checkpoint_path, 'training-checkpoint'))
        G_XtoY.load_state_dict(checkpoint['G_XtoY'])
        G_YtoX.load_state_dict(checkpoint['G_YtoX'])
        Dx.load_state_dict(checkpoint['Dx'])
        Dy.load_state_dict(checkpoint['Dy'])
        G_XtoY_optimizer.load_state_dict(checkpoint['G_XtoY_optimizer'])
        G_YtoX_optimizer.load_state_dict(checkpoint['G_YtoX_optimizer'])
        Dx_optimizer.load_state_dict(checkpoint['Dx_optimizer'])
        Dy_optimizer.load_state_dict(checkpoint['Dy_optimizer'])
        Dx_optimizer_scheduler.load_state_dict(checkpoint['Dx_optimizer_scheduler'])
        Dy_optimizer_scheduler.load_state_dict(checkpoint['Dy_optimizer_scheduler'])
        G_XtoY_optimizer_scheduler.load_state_dict(checkpoint['G_XtoY_optimizer_scheduler'])
        G_YtoX_optimizer_scheduler.load_state_dict(checkpoint['G_YtoX_optimizer_scheduler'])
        CURRENT_EPOCH = checkpoint['epoch']
        print('Latest checkpoint of epoch {} restored!!'.format(CURRENT_EPOCH))

    # 从当前epoch训练至结束
    for epoch in range(CURRENT_EPOCH, EPOCHs + 1):

        # 记录当前epoch开始时间
        start = time.time()
        print('Start of epoch %d' % (epoch,))

        # 重置dataloader
        iter_train_image_X = iter(train_image_loader_X)
        iter_train_image_Y = iter(train_image_loader_Y)
        # 当前epoch的各类平均损失
        G_XtoY_loss_mean = 0
        G_YtoX_loss_mean = 0
        Dx_loss_mean = 0
        Dy_loss_mean = 0
        for step in range(training_steps):

            # 使用以前图片来训练discriminator的概率
            replay_previous = True if REPLAY_PROB > random.random() else False

            # 获取当前step的训练图片X，Y
            real_image_X = iter_train_image_X.next().to(device)
            real_image_Y = iter_train_image_Y.next().to(device)

            # 使用generator生成伪造图片
            fake_image_X = G_YtoX(real_image_Y).detach()
            fake_image_Y = G_XtoY(real_image_X).detach()

            # 更新image buffer
            image_buffer.add(real_image_X, fake_image_X, real_image_Y, fake_image_Y)

            # 重置梯度
            Dx_optimizer.zero_grad()
            Dy_optimizer.zero_grad()

            if USE_BUFFER and replay_previous:
                # 从image buffer中获取伪造图片
                buffered_images = image_buffer.sample()

                # 计算discriminator loss
                real_buffer_image_X = buffered_images.real_image_X
                fake_buffer_image_X = buffered_images.fake_image_X
                real_buffer_image_Y = buffered_images.real_image_Y
                fake_buffer_image_Y = buffered_images.fake_image_Y
                Dx_real_buffer = Dx(real_buffer_image_X)
                Dx_fake_buffer = Dx(fake_buffer_image_X)
                Dx_loss = discriminator_loss(Dx_real_buffer, Dx_fake_buffer)
                Dy_real_buffer = Dy(real_buffer_image_Y)
                Dy_fake_buffer = Dy(fake_buffer_image_Y)
                Dy_loss = discriminator_loss(Dy_real_buffer, Dy_fake_buffer)
            else:
                # 直接使用当前图片计算discriminator loss
                Dx_real = Dx(real_image_X)
                Dx_fake = Dx(fake_image_X)
                Dy_real = Dy(real_image_Y)
                Dy_fake = Dy(fake_image_Y)
                Dx_loss = discriminator_loss(Dx_real, Dx_fake)
                Dy_loss = discriminator_loss(Dy_real, Dy_fake)

            # 更新判别器
            Dx_loss.backward()
            Dy_loss.backward()
            Dx_optimizer.step()
            Dy_optimizer.step()

            # 计算生成器损失
            G_XtoY_optimizer.zero_grad()
            G_YtoX_optimizer.zero_grad()

            fake_image_Y = G_XtoY(real_image_X)
            fake_image_X = G_YtoX(real_image_Y)
            dis_fake_image_Y = Dy(fake_image_Y)
            dis_fake_image_X = Dx(fake_image_X)

            G_XtoY_loss = generator_loss(dis_fake_image_Y)
            G_YtoX_loss = generator_loss(dis_fake_image_X)

            # 计算cycle consistency loss
            cycled_XtoYtoX = G_YtoX(fake_image_Y)
            cycled_YtoXtoY = G_XtoY(fake_image_X)
            cycled_XtoY_loss = cycle_consistency_loss(real_image_X, cycled_XtoYtoX)
            cycled_YtoX_loss = cycle_consistency_loss(real_image_Y, cycled_YtoXtoY)
            total_cycle_loss = cycled_XtoY_loss + cycled_YtoX_loss

            # 计算identity loss
            same_image_Y = G_XtoY(real_image_Y)
            same_image_X = G_YtoX(real_image_X)
            identity_loss_for_YtoX = identity_loss(real_image_X, same_image_X)
            identity_loss_for_XtoY = identity_loss(real_image_Y, same_image_Y)

            # 计算生成器损失
            total_G_XtoY_loss = G_XtoY_loss + identity_loss_for_XtoY
            total_G_YtoX_loss = G_YtoX_loss + identity_loss_for_YtoX
            total_G_losses = total_G_XtoY_loss + total_G_YtoX_loss + total_cycle_loss

            # 更新生成器
            total_G_losses.backward()
            G_XtoY_optimizer.step()
            G_YtoX_optimizer.step()

            # 叠加损失
            G_XtoY_loss_mean = G_XtoY_loss_mean + total_G_XtoY_loss.item() + total_cycle_loss.item()
            G_YtoX_loss_mean = G_YtoX_loss_mean + total_G_YtoX_loss.item() + total_cycle_loss.item()
            Dx_loss_mean += Dx_loss.item()
            Dy_loss_mean += Dy_loss.item()

        # 保存并输出各类损失
        print('G_XtoY_loss {} for epoch {}'.format(G_XtoY_loss_mean / training_steps, epoch))
        G_XtoY_loss_writer.writelines(["{}\n".format(G_XtoY_loss_mean / training_steps)])
        G_XtoY_loss_writer.flush()
        print('G_YtoX_loss {} for epoch {}'.format(G_YtoX_loss_mean / training_steps, epoch))
        G_YtoX_loss_writer.flush()
        G_YtoX_loss_writer.writelines(["{}\n".format(G_YtoX_loss_mean / training_steps)])
        print('Dx_loss {} for epoch {}'.format(Dx_loss_mean / training_steps, epoch))
        Dx_loss_writer.writelines(["{}\n".format(Dx_loss_mean / training_steps)])
        Dx_loss_writer.flush()
        print('Dy_loss {} for epoch {}'.format(Dy_loss_mean / training_steps, epoch))
        Dy_loss_writer.writelines(["{}\n".format(Dy_loss_mean / training_steps)])
        Dy_loss_writer.flush()

        # 更新scheduler
        Dx_optimizer_scheduler.step()
        Dy_optimizer_scheduler.step()
        G_XtoY_optimizer_scheduler.step()
        G_YtoX_optimizer_scheduler.step()

        # 每10个epoch打印出样例图片风格迁移后的结果
        save_point = epoch % 10 == 0
        save_test_images(G_XtoY, G_YtoX, sample_X, folder_name='XtoY/epoch_{}'.format(epoch), step=step,
                         save=save_point, show_result=save_point)
        save_test_images(G_YtoX, G_XtoY, sample_Y, folder_name='YtoX/epoch_{}'.format(epoch), step=step,
                         save=save_point, show_result=save_point)

        # 每隔一定epoch更新checkpoint
        if epoch % SAVE_EVERY_N_EPOCH == 0:
            save_training_checkpoint(epoch)
            print('Saving checkpoint for epoch {} at {}'.format(epoch, checkpoint_path))

        # 计算当前epoch所花费时间
        print('Time taken for epoch {} is {} sec\n'.format(epoch, time.time() - start))
        gc.collect()


if __name__ == '__main__':
    G_XtoY = ResNetGenerator(input_channel=3, output_channel=3, filters=64, n_blocks=9).to(device)
    G_YtoX = ResNetGenerator(input_channel=3, output_channel=3, filters=64, n_blocks=9).to(device)

    # Handle multi-gpu if desired
    if (device.type == 'cuda') and (ngpu > 1):
        G_XtoY = nn.DataParallel(G_XtoY, list(range(ngpu)))
        G_YtoX = nn.DataParallel(G_YtoX, list(range(ngpu)))

    # Apply the weights_init function to randomly initialize all weights
    #  to mean=0, stdev=0.2.
    G_XtoY.apply(weights_init)
    G_YtoX.apply(weights_init)

    preprocess_train_transformations = transforms.Compose([
        transforms.CenterCrop(INPUT_SHAPE),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])

    preprocess_test_transformations = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])

    train_data_X = GeneratorDataset(root_dir=os.path.join(DATASET_PATH, "trainA"),
                                    transform=preprocess_train_transformations)

    train_data_Y = GeneratorDataset(root_dir=os.path.join(DATASET_PATH, "trainB"),
                                    transform=preprocess_train_transformations)

    test_data_X = GeneratorDataset(root_dir=os.path.join(DATASET_PATH, "testA"),
                                   transform=preprocess_test_transformations)

    test_data_Y = GeneratorDataset(root_dir=os.path.join(DATASET_PATH, "testB"),
                                   transform=preprocess_test_transformations)

    train_image_loader_X = torch.utils.data.DataLoader(train_data_X, batch_size=BATCH_SIZE,
                                                       shuffle=True, num_workers=0)
    train_image_loader_Y = torch.utils.data.DataLoader(train_data_Y, batch_size=BATCH_SIZE,
                                                       shuffle=True, num_workers=0)
    test_image_loader_X = torch.utils.data.DataLoader(test_data_X, batch_size=BATCH_SIZE,
                                                      shuffle=False, num_workers=0)
    test_image_loader_Y = torch.utils.data.DataLoader(test_data_Y, batch_size=BATCH_SIZE,
                                                      shuffle=False, num_workers=0)

    print("Found {} images in {}".format(len(train_data_X), 'trainA'))
    print("Found {} images in {}".format(len(train_data_Y), 'trainB'))
    print("Found {} images in {}".format(len(test_data_X), 'testA'))
    print("Found {} images in {}".format(len(test_data_Y), 'testB'))

    Dx = PatchGANDiscriminator(input_channel=3, filters=64).to(device)
    Dy = PatchGANDiscriminator(input_channel=3, filters=64).to(device)

    # Handle multi-gpu if desired
    if (device.type == 'cuda') and (ngpu > 1):
        Dx = nn.DataParallel(Dx, list(range(ngpu)))
        Dy = nn.DataParallel(Dy, list(range(ngpu)))

    # Apply the weights_init function to randomly initialize all weights
    #  to mean=0, stdev=0.2.
    Dx.apply(weights_init)
    Dy.apply(weights_init)

    iter_train_image_X = iter(train_image_loader_X)
    iter_train_image_Y = iter(train_image_loader_Y)
    iter_test_image_X = iter(test_image_loader_X)
    iter_test_image_Y = iter(test_image_loader_Y)


    Dx_optimizer = optim.Adam(Dx.parameters(), lr=0.0002, betas=(0.5, 0.999))
    Dy_optimizer = optim.Adam(Dy.parameters(), lr=0.0002, betas=(0.5, 0.999))
    G_XtoY_optimizer = optim.Adam(G_XtoY.parameters(), lr=0.0002, betas=(0.5, 0.999))
    G_YtoX_optimizer = optim.Adam(G_YtoX.parameters(), lr=0.0002, betas=(0.5, 0.999))

    Dx_optimizer_scheduler = lr_scheduler.LambdaLR(Dx_optimizer, lr_lambda=lambda_rule)
    Dy_optimizer_scheduler = lr_scheduler.LambdaLR(Dy_optimizer, lr_lambda=lambda_rule)
    G_XtoY_optimizer_scheduler = lr_scheduler.LambdaLR(G_XtoY_optimizer, lr_lambda=lambda_rule)
    G_YtoX_optimizer_scheduler = lr_scheduler.LambdaLR(G_YtoX_optimizer, lr_lambda=lambda_rule)

    image_buffer = ImageBuffer(buffer_size=BUFFER_SIZE)
    train()

    G_XtoY_loss_writer.close()
    G_YtoX_loss_writer.close()
    Dx_loss_writer.close()
    Dy_loss_writer.close()

    for step, image in enumerate(tqdm(test_image_loader_X)):
        show_result = True if (step + 1) >= len(test_data_X) else False
        save_test_images(G_XtoY, G_YtoX, image, folder_name='test_XtoY', step=step, save=True, show_result=show_result)

    for step, image in enumerate(tqdm(test_image_loader_Y)):
        show_result = True if (step + 1) >= len(test_data_Y) else False
        save_test_images(G_YtoX, G_XtoY, image, folder_name='test_YtoX', step=step, save=True, show_result=show_result)