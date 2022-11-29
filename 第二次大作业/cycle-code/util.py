from __future__ import division
import os
import numpy as np
import matplotlib.pyplot as plt

import torch

from main import device, EPOCHs, LR_DECAY_EPOCH, G_XtoY, G_YtoX, Dx, Dy, G_XtoY_optimizer, G_YtoX_optimizer, \
    Dx_optimizer, Dy_optimizer, Dx_optimizer_scheduler, Dy_optimizer_scheduler, G_XtoY_optimizer_scheduler, \
    G_YtoX_optimizer_scheduler, checkpoint_path, OUTPUT_PATH


def denormalize(images, std=0.5, mean=0.5):
    # For plot
    images = (images * std) + mean
    return images

def deprocess(input_tensor):
    if len(input_tensor.shape) == 3:
        return np.transpose(denormalize(input_tensor.to(device).cpu()), (1,2,0))
    elif len(input_tensor.shape) == 4:
        return np.transpose(denormalize(input_tensor.to(device).cpu()), (0, 2,3,1))

def lambda_rule(epoch):
    k = EPOCHs / LR_DECAY_EPOCH - 1
    lr = 1.0 - max(0, epoch + 1 - LR_DECAY_EPOCH) / float(k * LR_DECAY_EPOCH + 1)
    return lr


def save_training_checkpoint(epoch):
    state_dict = {
        'G_XtoY': G_XtoY.state_dict(),
        'G_YtoX': G_YtoX.state_dict(),
        'Dx': Dx.state_dict(),
        'Dy': Dy.state_dict(),
        'G_XtoY_optimizer': G_XtoY_optimizer.state_dict(),
        'G_YtoX_optimizer': G_YtoX_optimizer.state_dict(),
        'Dx_optimizer': Dx_optimizer.state_dict(),
        'Dy_optimizer': Dy_optimizer.state_dict(),
        'Dx_optimizer_scheduler': Dx_optimizer_scheduler.state_dict(),
        'Dy_optimizer_scheduler': Dy_optimizer_scheduler.state_dict(),
        'G_XtoY_optimizer_scheduler': G_XtoY_optimizer_scheduler.state_dict(),
        'G_YtoX_optimizer_scheduler': G_YtoX_optimizer_scheduler.state_dict(),
        'epoch': epoch
    }

    save_path = os.path.join(checkpoint_path, 'training-checkpoint')
    torch.save(state_dict, save_path)


def save_models():
    state_dict = {
        'G_XtoY': G_XtoY,
        'G_YtoX': G_YtoX
    }
    save_path = os.path.join(checkpoint_path, 'model')
    torch.save(state_dict, checkpoint_path)


def generate_images(model, test_input, img_name='img', step=0):
    prediction = model(test_input.to(device)).cpu().detach()
    plt.figure(figsize=(12, 12))

    display_list = [test_input, prediction]
    title = ['Input Image', 'Predicted Image']

    for i in range(2):
        plt.subplot(1, 2, i + 1)
        plt.title(title[i])
        # getting the pixel values between [0, 1] to plot it.
        plt.imshow(deprocess(display_list[i])[0])
        plt.axis('off')
    plt.show()


def generate_test_images(model1, model2, test_input, img_name='img', step=0, show_result=False):
    '''
        Generate images and cycled images, then save them to tensorboard
    '''
    with torch.no_grad():
        prediction1 = model1(test_input.to(device))
        prediction2 = model2(prediction1)

    test_input = test_input.cpu()
    prediction1 = prediction1.cpu()
    prediction2 = prediction2.cpu()
    display_list = [test_input, prediction1, prediction2]
    title = ['Input Image', 'Predicted Image', 'Cycled Image']

    if show_result:
        plt.figure(figsize=(12, 12))
        for i in range(3):
            plt.subplot(1, 3, i + 1)
            plt.title(title[i])
            # getting the pixel values between [0, 1] to plot it.
            plt.imshow(deprocess(display_list[i])[0])
            plt.axis('off')
        plt.show()


def save_test_images(model1, model2, test_input, folder_name='img', step=0, save=False, show_result=False):
    '''
        Generate images and cycled images, then save them as jpg
    '''
    with torch.no_grad():
        prediction1 = model1(test_input.to(device))
        prediction2 = model2(prediction1)

    test_input = test_input.cpu()
    prediction1 = prediction1.cpu()
    prediction2 = prediction2.cpu()

    display_list = [test_input, prediction1, prediction2]
    title = ['original', 'predicted', 'cycled']
    figure_title = ['Input Image', 'Predicted Image', 'Cycled Image']

    base_folder = os.path.join(OUTPUT_PATH, folder_name)
    if not os.path.exists(base_folder):
        os.makedirs(base_folder)

    if save:
        for img, title in zip(display_list, title):
            save_folder = os.path.join(base_folder, title)
            if not os.path.exists(save_folder):
                os.makedirs(save_folder)
            img = deprocess(img)[0]
            plt.imsave(os.path.join(save_folder, '{}.jpg'.format(step)), img.numpy())

    if show_result:
        plt.figure(figsize=(12, 12))
        for i in range(3):
            plt.subplot(1, 3, i + 1)
            plt.title(figure_title[i])
            # getting the pixel values between [0, 1] to plot it.
            plt.imshow(deprocess(display_list[i])[0])
            plt.axis('off')
        plt.show()