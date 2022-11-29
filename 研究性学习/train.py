
import data_downloading as data
import data_process as dp
import torchvision
import torch
import os
import tools
import matplotlib.pyplot as plt
from torch import nn
import SE_ResNet as se
import SE_ResNet_idea1 as se1
import SE_ResNet_idea2 as se2
import SE_ResNet_idea3 as se3
import SE_ResNet_idea4 as se4
import SE_ResNet_idea5 as se5

import matplotlib
from IPython.display import display, clear_output
plt.ion()

batch_size = 32
valid_ratio = 0.1

dp.reorg_data(data.data_dir, valid_ratio)

transform_train = torchvision.transforms.Compose([
torchvision.transforms.Resize([224,224]),
torchvision.transforms.RandomHorizontalFlip(),
torchvision.transforms.ToTensor(),
torchvision.transforms.Normalize([0.4914, 0.4822, 0.4465],[0.2023, 0.1994, 0.2010])])

transform_test = torchvision.transforms.Compose([torchvision.transforms.ToTensor(),torchvision.transforms.Normalize([0.4914, 0.4822, 0.4465],[0.2023, 0.1994, 0.2010])])

train_ds, train_valid_ds = [torchvision.datasets.ImageFolder(os.path.join(data.data_dir, 'train_valid_test', folder),transform=transform_train) for folder in ['train', 'train_valid']]

valid_ds, test_ds = [torchvision.datasets.ImageFolder(os.path.join(data.data_dir, 'train_valid_test', folder),transform=transform_test) for folder in ['valid', 'test']]

train_iter, train_valid_iter = [torch.utils.data.DataLoader(dataset, batch_size, shuffle=True, drop_last=True) for dataset in (train_ds, train_valid_ds)]

valid_iter = torch.utils.data.DataLoader(valid_ds, batch_size, shuffle=False,drop_last=True)

test_iter = torch.utils.data.DataLoader(test_ds, batch_size, shuffle=False,drop_last=False)

net = torchvision.models.resnet18(pretrained=True)
loss = nn.CrossEntropyLoss(reduction="none")



def train(net, train_iter, valid_iter, num_epochs, lr, wd, devices, lr_period,lr_decay):
    trainer = torch.optim.SGD(net.parameters(), lr=lr, momentum=0.9,weight_decay=wd)
    scheduler = torch.optim.lr_scheduler.StepLR(trainer, lr_period, lr_decay)
    num_batches, timer = len(train_iter), tools.Timer()
    legend = ['train loss', 'train acc']
    if valid_iter is not None:
        legend.append('valid acc')
    animator =tools.Animator(xlabel='epoch', xlim=[1, num_epochs],legend=legend)
    net = nn.DataParallel(net, device_ids=devices).to(devices[0])
    for epoch in range(num_epochs):
        net.train()
        metric = tools.Accumulator(3)
        for i, (features, labels) in enumerate(train_iter):
            timer.start()
            l, acc = tools.train_batch(net, features, labels,loss, trainer, devices)
            metric.add(l, acc, labels.shape[0])
            timer.stop()
            if (i + 1) % (num_batches // 5) == 0 or i == num_batches - 1:
                animator.add(epoch + (i + 1) / num_batches,(metric[0] / metric[2], metric[1] / metric[2],None))
        if valid_iter is not None:
            valid_acc = tools.evaluate_accuracy_gpu(net, valid_iter)
            animator.add(epoch + 1, (None, None, valid_acc))
        scheduler.step()
        print("epoch : {arg1}".format(arg1=epoch))
        plt.pause(0.5)
    measures = (f'train loss {metric[0] / metric[2]:.3f}, ' f'train acc {metric[1] / metric[2]:.3f}')
    if valid_iter is not None:
            measures += f', valid acc {valid_acc:.3f}'
    print(measures + f'\n{metric[2] * num_epochs / timer.sum():.1f}' f' examples/sec on {str(devices)}')

devices, num_epochs, lr, wd = tools.try_all_gpus(), 100, 2e-4, 5e-4
lr_period, lr_decay = 4, 0.9
train(net, train_iter, valid_iter, num_epochs, lr, wd, devices, lr_period,lr_decay)
plt.show()