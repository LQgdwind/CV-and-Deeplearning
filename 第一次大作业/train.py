import k_fold
import torch
import torchvision
import trainBatch as tb
import dataDisplay as dd
import matplotlib.pyplot as plt
from torch import nn
import os
import loss_func as lf
import SE_ResNet_dyrelu as senets1

os.environ['CUDA_VISIBLE_DEVICES'] = '0'

net = senets1.se_resnet_18_dyrelu()
figure, faxes = plt.subplots(2, 3, figsize=(8, 9), sharex=False, sharey=False)
loss = lf.cross_entropy_loss(reduction="none")


def train(net, num_epochs, lr, wd, devices, lr_period, lr_decay):
    test_top1_acc, test_top5_acc = 0.0, 0.0
    trainer = torch.optim.SGD(net.parameters(), lr=lr, momentum=0.9, weight_decay=wd)
    scheduler = torch.optim.lr_scheduler.StepLR(trainer, lr_period, lr_decay)
    # net.load_state_dict(torch.load("/content/models/rubbish_classification.pkl"))
    # net.eval()
    test_legend = ['top1 acc', 'top5 acc']
    test_animator = dd.Multi_Animator(xlabel='round', xlim=[1, 5], legend=test_legend, fig_main=figure, axes_main=faxes, rows=0,
                             cols=0)

    for k in range(5):
        legend = ['train loss', 'train acc', 'valid acc']
        animator = dd.Multi_Animator(xlabel=str(k) + 'th_round_epoch', xlim=[1, num_epochs], legend=legend, fig_main=figure,
                            axes_main=faxes, rows=(k + 1) // 3, cols=(k + 1) % 3)
        train_iter, valid_iter, test_iter = k_fold.k_fold(k, batch_size=512, num_k=10)
        num_batches = len(train_iter)
        for epoch in range(num_epochs):
            net.train()
            metric = dd.Accumulator(3)
            for i, (features, labels) in enumerate(train_iter):
                l, acc = tb.train_batch(net, features, labels, loss, trainer, devices)
                metric.add(l, acc, labels.shape[0])
                if (i + 1) % (num_batches // 5) == 0 or i == num_batches - 1:
                    animator.add(epoch + (i + 1) / num_batches,
                                 (metric[0] / metric[2], metric[1] / metric[2], None))

                valid_acc = tb.evaluate_accuracy_gpu(net, valid_iter, mode=1)
                animator.add(epoch + 1, (None, None, valid_acc))
                torch.save(net.state_dict(), "//content//models//rubbish_classification.pkl")
            scheduler.step()
            print("epoch : {arg1}".format(arg1=epoch))
        measures = (f'train loss {metric[0] / metric[2]:.3f}, ' f'train acc {metric[1] / metric[2]:.3f}')
        measures += f', valid acc {valid_acc:.3f}'
        for avg in range(5):
            # 五次取平均
            test_top1_acc += tb.evaluate_accuracy_gpu(net, test_iter, mode=1)
            test_top5_acc += tb.evaluate_accuracy_gpu(net, test_iter, mode=5)
        test_animator.add(k + 1, (test_top1_acc, test_top5_acc))
        print("test_top1_avgacc: " + str(test_top1_acc))
        print("test_top5_avgacc: " + str(test_top5_acc))
    test_top1_acc, test_top5_acc = 0.0, 0.0


devices, num_epochs, lr, wd = tb.try_all_gpus(), 10, 2e-4, 5e-4
print(devices)
lr_period, lr_decay = 4, 0.9
train(net, num_epochs, lr, wd, devices, lr_period, lr_decay)