import dataLoading as dl
import dataEnhance as de
import dataProcess as dp
import merge_dataset as md
import torch
import torchvision
from torch.utils import data

def k_fold(num, batch_size = 128, num_k =10):
    transform = torchvision.transforms.Compose(
        [torchvision.transforms.Resize([224,224]),
         torchvision.transforms.ToTensor(),
         torchvision.transforms.Normalize(
             [0.5420377, 0.5206433, 0.49089015],
             [0.21957076, 0.21739748, 0.2220452])])
    if num_k == 10:
        train_valid_ds = []
        for i in range(10):
            train_valid_ds.append(torchvision.datasets.ImageFolder(".//tiny_data_after_light_augment//train" + str(i),
                                                                   transform=transform))
        if num != 0:
            train_ds = torchvision.datasets.ImageFolder(".//tiny_data_after_light_augment//train0", transform=transform)
            for i in range(9):
                if num == i + 1:
                    continue
                else:
                    md.merge(train_ds, train_valid_ds[i + 1])
        else:
            train_ds = torchvision.datasets.ImageFolder(".//tiny_data_after_light_augment//train1",
                                                        transform=transform)
            for i in range(8):
                md.merge(train_ds, train_valid_ds[i + 2])

        valid_ds = train_valid_ds[num]
        test_ds = torchvision.datasets.ImageFolder(".//valid", transform=transform)
        train_iter = data.DataLoader(train_ds, batch_size=batch_size, shuffle=False, drop_last=True)
        valid_iter = data.DataLoader(valid_ds, batch_size=batch_size, shuffle=False, drop_last=False)
        test_iter = data.DataLoader(test_ds, batch_size=batch_size, shuffle=False, drop_last=False)
        print(len(train_iter))
        return train_iter, valid_iter, test_iter

    elif num_k == 5:
        train_valid_ds = []
        for i in range(10):
            train_valid_ds.append(torchvision.datasets.ImageFolder(".//tiny_data_after_light_augment//train" + str(i),
                                                                   transform=transform))
        if num != 0:
            train_ds = torchvision.datasets.ImageFolder(".//tiny_data_after_light_augment//train0", transform=transform)
            md.merge(train_ds,train_valid_ds[1])
            for i in range(4):
                if num == i:
                    continue
                else:
                    md.merge(train_ds, train_valid_ds[2 * i + 1])
                    md.merge(train_ds, train_valid_ds[2 * i])
        else:
            train_ds = torchvision.datasets.ImageFolder(".//tiny_data_after_light_augment//train2",
                                                        transform=transform)
            for i in range(7):
                md.merge(train_ds, train_valid_ds[i + 3])

        valid_ds = train_valid_ds[ 2 * num]
        md.merge(valid_ds,train_valid_ds[2 * num + 1])
        test_ds = torchvision.datasets.ImageFolder(".//valid", transform=transform)
        train_iter = data.DataLoader(train_ds, batch_size=batch_size, shuffle=False, drop_last=True)
        valid_iter = data.DataLoader(valid_ds, batch_size=batch_size, shuffle=False, drop_last=False)
        test_iter = data.DataLoader(test_ds, batch_size=batch_size, shuffle=False, drop_last=False)
        # print(len(train_iter))
        return train_iter, valid_iter, test_iter


