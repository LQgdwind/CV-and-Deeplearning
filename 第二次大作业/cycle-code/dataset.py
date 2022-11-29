from __future__ import division
import os
from PIL import Image
import torch
import torch.utils.data as data

class GeneratorFolderDataset(data.Dataset):

    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.filenames = os.listdir(root_dir)
        self.transform = transform

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        img_name = self.filenames[idx]
        img_path = os.path.join(self.root_dir, img_name)
        sample = Image.open(img_path)
        if self.transform:
            sample = self.transform(sample)
        return sample


class GeneratorDataset(data.Dataset):

    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.filenames = os.listdir(root_dir)
        self.transform = transform

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        img_name = self.filenames[idx]
        img_path = os.path.join(self.root_dir, img_name)
        sample = Image.open(img_path).convert('RGB')
        if self.transform:
            sample = self.transform(sample)
        return sample