import torch.utils.data as data
import torchvision.transforms as transforms
import os
from skimage import io

class ContentDataset(data.Dataset):
    def __init__(self, transform=transforms.ToTensor()):
        super().__init__()
        self.content_path = '/root/trans/content/'
        self.style_path = '/root/trans/style_total/'
        self.content_images = [f for f in os.listdir(self.content_path)]
        self.transform = transform

    def __getitem__(self, index):
        # print(content_path + self.content_images[index])
        self.content_path = '/root/trans/content/'
        self.style_path = '/root/trans/style_total/'
        content_img = io.imread(self.content_path + self.content_images[index])
        # content_img = cv2.imread(self.content_path + self.content_images[index],cv2.IMREAD_COLOR)
        # content_img = cv2.cvtColor(content_img, cv2.COLOR_BGR2RGB)
        # content_img = self.transform(content_img.astype(np.float32))
        content_img = self.transform(content_img)
        return content_img

    def __len__(self):
        return len(self.content_images)

class StyleDataset(data.Dataset):
    def __init__(self, transform=transforms.ToTensor()):
        self.content_path = '/root/trans/content/'
        self.style_path = '/root/trans/style_total/'
        super().__init__()
        self.style_images = [f for f in os.listdir(self.style_path)]
        self.transform = transform

    def __getitem__(self, index):
        self.content_path = '/root/trans/content/'
        self.style_path = '/root/trans/style_total/'
        # style_img = cv2.imread(self.style_path + self.style_images[index],cv2.IMREAD_COLOR)
        style_img = io.imread(self.style_path + self.style_images[index])
        # style_img = cv2.cvtColor(style_img, cv2.COLOR_BGR2RGB)
        # style_img = self.transform(style_img.astype(np.float32))
        style_img = self.transform(style_img)
        return style_img

    def __len__(self):
        return len(self.style_images)