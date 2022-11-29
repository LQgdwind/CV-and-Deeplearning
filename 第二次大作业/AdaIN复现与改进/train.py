import torch
import torch.nn as nn
import torch.utils.data as data
import torch.optim as optim
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import os
from model import AdaIN
from model import Encoder
from model import Decoder
from dataDisplaying import Multi_Animator
from dataLoading import ContentDataset
from dataLoading import StyleDataset
from dataDisplaying import mu
from dataDisplaying import sigma
from dataDisplaying import plot_result_images

class TrainModel(object):
    def __init__(self):
        # hyper parameters
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.num_epochs = 500
        self.batch_size = 40
        self.learning_rate = 1e-4
        self.lamda = 0.01

        # Initialisation
        self.adain = AdaIN().to(self.device)
        self.decoder = Decoder().to(self.device)
        self.encoder = Encoder().to(self.device)
        self.optimizer = optim.Adam(self.decoder.parameters(), lr=self.learning_rate)
        for param in self.encoder.parameters():
            param.requires_grad = False

        # loss
        self.mse = nn.MSELoss()

        # drawing
        self.figure, self.faxes = plt.subplots(6, 1, figsize=(30, 30), sharex=False, sharey=False, squeeze=False)
        loss_legend = ['Content Loss', 'Style Loss']
        self.loss_animator = Multi_Animator(xlabel='loss', xlim=[1, 5000], legend=loss_legend, fig_main=self.figure,
                                            axes_main=self.faxes, rows=1, cols=0)
        style_loss_legend = ['mu_loss', 'std_loss']
        self.style_loss_animator = Multi_Animator(xlabel='style_loss', xlim=[1, 5000], legend=style_loss_legend,
                                                  fig_main=self.figure, axes_main=self.faxes, rows=2, cols=0)
        epoch_legend = ['loss']
        self.epoch_animator = Multi_Animator(xlabel='epoch', xlim=[1, 200], legend=epoch_legend, fig_main=self.figure,
                                             axes_main=self.faxes, rows=0, cols=0)

    def load_data(self):
        transform = transforms.Compose([transforms.ToTensor(),
                                        transforms.Resize(512),
                                        transforms.RandomCrop(256)])
        content_dataset = ContentDataset(transform=transform)
        style_dataset = StyleDataset(transform=transform)
        self.content_loader = data.DataLoader(dataset=content_dataset,
                                              batch_size=self.batch_size,
                                              shuffle=True)
        self.style_loader = data.DataLoader(dataset=style_dataset,
                                            batch_size=self.batch_size,
                                            shuffle=True)

    def train_epoch(self, epoch=0):
        epoch_loss = 0
        i = 0
        for c, s in zip(self.content_loader, self.style_loader):
            i = i + 1
            c = c.to(self.device)
            s = s.to(self.device)
            _, _, _, f_c = self.encoder(c)
            p1_s, p2_s, p3_s, p4_s = self.encoder(s)
            t = self.adain(f_c, p4_s)
            g = self.decoder(t)
            p1_g, p2_g, p3_g, p4_g = self.encoder(g)
            content_loss = self.mse(p4_g, t)
            mu_loss = self.mse(mu(p1_g), mu(p1_s)) + self.mse(mu(p2_g), mu(p2_s)) + self.mse(mu(p3_g),
                                                                                             mu(p3_s)) + self.mse(
                mu(p4_g), mu(p4_s))
            std_loss = self.mse(sigma(p1_g), sigma(p1_s)) + self.mse(sigma(p2_g), sigma(p2_s)) + self.mse(sigma(p3_g),
                                                                                                          sigma(p3_s)) + self.mse(sigma(p4_g), sigma(p4_s))
            style_loss = mu_loss + std_loss
            loss = content_loss + self.lamda * style_loss

            epoch_loss = epoch_loss + loss.item()

            self.loss_animator.add(25 * epoch + i + 1, (content_loss.item(), style_loss.item()))
            self.style_loss_animator.add(25 * epoch + i + 1, (mu_loss.item(), std_loss.item()))

            del mu_loss, std_loss, style_loss, content_loss
            torch.cuda.empty_cache()

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

        self.epoch_animator.add(epoch + 1, epoch_loss)
        print('epoch:', epoch, 'loss:', epoch_loss)

        return c, s

    def pred_and_saving(self, epoch, c, s, saving_num, fig, ax):
        c = c.to(self.device)
        s = s.to(self.device)
        zlq, zlq, zlq, f_c = self.encoder(c)
        zlq, zlq, zlq, p4_s = self.encoder(s)
        t = self.adain(f_c, p4_s)
        g = self.decoder(t)
        content = c[0].permute(1, 2, 0).cpu().detach().numpy()
        style = s[0].permute(1, 2, 0).cpu().detach().numpy()
        out = g[0].permute(1, 2, 0).cpu().detach().numpy()
        if (epoch + 1) % saving_num == 0:
            plot_result_images(content, style, out, fig, ax, epoch + 1)
            torch.save(self.decoder.state_dict(), '/root/trans/decoder.pth')

    def train_full(self, saving_num=100):
        self.decoder.train()
        self.load_data()  # load data to DataLoaders
        if os.path.isfile(f"/root/trans/experiments/zlq_AdaIN_state_dict_epoch_500.pth"):
            params = torch.load(f"/root/trans/experiments/zlq_AdaIN_state_dict_epoch_500.pth", map_location=self.device)
            self.decoder.load_state_dict(params)

        print("*" * 60 + "Start training" + "*" * 60)
        for epoch in range(self.num_epochs):  # run epoch
            c, s = self.train_epoch(epoch)
            if (epoch + 1) % saving_num == 0:
                self.pred_and_saving(epoch, c, s, saving_num, self.figure, self.faxes)
        print("*" * 60 + "End training" + "*" * 60)


if __name__ == '__main__':
    model = TrainModel()
    model.train_full(saving_num=100)

