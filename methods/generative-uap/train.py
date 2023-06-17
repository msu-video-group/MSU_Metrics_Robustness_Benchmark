#!/usr/bin/env python
# -*- coding: utf-8 -*-



from torch import nn

import torch
from torch.autograd import Variable
from torch.utils.data import DataLoader

from tqdm import tqdm

from read_dataset import MyCustomDataset
from evaluate import jpeg_generator 

from evaluate import train_main_uap


class UnetGenerator(nn.Module):
    def __init__(self, input_nc, output_nc, ngf, norm_type='batch', act_type='selu'):
        super(UnetGenerator, self).__init__()
        self.name = 'unet'
        self.conv1 = nn.Conv2d(input_nc, ngf, 4, 2, 1)
        self.conv2 = nn.Conv2d(ngf, ngf * 2, 4, 2, 1)
        self.conv3 = nn.Conv2d(ngf * 2, ngf * 4, 4, 2, 1)
        self.conv4 = nn.Conv2d(ngf * 4, ngf * 8, 4, 2, 1)
        self.conv5 = nn.Conv2d(ngf * 8, ngf * 8, 4, 2, 1)
        self.conv6 = nn.Conv2d(ngf * 8, ngf * 8, 4, 2, 1)
        self.conv7 = nn.Conv2d(ngf * 8, ngf * 8, 4, 2, 1)
        self.conv8 = nn.Conv2d(ngf * 8, ngf * 8, 4, 2, 1)
        self.dconv1 = nn.ConvTranspose2d(ngf * 8, ngf * 8, 4, 2, 1)
        self.dconv2 = nn.ConvTranspose2d(ngf * 8 * 2, ngf * 8, 4, 2, 1)
        self.dconv3 = nn.ConvTranspose2d(ngf * 8 * 2, ngf * 8, 4, 2, 1)
        self.dconv4 = nn.ConvTranspose2d(ngf * 8 * 2, ngf * 8, 4, 2, 1)
        self.dconv5 = nn.ConvTranspose2d(ngf * 8 * 2, ngf * 4, 4, 2, 1)
        self.dconv6 = nn.ConvTranspose2d(ngf * 4 * 2, ngf * 2, 4, 2, 1)
        self.dconv7 = nn.ConvTranspose2d(ngf * 2 * 2, ngf, 4, 2, 1)
        self.dconv8 = nn.ConvTranspose2d(ngf * 2, output_nc, 4, 2, 1)

        if norm_type == 'batch':
            self.norm = nn.BatchNorm2d(ngf)
            self.norm2 = nn.BatchNorm2d(ngf * 2)
            self.norm4 = nn.BatchNorm2d(ngf * 4)
            self.norm8 = nn.BatchNorm2d(ngf * 8)
        elif norm_type == 'instance':
            self.norm = nn.InstanceNorm2d(ngf)
            self.norm2 = nn.InstanceNorm2d(ngf * 2)
            self.norm4 = nn.InstanceNorm2d(ngf * 4)
            self.norm8 = nn.InstanceNorm2d(ngf * 8)
        self.leaky_relu = nn.LeakyReLU(0.2, True)

        if act_type == 'selu':
            self.act = nn.SELU(True)
        else:
            self.act = nn.ReLU(True)

        self.dropout = nn.Dropout(0.5)

        self.tanh = nn.Tanh()

    def forward(self, input):
        # Encoder
        # Convolution layers:
        # input is (nc) x 512 x 1024
        e1 = self.conv1(input)
        # state size is (ngf) x 256 x 512
        e2 = self.norm2(self.conv2(self.leaky_relu(e1)))
        # state size is (ngf x 2) x 128 x 256
        e3 = self.norm4(self.conv3(self.leaky_relu(e2)))
        # state size is (ngf x 4) x 64 x 128
        e4 = self.norm8(self.conv4(self.leaky_relu(e3)))
        # state size is (ngf x 8) x 32 x 64
        e5 = self.norm8(self.conv5(self.leaky_relu(e4)))
        # state size is (ngf x 8) x 16 x 32
        e6 = self.norm8(self.conv6(self.leaky_relu(e5)))
        # state size is (ngf x 8) x 8 x 16
        e7 = self.norm8(self.conv7(self.leaky_relu(e6)))
        # state size is (ngf x 8) x 4 x 8
        # No batch norm on output of Encoder
        e8 = self.conv8(self.leaky_relu(e7))

        # Decoder
        # Deconvolution layers:
        # state size is (ngf x 8) x 2 x 4
        d1_ = self.dropout(self.norm8(self.dconv1(self.act(e8))))
        # state size is (ngf x 8) x 4 x 8
        d1 = torch.cat((d1_, e7), 1)
        d2_ = self.dropout(self.norm8(self.dconv2(self.act(d1))))
        # state size is (ngf x 8) x 8 x 16
        d2 = torch.cat((d2_, e6), 1)
        d3_ = self.dropout(self.norm8(self.dconv3(self.act(d2))))
        # state size is (ngf x 8) x 16 x 32
        d3 = torch.cat((d3_, e5), 1)
        d4_ = self.norm8(self.dconv4(self.act(d3)))
        # state size is (ngf x 8) x 32 x 64
        d4 = torch.cat((d4_, e4), 1)
        d5_ = self.norm4(self.dconv5(self.act(d4)))
        # state size is (ngf x 4) x 64 x 128
        d5 = torch.cat((d5_, e3), 1)
        d6_ = self.norm2(self.dconv6(self.act(d5)))
        # state size is (ngf x 2) x 128 x 256
        d6 = torch.cat((d6_, e2), 1)
        d7_ = self.norm(self.dconv7(self.act(d6)))
        # state size is (ngf) x 256 x 512
        d7 = torch.cat((d7_, e1), 1)
        d8 = self.dconv8(self.act(d7))
        # state size is (nc) x 512 x 1024
        output = self.tanh(d8)
        return output




def normalize_and_scale(delta_im, mode='train'):
    delta_im = (delta_im) * 10.0/255.0
    return delta_im




def train(metric_model, path_train, batch_size=8, is_fr=False, jpeg_quality=None, metric_range=100, device='cpu'):
    ds_train = MyCustomDataset(path_gt=path_train, device=device)
    dl_train = DataLoader(ds_train, batch_size=batch_size, shuffle=True)
    eps = 10.0/255.
    n_epoch = 1
    lr = 0.001
    universal_noise = torch.rand((1, 3, 256, 256)).to(device)
    universal_noise = Variable(universal_noise, requires_grad=True)
    netG = UnetGenerator(3, 3, 64, norm_type='instance', act_type='relu').to(device)
    optimizer = torch.optim.Adam(netG.parameters(), lr=lr)
    for epoch in range(n_epoch):
        for u in tqdm(range(len(dl_train))):
            y = next(iter(dl_train))
            
            for orig_image, jpeg_image in jpeg_generator(y, jpeg_quality):
                delta_im = netG(universal_noise)
                delta_im = normalize_and_scale(delta_im, 'train')
                if is_fr:
                    res = (jpeg_image.to(device) + delta_im).type(torch.FloatTensor)
                    res.data.clamp_(min=0.,max=1.)
                    a = 1 - metric_model(y, res).mean() / metric_range
                else:
                    res = y + delta_im
                    res.data.clamp_(min=0.,max=1.)
                    a = 1 - metric_model(res).mean() / metric_range
            optimizer.zero_grad()
            a.backward()
            optimizer.step()
            #universal_noise.data = universal_noise.data * min(1, eps/(universal_noise.data**2).mean()**0.5)
            universal_noise.data.clamp_(min=-eps, max=eps)
    delta_im = netG(universal_noise)
    delta_im = normalize_and_scale(delta_im, 'train')  
    return delta_im.squeeze().data.cpu().numpy().transpose(1, 2, 0)


if __name__ == "__main__":
    train_main_uap(train)