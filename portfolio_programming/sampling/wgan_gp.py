# -*- coding: utf-8 -*-
"""
Author: Hung-Hsin Chen <chen1116@gmail.com>

https://github.com/znxlwm/pytorch-generative-model-collections

WGAN
------
https://github.com/martinarjovsky/WassersteinGAN


WGAN-GP
---------
https://github.com/caogang/wgan-gp
https://github.com/LynnHo/Pytorch-WGAN-GP-DRAGAN-Celeba

RCGAN
---------
Real-valued (Medical) Time Series Generation with Recurrent Conditional GANs
https://arxiv.org/abs/1706.02633
https://github.com/ratschlab/RGAN

"""

from time import time

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import xarray as xr
from torch.autograd import (Variable, grad)

import portfolio_programming as pp


def print_network(net):
    num_params = 0
    for param in net.parameters():
        num_params += param.numel()
    print(net)
    print('Total number of parameters: {}'.format(num_params))


class Generator(nn.Module):
    def __init__(self):
        pass


class Discriminator(nn.Module):
    def __init__(self):
        pass


class WGAN(object):
    def __init__(self, args):
        # parameters
        self.epoch = args.epoch
        self.sample_num = 64
        self.batch_size = args.batch_size
        self.save_dir = args.save_dir
        self.result_dir = args.result_dir
        self.dataset = args.dataset
        self.log_dir = args.log_dir
        self.gpu_mode = args.gpu_mode
        self.model_name = args.gan_type
        # clipping value
        self.c = 0.01

        # the number of iterations of the critic per generator iteration
        self.n_critic = 5

        # networks init
        self.G = Generator(self.dataset)
        self.D = Discriminator(self.dataset)
        self.G_optimizer = optim.Adam(self.G.parameters(), lr=args.lrG,
                                      betas=(args.beta1, args.beta2))
        self.D_optimizer = optim.Adam(self.D.parameters(), lr=args.lrD,
                                      betas=(args.beta1, args.beta2))

        if self.gpu_mode:
            self.G.cuda()
            self.D.cuda()

        print('---------- Networks architecture -------------')
        print_network(self.G)
        print_network(self.D)
        print('-----------------------------------------------')


class WGAN_GP(nn.Module):
    def __init__(self, args):
        # parameters
        self.epoch = args.epoch
        self.sample_num = 64
        self.batch_size = args.batch_size
        self.save_dir = args.save_dir
        self.result_dir = args.result_dir
        self.dataset = args.dataset
        self.log_dir = args.log_dir
        self.gpu_mode = args.gpu_mode
        self.model_name = args.gan_type
        self.grad_penalty = 0.25

        # the number of iterations of the critic per generator iteration
        self.n_critic = 5

        # networks init
        self.G = Generator(self.dataset)
        self.D = Discriminator(self.dataset)
        self.G_optimizer = optim.Adam(self.G.parameters(), lr=args.lrG,
                                      betas=(args.beta1, args.beta2))
        self.D_optimizer = optim.Adam(self.D.parameters(), lr=args.lrD,
                                      betas=(args.beta1, args.beta2))

        if self.gpu_mode:
            self.G.cuda()
            self.D.cuda()

        print('---------- Networks architecture -------------')
        print_network(self.G)
        print_network(self.D)
        print('-----------------------------------------------')

    def train(self):
        self.train_hist = {}
        self.train_hist['D_loss'] = []
        self.train_hist['G_loss'] = []
        self.train_hist['per_epoch_time'] = []
        self.train_hist['total_time'] = []

        if self.gpu_mode:
            self.y_real_ = Variable(torch.ones(self.batch_size, 1).cuda())
            self.y_fake_ = Variable(torch.zeros(self.batch_size, 1).cuda())
        else:
            self.y_real_ = Variable(torch.ones(self.batch_size, 1))
            self.y_fake_ = Variable(torch.zeros(self.batch_size, 1))

        self.D.train()
        print('training start!!')
        start_time = time()

        for epoch in range(self.epoch):
            self.G.train()
            epoch_start_time = time()
            for iter, (x_, _) in enumerate(self.data_loader):
                if iter == self.data_loader.dataset.__len__() // self.batch_size:
                    break

                z_ = torch.rand((self.batch_size, self.z_dim))

                if self.gpu_mode:
                    x_ = Variable(x_.cuda())
                    z_ = Variable(z_.cuda())
                else:
                    x_ = Variable(x_)
                    z_ = Variable(z_)

                # update D network
                self.D_optimizer.zero_grad()

                D_real = self.D(x_)
                D_real_loss = -torch.mean(D_real)

                G_ = self.G(z_)
                D_fake = self.D(G_)
                D_fake_loss = torch.mean(D_fake)

                # gradient penalty
                if self.gpu_mode:
                    alpha = torch.rand(x_.size()).cuda()
                else:
                    alpha = torch.rand(x_.size())

                x_hat = Variable(alpha * x_.data + (1 - alpha) * G_.data,
                                 requires_grad=True)
                pred_hat = self.D(x_hat)
                if self.gpu_mode:
                    gradients = grad(outputs=pred_hat, inputs=x_hat,
                                     grad_outputs=torch.ones(
                                         pred_hat.size()).cuda(),
                                     create_graph=True, retain_graph=True,
                                     only_inputs=True)[0]
                else:
                    gradients = grad(outputs=pred_hat, inputs=x_hat,
                                     grad_outputs=torch.ones(pred_hat.size()),
                                     create_graph=True, retain_graph=True,
                                     only_inputs=True)[0]

                gradient_penalty = self.lambda_ * ((gradients.view(
                    gradients.size()[0], -1).norm(2, 1) - 1) ** 2).mean()

                D_loss = D_real_loss + D_fake_loss + gradient_penalty

                D_loss.backward()
                self.D_optimizer.step()

                if ((iter + 1) % self.n_critic) == 0:
                    # update G network
                    self.G_optimizer.zero_grad()

                    G_ = self.G(z_)
                    D_fake = self.D(G_)
                    G_loss = -torch.mean(D_fake)
                    self.train_hist['G_loss'].append(G_loss.data[0])

                    G_loss.backward()
                    self.G_optimizer.step()

                    self.train_hist['D_loss'].append(D_loss.data[0])

                if ((iter + 1) % 100) == 0:
                    print("Epoch: [%2d] [%4d/%4d] D_loss: %.8f, G_loss: %.8f" %
                          ((epoch + 1), (iter + 1),
                           self.data_loader.dataset.__len__() // self.batch_size,
                           D_loss.data[0], G_loss.data[0]))

            self.train_hist['per_epoch_time'].append(
                time.time() - epoch_start_time)
            self.visualize_results((epoch + 1))
        # end of epoch

        self.train_hist['total_time'].append(time() - start_time)
        print("Avg one epoch time: {:.2f}, total {} epochs time: {:.2f}".format(
            np.mean(self.train_hist['per_epoch_time']),
            self.epoch,
            self.train_hist['total_time'][0])
        )


def generate_data():
    risky_roi_xarr = xr.open_dataarray(
        pp.TAIEX_2005_LARGESTED_MARKET_CAP_DATA_NC)
    print(risky_roi_xarr)

    start_date = pp.EXP_START_DATE
    end_date = pp.EXP_END_DATE

    rolling_window_size = 240

    # How many critic iterations per generator iteration.
    critic_iter = 5

    # Gradient penalty lambda hyperparameter.
    grad_penalty = 10

    # Max number of data examples to load.
    MAX_N_EXAMPLES = 10000000  # 10000000


if __name__ == '__main__':
    generate_data()
