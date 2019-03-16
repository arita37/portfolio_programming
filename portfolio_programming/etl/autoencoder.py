# -*- coding: utf-8 -*-
"""
Author: Hung-Hsin Chen <chen1116@gmail.com>

https://github.com/SherlockLiao/pytorch-beginner/blob/master/08-AutoEncoder/simple_autoencoder.py
"""

import time
import pandas as pd
import torch
import torchvision
from torch import nn
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import MNIST
from torchvision.utils import save_image
import portfolio_programming as pp


def AE(n_epochs=1000, batch_size=60, learning_rate=1e-3):
    # Dimensions: 3090 (items) x 50 (major_axis) x 6 (minor_axis)
    # Items axis: 2005-01-03 00:00:00 to 2017-06-30 00:00:00
    pnl = pd.read_pickle(pp.TAIEX_2005_LARGESTED_MARKET_CAP_DATA_XARRAY)
    n_batch = int(pnl.items.size / batch_size)

    class autoencoder(nn.Module):
        def __init__(self):
            super(autoencoder, self).__init__()
            self.encoder = nn.Sequential(
                nn.Linear(50, 128),
                nn.Tanh(),
                nn.Linear(128, 64),
                nn.Tanh(),
                nn.Linear(64, 32),
                nn.Tanh(),
                nn.Linear(32, 16),
                nn.Tanh(),
                nn.Linear(16, 8),
                nn.Tanh(),
                nn.Linear(8, 4),
                nn.Tanh(),
                nn.Linear(4, 2),
            )

            self.decoder = nn.Sequential(
                nn.Linear(2, 4),
                nn.Tanh(),
                nn.Linear(4, 8),
                nn.Tanh(),
                nn.Linear(8, 16),
                nn.Tanh(),
                nn.Linear(16, 32),
                nn.Tanh(),
                nn.Linear(32, 64),
                nn.Tanh(),
                nn.Linear(64, 128),
                nn.Tanh(),
                nn.Linear(128, 50),
                nn.Sigmoid()
            )

        def forward(self, x):
            x = self.encoder(x)
            x = self.decoder(x)
            return x

    model = autoencoder().cuda()
    print(model)
    #model = autoencoder()
    criterion = nn.MSELoss()
    #criterion = nn.L1Loss()
    #criterion = nn.KLDivLoss()
    optimizer = torch.optim.Adam(
        model.parameters(), lr=learning_rate, weight_decay=1e-5)

    for epoch in range(n_epochs):
        t_start = time.time()
        for bdx in range(n_batch):
            sdx = bdx * batch_size
            edx = sdx + batch_size
            rois = pnl.ix[sdx:edx, :, 'simple_roi'] 
            input = Variable(torch.FloatTensor(rois.values.T)).cuda()
            #input = Variable(torch.FloatTensor(rois.values.T))
            # ===================forward=====================
            output = model(input)
            loss = criterion(output, input)
            # ===================backward====================
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        # ===================log========================
        print('epoch [{}/{}], loss:{:.8f} {:.4f} secs'
              .format(epoch + 1, n_epochs, loss.data[0], time.time()-t_start))
        # if epoch % 10 == 0:
        #     pic = to_img(output.cpu().data)
        #     save_image(pic, './mlp_img/image_{}.png'.format(epoch))

    # torch.save(model.state_dict(), './sim_autoencoder.pth')


if __name__ == '__main__':
    AE()
