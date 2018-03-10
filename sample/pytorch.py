# -*- coding: utf-8 -*-
"""
Author: Hung-Hsin Chen <chenhh@par.cse.nsysu.edu.tw>
License: GPL v3
"""

import numpy as np
import torch
from torch.autograd import (Variable, )
import torch.nn.functional as F
import matplotlib.pyplot as plt

def torch_tensor():
    # 构造一个未初始化的5*3的矩阵
    x = torch.Tensor(5, 3)
    print("Tensor:", x)

    x = torch.rand(5, 3)
    print("random:", x)
    print(x.size())

    y = torch.rand(5, 3)
    # 此处 将两个同形矩阵相加有两种语法结构
    # 语法一, operator overloading
    print("x+y=", x + y)

    # 语法二
    print("x+y=", torch.add(x, y))

    # 另外输出tensor也有两种写法
    result = torch.Tensor(5, 3)  # 语法一
    print("result=", result)
    torch.add(x, y, out=result)  # 语法二
    print("result=", result)

    # 特别注明：任何可以改变tensor内容的操作都会在方法名后加一个下划线'_'
    # 例如：x.copy_(y), x.t_(), 这俩都会改变x的值
    y.add_(x)  # 将y与x相加


def torch_to_numpy():
    # 此处演示tensor和numpy数据结构的相互转换
    a = torch.ones(5)
    b = a.numpy()
    print("a=", a)
    print("b=", b)

    # 轉成numpy只是轉換指標而已，a,b兩者指向同樣的內容
    # 只要改變一個，另一個就會隨之變動
    b[3] = 10
    print("a=", a)
    print("b=", b)

    a[1] = 99
    print("a=", a)
    print("b=", b)

    # array to tensor
    a = np.ones(5)
    b = torch.from_numpy(a)
    np.add(a, 1, out=a)
    print(a)
    print(b)


def torch_variable():
    # Variable in torch is to build a computational graph,
    # but this graph is dynamic compared with a static graph in Tensorflow or Theano.
    # So torch does not have placeholder, torch can just pass variable to the computational graph.

    tensor = torch.FloatTensor([[1, 2], [3, 4]])  # build a tensor

    # build a variable, usually for compute gradients
    variable = Variable(tensor, requires_grad=True)

    print(tensor)  # [torch.FloatTensor of size 2x2]
    print(variable)  # [torch.FloatTensor of size 2x2]

    # till now the tensor and variable seem the same.
    # However, the variable is a part of the graph, it's a part of the auto-gradient.

    t_out = torch.mean(tensor * tensor)  # x^2
    v_out = torch.mean(variable * variable)  # x^2
    print(t_out)
    print(v_out)  # 7.5

    v_out.backward()  # backpropagation from v_out
    # v_out = 1/4 * sum(variable*variable)
    # the gradients w.r.t the variable, d(v_out)/d(variable) = 1/4*2*variable = variable/2
    print(variable.grad)
    '''
     0.5000  1.0000
     1.5000  2.0000
    '''

    print(variable)  # this is data in variable format
    """
    Variable containing:
     1  2
     3  4
    [torch.FloatTensor of size 2x2]
    """

    print(variable.data)  # this is data in tensor format
    """
     1  2
     3  4
    [torch.FloatTensor of size 2x2]
    """

    print(variable.data.numpy())  # numpy format
    """
    [[ 1.  2.]
     [ 3.  4.]]
    """


def torch_auto_grad():
    from torch.autograd import Variable
    x = Variable(torch.ones(2, 2), requires_grad=True)
    y = x + 2

    # y 是作为一个操作的结果创建的因此y有一个creator
    # z = 3*y^2 = 3*(x^2 + 4x+ 4) = 3*x^2 + 12*x + 12
    # out = 1/4 (z_11 + z_12 + z_21 + z_22)
    z = y * y * 3
    out = z.mean()

    # 现在我们来使用反向传播
    # d(out)/dx = 1/4(
    out.backward()

    # 在此处输出 d(out)/dx
    # 最终得出的结果应该是一个全是4.5的矩阵
    print(x.grad)

def plot_activation_function():
    # fake data
    x = torch.linspace(-5, 5, 200)  # x data (tensor), shape=(100, 1)
    x = Variable(x)
    x_np = x.data.numpy()  # numpy array for plotting

    # following are popular activation functions
    y_relu = F.relu(x).data.numpy()
    y_sigmoid = F.sigmoid(x).data.numpy()
    y_tanh = F.tanh(x).data.numpy()
    y_softplus = F.softplus(x).data.numpy()
    # y_softmax = F.softmax(x)  softmax is a special kind of activation function, it is about probability

    # plt to visualize these activation function
    plt.figure(1, figsize=(8, 6))
    plt.subplot(221)
    plt.plot(x_np, y_relu, c='red', label='relu')
    plt.ylim((-1, 5))
    plt.legend(loc='best')

    plt.subplot(222)
    plt.plot(x_np, y_sigmoid, c='red', label='sigmoid')
    plt.ylim((-0.2, 1.2))
    plt.legend(loc='best')

    plt.subplot(223)
    plt.plot(x_np, y_tanh, c='red', label='tanh')
    plt.ylim((-1.2, 1.2))
    plt.legend(loc='best')

    plt.subplot(224)
    plt.plot(x_np, y_softplus, c='red', label='softplus')
    plt.ylim((-0.2, 6))
    plt.legend(loc='best')

    plt.show()


def torch_nn_regression():
    # torch.manual_seed(1)    # reproducible

    # x data (tensor), shape=(100, 1)
    x = torch.unsqueeze(torch.linspace(-1, 1, 100), dim=1)

    # noisy y data (tensor), shape=(100, 1)
    y = x.pow(2) + 0.2 * torch.rand(x.size())

    # torch can only train on Variable, so convert them to Variable
    x, y = Variable(x), Variable(y)

    # plt.scatter(x.data.numpy(), y.data.numpy())
    # plt.show()

    class Net(torch.nn.Module):
        def __init__(self, n_feature, n_hidden, n_output):
            super(Net, self).__init__()
            self.hidden = torch.nn.Linear(n_feature, n_hidden)  # hidden layer
            self.predict = torch.nn.Linear(n_hidden, n_output)  # output layer

        def forward(self, x):
            x = F.relu(self.hidden(x))  # activation function for hidden layer
            x = self.predict(x)  # linear output
            return x

    net = Net(n_feature=1, n_hidden=10, n_output=1)  # define the network
    print(net)  # net architecture

    optimizer = torch.optim.SGD(net.parameters(), lr=0.5)
    loss_func = torch.nn.MSELoss()  # this is for regression mean squared loss

    plt.ion()  # something about plotting

    for t in range(100):
        prediction = net(x)  # input x and predict based on x

        loss = loss_func(prediction, y)  # must be (1. nn output, 2. target)

        optimizer.zero_grad()  # clear gradients for next train
        loss.backward()  # backpropagation, compute gradients
        optimizer.step()  # apply gradients

        if t % 5 == 0:
            # plot and show learning process
            plt.cla()
            plt.scatter(x.data.numpy(), y.data.numpy())
            plt.plot(x.data.numpy(), prediction.data.numpy(), 'r-', lw=5)
            plt.text(0.5, 0, 'Loss=%.4f' % loss.data[0],
                     fontdict={'size': 20, 'color': 'red'})
            plt.pause(0.1)

    plt.ioff()
    plt.show()

def torch_nn_classification():
    # torch.manual_seed(1)    # reproducible

    # make fake data
    n_data = torch.ones(100, 2)
    x0 = torch.normal(2 * n_data, 1)  # class0 x data (tensor), shape=(100, 2)
    y0 = torch.zeros(100)  # class0 y data (tensor), shape=(100, 1)
    x1 = torch.normal(-2 * n_data, 1)  # class1 x data (tensor), shape=(100, 2)
    y1 = torch.ones(100)  # class1 y data (tensor), shape=(100, 1)
    x = torch.cat((x0, x1), 0).type(
        torch.FloatTensor)  # shape (200, 2) FloatTensor = 32-bit floating
    y = torch.cat((y0, y1), ).type(
        torch.LongTensor)  # shape (200,) LongTensor = 64-bit integer

    # torch can only train on Variable, so convert them to Variable
    x, y = Variable(x), Variable(y)

    # plt.scatter(x.data.numpy()[:, 0], x.data.numpy()[:, 1], c=y.data.numpy(), s=100, lw=0, cmap='RdYlGn')
    # plt.show()

    class Net(torch.nn.Module):
        def __init__(self, n_feature, n_hidden, n_output):
            super(Net, self).__init__()
            self.hidden = torch.nn.Linear(n_feature, n_hidden)  # hidden layer
            self.out = torch.nn.Linear(n_hidden, n_output)  # output layer

        def forward(self, x):
            x = F.relu(self.hidden(x))  # activation function for hidden layer
            x = self.out(x)
            return x

    net = Net(n_feature=2, n_hidden=10, n_output=2)  # define the network
    print(net)  # net architecture

    optimizer = torch.optim.SGD(net.parameters(), lr=0.02)
    loss_func = torch.nn.CrossEntropyLoss()  # the target label is NOT an one-hotted

    plt.ion()  # something about plotting

    for t in range(100):
        out = net(x)  # input x and predict based on x
        # must be (1. nn output, 2. target), the target label is NOT one-hotted
        loss = loss_func(out, y)

        optimizer.zero_grad()  # clear gradients for next train
        loss.backward()  # backpropagation, compute gradients
        optimizer.step()  # apply gradients

        if t % 2 == 0:
            # plot and show learning process
            plt.cla()
            prediction = torch.max(out, 1)[1]
            pred_y = prediction.data.numpy().squeeze()
            target_y = y.data.numpy()
            plt.scatter(x.data.numpy()[:, 0], x.data.numpy()[:, 1], c=pred_y,
                        s=100, lw=0, cmap='RdYlGn')
            accuracy = sum(pred_y == target_y) / 200.
            plt.text(1.5, -4, 'Accuracy=%.2f' % accuracy,
                     fontdict={'size': 20, 'color': 'red'})
            plt.pause(0.1)

    plt.ioff()
    plt.show()


def quick_nn_construction():
    # replace following class code with an easy sequential network
    class Net(torch.nn.Module):
        def __init__(self, n_feature, n_hidden, n_output):
            super(Net, self).__init__()
            self.hidden = torch.nn.Linear(n_feature, n_hidden)  # hidden layer
            self.predict = torch.nn.Linear(n_hidden, n_output)  # output layer

        def forward(self, x):
            x = F.relu(self.hidden(x))  # activation function for hidden layer
            x = self.predict(x)  # linear output
            return x

    net1 = Net(1, 10, 1)

    # easy and fast way to build your network
    net2 = torch.nn.Sequential(
        torch.nn.Linear(1, 10),
        torch.nn.ReLU(),
        torch.nn.Linear(10, 1)
    )

    print(net1)  # net1 architecture
    """
    Net (
      (hidden): Linear (1 -> 10)
      (predict): Linear (10 -> 1)
    )
    """

    print(net2)  # net2 architecture
    """
    Sequential (
      (0): Linear (1 -> 10)
      (1): ReLU ()
      (2): Linear (10 -> 1)
    )
    """


if __name__ == '__main__':
    # torch_tensor()
    # torch_to_numpy()
    # torch_variable()
    # torch_auto_grad()
    plot_activation_function()