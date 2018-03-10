# -*- coding: utf-8 -*-
"""
Author: Hung-Hsin Chen <chenhh@par.cse.nsysu.edu.tw>
License: GPL v3
"""

import torch
import numpy as np

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


if __name__ == '__main__':
    # torch_tensor()
    # torch_to_numpy()
    torch_auto_grad()