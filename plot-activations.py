#!/usr/bin/env python
# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
import matplotlib.pyplot as plt
plt.style.use('ggplot')

if __name__ == '__main__':
    fig, axes = plt.subplots(5)
    fig.set_size_inches(18.5, 10.5)
    fig.tight_layout()
    x = torch.arange(start=-10, end=10.1, step=0.1)

    # sigmoid
    sig = nn.Sigmoid()
    sig_out = sig(x)

    axes[0].plot(x, sig_out)
    axes[0].set(title='sigmoid')

    # tanh
    tanh = nn.Tanh()
    tanh_out = tanh(x)
    axes[1].plot(x, tanh_out)
    axes[1].set(title='tanh')

    # relu
    relu = nn.ReLU()
    relu_out = relu(x)
    axes[2].plot(x, relu_out)
    axes[2].set(title='ReLU')

    # leaky relu
    leaky_relu = nn.LeakyReLU()
    leaky_relu_out = leaky_relu(x)
    axes[3].plot(x, leaky_relu_out)
    axes[3].set(title='Leaky ReLU')

    # gelu
    gelu = nn.GELU()
    gelu_out = gelu(x)
    axes[4].plot(x, gelu_out)
    axes[4].set(title='GELU')


    # save fig
    fig.savefig('activation_fns.png')
