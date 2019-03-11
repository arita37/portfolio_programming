# -*- coding: utf-8 -*-
"""
Author: Hung-Hsin Chen <chen1116@gmail.com>
"""

import matplotlib.pyplot as plt
import numpy as np


def sine_wave():

    Fs = 150.0  # sampling rate
    Ts = 1.0 / Fs  # sampling interval
    t = np.arange(0, 1, Ts)  # time vector

    ff = 5  # frequency of the signal
    y = np.sin(2 * np.pi * ff * t)
    n = len(y)  # length of the signal
    k = np.arange(n)
    T = n / Fs
    frq = k / T  # two sides frequency range
    frq = frq[range(n // 2)]  # one side frequency range

    Y = np.fft.fft(y) / n  # fft computing and normalization
    Y = Y[range(n // 2)]

    fig, ax = plt.subplots(2, 1)
    ax[0].plot(t, y)
    ax[0].set_xlabel('Time')
    ax[0].set_ylabel('Amplitude')
    ax[1].plot(frq, abs(Y), 'r')  # plotting the spectrum
    ax[1].set_xlabel('Freq (Hz)')
    ax[1].set_ylabel('|Y(freq)|')
    plt.show()


def sine_wave2():
    # from scipy.fftpack import fft
    # Number of sample points
    N =1000

    # sample time length
    T = 1./N

    xs = np.linspace(0, N*T, N)
    ys = (np.sin(50.0 * 2.0*np.pi*xs) + 0.5*np.cos(200.0 * 2.0*np.pi*xs) +
          0.5 * np.sin(200.0 * 2.0 * np.pi * xs)
          )

    freq_xs = np.linspace(0.0, 1.0/(2. *T), N//2)

    freq_ys = np.fft.fft(ys) / N
    freq_ys = np.abs(freq_ys[0:N // 2])

    fig, ax = plt.subplots(2, 1)
    ax[0].plot(xs, ys)
    ax[0].set_xlabel('Time')
    ax[0].set_ylabel('Amplitude')
    ax[1].plot(freq_xs, abs(freq_ys), 'r')  # plotting the spectrum
    ax[1].set_xlabel('Freq (Hz)')
    ax[1].set_ylabel('|Y(freq)|')
    plt.show()

def plot_fft(ys, freq_ys):
    n_point = len(ys)
    xs = np.arange(n_point)

    fig, ax = plt.subplots(2, 1)
    ax[0].plot(xs, ys)
    ax[0].set_xlabel('Time')
    ax[0].set_ylabel('Amplitude')
    ax[1].plot(abs(freq_ys), 'r')  # plotting the spectrum
    ax[1].set_xlabel('Freq (Hz)')
    ax[1].set_ylabel('|Y(freq)|')

    plt.show()


def ar1(n_point=1000):
    # time domain
    pts = np.random.randn(n_point)
    pts2 = np.random.randn(n_point)
    xs = np.linspace(0, 1, 1000)
    ys = pts.cumsum() + 3*np.cos(200.0 * 2.0*np.pi*xs)

    # freq domain
    freq_ys = np.fft.fft(ys) / n_point
    freq_ys = freq_ys[range(int(n_point / 2))]
    print(freq_ys)
    plot_fft(ys, freq_ys)




if __name__ == '__main__':
    # sine_wave()
    # sine_wave2()
    ar1()