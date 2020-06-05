"""
Author: Pruthvi Suryadevara
Email:  pruthvi.suryadevara@tifr.res.in
Finding fourier transform of Step function

"""

import numpy as np
import matplotlib.pyplot as plt
import numpy.fft as ft


def f(x):                    # Defining Sinc Function
    y = np.zeros(len(x))
    for i in range(len(x)):
        if -1 < x[i] < 1:
            y[i] = 1
    return(y)


def four(f, x_r, n):      # Fourier transform in given range
    x = np.linspace(x_r[0], x_r[1], n)     # Defining x in the range
    y = f(x)
    ky = ft.fftshift(ft.fft(y))         # Fiding Fourier transform
    k = 2*np.pi*ft.fftshift(ft.fftfreq(n, x[1]-x[0]))
    c_factor = ((x[1]-x[0]) * np.exp(-1j*k*x_r[0])) / np.sqrt(2*np.pi)
    return([k, ky*c_factor])


x_p = np.linspace(-10, 10, 100)
y_p = f(x_p)
plt.subplot(2, 2, 1, title="Step Function")
plt.plot(x_p, y_p, 'k')
plt.xlabel("$x$")
plt.ylabel("$y$")

[k, ky] = four(f, [-10, 10], 128)
plt.subplot(2, 2, 2, title=" $x \in [-10,10] , \delta x = 0.157$")
plt.plot(k, ky.real, '.-b')
plt.xlabel("Frequency")
plt.ylabel("Fourier transform")


[k, ky] = four(f, [-5, 5], 64)
plt.subplot(2, 2, 3, title=" $x \in [-5, 5] , \delta x = 0.157$")
plt.plot(k, ky.real, '.-b')
plt.xlabel("Frequency")
plt.ylabel("Fourier transform")


[k, ky] = four(f, [-10, 10], 32)
plt.subplot(2, 2, 4, title=" $x \in [-10,10] , \delta x = 0.625$")
plt.plot(k, ky.real, '.-b')
plt.xlabel("Frequency")
plt.ylabel("Fourier transform")
plt.subplots_adjust(hspace=0.6, wspace=0.4)
plt.show()
