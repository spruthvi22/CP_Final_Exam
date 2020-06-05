"""
Author: Pruthvi Suryadevara
Email:  pruthvi.suryadevara@tifr.res.in
Generating random numbers and finding their power spectrum

"""

import numpy as np
import matplotlib.pyplot as plt
import numpy.fft as ft
import scipy.stats as st

# Generating random numbers between 0,1 and plotting them
n = 1024
y = np.random.rand(n)
plt.subplot(2, 1, 1, title="Sample data")
plt.hist(y)
plt.title("Random numbers using numpy.random.rand")
plt.xlabel("$x_i$")
plt.ylabel("number in each bin")
x_p = np.linspace(0, 1, 100)
y_p = n/10*np.ones(len(x_p))
plt.plot(x_p, y_p, 'r')

ky = ft.fftshift(ft.fft(y))                         # Taking FFT
k = 2*np.pi*ft.fftshift(ft.fftfreq(len(y), 1))      # Finding Frequency
print("max = ", np.max(k), "  min = ", np.min(k))
Pky = (np.abs(ky)**2)/(len(y))                      # Finding Periodogram
plt.subplot(2, 1, 2, title="Binned Periodogram")    # Binning Periodogram
ky_bin, k_be, binnumber = st.binned_statistic(k, Pky, bins=5)
k_bins = (k_be[0:len(k_be)-1]+k_be[1:len(k_be)])/2
plt.bar(k_bins, ky_bin, width=k_be[1]-k_be[0])
plt.xlabel("k")
plt.ylabel("P(k)")
plt.subplots_adjust(wspace=0.4, hspace=0.5)
plt.show()
