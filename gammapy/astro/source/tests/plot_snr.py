# Licensed under a 3-clause BSD style license - see LICENSE.rst
from __future__ import print_function, division
import numpy as np
from ..snr import SNR


def make_plots():
    import matplotlib.pyplot as plt
    snr = SNR()
    t = np.logspace(0, 5, 1000)
    r = snr.r_out(t)
    L = snr.L(t)

    fig1 = plt.figure(1)
    radPlt = fig1.add_subplot(1, 1, 1)
    radPlt.plot(t, r)
    radPlt.set_title('Radius of SNR')
    radPlt.set_xlabel('time [years]')
    radPlt.set_ylabel('radius [pc]')
    radPlt.loglog()

    fig2 = plt.figure(2)
    lumPlt = fig2.add_subplot(1, 1, 1)
    lumPlt.plot(t, L)
    lumPlt.set_title('Luminosity of SNR')
    lumPlt.set_xlabel('time [years]')
    lumPlt.set_ylabel('luminosity [ph*s^-1]')
    lumPlt.loglog()

    plt.show()

if __name__ == '__main__':
    make_plots()
