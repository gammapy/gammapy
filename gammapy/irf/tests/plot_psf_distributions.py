# Licensed under a 3-clause BSD style license - see LICENSE.rst
from __future__ import (absolute_import, division, print_function,
                        unicode_literals)
import numpy as np
from ...morphology import Gauss2DPDF, MultiGauss2D
from ...irf import HESSMultiGaussPSF


def make_theta(theta_max=5, n_bins=100):
    theta = np.linspace(0, theta_max, n_bins)
    theta2 = np.linspace(0, theta_max ** 2, n_bins)
    return theta, theta2


def plot_theta_distributions():
    import matplotlib.pyplot as plt
    theta, theta2 = make_theta()
    g = Gauss2DPDF(sigma=1)
    m = MultiGauss2D(sigmas=[1, 3], norms=[0.5, 0.5])
    plt.figure(figsize=(15, 8))
    for ylog in [0, 1]:
        plt.subplot(2, 3, 3 * ylog)
        plt.plot(theta2, g.dpdtheta2(theta2), label='sigma=1')
        plt.plot(theta2, m.dpdtheta2(theta2), label='sigma=1,3')
        plt.xlabel('theta2')
        plt.ylabel('dp / dtheta2')
        if ylog:
            plt.semilogy()
        plt.subplot(2, 3, 3 * ylog + 1)
        plt.plot(theta, g(theta, 0), label='sigma=1')
        plt.plot(theta, m(theta, 0), label='sigma=1,3')
        plt.xlabel('theta')
        plt.ylabel('dp / dtheta')
        if ylog:
            plt.semilogy()
        plt.subplot(2, 3, 3 * ylog + 2)
        plt.plot(theta2, theta2 * g.dpdtheta2(theta2), label='sigma=1')
        plt.plot(theta2, theta2 * m.dpdtheta2(theta2), label='sigma=1,3')
        plt.xlabel('theta2')
        plt.ylabel('theta2 * dp / dtheta2')
        if ylog:
            plt.semilogy()
    plt.legend()
    plt.savefig('output/plot_theta_distributions.pdf')


def plot_convolution():
    import matplotlib.pyplot as plt
    g = Gauss2DPDF(sigma=1)
    sigma = np.linspace(0, 5, 100)
    r80 = g.convolve(sigma).theta(0.68)
    plt.figure()
    plt.plot(sigma, r80)
    plt.xlabel('sigma')
    plt.ylabel('r80')
    plt.savefig('output/plot_convolution.pdf')


def plot_HESS_PSF_convolution(containment=0.8):
    import matplotlib.pyplot as plt
    m = HESSMultiGaussPSF('input/gc_psf.txt').to_MultiGauss2D(normalize=True)
    m_approx = Gauss2DPDF(m.match_sigma(containment))
    print('HESS PSF approx sigma = {0}'.format(m_approx.sigma))
    # 0.047 for containment 0.8
    # 0.055 for containment 0.9
    sigmas = np.linspace(0, 0.3, 100)
    # First compute the correct theta
    thetas = [m.convolve(sigma).theta(containment)
            for sigma in sigmas]
    # Now an approximate one using match_sigma
    thetas_approx = [m_approx.convolve(sigma).theta(containment)
                   for sigma in sigmas]
    plt.figure()
    plt.plot(sigmas, thetas, label='correct')
    plt.plot(sigmas, thetas_approx, label='approx')
    plt.xlabel('sigma')
    plt.ylabel('r%s' % (100 * containment))
    plt.savefig('output/plot_HESS_PSF_convolution.pdf')


if __name__ == '__main__':
    # plot_theta_distributions()
    # plot_convolution()
    plot_HESS_PSF_convolution()
