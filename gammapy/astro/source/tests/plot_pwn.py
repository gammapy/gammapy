# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""""Make a few example plots to see if PWN model works."""
from __future__ import print_function, division
import numpy as np
from ...population import PWN


def plot_hoppe(pwn):
    import matplotlib.pyplot as plt
    from gammapy.spectrum.models import PowerLaw
    from gammapy.spectrum.inverse_compton import InverseCompton
    from gammapy.spectrum.synchrotron import Synchrotron
    plt.subplot(311)
    injection_spectrum = PowerLaw()
    pwn = PWN(T=1e3, B=10, E_c=1000, injection_type='constant')
    e_fig = plt.figure(title='Electron Spectrum')
    gamma_fig = plt.figure(title='Gamma Spectrum')
    for age, color in zip([1e3, 1e4, 1e5],
                          ['blue', 'red', 'black']):
        pwn.evolve(age)
        sy_spec = Synchrotron(pwn.e_spec, pwn.B)
        ic_spec = InverseCompton()(pwn.e_spec)
        tot_spec = sy_spec + ic_spec

        pwn.e_spec.plot(e_fig, color=color)
        sy_spec.plot(gamma_fig, color=color)
        ic_spec.plot(gamma_fig, color=color)
        tot_spec.plot(gamma_fig, color=color)
    plt.show()


def plot_mattana(pwn):
    pwn = PWN(injection_type='pulsar')
    age = np.logspace(0, 5, 1e3)
    n_x = np.empty_like(age)
    n_g = np.empty_like(age)
    for i in range(age.size):
        pwn.evolve(age[i])
        n_x[i] = pwn.espec.integrate(1e42, 1e43)


def plot_electron_evolution():
    import matplotlib.pyplot as plt
    pwn = PWN()
    t, r = pwn.radius(T_min=1e2, T_max=1e3, dt=1)
    t, B = pwn.magnetic_field()
    # Run simulation
    e = np.logspace(0, 5, 1e3)
    n = np.zeros_like(e)
    # n = pwn.evolve(e, n,
    #               p=energy_losses(B=10),
    #               q=injection_spectrum(norm=10),
    #               age=age, dt=dt)
    # Plot results
    # plt.plot(e, e ** 2 * n, label='age = {0}'.format(age))
    plt.loglog()
    plt.xlim(1e1, 1e5)
    plt.ylim(1e-5, 1e5)
    plt.legend()
    plt.show()


def main():
    plot_hoppe()
    plot_mattana()
    # plot_electron_evolution()
