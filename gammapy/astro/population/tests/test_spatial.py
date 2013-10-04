# Licensed under a 3-clause BSD style license - see LICENSE.rst
from __future__ import print_function, division
import numpy as np
from numpy.testing import assert_approx_equal
from ....utils.distributions import normalize, density
from ..spatial import distributions

def test_call():
    # TODO: Verify numbers against Papers or Axel's thesis.
    assert_approx_equal(distributions['P90'](1), 0.03954258779836089)

def plot_spatial():
    import matplotlib.pyplot as plt
    max_radius = 20  # kpc
    r = np.linspace(0, max_radius, 100)
    plt.plot(r, normalize(density(distributions['P90']), 0, max_radius)(r), color='b',
            linestyle='-', label='Paczynski 1990')
    plt.plot(r, normalize(density(distributions['CB98']), 0, max_radius)(r), color='r',
            linestyle='--', label='Case&Battacharya 1998')
    plt.plot(r, normalize(density(distributions['YK04']), 0, max_radius)(r), color='g',
            linestyle='-.', label='Yusifov&Kucuk 2004')
    plt.plot(r, normalize(density(distributions['F06']), 0, max_radius)(r), color='m',
            linestyle='-', label='Faucher&Kaspi 2006')
    plt.plot(r, normalize(density(distributions['L06']), 0, max_radius)(r), color='k',
            linestyle=':', label='Lorimer 2006')
    plt.xlim(0, max_radius)
    plt.ylim(0, 0.28)
    plt.xlabel('Galactocentric Distance [kpc]')
    plt.ylabel('Surface Density')
    plt.title('Comparison Radial Distribution Models (Surface Density)')
    plt.legend(prop={'size': 10})
    # plt.show()
