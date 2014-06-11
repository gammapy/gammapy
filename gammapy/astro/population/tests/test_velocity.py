# Licensed under a 3-clause BSD style license - see LICENSE.rst
from __future__ import print_function, division
import numpy as np
from numpy.testing import assert_allclose
from ....utils.distributions import normalize
from ..velocity import H05, F06B, F06P


def test_call():
    # TODO: Verify numbers against Papers or Axel's thesis.
    assert_allclose(H05(1), 4.287452755806417e-08)


def plot_distributions(self):
    import matplotlib.pyplot as plt
    v_min, v_max = 10, 3000  # km / s
    v = np.linspace(v_min, v_max, 200)
    plt.plot(v, normalize(H05, v_min, v_max)(v), color='b', linestyle='-', label='H05')
    plt.plot(v, normalize(F06B, v_min, v_max)(v), color='r', linestyle=':', label='F06B')
    plt.plot(v, normalize(F06P, v_min, v_max)(v), color='k', linestyle='-', label='F06P')

    plt.xlim(v_min, v_max)
    plt.ylim(0, 0.004)
    plt.xlabel('Velocity [km/s]')
    plt.ylabel('PDF')
    plt.title('Comparison Velocity Distribution Models (PDF)')
    plt.semilogx()
    plt.legend(prop={'size': 10})
    # plt.show()
