# Licensed under a 3-clause BSD style license - see LICENSE.rst
from __future__ import print_function, division
from numpy.testing import assert_allclose
from ..effective_area import effective_area


def test_effective_area():
    energy = 0.1
    area = effective_area(energy, 'HESS')
    assert_allclose(area, 16546957.901469307)
    

def plot_effective_area():
    import numpy as np
    import matplotlib.pyplot as plt
    # Plot the effective area curves of the three experiments
    elim = [10 ** -3, 10 ** 3]
    # Build a vector of energies (TeV) with equal log spacing
    loge = np.linspace(0, np.log10(elim[1]), 100)
    energy = 10 ** loge
    for instrument in ['HESS', 'HESS2', 'CTA']:
        a_eff_hess = effective_area(energy, instrument)
        plt.plot(energy, a_eff_hess, label=instrument)
    plt.loglog()
    plt.xlabel('Energy (TeV)')
    plt.ylabel('Effective Area (cm^2)')
    plt.xlim(elim)
    plt.ylim([1e3, 1e12])
    plt.legend()
    plt.show()
