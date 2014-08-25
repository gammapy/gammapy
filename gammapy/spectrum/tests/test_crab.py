# Licensed under a 3-clause BSD style license - see LICENSE.rst
from __future__ import print_function, division
import numpy as np
from numpy.testing import assert_allclose
from astropy.tests.helper import pytest
from astropy.units import Unit
from ...spectrum import crab

try:
    import scipy
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False


@pytest.mark.skipif('not HAS_SCIPY')
def test_eval():
    e, e1, e2 = 2, 1, 1e3

    # This is just a test that the functions run
    # These results are not verified
    vals = dict()
    vals['meyer'] = [5.57243750e-12, 2.07445434e-11, 2.63159544e+00]
    vals['hegra'] = [4.60349681e-12, 1.74688947e-11, 2.62000000e+00]
    vals['hess_pl'] = [5.57327158e-12, 2.11653715e-11, 2.63000000e+00]
    vals['hess_ecpl'] = [6.23714253e-12, 2.26797344e-11, 2.52993006e+00]

    for reference in crab.CRAB_REFERENCES:
        f = crab.crab_flux(e, reference)
        I = crab.crab_integral_flux(e1, e2, reference)
        g = crab.crab_spectral_index(e, reference)
        assert_allclose([f, I, g], vals[reference])


def plot_spectra(what="flux"):
    import matplotlib.pyplot as plt
    plt.clf()
    e = np.logspace(-2, 3, 100)
    for reference in crab.CRAB_REFERENCES:
        if what == 'flux':
            y = Unit('TeV').to('erg') * e ** 2 * crab.crab_flux(e, reference)
        elif what == 'int_flux':
            # @todo there are integration problems!
            e2 = 1e4 * np.ones_like(e)
            y = crab.crab_integral_flux(e, e2, reference=reference)
        if what == 'ratio':
            y = (crab.crab_flux(e, reference) /
                 crab.crab_flux(e, 'meyer'))
        elif what == 'index':
            y = crab.crab_spectral_index(e, reference)
        plt.plot(e, y, label=reference)

    plt.xlabel('Energy (TeV)')
    if what == 'int_flux':
        plt.ylabel('Integral Flux (cm^-2 s^-1)')
        plt.ylim(1e-15, 1e-8)
        plt.loglog()
        filename = 'crab_int_flux.pdf'
    elif what == 'flux':
        plt.ylabel('Flux (erg cm^-2 s^-1)')
        plt.ylim(1e-12, 1e-9)
        plt.loglog()
        filename = 'crab_flux.pdf'
    elif what == 'ratio':
        plt.ylabel('Flux Ratio wrt. Meyer')
        plt.ylim(1e-1, 1e1)
        plt.loglog()
        filename = 'crab_ratio.pdf'
    elif what == 'index':
        plt.ylabel('Flux (erg cm^-2 s^-1)')
        plt.ylim(1, 5)
        plt.semilogx()
        filename = 'crab_index.pdf'

    plt.grid(which='both')
    plt.legend(loc='best')
    plt.savefig(filename)
