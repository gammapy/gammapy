# Licensed under a 3-clause BSD style license - see LICENSE.rst
from __future__ import print_function, division
import pytest
import numpy as np
from astropy.units import Unit
from .. import crab

try:
    import scipy
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False


@pytest.mark.skipif('not HAS_SCIPY')
def test_eval():
    # Check diff_flux and int_flux functions
    # against dict values containing published numbers.
    e, e1, e2 = 1, 1, 1e3
    for ref, values in crab.refs.items():
        f = crab.diff_flux(e, ref)
        I = crab.int_flux(e1, e2, ref)
        g = crab.spectral_index(e, ref)
        f_ref = values['diff_flux']
        if ref == 'hess_ecpl':
            f_ref *= np.exp(-e / values['cutoff'])

        I_ref = values['int_flux']
        g_ref = values['index']
        f_err = (f - f_ref) / f_ref
        I_err = (I - I_ref) / I_ref
        g_err = g - g_ref
        # TODO: add asserts
        # print(('%15s ' + '%13.5g' * 6) %
        #      (ref, f, I, g, f_err, I_err, g_err))


def plot_spectra(what="flux"):
    import matplotlib.pyplot as plt
    plt.clf()
    e = np.logspace(-2, 3, 100)
    for ref in crab.refs.keys():
        if what == 'flux':
            y = Unit('TeV').to('erg') * e ** 2 * crab.diff_flux(e, ref)
        elif what == 'int_flux':
            # @todo there are integration problems!
            e2 = 1e4 * np.ones_like(e)
            y = crab.int_flux(e, e2, ref=ref)
        if what == 'ratio':
            y = (crab.diff_flux(e, ref) /
                 crab.diff_flux(e, 'meyer'))
        elif what == 'index':
            y = crab.spectral_index(e, ref)
        plt.plot(e, y, label=ref)

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

if __name__ == '__main__':
    test_eval()
    plot_spectra('flux')
    plot_spectra('int_flux')
    plot_spectra('ratio')
    plot_spectra('index')
