# Licensed under a 3-clause BSD style license - see LICENSE.rst
from __future__ import absolute_import, division, print_function, unicode_literals
import numpy as np
import astropy.units as u
from astropy.tests.helper import assert_quantity_allclose, pytest
from ...utils.testing import requires_dependency
from ...spectrum import CrabSpectrum


desired = dict()
desired['meyer'] = [u.Quantity(5.572437502365652e-12, 'cm-2 s-1 TeV-1'),
                    u.Quantity(2.0744425607240974e-11, 'cm-2 s-1'),
                    2.631535530090332]
desired['hegra'] = [u.Quantity(4.60349681e-12, 'cm-2 s-1 TeV-1'),
                    u.Quantity(1.74688947e-11, 'cm-2 s-1'),
                    2.62000000]
desired['hess_pl'] = [u.Quantity(5.57327158e-12, 'cm-2 s-1 TeV-1'),
                      u.Quantity(2.11653715e-11, 'cm-2 s-1'),
                      2.63000000]
desired['hess_ecpl'] = [u.Quantity(6.23714253e-12, 'cm-2 s-1 TeV-1'),
                        u.Quantity(2.267957713046026e-11, 'cm-2 s-1'),
                        2.529860258102417]


class TestCrabSpectrum(object):

    @pytest.mark.parametrize('reference', ['meyer', 'hegra', 'hess_pl', 'hess_ecpl'])
    def test_evaluate(self, reference):
        e = 2 * u.TeV
        emin, emax = [1, 1E3] * u.TeV

        crab_spectrum = CrabSpectrum(reference)
        f = crab_spectrum.model(e)
        I = crab_spectrum.model.integral(emin, emax)
        g = crab_spectrum.model.spectral_index(e)
        assert_quantity_allclose(desired[reference][0], f)
        assert_quantity_allclose(desired[reference][1], I)
        assert_quantity_allclose(desired[reference][2], g)



# TODO: move this to the docs (this is not a test)
def plot_spectra(what="flux"):
    import matplotlib.pyplot as plt
    plt.clf()
    e = np.logspace(-2, 3, 100)
    for reference in crab.CRAB_REFERENCES:
        if what == 'flux':
            y = Unit('TeV').to('erg') * e ** 2 * crab.crab_flux(e, reference)
        elif what == 'int_flux':
            # TODO: there are integration problems!
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
