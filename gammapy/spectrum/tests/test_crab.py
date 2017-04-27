# Licensed under a 3-clause BSD style license - see LICENSE.rst
from __future__ import absolute_import, division, print_function, unicode_literals
from collections import OrderedDict
import astropy.units as u
from astropy.tests.helper import assert_quantity_allclose, pytest
from ...spectrum import CrabSpectrum

desired = OrderedDict()
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


class TestCrabSpectrum:
    @pytest.mark.parametrize('reference', ['meyer', 'hegra', 'hess_pl', 'hess_ecpl'])
    def test_evaluate(self, reference):
        energy = 2 * u.TeV
        emin, emax = [1, 1e3] * u.TeV

        crab_spectrum = CrabSpectrum(reference)
        f = crab_spectrum.model(energy)
        I = crab_spectrum.model.integral(emin, emax)
        g = crab_spectrum.model.spectral_index(energy)
        assert_quantity_allclose(desired[reference][0], f)
        assert_quantity_allclose(desired[reference][1], I)
        assert_quantity_allclose(desired[reference][2], g, rtol=1e-5)
