# Licensed under a 3-clause BSD style license - see LICENSE.rst
from __future__ import absolute_import, division, print_function, unicode_literals
import pytest
import astropy.units as u
from ...utils.testing import assert_quantity_allclose
from ...spectrum import CrabSpectrum

CRAB_SPECTRA = [
    dict(
        name="meyer",
        dnde=u.Quantity(5.572437502365652e-12, "cm-2 s-1 TeV-1"),
        flux=u.Quantity(2.0744425607240974e-11, "cm-2 s-1"),
        index=2.631535530090332,
    ),
    dict(
        name="hegra",
        dnde=u.Quantity(4.60349681e-12, "cm-2 s-1 TeV-1"),
        flux=u.Quantity(1.74688947e-11, "cm-2 s-1"),
        index=2.62000000,
    ),
    dict(
        name="hess_pl",
        dnde=u.Quantity(5.57327158e-12, "cm-2 s-1 TeV-1"),
        flux=u.Quantity(2.11653715e-11, "cm-2 s-1"),
        index=2.63000000,
    ),
    dict(
        name="hess_ecpl",
        dnde=u.Quantity(6.23714253e-12, "cm-2 s-1 TeV-1"),
        flux=u.Quantity(2.267957713046026e-11, "cm-2 s-1"),
        index=2.529860258102417,
    ),
    dict(
        name="magic_lp",
        dnde=u.Quantity(5.5451060834144166e-12, "cm-2 s-1 TeV-1"),
        flux=u.Quantity(2.028222279117e-11, "cm-2 s-1"),
        index=2.614495440236207,
    ),
    dict(
        name="magic_ecpl",
        dnde=u.Quantity(5.88494595619e-12, "cm-2 s-1 TeV-1"),
        flux=u.Quantity(2.070767119534948e-11, "cm-2 s-1"),
        index=2.5433349999859405,
    ),
]


@pytest.mark.parametrize("spec", CRAB_SPECTRA, ids=lambda _: _["name"])
def test_crab_spectrum(spec):
    energy = 2 * u.TeV
    emin, emax = [1, 1e3] * u.TeV

    crab_spectrum = CrabSpectrum(spec["name"])

    dnde = crab_spectrum.model(energy)
    assert_quantity_allclose(dnde, spec["dnde"])

    flux = crab_spectrum.model.integral(emin, emax)
    assert_quantity_allclose(flux, spec["flux"])

    index = crab_spectrum.model.spectral_index(energy)
    assert_quantity_allclose(index, spec["index"], rtol=1e-5)
