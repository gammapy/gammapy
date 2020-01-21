# Licensed under a 3-clause BSD style license - see LICENSE.rst
import pytest
import astropy.units as u
from gammapy.modeling.models import create_crab_spectral_model
from gammapy.utils.testing import assert_quantity_allclose

CRAB_SPECTRA = [
    {
        "name": "meyer",
        "dnde": u.Quantity(5.572437502365652e-12, "cm-2 s-1 TeV-1"),
        "flux": u.Quantity(2.0744425607240974e-11, "cm-2 s-1"),
        "index": 2.631535530090332,
    },
    {
        "name": "hegra",
        "dnde": u.Quantity(4.60349681e-12, "cm-2 s-1 TeV-1"),
        "flux": u.Quantity(1.74688947e-11, "cm-2 s-1"),
        "index": 2.62000000,
    },
    {
        "name": "hess_pl",
        "dnde": u.Quantity(5.57327158e-12, "cm-2 s-1 TeV-1"),
        "flux": u.Quantity(2.11653715e-11, "cm-2 s-1"),
        "index": 2.63000000,
    },
    {
        "name": "hess_ecpl",
        "dnde": u.Quantity(6.23714253e-12, "cm-2 s-1 TeV-1"),
        "flux": u.Quantity(2.267957713046026e-11, "cm-2 s-1"),
        "index": 2.529860258102417,
    },
    {
        "name": "magic_lp",
        "dnde": u.Quantity(5.5451060834144166e-12, "cm-2 s-1 TeV-1"),
        "flux": u.Quantity(2.028222279117e-11, "cm-2 s-1"),
        "index": 2.614495440236207,
    },
    {
        "name": "magic_ecpl",
        "dnde": u.Quantity(5.88494595619e-12, "cm-2 s-1 TeV-1"),
        "flux": u.Quantity(2.070767119534948e-11, "cm-2 s-1"),
        "index": 2.5433349999859405,
    },
]


@pytest.mark.parametrize("spec", CRAB_SPECTRA, ids=lambda _: _["name"])
def test_crab_spectrum(spec):
    crab_spectrum = create_crab_spectral_model(reference=spec["name"])

    dnde = crab_spectrum(2 * u.TeV)
    assert_quantity_allclose(dnde, spec["dnde"])

    flux = crab_spectrum.integral(1 * u.TeV, 1e3 * u.TeV)
    assert_quantity_allclose(flux, spec["flux"], rtol=1e-6)

    index = crab_spectrum.spectral_index(2 * u.TeV)
    assert_quantity_allclose(index, spec["index"], rtol=1e-5)


def test_invalid_format():
    with pytest.raises(ValueError):
        create_crab_spectral_model("spam")
