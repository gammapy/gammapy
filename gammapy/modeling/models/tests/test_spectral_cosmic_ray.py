# Licensed under a 3-clause BSD style license - see LICENSE.rst
import pytest
from numpy.testing import assert_allclose
import astropy.units as u
from gammapy.modeling.models import create_cosmic_ray_spectral_model

COSMIC_RAY_SPECTRA = [
    {"name": "proton", "dnde": 1.856522e-01, "flux": 7.096247e-01, "index": 2.70},
    {"name": "N", "dnde": 1.449504e-01, "flux": 5.509215e-01, "index": 2.64},
    {"name": "Si", "dnde": 5.646618e-02, "flux": 2.149887e-01, "index": 2.66},
    {"name": "Fe", "dnde": 2.720231e-02, "flux": 1.03305e-01, "index": 2.63},
    {"name": "electron", "dnde": 1.013671e-04, "flux": 4.691975e-04, "index": 3.428318},
]


@pytest.mark.parametrize("spec", COSMIC_RAY_SPECTRA, ids=lambda _: _["name"])
def test_cosmic_ray_spectrum(spec):
    cr_spectrum = create_cosmic_ray_spectral_model(particle=spec["name"])

    dnde = cr_spectrum(2 * u.TeV)
    assert_allclose(dnde.value, spec["dnde"], rtol=1e-3)
    assert dnde.unit == "m-2 s-1 TeV-1"

    flux = cr_spectrum.integral(1 * u.TeV, 1e3 * u.TeV)
    assert_allclose(flux.value, spec["flux"], rtol=1e-3)
    assert flux.unit == "m-2 s-1"

    index = cr_spectrum.spectral_index(2 * u.TeV)
    assert_allclose(index.value, spec["index"], rtol=1e-3)
    assert index.unit == ""
