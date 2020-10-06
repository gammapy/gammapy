# Licensed under a 3-clause BSD style license - see LICENSE.rst
from numpy.testing import assert_allclose
from astropy.units import Quantity
from gammapy.irf import load_cta_irfs
from gammapy.utils.testing import requires_data


@requires_data()
def test_cta_irf():
    """Test that CTA IRFs can be loaded and evaluated."""
    irf = load_cta_irfs(
        "$GAMMAPY_DATA/cta-1dc/caldb/data/cta/1dc/bcf/South_z20_50h/irf_file.fits"
    )

    energy = Quantity(1, "TeV")
    offset = Quantity(3, "deg")

    val = irf["aeff"].data.evaluate(energy_true=energy, offset=offset)
    assert_allclose(val.value, 545269.4675, rtol=1e-5)
    assert val.unit == "m2"

    val = irf["edisp"].data.evaluate(offset=offset, energy_true=energy, migra=1)
    assert_allclose(val.value, 3183.6882, rtol=1e-5)
    assert val.unit == ""

    psf = irf["psf"].psf_at_energy_and_theta(energy=energy, theta=offset)
    val = psf(Quantity(0.1, "deg"))
    assert_allclose(val, 3.56989, rtol=1e-5)

    val = irf["bkg"].data.evaluate(energy=energy, fov_lon=offset, fov_lat="0 deg")
    assert_allclose(val.value, 9.400071e-05, rtol=1e-5)
    assert val.unit == "1 / (MeV s sr)"
