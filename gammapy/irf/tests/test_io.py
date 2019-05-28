# Licensed under a 3-clause BSD style license - see LICENSE.rst
from numpy.testing import assert_allclose
from astropy.units import Quantity
from ...utils.testing import requires_data
from ..io import load_cta_irfs


@requires_data()
def test_cta_irf():
    """Test that CTA IRFs can be loaded and evaluated."""
    irf = load_cta_irfs(
        "$GAMMAPY_DATA/cta-1dc/caldb/data/cta/1dc/bcf/South_z20_50h/irf_file.fits"
    )

    energy = Quantity(1, "TeV")
    offset = Quantity(3, "deg")
    rad = Quantity(0.1, "deg")
    migra = 1

    val = irf["aeff"].data.evaluate(energy=energy, offset=offset)
    assert_allclose(val.value, 545269.4675532734)
    assert val.unit == "m2"

    val = irf["edisp"].data.evaluate(offset=offset, e_true=energy, migra=migra)
    assert_allclose(val.value, 3183.688224335546)
    assert val.unit == ""

    psf = irf["psf"].psf_at_energy_and_theta(energy=energy, theta=offset)
    val = psf(rad)
    assert_allclose(val, 4.868832183085153)

    val = irf["bkg"].data.evaluate(
        energy=energy, fov_lon=offset, fov_lat=Quantity(0, "deg")
    )
    assert_allclose(val.value, 9.400071082017065e-05)
    assert val.unit == "1 / (MeV s sr)"
