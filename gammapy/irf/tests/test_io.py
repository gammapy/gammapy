# Licensed under a 3-clause BSD style license - see LICENSE.rst
from __future__ import absolute_import, division, print_function, unicode_literals
from astropy.units import Quantity
from ...utils.testing import requires_data, requires_dependency, mpl_plot_check
from ...utils.testing import assert_quantity_allclose
from ..io import CTAIrf, CTAPerf


@requires_dependency("scipy")
@requires_data("gammapy-extra")
def test_cta_irf():
    """Test that CTA IRFs can be loaded and evaluated."""

    filename = "$GAMMAPY_EXTRA/datasets/cta/perf_prod2/South_5h/irf_file.fits.gz"
    irf = CTAIrf.read(filename)

    energy = Quantity(1, "TeV")
    offset = Quantity(3, "deg")
    rad = Quantity(0.1, "deg")
    migra = 1

    val = irf.aeff.data.evaluate(energy=energy, offset=offset)
    assert_quantity_allclose(val, Quantity(235929.697018741, "m^2"))

    val = irf.edisp.data.evaluate(offset=offset, e_true=energy, migra=migra)
    assert_quantity_allclose(val, 0.13111834249544874)

    # TODO: clean up these PSF classes, e.g. to return quantities.
    # Also: add an `evaluate(energy, offset, theta)` to the loaded PSF class.
    # psf = irf.psf.to_energy_dependent_table_psf(offset=offset)
    psf = irf.psf.psf_at_energy_and_theta(energy=energy, theta=offset)
    val = psf(rad)
    assert_quantity_allclose(val, 5.423486126591272)

    # TODO: Background cube class doesn't have a working evaluate yet
    # val = irf.bkg.evaluate(energy=energy, x=offset, y=Quantity(0, 'deg'))
    # assert_quantity_allclose(val, Quantity(247996.974414962, 'm^2'))


@requires_dependency("matplotlib")
@requires_data("gammapy-extra")
def test_point_like_perf():
    filename = "$GAMMAPY_EXTRA/datasets/cta/perf_prod2/point_like_non_smoothed/South_5h.fits.gz"
    cta_perf = CTAPerf.read(filename)
    with mpl_plot_check():
        cta_perf.peek()
