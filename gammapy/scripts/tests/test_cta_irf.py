# Licensed under a 3-clause BSD style license - see LICENSE.rst
from __future__ import absolute_import, division, print_function, unicode_literals
import pytest
from numpy.testing import assert_allclose, assert_equal
import astropy.units as u
import numpy as np
from ...utils.testing import requires_data, requires_dependency
from ...scripts import CTAIrf, CTAPerf


# TODO: fix this test - currently fails like this:
# https://travis-ci.org/gammapy/gammapy/jobs/275886064#L2499
@pytest.mark.xfail
@requires_dependency('scipy')
@requires_data('gammapy-extra')
def test_cta_irf():
    """Test that CTA IRFs can be loaded and evaluated."""

    filename = '$GAMMAPY_EXTRA/datasets/cta/perf_prod2/South_5h/irf_file.fits.gz'
    irf = CTAIrf.read(filename)

    energy = 1 * u.TeV
    offset = 3 * u.deg
    rad = 0.1 * u.deg
    migra = 1

    val = irf.aeff.data.evaluate(energy=energy, offset=offset)
    assert_quantity_allclose(val, Quantity(235929.697018741, 'm^2'))

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


@requires_dependency('matplotlib')
@requires_data('gammapy-extra')
def test_point_like_perf():
    filename = '$GAMMAPY_EXTRA/datasets/cta/perf_prod2/point_like_non_smoothed/South_5h.fits.gz'
    perf = CTAPerf.read(filename)
    perf.peek()


@requires_data('gammapy-extra')
def test_point_like_perf_bkg_reproj():
    filename = '$GAMMAPY_EXTRA/datasets/cta/perf_prod2/point_like_non_smoothed/South_5h.fits.gz'
    e_reco = np.logspace(np.log10(0.02), np.log10(100), 50) * u.TeV
    perf = CTAPerf.read(filename, e_reco=e_reco)
    assert_equal(len(perf.bkg.data.data), 49)
