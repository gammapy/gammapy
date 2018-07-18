# Licensed under a 3-clause BSD style license - see LICENSE.rst
from __future__ import absolute_import, division, print_function, unicode_literals
import pytest
import numpy as np
from numpy.testing.utils import assert_allclose
from astropy.convolution import Gaussian2DKernel
from ...utils.testing import requires_dependency, requires_data
from ...utils.scripts import make_path
from ...maps.utils import read_fits_hdus
from ...detect import TSMapEstimator, compute_ts_image_multiscale


@requires_dependency('scipy')
@requires_dependency('skimage')
@requires_data('gammapy-extra')
def test_compute_ts_map():
    """Minimal test of compute_ts_image"""
    filename = '$GAMMAPY_EXTRA/test_datasets/unbundled/poisson_stats_image/input_all.fits.gz'
    maps = read_fits_hdus(filename)

    kernel = Gaussian2DKernel(5)

    ts_estimator = TSMapEstimator(method='leastsq iter', n_jobs=4)
    result = ts_estimator.run(maps, kernel=kernel)

    assert_allclose(1714.2325, result['ts'].data[99, 99], rtol=1e-3)
    assert_allclose([[99], [99]], np.where(result['ts'].data == np.nanmax(result['ts'].data)))
    assert_allclose(3, result['niter'].data[99, 99])
    assert_allclose(1.02596e-09, result['flux'].data[99, 99], rtol=1e-3)
    assert_allclose(3.846415e-11, result['flux_err'].data[99, 99], rtol=1e-3)
    assert_allclose(1.102886e-09, result['flux_ul'].data[99, 99], rtol=1e-3)


@requires_dependency('scipy')
@requires_dependency('skimage')
@requires_data('gammapy-extra')
@pytest.mark.parametrize('scale', ['0.000', '0.050', '0.100', '0.200'])
def test_compute_ts_image_multiscale(tmpdir, scale):
    path = make_path('$GAMMAPY_EXTRA/test_datasets/unbundled/poisson_stats_image')
    filename = path / 'input_all.fits.gz'
    maps_in = read_fits_hdus(filename)

    psf_parameters = {
        "psf1": {"fwhm": 7.0644601350928475, "ampl": 1},
        "psf2": {"fwhm": 1e-05, "ampl": 0},
        "psf3": {"fwhm": 1e-05, "ampl": 0},
    }

    maps_out = compute_ts_image_multiscale(
        maps_in, psf_parameters, [float(scale)],
    )
    # Function returns a Python list of `SkyImageList`. TODO: simplify
    maps_out = maps_out[0]

    maps_ref = read_fits_hdus(path / 'expected_ts_{}.fits.gz'.format(scale))

    opts = dict(rtol=1e-2, atol=1e-5, equal_nan=True)
    assert_allclose(maps_out['ts'].data, maps_ref['ts'].data, **opts)
    assert_allclose(maps_out['sqrt_ts'].data, maps_ref['sqrt_ts'].data, **opts)
    assert_allclose(maps_out['flux'].data, maps_ref['amplitude'].data, **opts)
    assert 'niter' in maps_out
