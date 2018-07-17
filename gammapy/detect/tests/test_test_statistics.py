# Licensed under a 3-clause BSD style license - see LICENSE.rst
from __future__ import absolute_import, division, print_function, unicode_literals
import pytest
import numpy as np
from numpy.testing.utils import assert_allclose
from astropy.convolution import Gaussian2DKernel
from ...utils.testing import requires_dependency, requires_data
from ...utils.scripts import make_path
from ...image import SkyImageList
from ...detect import TSImageEstimator, compute_ts_image_multiscale


@requires_dependency('scipy')
@requires_dependency('skimage')
@requires_data('gammapy-extra')
def test_compute_ts_map():
    """Minimal test of compute_ts_image"""
    filename = '$GAMMAPY_EXTRA/test_datasets/unbundled/poisson_stats_image/input_all.fits.gz'
    images = SkyImageList.read(filename)

    kernel = Gaussian2DKernel(2.5)

    images['counts'] = images['counts'].downsample(2, np.nansum)
    images['background'] = images['background'].downsample(2, np.nansum)
    images['exposure'] = images['exposure'].downsample(2, np.mean)

    ts_estimator = TSImageEstimator(method='leastsq iter')
    result = ts_estimator.run(images, kernel=kernel)

    for name, order in zip(['ts', 'flux', 'flux_err', 'flux_ul', 'niter'], [2, 5, 5, 5, 0]):
        result[name].data = np.nan_to_num(result[name].data)
        result[name] = result[name].upsample(2, order=order)

    assert_allclose(1705.840212274973, result['ts'].data[99, 99], rtol=1e-3)
    assert_allclose([[99], [99]], np.where(result['ts'].data == result['ts'].data.max()))
    assert_allclose(3, result['niter'].data[99, 99])
    assert_allclose(1.0227934338735763e-09, result['flux'].data[99, 99], rtol=1e-3)
    assert_allclose(3.842162268386843e-11, result['flux_err'].data[99, 99], rtol=1e-3)
    assert_allclose(1.0996355030292762e-09, result['flux_ul'].data[99, 99], rtol=1e-3)


@requires_dependency('scipy')
@requires_dependency('skimage')
@requires_data('gammapy-extra')
@pytest.mark.parametrize('scale', ['0.000', '0.050', '0.100', '0.200'])
def test_compute_ts_image_multiscale(tmpdir, scale):
    path = make_path('$GAMMAPY_EXTRA/test_datasets/unbundled/poisson_stats_image')
    images_in = SkyImageList.read(path / 'input_all.fits.gz')

    psf_parameters = {
        "psf1": {"fwhm": 7.0644601350928475, "ampl": 1},
        "psf2": {"fwhm": 1e-05, "ampl": 0},
        "psf3": {"fwhm": 1e-05, "ampl": 0},
    }

    images_out = compute_ts_image_multiscale(
        images_in, psf_parameters, [float(scale)],
    )
    # Function returns a Python list of `SkyImageList`. TODO: simplify
    images_out = images_out[0]

    images_ref = SkyImageList.read(path / 'expected_ts_{}.fits.gz'.format(scale))

    opts = dict(rtol=1e-2, atol=1e-5, equal_nan=True)
    assert_allclose(images_out['ts'].data, images_ref['ts'].data, **opts)
    assert_allclose(images_out['sqrt_ts'].data, images_ref['sqrt_ts'].data, **opts)
    assert_allclose(images_out['flux'].data, images_ref['amplitude'].data, **opts)
    assert 'niter' in images_out.names
