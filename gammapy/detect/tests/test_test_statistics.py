# Licensed under a 3-clause BSD style license - see LICENSE.rst
from __future__ import absolute_import, division, print_function, unicode_literals
import numpy as np
from numpy.testing.utils import assert_allclose
from astropy.convolution import Gaussian2DKernel
from ...utils.testing import requires_dependency, requires_data
from ...detect import compute_ts_image
from ...image import SkyImageList


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

    result = compute_ts_image(
        images['counts'], images['background'], images['exposure'], kernel,
        method='leastsq iter',
    )
    for name, order in zip(['ts', 'amplitude', 'niter'], [2, 5, 0]):
        result[name].data = np.nan_to_num(result[name].data)
        result[name] = result[name].upsample(2, order=order)

    assert_allclose(1705.840212274973, result['ts'].data[99, 99], rtol=1e-3)
    assert_allclose([[99], [99]], np.where(result['ts'].data == result['ts'].data.max()))
    assert_allclose(3, result['niter'].data[99, 99])
    assert_allclose(1.0227934338735763e-09, result['amplitude'].data[99, 99], rtol=1e-3)
