# Licensed under a 3-clause BSD style license - see LICENSE.rst
from __future__ import absolute_import, division, print_function, unicode_literals
import pytest
from numpy.testing.utils import assert_allclose
from astropy.convolution import Gaussian2DKernel
from ...utils.testing import requires_data
from ...maps import Map
from ...detect import TSMapEstimator

pytest.importorskip("scipy")


@pytest.fixture(scope="session")
def input_maps():
    filename = (
        "$GAMMAPY_EXTRA/test_datasets/unbundled/poisson_stats_image/input_all.fits.gz"
    )
    return {
        "counts": Map.read(filename, hdu="counts"),
        "exposure": Map.read(filename, hdu="exposure"),
        "background": Map.read(filename, hdu="background"),
    }


@requires_data("gammapy-extra")
def test_compute_ts_map(input_maps):
    """Minimal test of compute_ts_image"""
    kernel = Gaussian2DKernel(5)

    ts_estimator = TSMapEstimator(method="leastsq iter", n_jobs=4, threshold=1)
    result = ts_estimator.run(input_maps, kernel=kernel)

    assert "leastsq iter" in repr(ts_estimator)
    assert_allclose(result["ts"].data[99, 99], 1714.23, rtol=1e-2)
    assert_allclose(result["niter"].data[99, 99], 3)
    assert_allclose(result["flux"].data[99, 99], 1.02e-09, rtol=1e-2)
    assert_allclose(result["flux_err"].data[99, 99], 3.84e-11, rtol=1e-2)
    assert_allclose(result["flux_ul"].data[99, 99], 1.10e-09, rtol=1e-2)


@requires_data("gammapy-extra")
def test_compute_ts_map_downsampled(input_maps):
    """Minimal test of compute_ts_image"""
    kernel = Gaussian2DKernel(2.5)

    ts_estimator = TSMapEstimator(
        method="root brentq", n_jobs=4, error_method="conf", ul_method="conf"
    )
    result = ts_estimator.run(input_maps, kernel=kernel, downsampling_factor=2)

    assert_allclose(result["ts"].data[99, 99], 1675.28, rtol=1e-2)
    assert_allclose(result["niter"].data[99, 99], 7)
    assert_allclose(result["flux"].data[99, 99], 1.02e-09, rtol=1e-2)
    assert_allclose(result["flux_err"].data[99, 99], 3.84e-11, rtol=1e-2)
    assert_allclose(result["flux_ul"].data[99, 99], 1.10e-09, rtol=1e-2)


@requires_data("gammapy-extra")
def test_large_kernel(input_maps):
    """Minimal test of compute_ts_image"""
    kernel = Gaussian2DKernel(100)
    ts_estimator = TSMapEstimator()

    with pytest.raises(ValueError) as err:
        ts_estimator.run(input_maps, kernel=kernel)
        assert "Kernel shape larger" in str(err.value)
