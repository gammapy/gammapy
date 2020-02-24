# Licensed under a 3-clause BSD style license - see LICENSE.rst
import pytest
import numpy as np
from numpy.testing import assert_allclose
import astropy.units as u
from astropy.convolution import Gaussian2DKernel
from gammapy.datasets import MapDataset
from gammapy.detect import TSMapEstimator
from gammapy.maps import Map, MapAxis
from gammapy.modeling.models import BackgroundModel
from gammapy.utils.testing import requires_data


@pytest.fixture(scope="session")
def input_dataset():
    filename = "$GAMMAPY_DATA/tests/unbundled/poisson_stats_image/input_all.fits.gz"

    energy = MapAxis.from_energy_bounds("0.1 TeV", "1 TeV", 1)

    counts2D = Map.read(filename, hdu="counts")
    counts = Map.from_geom(
        counts2D.geom.to_cube([energy]),
        data=counts2D.data[np.newaxis, :, :],
        unit=counts2D.unit,
    )
    exposure2D = Map.read(filename, hdu="exposure")
    exposure = Map.from_geom(
        exposure2D.geom.to_cube([energy]),
        data=exposure2D.data[np.newaxis, :, :],
        unit="cm2s",  # no unit in header?
    )

    background2D = Map.read(filename, hdu="background")
    background = Map.from_geom(
        background2D.geom.to_cube([energy]),
        data=background2D.data[np.newaxis, :, :],
        unit=background2D.unit,
    )
    background_model = BackgroundModel(background)

    # add mask
    mask2D = np.ones_like(background2D.data).astype("bool")
    mask2D[0:40, :] = False
    mask = Map.from_geom(
        background2D.geom.to_cube([energy]), data=mask2D[np.newaxis, :, :],
    )

    return MapDataset(
        counts=counts,
        exposure=exposure,
        models=background_model,
        mask_safe=mask,
    )


@requires_data()
def test_compute_ts_map(input_dataset):
    """Minimal test of compute_ts_image"""
    kernel = Gaussian2DKernel(5)

    ts_estimator = TSMapEstimator(method="leastsq iter", threshold=1)
    result = ts_estimator.run(input_dataset, kernel=kernel)

    assert "leastsq iter" in repr(ts_estimator)
    assert_allclose(result["ts"].data[99, 99], 1714.23, rtol=1e-2)
    assert_allclose(result["niter"].data[99, 99], 3)
    assert_allclose(result["flux"].data[99, 99], 1.02e-09, rtol=1e-2)
    assert_allclose(result["flux_err"].data[99, 99], 3.84e-11, rtol=1e-2)
    assert_allclose(result["flux_ul"].data[99, 99], 1.10e-09, rtol=1e-2)

    assert result["flux"].unit == u.Unit("cm-2s-1")
    assert result["flux_err"].unit == u.Unit("cm-2s-1")
    assert result["flux_ul"].unit == u.Unit("cm-2s-1")

    # Check mask is correctly taken into account
    assert np.isnan(result["ts"].data[30, 40])


@requires_data()
def test_compute_ts_map_newton(input_dataset):
    """Minimal test of compute_ts_image"""
    kernel = Gaussian2DKernel(5)

    ts_estimator = TSMapEstimator(method="root newton", threshold=1)
    result = ts_estimator.run(input_dataset, kernel=kernel)

    assert "root newton" in repr(ts_estimator)
    assert_allclose(result["ts"].data[99, 99], 1714.23, rtol=1e-2)
    assert_allclose(result["niter"].data[99, 99], 0)
    assert_allclose(result["flux"].data[99, 99], 1.02e-09, rtol=1e-2)
    assert_allclose(result["flux_err"].data[99, 99], 3.84e-11, rtol=1e-2)
    assert_allclose(result["flux_ul"].data[99, 99], 1.10e-09, rtol=1e-2)

    assert result["flux"].unit == u.Unit("cm-2s-1")
    assert result["flux_err"].unit == u.Unit("cm-2s-1")
    assert result["flux_ul"].unit == u.Unit("cm-2s-1")

    # Check mask is correctly taken into account
    assert np.isnan(result["ts"].data[30, 40])


@requires_data()
def test_compute_ts_map_downsampled(input_dataset):
    """Minimal test of compute_ts_image"""
    kernel = Gaussian2DKernel(2.5)

    ts_estimator = TSMapEstimator(
        method="root brentq", error_method="conf", ul_method="conf"
    )
    result = ts_estimator.run(input_dataset, kernel=kernel, downsampling_factor=2)

    assert_allclose(result["ts"].data[99, 99], 1675.28, rtol=1e-2)
    assert_allclose(result["niter"].data[99, 99], 7)
    assert_allclose(result["flux"].data[99, 99], 1.02e-09, rtol=1e-2)
    assert_allclose(result["flux_err"].data[99, 99], 3.84e-11, rtol=1e-2)
    assert_allclose(result["flux_ul"].data[99, 99], 1.10e-09, rtol=1e-2)

    assert result["flux"].unit == u.Unit("cm-2s-1")
    assert result["flux_err"].unit == u.Unit("cm-2s-1")
    assert result["flux_ul"].unit == u.Unit("cm-2s-1")

    # Check mask is correctly taken into account
    assert np.isnan(result["ts"].data[30, 40])


@requires_data()
def test_large_kernel(input_dataset):
    """Minimal test of compute_ts_image"""
    kernel = Gaussian2DKernel(100)
    ts_estimator = TSMapEstimator()

    with pytest.raises(ValueError):
        ts_estimator.run(input_dataset, kernel=kernel)


def test_incorrect_method():
    with pytest.raises(ValueError):
        TSMapEstimator(method="bad")
    with pytest.raises(ValueError):
        TSMapEstimator(error_method="bad")
