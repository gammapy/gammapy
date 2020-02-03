# Licensed under a 3-clause BSD style license - see LICENSE.rst
import pytest
from numpy.testing import assert_allclose
import astropy.units as u
from astropy.convolution import Tophat2DKernel
from gammapy.detect import (
    compute_lima_image,
    compute_lima_on_off_image,
    SignificanceMapEstimator,
)
from gammapy.maps import Map, MapAxis, WcsGeom
from gammapy.cube import MapDataset, MapDatasetOnOff
from gammapy.utils.testing import requires_data


@pytest.fixture
def simple_dataset():
    axis = MapAxis.from_energy_bounds(0.1, 10, 1, unit="TeV")
    geom = WcsGeom.create(npix=50, binsz=0.02, axes=[axis])
    dataset = MapDataset.create(geom)
    dataset.mask_safe += 1
    dataset.counts += 2
    dataset.background_model.map += 1
    return dataset


@pytest.fixture
def simple_dataset_on_off():
    axis = MapAxis.from_energy_bounds(0.1, 10, 1, unit="TeV")
    geom = WcsGeom.create(npix=50, binsz=0.02, axes=[axis])
    dataset = MapDatasetOnOff.create(geom)
    dataset.mask_safe += 1
    dataset.counts += 2
    dataset.counts_off += 1
    dataset.acceptance += 1
    dataset.acceptance_off += 1
    return dataset


@requires_data()
def test_compute_lima_image():
    """
    Test Li & Ma image against TS image for Tophat kernel
    """
    filename = "$GAMMAPY_DATA/tests/unbundled/poisson_stats_image/input_all.fits.gz"
    counts = Map.read(filename, hdu="counts")
    background = Map.read(filename, hdu="background")

    kernel = Tophat2DKernel(5)
    result_lima = compute_lima_image(counts, background, kernel)

    assert_allclose(result_lima["significance"].data[100, 100], 30.814916, atol=1e-3)
    assert_allclose(result_lima["significance"].data[1, 1], 0.164, atol=1e-3)


@requires_data()
def test_compute_lima_on_off_image():
    """
    Test Li & Ma image with snippet from the H.E.S.S. survey data.
    """
    filename = "$GAMMAPY_DATA/tests/unbundled/hess/survey/hess_survey_snippet.fits.gz"
    n_on = Map.read(filename, hdu="ON")
    n_off = Map.read(filename, hdu="OFF")
    a_on = Map.read(filename, hdu="ONEXPOSURE")
    a_off = Map.read(filename, hdu="OFFEXPOSURE")
    significance = Map.read(filename, hdu="SIGNIFICANCE")

    kernel = Tophat2DKernel(5)
    results = compute_lima_on_off_image(n_on, n_off, a_on, a_off, kernel)

    # Reproduce safe significance threshold from HESS software
    results["significance"].data[results["n_on"].data < 5] = 0

    # crop the image at the boundaries, because the reference image
    # is cut out from a large map, there is no way to reproduce the
    # result with regular boundary handling
    actual = results["significance"].crop(kernel.shape).data
    desired = significance.crop(kernel.shape).data

    # Set boundary to NaN in reference image
    # The absolute tolerance is low because the method used here is slightly different from the one used in HGPS
    # n_off is convolved as well to ensure the method applies to true ON-OFF datasets
    assert_allclose(actual, desired, atol=0.2)


def test_significance_map_estimator_incorrect_dataset():
    estimator = SignificanceMapEstimator("0.1 deg")

    with pytest.raises(ValueError):
        estimator.run("bad")


def test_significance_map_estimator_map_dataset(simple_dataset):
    estimator = SignificanceMapEstimator(0.1 * u.deg)
    result = estimator.run(simple_dataset)

    assert_allclose(result["counts"].data[0, 25, 25], 162)
    assert_allclose(result["excess"].data[0, 25, 25], 81)
    assert_allclose(result["background"].data[0, 25, 25], 81)
    assert_allclose(result["significance"].data[0, 25, 25], 7.910732)


def test_significance_map_estimator_map_dataset_on_off(simple_dataset_on_off):
    estimator = SignificanceMapEstimator(0.1 * u.deg)
    result = estimator.run(simple_dataset_on_off)

    assert_allclose(result["n_on"].data[0, 25, 25], 162)
    assert_allclose(result["excess"].data[0, 25, 25], 81)
    assert_allclose(result["background"].data[0, 25, 25], 81)
    assert_allclose(result["significance"].data[0, 25, 25], 5.246298)
