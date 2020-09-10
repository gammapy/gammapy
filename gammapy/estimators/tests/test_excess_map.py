# Licensed under a 3-clause BSD style license - see LICENSE.rst
import pytest
import numpy as np
from numpy.testing import assert_allclose
import astropy.units as u
from gammapy.datasets import MapDataset, MapDatasetOnOff
from gammapy.estimators import ExcessMapEstimator
from gammapy.maps import Map, MapAxis, WcsGeom
from gammapy.modeling.models import BackgroundModel
from gammapy.utils.testing import requires_data


def image_to_cube(input_map, e_min, e_max):
    e_min = u.Quantity(e_min)
    e_max = u.Quantity(e_max)
    axis = MapAxis.from_energy_bounds(e_min, e_max, nbin=1)
    geom = input_map.geom.to_cube([axis])
    return Map.from_geom(geom, data=input_map.data[np.newaxis, :, :])


@pytest.fixture
def simple_dataset():
    axis = MapAxis.from_energy_bounds(0.1, 10, 1, unit="TeV")
    geom = WcsGeom.create(npix=20, binsz=0.02, axes=[axis])
    dataset = MapDataset.create(geom)
    dataset.mask_safe += np.ones(dataset.data_shape, dtype=bool)
    dataset.counts += 2
    dataset.background_model.map += 1
    return dataset


@pytest.fixture
def simple_dataset_on_off():
    axis = MapAxis.from_energy_bounds(0.1, 10, 2, unit="TeV")
    geom = WcsGeom.create(npix=20, binsz=0.02, axes=[axis])
    dataset = MapDatasetOnOff.create(geom)
    dataset.mask_safe += np.ones(dataset.data_shape, dtype=bool)
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
    counts = image_to_cube(counts, "1 GeV", "100 GeV")
    background = Map.read(filename, hdu="background")
    background = image_to_cube(background, "1 GeV", "100 GeV")
    background_model = BackgroundModel(background)
    dataset = MapDataset(counts=counts)
    background_model.datasets_names = [dataset.name]
    dataset.models = background_model

    estimator = ExcessMapEstimator("0.1 deg", selection_optional=None)
    result_lima = estimator.run(dataset)

    assert_allclose(result_lima["significance"].data[:, 100, 100], 30.814916, atol=1e-3)
    assert_allclose(result_lima["significance"].data[:, 1, 1], 0.164, atol=1e-3)


@requires_data()
def test_compute_lima_on_off_image():
    """
    Test Li & Ma image with snippet from the H.E.S.S. survey data.
    """
    filename = "$GAMMAPY_DATA/tests/unbundled/hess/survey/hess_survey_snippet.fits.gz"
    n_on = Map.read(filename, hdu="ON")
    counts = image_to_cube(n_on, "1 TeV", "100 TeV")
    n_off = Map.read(filename, hdu="OFF")
    counts_off = image_to_cube(n_off, "1 TeV", "100 TeV")
    a_on = Map.read(filename, hdu="ONEXPOSURE")
    acceptance = image_to_cube(a_on, "1 TeV", "100 TeV")
    a_off = Map.read(filename, hdu="OFFEXPOSURE")
    acceptance_off = image_to_cube(a_off, "1 TeV", "100 TeV")
    dataset = MapDatasetOnOff(
        counts=counts,
        counts_off=counts_off,
        acceptance=acceptance,
        acceptance_off=acceptance_off,
    )

    significance = Map.read(filename, hdu="SIGNIFICANCE")
    significance = image_to_cube(significance, "1 TeV", "10 TeV")
    estimator = ExcessMapEstimator("0.1 deg", selection_optional=None)
    results = estimator.run(dataset)

    # Reproduce safe significance threshold from HESS software
    results["significance"].data[results["counts"].data < 5] = 0

    # crop the image at the boundaries, because the reference image
    # is cut out from a large map, there is no way to reproduce the
    # result with regular boundary handling
    actual = results["significance"].crop((11, 11)).data
    desired = significance.crop((11, 11)).data

    # Set boundary to NaN in reference image
    # The absolute tolerance is low because the method used here is slightly different from the one used in HGPS
    # n_off is convolved as well to ensure the method applies to true ON-OFF datasets
    assert_allclose(actual, desired, atol=0.2)


def test_significance_map_estimator_map_dataset(simple_dataset):
    estimator = ExcessMapEstimator(0.1 * u.deg)
    result = estimator.run(simple_dataset)

    assert_allclose(result["counts"].data[0, 10, 10], 162)
    assert_allclose(result["excess"].data[0, 10, 10], 81)
    assert_allclose(result["background"].data[0, 10, 10], 81)
    assert_allclose(result["significance"].data[0, 10, 10], 7.910732, atol=1e-5)
    assert_allclose(result["err"].data[0, 10, 10], 12.727922, atol=1e-3)
    assert_allclose(result["errp"].data[0, 10, 10], 13.063328, atol=1e-3)
    assert_allclose(result["errn"].data[0, 10, 10], -12.396716, atol=1e-3)
    assert_allclose(result["ul"].data[0, 10, 10], 122.240837, atol=1e-3)

    estimator_image = ExcessMapEstimator(0.1 * u.deg, return_image=True)
    result_image = estimator_image.run(simple_dataset)
    assert result_image["counts"].data.shape == (1, 20, 20)
    assert_allclose(result_image["significance"].data[0, 10, 10], 7.910732, atol=1e-5)


def test_significance_map_estimator_map_dataset_on_off(simple_dataset_on_off):
    estimator = ExcessMapEstimator(
        0.11 * u.deg,
        selection_optional=None,
        e_edges=[0.1 * u.TeV, 1 * u.TeV, 10 * u.TeV],
    )
    result = estimator.run(simple_dataset_on_off)

    assert result["counts"].data.shape == (2, 20, 20)
    assert_allclose(result["counts"].data[:, 10, 10], 194)
    assert_allclose(result["excess"].data[:, 10, 10], 48.5)
    assert_allclose(result["background"].data[:, 10, 10], 145.5)
    assert_allclose(result["significance"].data[:, 10, 10], 4.967916, atol=1e-5)

    estimator_image = ExcessMapEstimator(0.11 * u.deg, e_edges=[0.1 * u.TeV, 1 * u.TeV])

    result_image = estimator_image.run(simple_dataset_on_off)
    assert result_image["counts"].data.shape == (1, 20, 20)
    assert_allclose(
        result_image["significance"].data[0, 10, 10], 4.967916823, atol=1e-3
    )

    mask_fit = Map.from_geom(
        simple_dataset_on_off._geom,
        data=np.ones(simple_dataset_on_off.counts.data.shape, dtype=bool),
    )
    mask_fit.data[:, :, 10] = False
    mask_fit.data[:, 10, :] = False
    simple_dataset_on_off.mask_fit = mask_fit

    estimator_image = ExcessMapEstimator(0.11 * u.deg, apply_mask_fit=True)

    simple_dataset_on_off.exposure.data = (
        np.ones(simple_dataset_on_off.exposure.data.shape) * 1e6
    )
    result_image = estimator_image.run(simple_dataset_on_off)
    assert result_image["counts"].data.shape == (1, 20, 20)

    assert_allclose(result_image["significance"].data[0, 10, 10], 6.218852, atol=1e-3)

    assert_allclose(result_image["counts"].data[0, 10, 10], 304)
    assert_allclose(result_image["excess"].data[0, 10, 10], 76)
    assert_allclose(result_image["background"].data[0, 10, 10], 228)

    assert result_image["flux"].unit == u.Unit("cm-2s-1")
    assert_allclose(result_image["flux"].data[0, 10, 10], 3.8e-9, rtol=1e-3)


def test_incorrect_selection():
    with pytest.raises(ValueError):
        ExcessMapEstimator(0.11 * u.deg, selection_optional=["bad"])

    with pytest.raises(ValueError):
        ExcessMapEstimator(0.11 * u.deg, selection_optional=["ul", "bad"])

    estimator = ExcessMapEstimator(0.11 * u.deg)
    with pytest.raises(ValueError):
        estimator.selection_optional = "bad"


def test_significance_map_estimator_incorrect_dataset():
    with pytest.raises(ValueError):
        ExcessMapEstimator("bad")
