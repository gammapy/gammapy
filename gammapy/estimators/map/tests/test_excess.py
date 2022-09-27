# Licensed under a 3-clause BSD style license - see LICENSE.rst
import pytest
import numpy as np
from numpy.testing import assert_allclose
import astropy.units as u
from gammapy.datasets import MapDataset, MapDatasetOnOff
from gammapy.estimators import ExcessMapEstimator
from gammapy.estimators.utils import estimate_exposure_reco_energy
from gammapy.irf import PSFMap
from gammapy.maps import Map, MapAxis, WcsGeom
from gammapy.modeling.models import (
    GaussianSpatialModel,
    PowerLawSpectralModel,
    SkyModel,
)
from gammapy.utils.testing import requires_data


def image_to_cube(input_map, energy_min, energy_max):
    energy_min = u.Quantity(energy_min)
    energy_max = u.Quantity(energy_max)
    axis = MapAxis.from_energy_bounds(energy_min, energy_max, nbin=1)
    geom = input_map.geom.to_cube([axis])
    return Map.from_geom(geom, data=input_map.data[np.newaxis, :, :])


@pytest.fixture
def simple_dataset():
    axis = MapAxis.from_energy_bounds(0.1, 10, 1, unit="TeV")
    geom = WcsGeom.create(npix=20, binsz=0.02, axes=[axis])
    dataset = MapDataset.create(geom)
    dataset.mask_safe += np.ones(dataset.data_shape, dtype=bool)
    dataset.counts += 2
    dataset.background += 1
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
    dataset.exposure.data += 1e6
    dataset.psf = None
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
    dataset = MapDataset(counts=counts, background=background)

    estimator = ExcessMapEstimator("0.1 deg")
    result_lima = estimator.run(dataset)

    assert_allclose(result_lima["sqrt_ts"].data[:, 100, 100], 30.814916, atol=1e-3)
    assert_allclose(result_lima["sqrt_ts"].data[:, 1, 1], 0.164, atol=1e-3)
    assert_allclose(result_lima["npred_background"].data[:, 1, 1], 37, atol=1e-3)
    assert_allclose(result_lima["npred"].data[:, 1, 1], 38, atol=1e-3)
    assert_allclose(result_lima["npred_excess"].data[:, 1, 1], 1, atol=1e-3)


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
    estimator = ExcessMapEstimator("0.1 deg", correlate_off=False)
    results = estimator.run(dataset)

    # Reproduce safe significance threshold from HESS software
    results["sqrt_ts"].data[results["npred"].data < 5] = 0

    # crop the image at the boundaries, because the reference image
    # is cut out from a large map, there is no way to reproduce the
    # result with regular boundary handling
    actual = results["sqrt_ts"].crop((11, 11)).data
    desired = significance.crop((11, 11)).data

    # Set boundary to NaN in reference image
    # The absolute tolerance is low because the method used here is slightly
    # different from the one used in HGPS n_off is convolved as well to ensure
    # the method applies to true ON-OFF datasets
    assert_allclose(actual, desired, atol=0.2, rtol=1e-5)

    actual = np.nan_to_num(results["npred_background"].crop((11, 11)).data)
    background_corr = image_to_cube(
        Map.read(filename, hdu="BACKGROUNDCORRELATED"), "1 TeV", "100 TeV"
    )
    desired = background_corr.crop((11, 11)).data

    # Set boundary to NaN in reference image
    # The absolute tolerance is low because the method used here is slightly different
    # from the one used in HGPS n_off is convolved as well to ensure the method applies
    # to true ON-OFF datasets
    assert_allclose(actual, desired, atol=0.2, rtol=1e-5)


def test_significance_map_estimator_map_dataset(simple_dataset):
    simple_dataset.exposure = None
    estimator = ExcessMapEstimator(0.1 * u.deg, selection_optional=["all"])
    result = estimator.run(simple_dataset)

    assert_allclose(result["npred"].data[0, 10, 10], 162)
    assert_allclose(result["npred_excess"].data[0, 10, 10], 81)
    assert_allclose(result["npred_background"].data[0, 10, 10], 81)
    assert_allclose(result["sqrt_ts"].data[0, 10, 10], 7.910732, atol=1e-5)

    assert_allclose(result["npred_excess_err"].data[0, 10, 10], 12.727922, atol=1e-3)
    assert_allclose(result["npred_excess_errp"].data[0, 10, 10], 13.063328, atol=1e-3)
    assert_allclose(result["npred_excess_errn"].data[0, 10, 10], 12.396716, atol=1e-3)
    assert_allclose(result["npred_excess_ul"].data[0, 10, 10], 107.806275, atol=1e-3)


def test_significance_map_estimator_map_dataset_exposure(simple_dataset):
    simple_dataset.exposure += 1e10 * u.cm**2 * u.s
    axis = simple_dataset.exposure.geom.axes[0]
    simple_dataset.psf = PSFMap.from_gauss(axis, sigma="0.05 deg")

    model = SkyModel(
        PowerLawSpectralModel(amplitude="1e-9 cm-2 s-1 TeV-1"),
        GaussianSpatialModel(
            lat_0=0.0 * u.deg, lon_0=0.0 * u.deg, sigma=0.1 * u.deg, frame="icrs"
        ),
        name="sky_model",
    )

    simple_dataset.models = [model]
    simple_dataset.npred()

    estimator = ExcessMapEstimator(0.1 * u.deg, selection_optional="all")
    result = estimator.run(simple_dataset)

    assert_allclose(result["npred_excess"].data.sum(), 19733.602, rtol=1e-3)
    assert_allclose(result["sqrt_ts"].data[0, 10, 10], 4.217129, rtol=1e-3)

    # without mask safe
    simple_dataset_no_mask = MapDataset(
        counts=simple_dataset.counts,
        exposure=simple_dataset.exposure,
        background=simple_dataset.background,
        psf=simple_dataset.psf,
        edisp=simple_dataset.edisp,
    )

    simple_dataset_no_mask.models = [model]

    result_no_mask = estimator.run(simple_dataset_no_mask)
    assert_allclose(result_no_mask["npred_excess"].data.sum(), 19733.602, rtol=1e-3)
    assert_allclose(result_no_mask["sqrt_ts"].data[0, 10, 10], 4.217129, rtol=1e-3)


def test_excess_map_estimator_map_dataset_on_off_no_correlation(
    simple_dataset_on_off,
):
    # Test with exposure
    estimator_image = ExcessMapEstimator(
        0.11 * u.deg, energy_edges=[0.1, 1] * u.TeV, correlate_off=False
    )

    result_image = estimator_image.run(simple_dataset_on_off)
    assert result_image["npred"].data.shape == (1, 20, 20)
    assert_allclose(result_image["npred"].data[0, 10, 10], 194)
    assert_allclose(result_image["npred_excess"].data[0, 10, 10], 97)
    assert_allclose(result_image["sqrt_ts"].data[0, 10, 10], 0.780125, atol=1e-3)
    assert_allclose(result_image["flux"].data[:, 10, 10], 9.7e-9, atol=1e-5)


def test_excess_map_estimator_map_dataset_on_off_with_correlation_no_exposure(
    simple_dataset_on_off,
):
    # First without exposure
    simple_dataset_on_off.exposure = None

    estimator = ExcessMapEstimator(
        0.11 * u.deg, energy_edges=[0.1, 1, 10] * u.TeV, correlate_off=True
    )
    result = estimator.run(simple_dataset_on_off)

    assert result["npred"].data.shape == (2, 20, 20)
    assert_allclose(result["npred"].data[:, 10, 10], 194)
    assert_allclose(result["npred_excess"].data[:, 10, 10], 97)
    assert_allclose(result["sqrt_ts"].data[:, 10, 10], 5.741116, atol=1e-5)


def test_excess_map_estimator_map_dataset_on_off_with_correlation(
    simple_dataset_on_off,
):
    simple_dataset_on_off.exposure = None

    estimator_image = ExcessMapEstimator(
        0.11 * u.deg, energy_edges=[0.1, 1] * u.TeV, correlate_off=True
    )

    result_image = estimator_image.run(simple_dataset_on_off)
    assert result_image["npred"].data.shape == (1, 20, 20)
    assert_allclose(result_image["npred"].data[0, 10, 10], 194)
    assert_allclose(result_image["npred_excess"].data[0, 10, 10], 97)
    assert_allclose(result_image["sqrt_ts"].data[0, 10, 10], 5.741116, atol=1e-3)
    assert_allclose(result_image["flux"].data[:, 10, 10], 9.7e-9, atol=1e-5)


def test_excess_map_estimator_map_dataset_on_off_with_correlation_mask_fit(
    simple_dataset_on_off,
):
    geom = simple_dataset_on_off.counts.geom
    mask_fit = Map.from_geom(geom, data=1, dtype=bool)
    mask_fit.data[:, :, 10] = False
    mask_fit.data[:, 10, :] = False
    simple_dataset_on_off.mask_fit = mask_fit

    estimator_image = ExcessMapEstimator(0.11 * u.deg, correlate_off=True)

    result_image = estimator_image.run(simple_dataset_on_off)
    assert result_image["npred"].data.shape == (1, 20, 20)

    assert_allclose(result_image["sqrt_ts"].data[0, 10, 10], np.nan, atol=1e-3)
    assert_allclose(result_image["npred"].data[0, 10, 10], np.nan)
    assert_allclose(result_image["npred_excess"].data[0, 10, 10], np.nan)
    assert_allclose(result_image["sqrt_ts"].data[0, 9, 9], 7.186745, atol=1e-3)
    assert_allclose(result_image["npred"].data[0, 9, 9], 304)
    assert_allclose(result_image["npred_excess"].data[0, 9, 9], 152)

    assert result_image["flux"].unit == u.Unit("cm-2s-1")
    assert_allclose(result_image["flux"].data[0, 9, 9], 1.190928e-08, rtol=1e-3)


def test_excess_map_estimator_map_dataset_on_off_with_correlation_model(
    simple_dataset_on_off,
):
    model = SkyModel(
        PowerLawSpectralModel(amplitude="1e-9 cm-2 s-1TeV-1"),
        GaussianSpatialModel(
            lat_0=0.0 * u.deg, lon_0=0.0 * u.deg, sigma=0.1 * u.deg, frame="icrs"
        ),
        name="sky_model",
    )

    simple_dataset_on_off.models = [model]

    estimator_mod = ExcessMapEstimator(0.11 * u.deg, correlate_off=True)

    result_mod = estimator_mod.run(simple_dataset_on_off)
    assert result_mod["npred"].data.shape == (1, 20, 20)

    assert_allclose(result_mod["sqrt_ts"].data[0, 10, 10], 6.240846, atol=1e-3)

    assert_allclose(result_mod["npred"].data[0, 10, 10], 388)
    assert_allclose(result_mod["npred_excess"].data[0, 10, 10], 148.68057)

    assert result_mod["flux"].unit == "cm-2s-1"
    assert_allclose(result_mod["flux"].data[0, 10, 10], 1.486806e-08, rtol=1e-3)
    assert_allclose(result_mod["flux"].data.sum(), 5.254442e-06, rtol=1e-3)


def test_excess_map_estimator_map_dataset_on_off_reco_exposure(
    simple_dataset_on_off,
):

    # TODO: this has never worked...
    model = SkyModel(
        PowerLawSpectralModel(amplitude="1e-9 cm-2 s-1TeV-1"),
        GaussianSpatialModel(
            lat_0=0.0 * u.deg, lon_0=0.0 * u.deg, sigma=0.1 * u.deg, frame="icrs"
        ),
        name="sky_model",
    )

    simple_dataset_on_off.models = [model]

    spectral_model = PowerLawSpectralModel(index=15)
    estimator_mod = ExcessMapEstimator(
        0.11 * u.deg,
        correlate_off=True,
        spectral_model=spectral_model,
    )
    result_mod = estimator_mod.run(simple_dataset_on_off)

    assert result_mod["flux"].unit == "cm-2s-1"
    assert_allclose(result_mod["flux"].data.sum(), 5.254442e-06, rtol=1e-3)

    reco_exposure = estimate_exposure_reco_energy(
        simple_dataset_on_off, spectral_model=spectral_model
    )
    assert_allclose(reco_exposure.data.sum(), 8e12, rtol=0.001)


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
