# Licensed under a 3-clause BSD style license - see LICENSE.rst
import pytest
import numpy as np
from numpy.testing import assert_allclose
import astropy.units as u
from astropy.coordinates import Angle
from gammapy.datasets import MapDataset
from gammapy.estimators import TSMapEstimator
from gammapy.irf import EnergyDependentTablePSF, PSFMap
from gammapy.maps import Map, MapAxis
from gammapy.modeling.models import (
    BackgroundModel,
    GaussianSpatialModel,
    PowerLawSpectralModel,
    SkyModel,
)
from gammapy.utils.testing import requires_data


@pytest.fixture(scope="session")
def input_dataset():
    filename = "$GAMMAPY_DATA/tests/unbundled/poisson_stats_image/input_all.fits.gz"

    energy = MapAxis.from_energy_bounds("0.1 TeV", "1 TeV", 1)
    energy_true = MapAxis.from_energy_bounds("0.1 TeV", "1 TeV", 1, name="energy_true")

    counts2D = Map.read(filename, hdu="counts")
    counts = Map.from_geom(
        counts2D.geom.to_cube([energy]),
        data=counts2D.data[np.newaxis, :, :],
        unit=counts2D.unit,
    )
    exposure2D = Map.read(filename, hdu="exposure")
    exposure = Map.from_geom(
        exposure2D.geom.to_cube([energy_true]),
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
        counts=counts, exposure=exposure, models=background_model, mask_safe=mask,
    )


@pytest.fixture(scope="session")
def fermi_dataset():
    size = Angle("3 deg", "3.5 deg")
    counts = Map.read("$GAMMAPY_DATA/fermi-3fhl-gc/fermi-3fhl-gc-counts-cube.fits.gz")
    counts = counts.cutout(counts.geom.center_skydir, size)

    background = Map.read(
        "$GAMMAPY_DATA/fermi-3fhl-gc/fermi-3fhl-gc-background-cube.fits.gz"
    )
    background = background.cutout(background.geom.center_skydir, size)
    background = BackgroundModel(background, datasets_names=["fermi-3fhl-gc"])

    exposure = Map.read(
        "$GAMMAPY_DATA/fermi-3fhl-gc/fermi-3fhl-gc-exposure-cube.fits.gz"
    )
    exposure = exposure.cutout(exposure.geom.center_skydir, size)
    exposure.unit = "cm2s"
    mask_safe = counts.copy(data=np.ones_like(counts.data).astype("bool"))

    psf = EnergyDependentTablePSF.read(
        "$GAMMAPY_DATA/fermi-3fhl-gc/fermi-3fhl-gc-psf-cube.fits.gz"
    )
    psfmap = PSFMap.from_energy_dependent_table_psf(psf)

    dataset = MapDataset(
        counts=counts,
        models=[background],
        exposure=exposure,
        mask_safe=mask_safe,
        psf=psfmap,
        name="fermi-3fhl-gc",
    )
    dataset = dataset.to_image()

    return dataset


@requires_data()
def test_compute_ts_map(input_dataset):
    """Minimal test of compute_ts_image"""
    spatial_model = GaussianSpatialModel(sigma="0.1 deg")
    spectral_model = PowerLawSpectralModel(index=2)
    model = SkyModel(spatial_model=spatial_model, spectral_model=spectral_model)
    ts_estimator = TSMapEstimator(
        model=model, method="leastsq iter", threshold=1, kernel_width="1 deg"
    )
    result = ts_estimator.run(input_dataset)

    assert "leastsq iter" in repr(ts_estimator)
    assert_allclose(result["ts"].data[99, 99], 1704.23, rtol=1e-2)
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
def test_compute_ts_map_psf(fermi_dataset):
    estimator = TSMapEstimator(kernel_width="1 deg")
    result = estimator.run(fermi_dataset)

    assert "root brentq" in repr(estimator)
    assert_allclose(result["ts"].data[29, 29], 852.1548, rtol=1e-2)
    assert_allclose(result["niter"].data[29, 29], 7)
    assert_allclose(result["flux"].data[29, 29], 1.419909e-09, rtol=1e-2)
    assert_allclose(result["flux_err"].data[29, 29], 8.245766e-11, rtol=1e-2)
    assert_allclose(result["flux_ul"].data[29, 29], 1.584825e-09, rtol=1e-2)
    assert result["flux"].unit == u.Unit("cm-2s-1")
    assert result["flux_err"].unit == u.Unit("cm-2s-1")
    assert result["flux_ul"].unit == u.Unit("cm-2s-1")


@requires_data()
def test_compute_ts_map_newton(input_dataset):
    """Minimal test of compute_ts_image"""
    spatial_model = GaussianSpatialModel(sigma="0.1 deg")
    spectral_model = PowerLawSpectralModel(index=2)
    model = SkyModel(spatial_model=spatial_model, spectral_model=spectral_model)

    ts_estimator = TSMapEstimator(
        model=model, method="root newton", threshold=1, kernel_width="1 deg"
    )
    result = ts_estimator.run(input_dataset)

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
    spatial_model = GaussianSpatialModel(sigma="0.11 deg")
    spectral_model = PowerLawSpectralModel(index=2)
    model = SkyModel(spatial_model=spatial_model, spectral_model=spectral_model)

    ts_estimator = TSMapEstimator(
        model=model,
        downsampling_factor=2,
        method="root brentq",
        error_method="conf",
        ul_method="conf",
        kernel_width="1 deg",
    )
    result = ts_estimator.run(input_dataset)

    assert_allclose(result["ts"].data[99, 99], 1661.49, rtol=1e-2)
    assert_allclose(result["niter"].data[99, 99], 9)
    assert_allclose(result["flux"].data[99, 99], 1.065988e-09, rtol=1e-2)
    assert_allclose(result["flux_err"].data[99, 99], 4.005628e-11, rtol=1e-2)
    assert_allclose(result["flux_ul"].data[99, 99], 1.147133e-09, rtol=1e-2)

    assert result["flux"].unit == u.Unit("cm-2s-1")
    assert result["flux_err"].unit == u.Unit("cm-2s-1")
    assert result["flux_ul"].unit == u.Unit("cm-2s-1")

    # Check mask is correctly taken into account
    assert np.isnan(result["ts"].data[30, 40])


@requires_data()
def test_large_kernel(input_dataset):
    """Minimal test of compute_ts_image"""
    spatial_model = GaussianSpatialModel(sigma="4 deg")
    spectral_model = PowerLawSpectralModel(index=2)
    model = SkyModel(spatial_model=spatial_model, spectral_model=spectral_model)
    ts_estimator = TSMapEstimator(model=model, kernel_width="4 deg")

    with pytest.raises(ValueError):
        ts_estimator.run(input_dataset)


def test_incorrect_method():
    model = GaussianSpatialModel(sigma="0.2 deg")
    with pytest.raises(ValueError):
        TSMapEstimator(model, method="bad")
    with pytest.raises(ValueError):
        TSMapEstimator(model, error_method="bad")
