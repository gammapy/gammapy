# Licensed under a 3-clause BSD style license - see LICENSE.rst
import pytest
import numpy as np
from numpy.testing import assert_allclose
import astropy.units as u
from astropy.coordinates import Angle
from gammapy.datasets import MapDataset
from gammapy.estimators import TSMapEstimator
from gammapy.irf import EnergyDependentTablePSF, PSFMap, EDispKernelMap
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
    counts = counts2D.to_cube([energy])
    exposure2D = Map.read(filename, hdu="exposure")
    exposure2D.unit = "cm2s"
    exposure = exposure2D.to_cube([energy_true])
    background2D = Map.read(filename, hdu="background")
    background = background2D.to_cube([energy])
    name = "test-dataset"
    background_model = BackgroundModel(background, datasets_names=[name])

    # add mask
    mask2D_data = np.ones_like(background2D.data).astype("bool")
    mask2D_data[0:40, :] = False
    mask2D = Map.from_geom(geom=counts2D.geom, data=mask2D_data)
    mask = mask2D.to_cube([energy])
    return MapDataset(
        counts=counts,
        exposure=exposure,
        models=background_model,
        mask_safe=mask,
        name=name,
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
    exposure.unit = "cm2 s"
    mask_safe = counts.copy(data=np.ones_like(counts.data).astype("bool"))

    psf = EnergyDependentTablePSF.read(
        "$GAMMAPY_DATA/fermi-3fhl-gc/fermi-3fhl-gc-psf-cube.fits.gz"
    )
    psfmap = PSFMap.from_energy_dependent_table_psf(psf)
    edisp = EDispKernelMap.from_diagonal_response(
        energy_axis=counts.geom.get_axis_by_name("energy"),
        energy_axis_true=exposure.geom.get_axis_by_name("energy_true"),
    )

    dataset = MapDataset(
        counts=counts,
        models=[background],
        exposure=exposure,
        mask_safe=mask_safe,
        psf=psfmap,
        name="fermi-3fhl-gc",
        edisp=edisp
    )

    return dataset.to_image()


@requires_data()
def test_compute_ts_map(input_dataset):
    """Minimal test of compute_ts_image"""
    spatial_model = GaussianSpatialModel(sigma="0.1 deg")
    spectral_model = PowerLawSpectralModel(index=2)
    model = SkyModel(spatial_model=spatial_model, spectral_model=spectral_model)
    ts_estimator = TSMapEstimator(
        model=model, threshold=1, kernel_width="1 deg", selection_optional=[]
    )
    result = ts_estimator.run(input_dataset)

    assert_allclose(result["ts"].data[99, 99], 1704.23, rtol=1e-2)
    assert_allclose(result["niter"].data[99, 99], 8)
    assert_allclose(result["flux"].data[99, 99], 1.02e-09, rtol=1e-2)
    assert_allclose(result["flux_err"].data[99, 99], 3.84e-11, rtol=1e-2)

    assert result["flux"].unit == u.Unit("cm-2s-1")
    assert result["flux_err"].unit == u.Unit("cm-2s-1")

    # Check mask is correctly taken into account
    assert np.isnan(result["ts"].data[30, 40])


@requires_data()
def test_compute_ts_map_psf(fermi_dataset):
    estimator = TSMapEstimator(kernel_width="1 deg")
    result = estimator.run(fermi_dataset)

    assert_allclose(result["ts"].data[29, 29], 835.140605, rtol=1e-2)
    assert_allclose(result["niter"].data[29, 29], 7)
    assert_allclose(result["flux"].data[29, 29], 1.626651e-09, rtol=1e-2)
    assert_allclose(result["flux_err"].data[29, 29], 9.548939e-11, rtol=1e-2)
    assert_allclose(result["flux_errp"].data[29, 29], 9.540259e-11, rtol=1e-2)
    assert_allclose(result["flux_errn"].data[29, 29], 9.034366e-11, rtol=1e-2)
    assert_allclose(result["flux_ul"].data[29, 29], 1.961776e-10, rtol=1e-2)
    assert result["flux"].unit == u.Unit("cm-2s-1")
    assert result["flux_err"].unit == u.Unit("cm-2s-1")
    assert result["flux_ul"].unit == u.Unit("cm-2s-1")


@requires_data()
def test_compute_ts_map_downsampled(input_dataset):
    """Minimal test of compute_ts_image"""
    spatial_model = GaussianSpatialModel(sigma="0.11 deg")
    spectral_model = PowerLawSpectralModel(index=2)
    model = SkyModel(spatial_model=spatial_model, spectral_model=spectral_model)

    ts_estimator = TSMapEstimator(
        model=model, downsampling_factor=2, kernel_width="1 deg",
    )
    result = ts_estimator.run(input_dataset)

    assert_allclose(result["ts"].data[99, 99], 1661.49, rtol=1e-2)
    assert_allclose(result["niter"].data[99, 99], 7)
    assert_allclose(result["flux"].data[99, 99], 1.065988e-09, rtol=1e-2)
    assert_allclose(result["flux_err"].data[99, 99], 4.005628e-11, rtol=1e-2)
    assert_allclose(result["flux_ul"].data[99, 99], 8.220152e-11, rtol=1e-2)

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
