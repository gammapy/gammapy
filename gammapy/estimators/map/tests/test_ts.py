# Licensed under a 3-clause BSD style license - see LICENSE.rst
import pytest
import numpy as np
from numpy.testing import assert_allclose
import astropy.units as u
from astropy.coordinates import Angle
from gammapy.datasets import MapDataset
from gammapy.estimators import TSMapEstimator
from gammapy.irf import EDispKernelMap, PSFMap
from gammapy.maps import Map, MapAxis, WcsGeom
from gammapy.modeling.models import (
    GaussianSpatialModel,
    PointSpatialModel,
    PowerLawSpectralModel,
    SkyModel,
)
from gammapy.utils.testing import requires_data


@pytest.fixture(scope="session")
def fake_dataset():
    axis = MapAxis.from_energy_bounds(0.1, 10, 5, unit="TeV", name="energy")
    axis_true = MapAxis.from_energy_bounds(0.05, 20, 10, unit="TeV", name="energy_true")

    geom = WcsGeom.create(npix=50, binsz=0.02, axes=[axis])
    dataset = MapDataset.create(geom)
    dataset.psf = PSFMap.from_gauss(axis_true, sigma="0.05 deg")
    dataset.mask_safe += np.ones(dataset.data_shape, dtype=bool)
    dataset.background += 1
    dataset.exposure += 1e12 * u.cm**2 * u.s

    spatial_model = PointSpatialModel()
    spectral_model = PowerLawSpectralModel(amplitude="1e-10 cm-2s-1TeV-1", index=2)
    model = SkyModel(
        spatial_model=spatial_model, spectral_model=spectral_model, name="source"
    )
    dataset.models = [model]
    dataset.fake(random_state=42)
    return dataset


@pytest.fixture(scope="session")
def input_dataset():
    filename = "$GAMMAPY_DATA/tests/unbundled/poisson_stats_image/input_all.fits.gz"

    energy = MapAxis.from_energy_bounds("0.1 TeV", "1 TeV", 1)
    energy_true = MapAxis.from_energy_bounds("0.1 TeV", "1 TeV", 1, name="energy_true")

    counts2D = Map.read(filename, hdu="counts")
    counts = counts2D.to_cube([energy])
    exposure2D = Map.read(filename, hdu="exposure")
    exposure2D = Map.from_geom(exposure2D.geom, data=exposure2D.data, unit="cm2s")
    exposure = exposure2D.to_cube([energy_true])
    background2D = Map.read(filename, hdu="background")
    background = background2D.to_cube([energy])

    # add mask
    mask2D_data = np.ones_like(background2D.data).astype("bool")
    mask2D_data[0:40, :] = False
    mask2D = Map.from_geom(geom=counts2D.geom, data=mask2D_data)
    mask = mask2D.to_cube([energy])

    name = "test-dataset"
    return MapDataset(
        counts=counts,
        exposure=exposure,
        background=background,
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

    exposure = Map.read(
        "$GAMMAPY_DATA/fermi-3fhl-gc/fermi-3fhl-gc-exposure-cube.fits.gz"
    )
    exposure = exposure.cutout(exposure.geom.center_skydir, size)

    psfmap = PSFMap.read(
        "$GAMMAPY_DATA/fermi-3fhl-gc/fermi-3fhl-gc-psf-cube.fits.gz", format="gtpsf"
    )
    edisp = EDispKernelMap.from_diagonal_response(
        energy_axis=counts.geom.axes["energy"],
        energy_axis_true=exposure.geom.axes["energy_true"],
    )

    return MapDataset(
        counts=counts,
        background=background,
        exposure=exposure,
        psf=psfmap,
        name="fermi-3fhl-gc",
        edisp=edisp,
    )


@requires_data()
def test_compute_ts_map(input_dataset):
    """Minimal test of compute_ts_image"""
    spatial_model = GaussianSpatialModel(sigma="0.1 deg")
    spectral_model = PowerLawSpectralModel(index=2)
    model = SkyModel(spatial_model=spatial_model, spectral_model=spectral_model)
    ts_estimator = TSMapEstimator(model=model, threshold=1, selection_optional=[])

    kernel = ts_estimator.estimate_kernel(dataset=input_dataset)
    assert_allclose(kernel.geom.width, 1.22 * u.deg)
    assert_allclose(kernel.data.sum(), 1.0)

    result = ts_estimator.run(input_dataset)
    assert_allclose(result["ts"].data[0, 99, 99], 1704.23, rtol=1e-2)
    assert_allclose(result["niter"].data[0, 99, 99], 7)
    assert_allclose(result["flux"].data[0, 99, 99], 1.02e-09, rtol=1e-2)
    assert_allclose(result["flux_err"].data[0, 99, 99], 3.84e-11, rtol=1e-2)
    assert_allclose(result["npred"].data[0, 99, 99], 4744.020361, rtol=1e-2)
    assert_allclose(result["npred_excess"].data[0, 99, 99], 1026.874063, rtol=1e-2)
    assert_allclose(result["npred_excess_err"].data[0, 99, 99], 38.470995, rtol=1e-2)

    assert result["flux"].unit == u.Unit("cm-2s-1")
    assert result["flux_err"].unit == u.Unit("cm-2s-1")

    # Check mask is correctly taken into account
    assert np.isnan(result["ts"].data[0, 30, 40])

    energy_axis = result["ts"].geom.axes["energy"]
    assert_allclose(energy_axis.edges.value, [0.1, 1])


@requires_data()
def test_compute_ts_map_psf(fermi_dataset):
    spatial_model = PointSpatialModel()
    spectral_model = PowerLawSpectralModel(amplitude="1e-22 cm-2 s-1 keV-1")
    model = SkyModel(spatial_model=spatial_model, spectral_model=spectral_model)

    estimator = TSMapEstimator(
        model=model, kernel_width="1 deg", selection_optional="all"
    )
    result = estimator.run(fermi_dataset)

    assert_allclose(result["ts"].data[0, 29, 29], 833.38, rtol=2e-3)
    assert_allclose(result["niter"].data[0, 29, 29], 7)
    assert_allclose(result["flux"].data[0, 29, 29], 1.34984e-09, rtol=2e-3)
    assert_allclose(result["flux_err"].data[0, 29, 29], 7.93751176e-11, rtol=2e-3)
    assert_allclose(result["flux_errp"].data[0, 29, 29], 7.948953e-11, rtol=2e-3)
    assert_allclose(result["flux_errn"].data[0, 29, 29], 7.508168e-11, rtol=2e-3)
    assert_allclose(result["flux_ul"].data[0, 29, 29], 1.513062157e-09, rtol=2e-3)

    assert result["flux"].unit == u.Unit("cm-2s-1")
    assert result["flux_err"].unit == u.Unit("cm-2s-1")
    assert result["flux_ul"].unit == u.Unit("cm-2s-1")


@requires_data()
def test_compute_ts_map_energy(fermi_dataset):
    spatial_model = PointSpatialModel()
    spectral_model = PowerLawSpectralModel(amplitude="1e-22 cm-2 s-1 keV-1")
    model = SkyModel(spatial_model=spatial_model, spectral_model=spectral_model)

    estimator = TSMapEstimator(
        model=model,
        kernel_width="0.6 deg",
        energy_edges=[10, 100, 1000] * u.GeV,
        sum_over_energy_groups=False,
    )

    result = estimator.run(fermi_dataset)
    result.filter_success_nan = False

    assert_allclose(result.ts.data[1, 43, 30], 0.199291, atol=0.01)
    assert not result["success"].data[1, 43, 30]

    assert_allclose(result["ts"].data[:, 29, 29], [804.86171, 16.988756], rtol=1e-2)
    assert_allclose(
        result["flux"].data[:, 29, 29], [1.233119e-09, 3.590694e-11], rtol=1e-2
    )
    assert_allclose(
        result["flux_err"].data[:, 29, 29], [7.382305e-11, 1.338985e-11], rtol=1e-2
    )
    assert_allclose(result["niter"].data[:, 29, 29], [6, 6])

    energy_axis = result["ts"].geom.axes["energy"]
    assert_allclose(energy_axis.edges.to_value("GeV"), [10, 84.471641, 500], rtol=1e-4)


@requires_data()
def test_compute_ts_map_downsampled(input_dataset):
    """Minimal test of compute_ts_image"""
    spatial_model = GaussianSpatialModel(sigma="0.11 deg")
    spectral_model = PowerLawSpectralModel(index=2)
    model = SkyModel(spatial_model=spatial_model, spectral_model=spectral_model)

    ts_estimator = TSMapEstimator(
        model=model,
        downsampling_factor=2,
        kernel_width="1 deg",
        selection_optional=["ul"],
    )
    result = ts_estimator.run(input_dataset)

    assert_allclose(result["ts"].data[0, 99, 99], 1661.49, rtol=1e-2)
    assert_allclose(result["niter"].data[0, 99, 99], 7)
    assert_allclose(result["flux"].data[0, 99, 99], 1.065988e-09, rtol=1e-2)
    assert_allclose(result["flux_err"].data[0, 99, 99], 4.005628e-11, rtol=1e-2)
    assert_allclose(result["flux_ul"].data[0, 99, 99], 1.14818952e-09, rtol=1e-2)

    assert result["flux"].unit == u.Unit("cm-2s-1")
    assert result["flux_err"].unit == u.Unit("cm-2s-1")
    assert result["flux_ul"].unit == u.Unit("cm-2s-1")

    # Check mask is correctly taken into account
    assert np.isnan(result["ts"].data[0, 30, 40])


def test_ts_map_with_model(fake_dataset):
    model = fake_dataset.models["source"]

    fake_dataset.models = []

    estimator = TSMapEstimator(
        model,
        kernel_width="0.3 deg",
        selection_optional=["all"],
        energy_edges=[200, 3500] * u.GeV,
    )
    maps = estimator.run(fake_dataset)

    assert_allclose(maps["sqrt_ts"].data[:, 25, 25], 18.369942, atol=0.1)
    assert_allclose(maps["flux"].data[:, 25, 25], 3.513e-10, atol=1e-12)
    assert_allclose(maps["flux_err"].data[0, 0, 0], 2.494462e-11, rtol=1e-4)

    fake_dataset.models = [model]
    maps = estimator.run(fake_dataset)

    assert_allclose(maps["sqrt_ts"].data[:, 25, 25], -0.231187, atol=0.1)
    assert_allclose(maps["flux"].data[:, 25, 25], -5.899423e-12, atol=1e-12)

    # Try downsampling
    estimator = TSMapEstimator(
        model,
        kernel_width="0.3 deg",
        selection_optional=[],
        downsampling_factor=2,
        energy_edges=[200, 3500] * u.GeV,
    )
    maps = estimator.run(fake_dataset)
    assert_allclose(maps["sqrt_ts"].data[:, 25, 25], 0.323203, atol=0.1)
    assert_allclose(maps["flux"].data[:, 25, 25], 1.015509e-12, atol=1e-12)


@requires_data()
def test_compute_ts_map_with_hole(fake_dataset):
    """Test of compute_ts_image with a null exposure at the center of the map"""
    holes_dataset = fake_dataset.copy("holes_dataset")
    i, j, ie = holes_dataset.exposure.geom.center_pix
    holes_dataset.exposure.data[:, np.int_(i), np.int_(j)] = 0.0

    spatial_model = GaussianSpatialModel(sigma="0.1 deg")
    spectral_model = PowerLawSpectralModel(index=2)
    model = SkyModel(spatial_model=spatial_model, spectral_model=spectral_model)
    ts_estimator = TSMapEstimator(model=model, threshold=1, selection_optional=[])

    kernel = ts_estimator.estimate_kernel(dataset=holes_dataset)
    assert_allclose(kernel.geom.width, 1.0 * u.deg)
    assert_allclose(kernel.data.sum(), 1.0)

    holes_dataset.exposure.data[...] = 0.0
    with pytest.raises(ValueError):
        kernel = ts_estimator.estimate_kernel(dataset=holes_dataset)
