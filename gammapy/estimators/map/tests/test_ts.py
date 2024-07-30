# Licensed under a 3-clause BSD style license - see LICENSE.rst
from copy import deepcopy
import pytest
import numpy as np
from numpy.testing import assert_allclose
import astropy.units as u
from astropy.coordinates import Angle, SkyCoord
from gammapy.datasets import Datasets, MapDataset, MapDatasetOnOff
from gammapy.estimators import TSMapEstimator
from gammapy.estimators.utils import (
    approximate_profile_map,
    combine_flux_maps,
    get_combined_significance_maps,
    get_flux_map_from_profile,
)
from gammapy.irf import EDispKernelMap, PSFMap
from gammapy.maps import Map, MapAxis, WcsGeom
from gammapy.modeling.models import (
    ConstantSpatialModel,
    GaussianSpatialModel,
    PointSpatialModel,
    PowerLawSpectralModel,
    SkyModel,
    TemplateSpatialModel,
)
from gammapy.utils.testing import requires_data, requires_dependency


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
@requires_dependency("ray")
def test_compute_ts_map_parallel_ray(input_dataset):
    """Minimal test of compute_ts_image"""
    spatial_model = GaussianSpatialModel(sigma="0.1 deg")
    spectral_model = PowerLawSpectralModel(index=2)
    model = SkyModel(spatial_model=spatial_model, spectral_model=spectral_model)

    ts_estimator = TSMapEstimator(
        model=model,
        threshold=1,
        selection_optional=[],
        parallel_backend="ray",
        n_jobs=2,
    )
    assert ts_estimator.parallel_backend == "ray"
    assert ts_estimator.n_jobs == 2

    result = ts_estimator.run(input_dataset)
    assert_allclose(result["ts"].data[0, 99, 99], 1704.23, rtol=1e-2)
    assert_allclose(result["niter"].data[0, 99, 99], 7)
    assert_allclose(result["flux"].data[0, 99, 99], 1.02e-09, rtol=1e-2)
    assert_allclose(result["flux_err"].data[0, 99, 99], 3.84e-11, rtol=1e-2)
    assert_allclose(result["npred"].data[0, 99, 99], 4744.020361, rtol=1e-2)
    assert_allclose(result["npred_excess"].data[0, 99, 99], 1026.874063, rtol=1e-2)
    assert_allclose(result["npred_excess_err"].data[0, 99, 99], 38.470995, rtol=1e-2)


@requires_data()
def test_compute_ts_map_parallel_multiprocessing(input_dataset):
    """Minimal test of compute_ts_image"""
    spatial_model = GaussianSpatialModel(sigma="0.1 deg")
    spectral_model = PowerLawSpectralModel(index=2)
    model = SkyModel(spatial_model=spatial_model, spectral_model=spectral_model)

    ts_estimator = TSMapEstimator(
        model=model,
        threshold=1,
        selection_optional=[],
        n_jobs=3,
        parallel_backend="multiprocessing",
    )

    result = ts_estimator.run(input_dataset)

    assert ts_estimator.n_jobs == 3
    assert_allclose(result["ts"].data[0, 99, 99], 1704.23, rtol=1e-2)
    assert_allclose(result["niter"].data[0, 99, 99], 7)
    assert_allclose(result["flux"].data[0, 99, 99], 1.02e-09, rtol=1e-2)
    assert_allclose(result["flux_err"].data[0, 99, 99], 3.84e-11, rtol=1e-2)
    assert_allclose(result["npred"].data[0, 99, 99], 4744.020361, rtol=1e-2)
    assert_allclose(result["npred_excess"].data[0, 99, 99], 1026.874063, rtol=1e-2)
    assert_allclose(result["npred_excess_err"].data[0, 99, 99], 38.470995, rtol=1e-2)


@requires_data()
def test_compute_ts_map_psf(fermi_dataset):
    spatial_model = PointSpatialModel()
    spectral_model = PowerLawSpectralModel(amplitude="1e-22 cm-2 s-1 keV-1")
    model = SkyModel(spatial_model=spatial_model, spectral_model=spectral_model)

    estimator = TSMapEstimator(
        model=model, kernel_width="1 deg", selection_optional=["ul", "errn-errp"]
    )
    result = estimator.run(fermi_dataset)

    assert_allclose(result["ts"].data[0, 29, 29], 830.97957, rtol=2e-3)
    assert_allclose(result["niter"].data[0, 29, 29], 7)
    assert_allclose(result["flux"].data[0, 29, 29], 1.339426e-09, rtol=2e-3)
    assert_allclose(result["flux_err"].data[0, 29, 29], 7.883016e-11, rtol=2e-3)
    assert_allclose(result["flux_errp"].data[0, 29, 29], 7.913813e-11, rtol=2e-3)
    assert_allclose(result["flux_errn"].data[0, 29, 29], 7.453983e-11, rtol=2e-3)
    assert_allclose(result["flux_ul"].data[0, 29, 29], 1.501809e-09, rtol=2e-3)

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

    assert_allclose(result.ts.data[1, 43, 30], 0.212079, atol=0.01)
    assert not result["success"].data[1, 43, 30]

    assert_allclose(result["ts"].data[:, 29, 29], [795.815842, 17.52017], rtol=1e-2)
    assert_allclose(
        result["flux"].data[:, 29, 29], [1.233119e-09, 3.590694e-11], rtol=1e-2
    )
    assert_allclose(
        result["flux_err"].data[:, 29, 29], [7.382305e-11, 1.338985e-11], rtol=1e-2
    )
    assert_allclose(result["niter"].data[:, 29, 29], [6, 6])

    energy_axis = result["ts"].geom.axes["energy"]
    assert_allclose(energy_axis.edges.to_value("GeV"), [10, 84.471641, 500], rtol=1e-4)

    fermi_dataset_maksed = fermi_dataset.copy()
    mask_safe = Map.from_geom(fermi_dataset.counts.geom, dtype=bool)
    mask_safe.data[:-3, :, :] = True

    fermi_dataset_maksed.mask_safe = mask_safe

    result = estimator.run(fermi_dataset_maksed)
    result.filter_success_nan = False

    assert_allclose(result.ts.data[1, 43, 30], 0.164831, atol=0.01)
    assert not result["success"].data[1, 43, 30]

    assert_allclose(result["ts"].data[:, 29, 29], [795.815842, 8.777864], rtol=1e-2)
    assert_allclose(
        result["flux"].data[:, 29, 29], [1.223901e-09, 3.748007e-11], rtol=1e-2
    )
    assert_allclose(
        result["flux_err"].data[:, 29, 29], [7.363390e-11, 1.799367e-11], rtol=1e-2
    )
    assert_allclose(result["niter"].data[:, 29, 29], [6, 6])

    energy_axis = result["ts"].geom.axes["energy"]
    assert_allclose(energy_axis.edges.to_value("GeV"), [10, 84.471641, 500], rtol=1e-4)


@requires_data()
def test_compute_ts_map_invalid(fermi_dataset):
    spatial_model = PointSpatialModel()
    spectral_model = PowerLawSpectralModel(amplitude="1e-22 cm-2 s-1 keV-1")
    model = SkyModel(spatial_model=spatial_model, spectral_model=spectral_model)

    estimator = TSMapEstimator(
        model=model,
        kernel_width="0.6 deg",
        energy_edges=[10, 100, 1000] * u.GeV,
        sum_over_energy_groups=False,
    )

    fermi_dataset_empty = fermi_dataset.copy()
    fermi_dataset_empty.background = None
    with pytest.raises(ValueError):
        result = estimator.run(fermi_dataset_empty)
    fermi_dataset_empty = fermi_dataset.copy()
    fermi_dataset_empty.background.data = 0
    with pytest.raises(ValueError):
        result = estimator.run(fermi_dataset_empty)

    fermi_dataset_empty = fermi_dataset.copy()
    mask_safe = Map.from_geom(fermi_dataset.counts.geom, dtype=bool)
    fermi_dataset_empty.mask_safe = mask_safe
    with pytest.raises(ValueError):
        result = estimator.run(fermi_dataset_empty)

    spatial_model = ConstantSpatialModel(value=0 / u.sr)
    spectral_model = PowerLawSpectralModel(amplitude="1e-22 cm-2 s-1 keV-1")
    model = SkyModel(spatial_model=spatial_model, spectral_model=spectral_model)

    estimator = TSMapEstimator(
        model=model,
        kernel_width="0.6 deg",
        energy_edges=[10, 100, 1000] * u.GeV,
        sum_over_energy_groups=False,
    )

    result = estimator.run(fermi_dataset)
    assert_allclose(result["ts"].data, 0)


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


def test_ts_map_stat_scan(fake_dataset):
    model = fake_dataset.models["source"]

    dataset = fake_dataset.downsample(25)

    estimator_ref = TSMapEstimator(
        model,
        kernel_width="0.3 deg",
        energy_edges=[200, 3500] * u.GeV,
        selection_optional=["errn-errp", "ul"],
    )

    estimator = TSMapEstimator(
        model,
        kernel_width="0.3 deg",
        selection_optional=["stat_scan"],
        energy_edges=[200, 3500] * u.GeV,
    )

    maps_ref = estimator_ref.run(dataset)
    maps = estimator.run(dataset)
    success = maps.success.data

    assert maps.stat_scan.geom.data_shape == (1, 109, 2, 2)
    ts = np.abs(maps["stat_scan"].data.min(axis=1))
    assert_allclose(ts[success], maps_ref.ts.data[success], rtol=1e-3)

    dnde_ref = maps.dnde_ref.squeeze()
    assert maps.dnde_scan_values.unit == dnde_ref.unit

    ind_best = maps.stat_scan.data.argmin(axis=1)
    ij, ik, il = np.indices(ind_best.shape)
    norm = maps.dnde_scan_values.data[ij, ind_best, ik, il] / dnde_ref.value
    assert_allclose(norm[success], maps_ref.norm.data[success], rtol=1e-5)

    stat_scan_aprrox = approximate_profile_map(maps, sqrt_ts_threshold_ul="ignore")
    ts_aprrox = np.abs(stat_scan_aprrox.data.min(axis=1))
    assert_allclose(ts_aprrox[success], maps_ref.ts.data[success], rtol=1e-3)

    stat_scan_aprrox = approximate_profile_map(maps, sqrt_ts_threshold_ul=None)
    ts_aprrox = np.abs(stat_scan_aprrox.data.min(axis=1))
    assert_allclose(ts_aprrox[success], maps_ref.ts.data[success], rtol=1e-3)

    maps_from_scan = get_flux_map_from_profile(maps)
    assert_allclose(
        maps_from_scan.ts.data[success], maps_ref.ts.data[success], rtol=1e-3
    )

    combined_map = combine_flux_maps([maps, maps], method="profile")
    assert_allclose(combined_map.ts.data, 2 * ts, rtol=1e-4)
    assert_allclose(combined_map.norm.data[success], norm[success], rtol=5e-2)

    maps1 = deepcopy(maps)
    combined_map = combine_flux_maps([maps, maps1], method="gaussian_errors")
    assert_allclose(combined_map.ts.data[success], 2 * ts[success], rtol=1e-4)
    assert_allclose(combined_map.norm.data[success], norm[success], rtol=5e-2)

    combined_map = combine_flux_maps([maps, maps1], method="distrib")
    assert_allclose(combined_map.ts.data, 2 * ts, rtol=1e-4)
    assert_allclose(combined_map.norm.data[success], norm[success], rtol=5e-2)

    combined_map = combine_flux_maps([maps, maps1], method="profile")
    assert_allclose(combined_map.ts.data, 2 * ts, rtol=1e-4)
    assert_allclose(combined_map.norm.data[success], norm[success], rtol=5e-2)

    maps1._reference_model.parameters["amplitude"].value = 1

    combined_map = combine_flux_maps([maps, maps1], method="distrib")
    assert_allclose(combined_map.ts.data, 2 * ts, rtol=1e-4)
    assert_allclose(combined_map.norm.data[success], norm[success], rtol=5e-2)

    combined_map = combine_flux_maps([maps, maps1], method="profile")
    assert_allclose(combined_map.ts.data, 2 * ts, rtol=1e-4)
    assert_allclose(combined_map.norm.data[success], norm[success], rtol=5e-2)

    assert_allclose(maps.norm.data[success], maps_ref.norm.data[success], rtol=1e-4)
    assert_allclose(
        maps.norm_errn.data[success], maps_ref.norm_errn.data[success], rtol=5e-2
    )
    assert_allclose(
        maps.norm_errp.data[success], maps_ref.norm_errp.data[success], rtol=5e-2
    )
    assert_allclose(
        maps.norm_ul.data[success], maps_ref.norm_ul.data[success], rtol=5e-2
    )
    with pytest.raises(ValueError):
        combine_flux_maps([maps, maps1], method="test")


def test_ts_map_with_model(fake_dataset):
    model = fake_dataset.models["source"]
    fake_dataset = fake_dataset.copy()

    fake_dataset.models = []

    estimator = TSMapEstimator(
        model,
        kernel_width="0.3 deg",
        selection_optional=["ul", "errn-errp"],
        energy_edges=[200, 3500] * u.GeV,
    )
    maps = estimator.run(fake_dataset)

    assert_allclose(maps["sqrt_ts"].data[:, 25, 25], 18.369942, atol=0.1)
    assert_allclose(maps["flux"].data[:, 25, 25], 3.513e-10, atol=1e-12)
    assert_allclose(maps["flux_err"].data[0, 0, 0], 2.413244e-11, rtol=1e-4)

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
    assert_allclose(maps["sqrt_ts"].data[:, 25, 25], -0.279392, atol=0.1)
    assert_allclose(maps["flux"].data[:, 25, 25], -2.015715e-13, atol=1e-12)


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


def test_MapDatasetOnOff_error():
    """Test raise error when applying TSMapEStimator to MapDatasetOnOff"""
    axis = MapAxis.from_edges([1, 10] * u.TeV, name="energy")
    geom = WcsGeom.create(width=1, axes=[axis])
    dataset_on_off = MapDatasetOnOff.create(geom)

    ts_estimator = TSMapEstimator()
    with pytest.raises(TypeError):
        ts_estimator.run(dataset=dataset_on_off)


@requires_data()
def test_with_TemplateSpatialModel():
    # Test for bug reported in 4920
    dataset = MapDataset.read("$GAMMAPY_DATA/cta-1dc-gc/cta-1dc-gc.fits.gz")
    dataset = dataset.downsample(10)
    filename = "$GAMMAPY_DATA/catalogs/fermi/Extended_archive_v18/Templates/RXJ1713_2016_250GeV.fits"
    model = TemplateSpatialModel.read(filename, normalize=False)
    model.position = SkyCoord(0, 0, unit="deg", frame="galactic")
    sky_model = SkyModel(spatial_model=model, spectral_model=PowerLawSpectralModel())
    dataset.models = sky_model
    estimator = TSMapEstimator(
        model=sky_model,
        energy_edges=[1.0, 5.0] * u.TeV,
        n_jobs=4,
    )

    result = estimator.run(dataset)
    assert_allclose(result["sqrt_ts"].data[0, 12, 16], 22.932, rtol=1e-3)


def test_joint_ts_map(fake_dataset):
    model = fake_dataset.models["source"]
    fake_dataset = fake_dataset.copy()

    fake_dataset.models = [model]

    estimator = TSMapEstimator(
        model=model, selection_optional=[], sum_over_energy_groups=True
    )
    assert estimator.sum_over_energy_groups

    result = estimator.run(fake_dataset)
    assert_allclose(result["npred_excess"].data.sum(), 902.403647, rtol=1e-3)
    assert_allclose(result["sqrt_ts"].data[0, 10, 10], 1.360219, rtol=1e-3)

    result = get_combined_significance_maps(estimator, [fake_dataset, fake_dataset])

    assert_allclose(result["npred_excess"].data.sum(), 2 * 902.403647, rtol=1e-3)
    assert_allclose(result["significance"].data[10, 10], 1.414529, rtol=1e-3)
    assert_allclose(
        result["df"].data, 2 * (~np.isnan(result["significance"].data)), rtol=1e-3
    )

    estimator = TSMapEstimator(
        model=model, threshold=1, selection_optional="all", sum_over_energy_groups=True
    )
    result = estimator.run([fake_dataset, fake_dataset.copy()])
    assert_allclose(result["sqrt_ts"].data[0, 10, 10], 1.92364, rtol=1e-3)


@requires_data()
def test_joint_ts_map_hawc():
    datasets = Datasets.read("$GAMMAPY_DATA/hawc/DL4/HAWC_pass4_public_Crab.yaml")
    datasets = Datasets(datasets[-2:])

    estimator = TSMapEstimator(kernel_width=2 * u.deg, sum_over_energy_groups=False)
    result = estimator.run(datasets)
    assert_allclose(result["flux"].data[0, 59, 59], 1.909396e-13, rtol=1e-3)
    assert_allclose(result["sqrt_ts"].data[0, 59, 59], 10.878956, rtol=1e-3)

    estimator = TSMapEstimator(kernel_width=2 * u.deg, sum_over_energy_groups=True)
    result = estimator.run(datasets)
    assert_allclose(result["flux"].data[0, 59, 59], 1.99452e-13, rtol=1e-3)
    assert_allclose(result["sqrt_ts"].data[0, 59, 59], 11.997135, rtol=1e-3)
