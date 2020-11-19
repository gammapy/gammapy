# Licensed under a 3-clause BSD style license - see LICENSE.rst
import pytest
import numpy as np
from numpy.testing import assert_allclose
from astropy import units as u
from astropy.coordinates import SkyCoord
from gammapy.data import Observation
from gammapy.datasets import MapDataset, SpectrumDatasetOnOff
from gammapy.estimators import FluxPointsEstimator
from gammapy.irf import EDispKernelMap, EffectiveAreaTable, load_cta_irfs
from gammapy.makers import MapDatasetMaker
from gammapy.maps import MapAxis, RegionGeom, RegionNDMap, WcsGeom
from gammapy.modeling.models import (
    ExpCutoffPowerLawSpectralModel,
    FoVBackgroundModel,
    GaussianSpatialModel,
    PowerLawSpectralModel,
    SkyModel,
)
from gammapy.utils.testing import requires_data, requires_dependency


# TODO: use pregenerate data instead
def simulate_spectrum_dataset(model, random_state=0):
    energy_edges = np.logspace(-0.5, 1.5, 21) * u.TeV
    energy_axis = MapAxis.from_edges(energy_edges, interp="log", name="energy")

    aeff = EffectiveAreaTable.from_parametrization(energy=energy_edges).to_region_map()
    bkg_model = SkyModel(
        spectral_model=PowerLawSpectralModel(
            index=2.5, amplitude="1e-12 cm-2 s-1 TeV-1"
        ),
        name="background",
    )
    bkg_model.spectral_model.amplitude.frozen = True
    bkg_model.spectral_model.index.frozen = True

    geom = RegionGeom(region=None, axes=[energy_axis])
    acceptance = RegionNDMap.from_geom(geom=geom, data=1)
    edisp = EDispKernelMap.from_diagonal_response(
        energy_axis=energy_axis,
        energy_axis_true=energy_axis.copy(name="energy_true"),
        geom=geom,
    )

    livetime = 100 * u.h
    exposure = aeff * livetime

    dataset = SpectrumDatasetOnOff(
        name="test_onoff",
        exposure=exposure,
        acceptance=acceptance,
        acceptance_off=5,
        edisp=edisp,
    )
    dataset.models = bkg_model
    bkg_npred = dataset.npred_signal()

    dataset.models = model
    dataset.fake(
        random_state=random_state, npred_background=bkg_npred,
    )
    return dataset


def create_fpe(model):
    model = SkyModel(spectral_model=model, name="source")
    dataset = simulate_spectrum_dataset(model)
    energy_edges = [0.1, 1, 10, 100] * u.TeV
    dataset.models = model
    fpe = FluxPointsEstimator(
        energy_edges=energy_edges, norm_n_values=11, source="source"
    )
    datasets = [dataset]
    return datasets, fpe


def simulate_map_dataset(random_state=0, name=None):
    irfs = load_cta_irfs(
        "$GAMMAPY_DATA/cta-1dc/caldb/data/cta/1dc/bcf/South_z20_50h/irf_file.fits"
    )

    skydir = SkyCoord("0 deg", "0 deg", frame="galactic")
    energy_edges = np.logspace(-1, 2, 15) * u.TeV
    energy_axis = MapAxis.from_edges(edges=energy_edges, name="energy", interp="log")

    geom = WcsGeom.create(
        skydir=skydir, width=(4, 4), binsz=0.1, axes=[energy_axis], frame="galactic"
    )

    gauss = GaussianSpatialModel(
        lon_0="0 deg", lat_0="0 deg", sigma="0.4 deg", frame="galactic"
    )
    pwl = PowerLawSpectralModel(amplitude="1e-11 cm-2 s-1 TeV-1")
    skymodel = SkyModel(spatial_model=gauss, spectral_model=pwl, name="source")

    obs = Observation.create(pointing=skydir, livetime=1 * u.h, irfs=irfs)
    empty = MapDataset.create(geom, name=name)
    maker = MapDatasetMaker(selection=["exposure", "background", "psf", "edisp"])
    dataset = maker.run(empty, obs)

    bkg_model = FoVBackgroundModel(dataset_name=dataset.name)

    dataset.models = [bkg_model, skymodel]
    dataset.fake(random_state=random_state)
    return dataset


@pytest.fixture(scope="session")
def fpe_map_pwl():
    dataset_1 = simulate_map_dataset(name="test-map-pwl")
    dataset_2 = dataset_1.copy(name="test-map-pwl-2")
    dataset_2.models = dataset_1.models

    dataset_2.mask_safe = RegionNDMap.from_geom(dataset_2.counts.geom, dtype=bool)

    energy_edges = [0.1, 1, 10, 100] * u.TeV
    datasets = [dataset_1, dataset_2]
    fpe = FluxPointsEstimator(
        energy_edges=energy_edges, norm_n_values=3, source="source"
    )
    return datasets, fpe


@pytest.fixture(scope="session")
def fpe_map_pwl_reoptimize():
    dataset = simulate_map_dataset()
    energy_edges = [1, 10] * u.TeV
    dataset.models.parameters["lon_0"].frozen = True
    dataset.models.parameters["lat_0"].frozen = True
    #    dataset.models.parameters["index"].frozen = True
    dataset.models.parameters["sigma"].frozen = True
    datasets = [dataset]
    fpe = FluxPointsEstimator(
        energy_edges=energy_edges, norm_values=[1], reoptimize=True, source="source"
    )
    return datasets, fpe


@pytest.fixture(scope="session")
def fpe_pwl():
    return create_fpe(PowerLawSpectralModel())


@pytest.fixture(scope="session")
def fpe_ecpl():
    return create_fpe(ExpCutoffPowerLawSpectralModel(lambda_="1 TeV-1"))


def test_str(fpe_pwl):
    datasets, fpe = fpe_pwl
    assert "FluxPointsEstimator" in str(fpe)


@requires_dependency("iminuit")
def test_run_pwl(fpe_pwl):
    datasets, fpe = fpe_pwl

    fp = fpe.run(datasets)

    actual = fp.table["e_min"].data
    assert_allclose(actual, [0.316228, 1.0, 10.0], rtol=1e-5)

    actual = fp.table["e_max"].data
    assert_allclose(actual, [1.0, 10.0, 31.622777], rtol=1e-5)

    actual = fp.table["e_ref"].data
    assert_allclose(actual, [0.562341, 3.162278, 17.782794], rtol=1e-3)

    actual = fp.table["ref_flux"].quantity
    desired = [2.162278e-12, 9.000000e-13, 6.837722e-14] * u.Unit("1 / (cm2 s)")
    assert_allclose(actual, desired, rtol=1e-3)

    actual = fp.table["ref_dnde"].quantity
    desired = [3.162278e-12, 1.000000e-13, 3.162278e-15] * u.Unit("1 / (cm2 s TeV)")
    assert_allclose(actual, desired, rtol=1e-3)

    actual = fp.table["ref_e2dnde"].quantity
    assert_allclose(actual, 1e-12 * u.Unit("TeV / (cm2 s)"), rtol=1e-3)

    actual = fp.table["ref_eflux"].quantity
    desired = [1.151293e-12, 2.302585e-12, 1.151293e-12] * u.Unit("TeV / (cm2 s)")
    assert_allclose(actual, desired, rtol=1e-3)

    actual = fp.table["norm"].data
    assert_allclose(actual, [1.081434, 0.91077, 0.922176], rtol=1e-3)

    actual = fp.table["norm_err"].data
    assert_allclose(actual, [0.066374, 0.061025, 0.179729], rtol=1e-2)

    actual = fp.table["norm_errn"].data
    assert_allclose(actual, [0.065803, 0.060403, 0.171376], rtol=1e-2)

    actual = fp.table["norm_errp"].data
    assert_allclose(actual, [0.06695, 0.061652, 0.18839], rtol=1e-2)

    actual = fp.table["counts"].data.squeeze()
    assert_allclose(actual, [1490, 748, 43])

    actual = fp.table["norm_ul"].data
    assert_allclose(actual, [1.216227, 1.035472, 1.316878], rtol=1e-2)

    actual = fp.table["sqrt_ts"].data
    assert_allclose(actual, [18.568429, 18.054651, 7.057121], rtol=1e-2)

    actual = fp.table["norm_scan"][0][[0, 5, -1]]
    assert_allclose(actual, [0.2, 1, 5])

    actual = fp.table["stat_scan"][0][[0, 5, -1]]
    assert_allclose(actual, [220.368653, 4.301011, 1881.626454], rtol=1e-2)


@requires_dependency("iminuit")
def test_run_ecpl(fpe_ecpl):
    datasets, fpe = fpe_ecpl

    fp = fpe.run(datasets)

    actual = fp.table["ref_flux"].quantity
    desired = [9.024362e-13, 1.781341e-13, 1.260298e-18] * u.Unit("1 / (cm2 s)")
    assert_allclose(actual, desired, rtol=1e-3)

    actual = fp.table["ref_dnde"].quantity
    desired = [1.351382e-12, 7.527318e-15, 2.523659e-22] * u.Unit("1 / (cm2 s TeV)")
    assert_allclose(actual, desired, rtol=1e-3)

    actual = fp.table["ref_e2dnde"].quantity
    desired = [4.273446e-13, 7.527318e-14, 7.980510e-20] * u.Unit("TeV / (cm2 s)")
    assert_allclose(actual, desired, rtol=1e-3)

    actual = fp.table["ref_eflux"].quantity
    desired = [4.770557e-13, 2.787695e-13, 1.371963e-17] * u.Unit("TeV / (cm2 s)")
    assert_allclose(actual, desired, rtol=1e-3)

    actual = fp.table["norm"].data
    assert_allclose(actual, [1.001683, 1.061821, 1.237512e03], rtol=1e-3)

    actual = fp.table["norm_err"].data
    assert_allclose(actual, [1.386091e-01, 2.394241e-01, 3.259756e03], rtol=1e-2)

    actual = fp.table["norm_errn"].data
    assert_allclose(actual, [1.374962e-01, 2.361246e-01, 2.888978e03], rtol=1e-2)

    actual = fp.table["norm_errp"].data
    assert_allclose(actual, [1.397358e-01, 2.428481e-01, 3.716550e03], rtol=1e-2)

    actual = fp.table["norm_ul"].data
    assert_allclose(actual, [1.283433e00, 1.555117e00, 9.698645e03], rtol=1e-2)

    actual = fp.table["sqrt_ts"].data
    assert_allclose(actual, [7.678454, 4.735691, 0.399243], rtol=1e-2)


@requires_dependency("iminuit")
@requires_data()
def test_run_map_pwl(fpe_map_pwl):
    datasets, fpe = fpe_map_pwl
    fp = fpe.run(datasets)

    actual = fp.table["e_min"].data
    assert_allclose(actual, [0.1, 1.178769, 8.48342], rtol=1e-5)

    actual = fp.table["e_max"].data
    assert_allclose(actual, [1.178769, 8.483429, 100.0], rtol=1e-5)

    actual = fp.table["e_ref"].data
    assert_allclose(actual, [0.343332, 3.162278, 29.126327], rtol=1e-5)

    actual = fp.table["norm"].data
    assert_allclose(actual, [0.974726, 0.96342, 0.994251], rtol=1e-2)

    actual = fp.table["norm_err"].data
    assert_allclose(actual, [0.067637, 0.052022, 0.087059], rtol=3e-2)

    actual = fp.table["counts"].data
    assert_allclose(actual, [[44611, 0], [1923, 0], [282, 0]])

    actual = fp.table["norm_ul"].data
    assert_allclose(actual, [1.111852, 1.07004, 1.17829], rtol=1e-2)

    actual = fp.table["sqrt_ts"].data
    assert_allclose(actual, [16.681221, 28.408676, 21.91912], rtol=1e-2)

    actual = fp.table["norm_scan"][0]
    assert_allclose(actual, [0.2, 1, 5])

    actual = fp.table["stat_scan"][0] - fp.table["stat"][0]
    assert_allclose(actual, [1.628530e02, 1.436323e-01, 2.007461e03], rtol=1e-2)


@requires_dependency("iminuit")
@requires_data()
def test_run_map_pwl_reoptimize(fpe_map_pwl_reoptimize):
    datasets, fpe = fpe_map_pwl_reoptimize
    fpe = fpe.copy()
    fpe.selection = ["scan"]

    fp = fpe.run(datasets)

    actual = fp.table["norm"].data
    assert_allclose(actual, 0.962368, rtol=1e-2)

    actual = fp.table["norm_err"].data
    assert_allclose(actual, 0.051955, rtol=1e-2)

    actual = fp.table["sqrt_ts"].data
    assert_allclose(actual, 28.408426, rtol=1e-2)

    actual = fp.table["norm_scan"][0]
    assert_allclose(actual, 1)

    actual = fp.table["stat_scan"][0] - fp.table["stat"][0]
    assert_allclose(actual, 0.489359, rtol=1e-2)


@requires_dependency("iminuit")
@requires_data()
def test_flux_points_estimator_no_norm_scan(fpe_pwl):
    datasets, fpe = fpe_pwl
    fpe.selection_optional = None

    fp = fpe.run(datasets)

    assert fp.sed_type == "dnde"
    assert "norm_scan" not in fp.table.colnames


def test_no_likelihood_contribution():
    dataset = simulate_spectrum_dataset(
        SkyModel(spectral_model=PowerLawSpectralModel(), name="source")
    )

    dataset_2 = dataset.slice_by_idx(slices={"energy": slice(0, 5)})

    dataset.mask_safe = RegionNDMap.from_geom(dataset.counts.geom, dtype=bool)

    fpe = FluxPointsEstimator(energy_edges=[1, 3, 10] * u.TeV, source="source")
    fp = fpe.run([dataset, dataset_2])

    assert np.isnan(fp.table["norm"]).all()
    assert np.isnan(fp.table["norm_err"]).all()
    assert np.isnan(fp.table["norm_ul"]).all()
    assert np.isnan(fp.table["norm_scan"]).all()
    assert_allclose(fp.table["counts"], 0)


def test_mask_shape():
    axis = MapAxis.from_edges([1, 3, 10], unit="TeV", interp="log", name="energy")
    geom_1 = WcsGeom.create(binsz=1, width=3, axes=[axis])
    geom_2 = WcsGeom.create(binsz=1, width=5, axes=[axis])

    dataset_1 = MapDataset.create(geom_1)
    dataset_2 = MapDataset.create(geom_2)
    dataset_1.psf = None
    dataset_2.psf = None
    dataset_1.edisp = None
    dataset_2.edisp = None

    model = SkyModel(
        spectral_model=PowerLawSpectralModel(),
        spatial_model=GaussianSpatialModel(),
        name="source",
    )

    dataset_1.models = model
    dataset_2.models = model

    fpe = FluxPointsEstimator(energy_edges=[1, 10] * u.TeV, source="source")

    fp = fpe.run([dataset_2, dataset_1])

    assert_allclose(fp.table["counts"], 0)
