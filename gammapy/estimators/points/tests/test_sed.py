# Licensed under a 3-clause BSD style license - see LICENSE.rst
import pytest
import numpy as np
from numpy.testing import assert_allclose
from astropy import units as u
from astropy.coordinates import EarthLocation, SkyCoord
from astropy.table import Table
from gammapy.data import Observation
from gammapy.data.pointing import FixedPointingInfo
from gammapy.datasets import (
    Datasets,
    FluxPointsDataset,
    MapDataset,
    SpectrumDatasetOnOff,
)
from gammapy.datasets.spectrum import SpectrumDataset
from gammapy.estimators import FluxPoints, FluxPointsEstimator
from gammapy.irf import EDispKernelMap, EffectiveAreaTable2D, load_irf_dict_from_file
from gammapy.makers import MapDatasetMaker
from gammapy.makers.utils import make_map_exposure_true_energy
from gammapy.maps import MapAxis, RegionGeom, RegionNDMap, WcsGeom
from gammapy.modeling import Fit
from gammapy.modeling.models import (
    ExpCutoffPowerLawSpectralModel,
    FoVBackgroundModel,
    GaussianSpatialModel,
    Models,
    PiecewiseNormSpectralModel,
    PowerLawSpectralModel,
    SkyModel,
    TemplateNPredModel,
    TemplateSpatialModel,
)
from gammapy.utils import parallel
from gammapy.utils.testing import requires_data, requires_dependency


@pytest.fixture()
def fermi_datasets():
    filename = "$GAMMAPY_DATA/fermi-3fhl-crab/Fermi-LAT-3FHL_datasets.yaml"
    filename_models = "$GAMMAPY_DATA/fermi-3fhl-crab/Fermi-LAT-3FHL_models.yaml"
    return Datasets.read(filename=filename, filename_models=filename_models)


# TODO: use pre-generated data instead
def simulate_spectrum_dataset(model, random_state=0):
    energy_edges = np.logspace(-0.5, 1.5, 21) * u.TeV
    energy_axis = MapAxis.from_edges(energy_edges, interp="log", name="energy")
    energy_axis_true = energy_axis.copy(name="energy_true")

    aeff = EffectiveAreaTable2D.from_parametrization(energy_axis_true=energy_axis_true)

    bkg_model = SkyModel(
        spectral_model=PowerLawSpectralModel(
            index=2.5, amplitude="1e-12 cm-2 s-1 TeV-1"
        ),
        name="background",
    )
    bkg_model.spectral_model.amplitude.frozen = True
    bkg_model.spectral_model.index.frozen = True

    geom = RegionGeom.create(region="icrs;circle(0, 0, 0.1)", axes=[energy_axis])
    acceptance = RegionNDMap.from_geom(geom=geom, data=1)
    edisp = EDispKernelMap.from_diagonal_response(
        energy_axis=energy_axis,
        energy_axis_true=energy_axis_true,
        geom=geom,
    )

    geom_true = RegionGeom.create(
        region="icrs;circle(0, 0, 0.1)", axes=[energy_axis_true]
    )
    exposure = make_map_exposure_true_energy(
        pointing=SkyCoord("0d", "0d"), aeff=aeff, livetime=100 * u.h, geom=geom_true
    )

    mask_safe = RegionNDMap.from_geom(geom=geom, dtype=bool)
    mask_safe.data += True

    acceptance_off = RegionNDMap.from_geom(geom=geom, data=5)
    dataset = SpectrumDatasetOnOff(
        name="test_onoff",
        exposure=exposure,
        acceptance=acceptance,
        acceptance_off=acceptance_off,
        edisp=edisp,
        mask_safe=mask_safe,
    )
    dataset.models = bkg_model
    bkg_npred = dataset.npred_signal()

    dataset.models = model
    dataset.fake(
        random_state=random_state,
        npred_background=bkg_npred,
    )
    return dataset


def create_fpe(model):
    model = SkyModel(spectral_model=model, name="source")
    dataset = simulate_spectrum_dataset(model)
    energy_edges = [0.1, 1, 10, 100] * u.TeV
    dataset.models = model
    fpe = FluxPointsEstimator(
        energy_edges=energy_edges,
        source="source",
        selection_optional="all",
        fit=Fit(backend="minuit", optimize_opts=dict(tol=0.2, strategy=1)),
    )
    fpe.norm.scan_n_values = 11
    datasets = [dataset]
    return datasets, fpe


def simulate_map_dataset(random_state=0, name=None):
    irfs = load_irf_dict_from_file(
        "$GAMMAPY_DATA/cta-1dc/caldb/data/cta/1dc/bcf/South_z20_50h/irf_file.fits"
    )

    skydir = SkyCoord("0 deg", "0 deg", frame="galactic")
    pointing = FixedPointingInfo(fixed_icrs=skydir.icrs)
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

    obs = Observation.create(
        pointing=pointing,
        livetime=1 * u.h,
        irfs=irfs,
        location=EarthLocation(lon="-70d18m58.84s", lat="-24d41m0.34s", height="2000m"),
    )
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
        energy_edges=energy_edges,
        source="source",
        selection_optional="all",
    )
    fpe.norm.scan_n_values = 3

    return datasets, fpe


@pytest.fixture(scope="session")
def fpe_map_pwl_ray():
    """duplicate of fpe_map_pwl to avoid fails due to execution order"""
    dataset_1 = simulate_map_dataset(name="test-map-pwl")
    dataset_2 = dataset_1.copy(name="test-map-pwl-2")
    dataset_2.models = dataset_1.models

    dataset_2.mask_safe = RegionNDMap.from_geom(dataset_2.counts.geom, dtype=bool)

    energy_edges = [0.1, 1, 10, 100] * u.TeV
    datasets = [dataset_1, dataset_2]
    fpe = FluxPointsEstimator(
        energy_edges=energy_edges,
        source="source",
        selection_optional="all",
    )
    fpe.norm.scan_n_values = 3

    return datasets, fpe


@pytest.fixture(scope="session")
def fpe_map_pwl_reoptimize():
    dataset = simulate_map_dataset()
    energy_edges = [1, 10] * u.TeV
    dataset.models.parameters["lon_0"].frozen = True
    dataset.models.parameters["lat_0"].frozen = True
    dataset.models.parameters["sigma"].frozen = True
    datasets = [dataset]
    fpe = FluxPointsEstimator(
        energy_edges=energy_edges,
        reoptimize=True,
        source="source",
    )
    fpe.norm.scan_values = [0.8, 1, 1.2]
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


def test_run_pwl(fpe_pwl, tmpdir):
    datasets, fpe = fpe_pwl

    fp = fpe.run(datasets)
    table = fp.to_table()

    actual = table["e_min"].data
    assert_allclose(actual, [0.316228, 1.0, 10.0], rtol=1e-5)

    actual = table["e_max"].data
    assert_allclose(actual, [1.0, 10.0, 31.622777], rtol=1e-5)

    actual = table["e_ref"].data
    assert_allclose(actual, [0.562341, 3.162278, 17.782794], rtol=1e-3)

    actual = table["ref_flux"].quantity
    desired = [2.162278e-12, 9.000000e-13, 6.837722e-14] * u.Unit("1 / (cm2 s)")
    assert_allclose(actual, desired, rtol=1e-3)

    actual = table["ref_dnde"].quantity
    desired = [3.162278e-12, 1.000000e-13, 3.162278e-15] * u.Unit("1 / (cm2 s TeV)")
    assert_allclose(actual, desired, rtol=1e-3)

    actual = table["ref_eflux"].quantity
    desired = [1.151293e-12, 2.302585e-12, 1.151293e-12] * u.Unit("TeV / (cm2 s)")
    assert_allclose(actual, desired, rtol=1e-3)

    actual = table["norm"].data
    assert_allclose(actual, [1.081434, 0.91077, 0.922176], rtol=1e-3)

    actual = table["norm_err"].data
    assert_allclose(actual, [0.066374, 0.061025, 0.179729], rtol=1e-2)

    actual = table["norm_errn"].data
    assert_allclose(actual, [0.065803, 0.060403, 0.171376], rtol=1e-2)

    actual = table["norm_errp"].data
    assert_allclose(actual, [0.06695, 0.061652, 0.18839], rtol=1e-2)

    actual = table["counts"].data.squeeze()
    assert_allclose(actual, [1490, 748, 43])

    actual = table["norm_ul"].data
    assert_allclose(actual, [1.216227, 1.035472, 1.316878], rtol=1e-2)

    actual = table["sqrt_ts"].data
    assert_allclose(actual, [18.568429, 18.054651, 7.057121], rtol=1e-2)

    actual = table["ts"].data
    assert_allclose(actual, [344.7866, 325.9704, 49.8029], rtol=1e-2)

    actual = table["stat"].data
    assert_allclose(actual, [2.76495, 13.11912, 3.70128], rtol=1e-2)

    actual = table["stat_null"].data
    assert_allclose(actual, [347.55159, 339.08952, 53.50424], rtol=1e-2)

    actual = table["norm_scan"][0][[0, 5, -1]]
    assert_allclose(actual, [0.2, 1.0, 5.0])

    actual = table["stat_scan"][0][[0, 5, -1]]
    assert_allclose(actual, [220.369, 4.301, 1881.626], rtol=1e-2)

    actual = table["npred"].data
    assert_allclose(actual, [[1492.966], [749.459], [43.105]], rtol=1e-3)

    actual = table["npred_excess"].data
    assert_allclose(actual, [[660.5625], [421.5402], [34.3258]], rtol=1e-3)

    actual = table.meta["UL_CONF"]
    assert_allclose(actual, 0.9544997)

    npred_excess_err = fp.npred_excess_err.data.squeeze()
    assert_allclose(npred_excess_err, [40.541334, 28.244024, 6.690005], rtol=1e-3)

    npred_excess_errp = fp.npred_excess_errp.data.squeeze()
    assert_allclose(npred_excess_errp, [40.838806, 28.549508, 7.013377], rtol=1e-3)

    npred_excess_errn = fp.npred_excess_errn.data.squeeze()
    assert_allclose(npred_excess_errn, [40.247313, 27.932033, 6.378465], rtol=1e-3)

    npred_excess_ul = fp.npred_excess_ul.data.squeeze()
    assert_allclose(npred_excess_ul, [742.87486, 479.169719, 49.019125], rtol=1e-3)

    # test GADF I/O
    fp.write(tmpdir / "test.fits")
    fp_new = FluxPoints.read(tmpdir / "test.fits")
    assert fp_new.meta["sed_type_init"] == "likelihood"

    # test datasets stat
    fp_dataset = FluxPointsDataset(data=fp, models=fp.reference_model)
    fp_dataset.stat_type = "chi2"
    assert_allclose(fp_dataset.stat_sum(), 3.82374, rtol=1e-4)
    fp_dataset.stat_type = "profile"
    assert_allclose(fp_dataset.stat_sum(), 3.790053, rtol=1e-4)
    fp_dataset.stat_type = "distrib"
    assert_allclose(fp_dataset.stat_sum(), 3.783325, rtol=1e-4)


def test_run_ecpl(fpe_ecpl, tmpdir):
    datasets, fpe = fpe_ecpl

    fp = fpe.run(datasets)

    table = fp.to_table()

    actual = table["ref_flux"].quantity
    desired = [9.024362e-13, 1.781341e-13, 1.260298e-18] * u.Unit("1 / (cm2 s)")
    assert_allclose(actual, desired, rtol=1e-3)

    actual = table["ref_dnde"].quantity
    desired = [1.351382e-12, 7.527318e-15, 2.523659e-22] * u.Unit("1 / (cm2 s TeV)")
    assert_allclose(actual, desired, rtol=1e-3)

    actual = table["ref_eflux"].quantity
    desired = [4.770557e-13, 2.787695e-13, 1.371963e-17] * u.Unit("TeV / (cm2 s)")
    assert_allclose(actual, desired, rtol=1e-3)

    actual = table["norm"].data
    assert_allclose(actual, [1.001683, 1.061821, 1.237512e03], rtol=1e-3)

    actual = table["norm_err"].data
    assert_allclose(actual, [1.386091e-01, 2.394241e-01, 3.259756e03], rtol=1e-2)

    actual = table["norm_errn"].data
    assert_allclose(actual, [1.374962e-01, 2.361246e-01, 2.888978e03], rtol=1e-2)

    actual = table["norm_errp"].data
    assert_allclose(actual, [1.397358e-01, 2.428481e-01, 3.716550e03], rtol=1e-2)

    actual = table["norm_ul"].data
    assert_allclose(actual, [1.283433e00, 1.555117e00, 9.698645e03], rtol=1e-2)

    actual = table["sqrt_ts"].data
    assert_allclose(actual, [7.678454, 4.735691, 0.399243], rtol=1e-2)

    # test GADF I/O
    fp.write(tmpdir / "test.fits")
    fp_new = FluxPoints.read(tmpdir / "test.fits")
    assert fp_new.meta["sed_type_init"] == "likelihood"


@requires_data()
def test_run_map_pwl(fpe_map_pwl, tmpdir):
    datasets, fpe = fpe_map_pwl
    fp = fpe.run(datasets)

    table = fp.to_table()

    actual = table["e_min"].data
    assert_allclose(actual, [0.1, 1.178769, 8.48342], rtol=1e-5)

    actual = table["e_max"].data
    assert_allclose(actual, [1.178769, 8.483429, 100.0], rtol=1e-5)

    actual = table["e_ref"].data
    assert_allclose(actual, [0.343332, 3.162278, 29.126327], rtol=1e-5)

    actual = table["norm"].data
    assert_allclose(actual, [0.974726, 0.96342, 0.994251], rtol=1e-2)

    actual = table["norm_err"].data
    assert_allclose(actual, [0.067637, 0.052022, 0.087059], rtol=3e-2)

    actual = table["counts"].data
    assert_allclose(actual, [[44611, 0], [1923, 0], [282, 0]])

    actual = table["norm_ul"].data
    assert_allclose(actual, [1.111852, 1.07004, 1.17829], rtol=1e-2)

    actual = table["sqrt_ts"].data
    assert_allclose(actual, [16.681221, 28.408676, 21.91912], rtol=1e-2)

    actual = table["norm_scan"][0]
    assert_allclose(actual, [0.2, 1.0, 5])

    actual = table["stat_scan"][0] - table["stat"][0]
    assert_allclose(actual, [1.628398e02, 1.452456e-01, 2.008018e03], rtol=1e-2)

    # test GADF I/O
    fp.write(tmpdir / "test.fits")
    fp_new = FluxPoints.read(tmpdir / "test.fits")
    assert fp_new.meta["sed_type_init"] == "likelihood"


@requires_data()
def test_run_map_pwl_reoptimize(fpe_map_pwl_reoptimize):
    datasets, fpe = fpe_map_pwl_reoptimize
    fpe = fpe.copy()
    fpe.selection_optional = ["scan"]

    fp = fpe.run(datasets)
    table = fp.to_table()

    actual = table["norm"].data
    assert_allclose(actual, 0.962368, rtol=1e-2)

    actual = table["norm_err"].data
    assert_allclose(actual, 0.053878, rtol=1e-2)

    actual = table["sqrt_ts"].data
    assert_allclose(actual, 25.196585, rtol=1e-2)

    actual = table["norm_scan"][0]
    assert_allclose(actual, [0.8, 1, 1.2])

    actual = table["stat_scan"][0] - table["stat"][0]
    assert_allclose(actual, [9.788123, 0.486066, 17.603708], rtol=1e-2)


def test_run_no_edip(fpe_pwl, tmpdir):
    datasets, fpe = fpe_pwl

    datasets = datasets.copy()

    datasets[0].models["source"].apply_irf["edisp"] = False
    fp = fpe.run(datasets)
    table = fp.to_table()
    actual = table["norm"].data
    assert_allclose(actual, [1.081434, 0.91077, 0.922176], rtol=1e-3)

    datasets[0].edisp = None

    fp = fpe.run(datasets)
    table = fp.to_table()
    actual = table["norm"].data
    assert_allclose(actual, [1.081434, 0.91077, 0.922176], rtol=1e-3)

    datasets[0].models["source"].apply_irf["edisp"] = True
    fp = fpe.run(datasets)
    table = fp.to_table()
    actual = table["norm"].data
    assert_allclose(actual, [1.081434, 0.91077, 0.922176], rtol=1e-3)


@requires_dependency("iminuit")
@requires_data()
def test_run_template_npred(fpe_map_pwl, tmpdir):
    datasets, fpe = fpe_map_pwl
    dataset = datasets[0]
    models = Models(dataset.models)
    model = TemplateNPredModel(dataset.background, datasets_names=[dataset.name])
    models.append(model)
    dataset.models = models
    dataset.background.data = 0

    fp = fpe.run(dataset)

    table = fp.to_table()
    actual = table["norm"].data
    assert_allclose(actual, [0.974726, 0.96342, 0.994251], rtol=1e-2)

    fpe.sum_over_energy_groups = True
    fp = fpe.run(dataset)

    table = fp.to_table()
    actual = table["norm"].data
    assert_allclose(actual, [0.955512, 0.965328, 0.995237], rtol=1e-2)


@requires_data()
def test_reoptimize_no_free_parameters(fpe_pwl, caplog):
    datasets, fpe = fpe_pwl
    fpe.reoptimize = True
    with pytest.raises(ValueError, match="No free parameters for fitting"):
        fpe.run(datasets)
    fpe.reoptimize = False


@requires_data()
def test_flux_points_estimator_no_norm_scan(fpe_pwl, tmpdir):
    datasets, fpe = fpe_pwl
    fpe.selection_optional = None

    fp = fpe.run(datasets)

    assert_allclose(fpe.fit.optimize_opts["tol"], 0.2)

    assert fp.sed_type_init == "likelihood"
    assert "stat_scan" not in fp._data

    # test GADF I/O
    fp.write(tmpdir / "test.fits")
    fp_new = FluxPoints.read(tmpdir / "test.fits")
    assert fp_new.meta["sed_type_init"] == "likelihood"


def test_no_likelihood_contribution():
    dataset = simulate_spectrum_dataset(
        SkyModel(spectral_model=PowerLawSpectralModel(), name="source")
    )

    dataset_2 = dataset.slice_by_idx(slices={"energy": slice(0, 5)})

    dataset.mask_safe = RegionNDMap.from_geom(dataset.counts.geom, dtype=bool)

    fpe = FluxPointsEstimator(energy_edges=[1.0, 3.0, 10.0] * u.TeV, source="source")
    table = fpe.run([dataset, dataset_2]).to_table()

    assert np.isnan(table["norm"]).all()
    assert np.isnan(table["norm_err"]).all()
    assert_allclose(table["counts"], 0)


def test_mask_shape():
    axis = MapAxis.from_edges([1.0, 3.0, 10.0], unit="TeV", interp="log", name="energy")
    geom_1 = WcsGeom.create(binsz=1, width=3, axes=[axis])
    geom_2 = WcsGeom.create(binsz=1, width=5, axes=[axis])

    dataset_1 = MapDataset.create(geom_1)
    dataset_2 = MapDataset.create(geom_2)
    dataset_1.gti = None
    dataset_2.gti = None
    dataset_1.psf = None
    dataset_2.psf = None
    dataset_1.edisp = None
    dataset_2.edisp = None
    dataset_2.mask_safe = None

    model = SkyModel(
        spectral_model=PowerLawSpectralModel(),
        spatial_model=GaussianSpatialModel(),
        name="source",
    )

    dataset_1.models = model
    dataset_2.models = model

    fpe = FluxPointsEstimator(energy_edges=[1, 10] * u.TeV, source="source")

    fp = fpe.run([dataset_2, dataset_1])
    table = fp.to_table()

    assert_allclose(table["counts"], 0)
    assert_allclose(table["npred"], 0)


def test_run_pwl_parameter_range(fpe_pwl):
    pl = PowerLawSpectralModel(amplitude="1e-16 cm-2s-1TeV-1")

    datasets, fpe = create_fpe(pl)

    fp = fpe.run(datasets)

    table_no_bounds = fp.to_table()

    fpe.norm.min = 0
    fpe.norm.max = 1e4
    fp = fpe.run(datasets)
    table_with_bounds = fp.to_table()

    actual = table_with_bounds["norm"].data
    assert_allclose(actual, [0.0, 0.0, 0.0], atol=1e-2)

    actual = table_with_bounds["norm_errp"].data
    assert_allclose(actual, [212.593368, 298.383045, 449.951747], rtol=1e-2)

    actual = table_with_bounds["norm_ul"].data
    assert_allclose(actual, [640.067576, 722.571371, 1414.22209], rtol=1e-2)

    actual = table_with_bounds["sqrt_ts"].data
    assert_allclose(actual, [0.0, 0.0, 0.0], atol=1e-2)

    actual = table_no_bounds["norm"].data
    assert_allclose(actual, [-511.76675, -155.75408, -853.547117], rtol=1e-3)

    actual = table_no_bounds["norm_err"].data
    assert_allclose(actual, [504.601499, 416.69248, 851.223077], rtol=1e-2)

    actual = table_no_bounds["norm_ul"].data
    assert_allclose(actual, [514.957128, 707.888477, 1167.105962], rtol=1e-2)

    actual = table_no_bounds["sqrt_ts"].data
    assert_allclose(actual, [-1.006081, -0.364848, -0.927819], rtol=1e-2)


def test_flux_points_estimator_small_edges():
    pl = PowerLawSpectralModel(amplitude="1e-11 cm-2s-1TeV-1")

    datasets, fpe = create_fpe(pl)

    fpe.energy_edges = datasets[0].counts.geom.axes["energy"].upsample(3).edges[1:4]
    fpe.selection_optional = []

    fp = fpe.run(datasets)

    assert_allclose(fp.ts.data[0, 0, 0], 2156.96959291)
    assert np.isnan(fp.ts.data[1, 0, 0])
    assert np.isnan(fp.npred.data[1, 0, 0])


def test_flux_points_recompute_ul(fpe_pwl):
    datasets, fpe = fpe_pwl
    fpe.selection_optional = ["all"]
    fp = fpe.run(datasets)
    assert_allclose(
        fp.flux_ul.data,
        [[[2.629819e-12]], [[9.319243e-13]], [[9.004449e-14]]],
        rtol=1e-3,
    )
    fp1 = fp.recompute_ul(n_sigma_ul=4)
    assert_allclose(
        fp1.flux_ul.data,
        [[[2.92877891e-12]], [[1.04993236e-12]], [[1.22089744e-13]]],
        rtol=1e-3,
    )
    assert fp1.meta["n_sigma_ul"] == 4

    # check that it returns a sensible value
    fp2 = fp.recompute_ul(n_sigma_ul=2)
    assert_allclose(fp2.flux_ul.data, fp.flux_ul.data, rtol=1e-2)


def test_flux_points_parallel_multiprocessing(fpe_pwl):
    datasets, fpe = fpe_pwl
    fpe.selection_optional = ["all"]
    fpe.n_jobs = 2
    assert fpe.n_jobs == 2

    fp = fpe.run(datasets)
    assert_allclose(
        fp.flux_ul.data,
        [[[2.629819e-12]], [[9.319243e-13]], [[9.004449e-14]]],
        rtol=1e-3,
    )


def test_global_n_jobs_default_handling():
    fpe = FluxPointsEstimator(energy_edges=[1, 3, 10] * u.TeV)

    assert fpe.n_jobs == 1

    parallel.N_JOBS_DEFAULT = 2
    assert fpe.n_jobs == 2

    fpe.n_jobs = 5
    assert fpe.n_jobs == 5

    fpe.n_jobs = None
    assert fpe.n_jobs == 2
    assert fpe._n_jobs is None

    parallel.N_JOBS_DEFAULT = 1
    assert fpe.n_jobs == 1


@requires_dependency("ray")
def test_flux_points_parallel_ray(fpe_pwl):
    datasets, fpe = fpe_pwl
    fpe.selection_optional = ["all"]
    fpe.parallel_backend = "ray"
    fpe.n_jobs = 2
    fp = fpe.run(datasets)
    assert_allclose(
        fp.flux_ul.data,
        [[[2.629819e-12]], [[9.319243e-13]], [[9.004449e-14]]],
        rtol=1e-3,
    )


@requires_dependency("ray")
def test_flux_points_parallel_ray_actor_spectrum(fpe_pwl):
    from gammapy.datasets.actors import DatasetsActor

    datasets, fpe = fpe_pwl
    with pytest.raises(TypeError):
        DatasetsActor(datasets)


@requires_data()
@requires_dependency("ray")
def test_flux_points_parallel_ray_actor_map(fpe_map_pwl_ray):
    from gammapy.datasets.actors import DatasetsActor

    datasets, fpe = fpe_map_pwl_ray
    actors = DatasetsActor(datasets)

    fp = fpe.run(actors)

    table = fp.to_table()

    actual = table["e_min"].data
    assert_allclose(actual, [0.1, 1.178769, 8.48342], rtol=1e-5)

    actual = table["e_max"].data
    assert_allclose(actual, [1.178769, 8.483429, 100.0], rtol=1e-5)

    actual = table["e_ref"].data
    assert_allclose(actual, [0.343332, 3.162278, 29.126327], rtol=1e-5)

    actual = table["norm"].data
    assert_allclose(actual, [0.974726, 0.96342, 0.994251], rtol=1e-2)

    actual = table["norm_err"].data
    assert_allclose(actual, [0.067637, 0.052022, 0.087059], rtol=3e-2)

    actual = table["counts"].data
    assert_allclose(actual, [[44611, 0], [1923, 0], [282, 0]])

    actual = table["norm_ul"].data
    assert_allclose(actual, [1.111852, 1.07004, 1.17829], rtol=1e-2)

    actual = table["sqrt_ts"].data
    assert_allclose(actual, [16.681221, 28.408676, 21.91912], rtol=1e-2)

    actual = table["norm_scan"][0]
    assert_allclose(actual, [0.2, 1.0, 5])

    actual = table["stat_scan"][0] - table["stat"][0]
    assert_allclose(actual, [1.628398e02, 1.452456e-01, 2.008018e03], rtol=1e-2)


def test_fpe_non_aligned_energy_axes():
    energy_axis = MapAxis.from_energy_bounds("1 TeV", "10 TeV", nbin=10)
    geom_1 = RegionGeom.create("icrs;circle(0, 0, 0.1)", axes=[energy_axis])
    dataset_1 = SpectrumDataset.create(geom=geom_1)

    energy_axis = MapAxis.from_energy_bounds("1 TeV", "10 TeV", nbin=7)
    geom_2 = RegionGeom.create("icrs;circle(0, 0, 0.1)", axes=[energy_axis])
    dataset_2 = SpectrumDataset.create(geom=geom_2)

    fpe = FluxPointsEstimator(energy_edges=[1, 3, 10] * u.TeV)

    with pytest.raises(ValueError, match="must have aligned energy axes"):
        fpe.run(datasets=[dataset_1, dataset_2])


def test_fpe_non_uniform_datasets():
    energy_axis = MapAxis.from_energy_bounds("1 TeV", "10 TeV", nbin=10)
    geom_1 = RegionGeom.create("icrs;circle(0, 0, 0.1)", axes=[energy_axis])
    dataset_1 = SpectrumDataset.create(
        geom=geom_1, meta_table=Table({"TELESCOP": ["CTA"]})
    )

    energy_axis = MapAxis.from_energy_bounds("1 TeV", "10 TeV", nbin=10)
    geom_2 = RegionGeom.create("icrs;circle(0, 0, 0.1)", axes=[energy_axis])
    dataset_2 = SpectrumDataset.create(
        geom=geom_2, meta_table=Table({"TELESCOP": ["CTB"]})
    )

    fpe = FluxPointsEstimator(energy_edges=[1, 3, 10] * u.TeV)

    with pytest.raises(ValueError, match="same value of the 'TELESCOP' meta keyword"):
        fpe.run(datasets=[dataset_1, dataset_2])


@requires_data()
def test_flux_points_estimator_norm_spectral_model(fermi_datasets):
    energy_edges = [10, 30, 100, 300, 1000] * u.GeV

    model_ref = fermi_datasets.models["Crab Nebula"]
    estimator = FluxPointsEstimator(
        energy_edges=energy_edges,
        source="Crab Nebula",
        selection_optional=[],
        reoptimize=True,
    )
    flux_points = estimator.run(fermi_datasets[0])
    flux_points_dataset = FluxPointsDataset(data=flux_points, models=model_ref)
    flux_pred_ref = flux_points_dataset.flux_pred()

    models = Models([model_ref])
    geom = fermi_datasets[0].exposure.geom.to_image()
    energy_axis = MapAxis.from_energy_bounds(
        3 * u.GeV, 1.7 * u.TeV, nbin=30, per_decade=True, name="energy_true"
    )
    geom = geom.to_cube([energy_axis])

    model = models.to_template_sky_model(geom, name="test")
    fermi_datasets.models = [fermi_datasets[0].background_model, model]
    estimator = FluxPointsEstimator(
        energy_edges=energy_edges, source="test", selection_optional=[], reoptimize=True
    )
    flux_points = estimator.run(fermi_datasets[0])

    flux_points_dataset = FluxPointsDataset(data=flux_points, models=model)
    flux_pred = flux_points_dataset.flux_pred()
    assert_allclose(flux_pred, flux_pred_ref, rtol=2e-4)

    # test model 2d
    norms = (
        model.spatial_model.map.data.sum(axis=(1, 2))
        / model.spatial_model.map.data.sum()
    )
    model.spatial_model = TemplateSpatialModel(
        model.spatial_model.map.reduce_over_axes(), normalize=False
    )
    model.spectral_model = PiecewiseNormSpectralModel(geom.axes[0].center, norms)
    flux_points_dataset = FluxPointsDataset(data=flux_points, models=model)
    flux_pred = flux_points_dataset.flux_pred()
    assert_allclose(flux_pred, flux_pred_ref, rtol=2e-4)


@requires_data()
def test_fpe_diff_lengths():
    dataset = SpectrumDatasetOnOff.read(
        "$GAMMAPY_DATA/joint-crab/spectra/hess/pha_obs23523.fits"
    )
    dataset1 = SpectrumDatasetOnOff.read(
        "$GAMMAPY_DATA/joint-crab/spectra/hess/pha_obs23559.fits"
    )

    dataset.meta_table = Table(names=["NAME", "TELESCOP"], data=[["23523"], ["hess"]])
    dataset1.meta_table = Table(names=["NAME", "TELESCOP"], data=[["23559"], ["hess"]])
    dataset2 = Datasets([dataset, dataset1]).stack_reduce(name="dataset2")

    dataset3 = dataset1.copy()
    dataset3.meta_table = None

    pwl = PowerLawSpectralModel()

    datasets = Datasets([dataset1, dataset2, dataset3])

    datasets.models = SkyModel(spectral_model=pwl, name="test")
    energy_edges = [1, 2, 4, 10] * u.TeV
    fpe = FluxPointsEstimator(energy_edges=energy_edges, source="test")

    fp = fpe.run(datasets)

    assert_allclose(
        fp.dnde.data,
        [[[2.034323e-11]], [[3.39049716e-12]], [[2.96231326e-13]]],
        rtol=1e-3,
    )

    dataset4 = dataset1.copy()
    dataset4.meta_table = Table(names=["NAME", "TELESCOP"], data=[["23523"], ["not"]])
    datasets = Datasets([dataset1, dataset2, dataset3, dataset4])
    with pytest.raises(ValueError):
        fp = fpe.run(datasets)
