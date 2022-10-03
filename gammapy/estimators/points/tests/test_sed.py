# Licensed under a 3-clause BSD style license - see LICENSE.rst
import pytest
import numpy as np
from numpy.testing import assert_allclose
from astropy import units as u
from astropy.coordinates import EarthLocation, SkyCoord
from astropy.table import Table
from gammapy.data import Observation
from gammapy.datasets import MapDataset, SpectrumDatasetOnOff
from gammapy.datasets.spectrum import SpectrumDataset
from gammapy.estimators import FluxPoints, FluxPointsEstimator
from gammapy.irf import EDispKernelMap, EffectiveAreaTable2D, load_cta_irfs
from gammapy.makers import MapDatasetMaker
from gammapy.makers.utils import make_map_exposure_true_energy
from gammapy.maps import MapAxis, RegionGeom, RegionNDMap, WcsGeom
from gammapy.modeling import Fit
from gammapy.modeling.models import (
    ExpCutoffPowerLawSpectralModel,
    FoVBackgroundModel,
    GaussianSpatialModel,
    PowerLawSpectralModel,
    SkyModel,
)
from gammapy.utils.testing import requires_data


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
        norm_n_values=11,
        source="source",
        selection_optional="all",
        fit=Fit(backend="minuit", optimize_opts=dict(tol=0.2, strategy=1)),
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

    obs = Observation.create(
        pointing=skydir,
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
        norm_n_values=3,
        source="source",
        selection_optional="all",
    )
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
        norm_values=[0.8, 1, 1.2],
        reoptimize=True,
        source="source",
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
    fp.write(tmpdir / "test.fits", format="gadf-sed")
    fp_new = FluxPoints.read(tmpdir / "test.fits")
    assert fp_new.meta["sed_type_init"] == "likelihood"


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
    fp.write(tmpdir / "test.fits", format="gadf-sed")
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
    fp.write(tmpdir / "test.fits", format="gadf-sed")
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
    assert_allclose(fpe.fit.minuit.tol, 0.2)

    assert fp.sed_type_init == "likelihood"
    assert "stat_scan" not in fp._data

    # test GADF I/O
    fp.write(tmpdir / "test.fits", format="gadf-sed")
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

    pl.amplitude.min = 0
    pl.amplitude.max = 1e-12

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
        [[[2.93166296e-12]], [[1.05421128e-12]], [[1.22660055e-13]]],
        rtol=1e-3,
    )
    assert fp1.meta["n_sigma_ul"] == 4

    # check that it returns a sensible value
    fp2 = fp.recompute_ul(n_sigma_ul=2)
    assert_allclose(fp2.flux_ul.data, fp.flux_ul.data, rtol=1e-2)


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
