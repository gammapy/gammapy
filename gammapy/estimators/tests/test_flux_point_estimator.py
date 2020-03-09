# Licensed under a 3-clause BSD style license - see LICENSE.rst
import pytest
import numpy as np
from numpy.testing import assert_allclose
from astropy import units as u
from astropy.coordinates import SkyCoord
from gammapy.data import Observation
from gammapy.datasets import MapDataset, SpectrumDatasetOnOff
from gammapy.irf import EffectiveAreaTable, load_cta_irfs
from gammapy.makers import MapDatasetMaker
from gammapy.maps import MapAxis, WcsGeom, RegionGeom, RegionNDMap
from gammapy.modeling.models import (
    ExpCutoffPowerLawSpectralModel,
    GaussianSpatialModel,
    PowerLawSpectralModel,
    SkyModel,
)
from gammapy.estimators import FluxPointsEstimator
from gammapy.utils.testing import requires_data, requires_dependency


# TODO: use pregenerate data instead
def simulate_spectrum_dataset(model, random_state=0):
    edges = np.logspace(-0.5, 1.5, 21) * u.TeV
    energy_axis = MapAxis.from_edges(edges, interp="log", name="energy")

    aeff = EffectiveAreaTable.from_parametrization(energy=edges)
    bkg_model = SkyModel(
        spectral_model=PowerLawSpectralModel(
            index=2.5, amplitude="1e-12 cm-2 s-1 TeV-1"
        ),
        name="background"
    )
    bkg_model.spectral_model.amplitude.frozen = True
    bkg_model.spectral_model.index.frozen = True

    geom = RegionGeom(region=None, axes=[energy_axis])
    acceptance = RegionNDMap.from_geom(geom=geom, data=1)

    dataset = SpectrumDatasetOnOff(
        aeff=aeff, livetime=100 * u.h, acceptance=acceptance, acceptance_off=5
    )
    dataset.models = bkg_model
    bkg_npred = dataset.npred_sig()

    dataset.models = model
    dataset.fake(random_state=random_state, background_model=bkg_npred)
    return dataset


def create_fpe(model):
    model = SkyModel(spectral_model=model, name="source")
    dataset = simulate_spectrum_dataset(model)
    e_edges = [0.1, 1, 10, 100] * u.TeV
    dataset.models = model
    return FluxPointsEstimator(
        datasets=[dataset], e_edges=e_edges, norm_n_values=11, source="source"
    )


def simulate_map_dataset(random_state=0, name=None):
    irfs = load_cta_irfs(
        "$GAMMAPY_DATA/cta-1dc/caldb/data/cta/1dc/bcf/South_z20_50h/irf_file.fits"
    )

    skydir = SkyCoord("0 deg", "0 deg", frame="galactic")
    edges = np.logspace(-1, 2, 15) * u.TeV
    energy_axis = MapAxis.from_edges(edges=edges, name="energy", interp="log")

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

    dataset.models.append(skymodel)
    dataset.fake(random_state=random_state)
    return dataset


@pytest.fixture(scope="session")
def fpe_map_pwl():
    dataset_1 = simulate_map_dataset()
    dataset_2 = dataset_1.copy()
    dataset_2.mask_safe = RegionNDMap.from_geom(dataset_2.counts.geom, dtype=bool)

    e_edges = [0.1, 1, 10, 100] * u.TeV
    return FluxPointsEstimator(
        datasets=[dataset_1, dataset_2],
        e_edges=e_edges,
        norm_n_values=3,
        source="source",
    )


@pytest.fixture(scope="session")
def fpe_map_pwl_reoptimize():
    dataset = simulate_map_dataset()
    e_edges = [1, 10] * u.TeV
    dataset.models.parameters["lon_0"].frozen = True
    dataset.models.parameters["lat_0"].frozen = True
#    dataset.models.parameters["index"].frozen = True
    dataset.models.parameters["sigma"].frozen = True
    return FluxPointsEstimator(
        datasets=[dataset],
        e_edges=e_edges,
        norm_values=[1],
        reoptimize=True,
        source="source",
    )


@pytest.fixture(scope="session")
def fpe_pwl():
    return create_fpe(PowerLawSpectralModel())


@pytest.fixture(scope="session")
def fpe_ecpl():
    return create_fpe(ExpCutoffPowerLawSpectralModel(lambda_="1 TeV-1"))


class TestFluxPointsEstimator:
    @staticmethod
    def test_str(fpe_pwl):
        assert "FluxPointsEstimator" in str(fpe_pwl)

    @pytest.mark.xfail
    @staticmethod
    @requires_dependency("iminuit")
    def test_run_pwl(fpe_pwl):
        fp = fpe_pwl.run()
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

    @staticmethod
    @requires_dependency("iminuit")
    def test_run_ecpl(fpe_ecpl):
        fp = fpe_ecpl.estimate_flux_point(fpe_ecpl.e_groups[1])
        assert_allclose(fp["norm"], 1, rtol=1e-1)

    @staticmethod
    @requires_dependency("iminuit")
    @requires_data()
    def test_run_map_pwl(fpe_map_pwl):
        fp = fpe_map_pwl.run()

        actual = fp.table["norm"].data
        assert_allclose(actual, [1.009629, 0.930886, 0.958678], rtol=1e-2)

        actual = fp.table["norm_err"].data
        assert_allclose(actual, [0.067375, 0.051074, 0.093575], rtol=3e-2)

        actual = fp.table["counts"].data
        assert_allclose(actual, [[44638, 0], [1898, 0], [258, 0]])

        actual = fp.table["norm_ul"].data
        assert_allclose(actual, [1.14644, 1.035613, 1.156623], rtol=1e-2)

        actual = fp.table["sqrt_ts"].data
        assert_allclose(actual, [17.475586, 27.891336, 19.149109], rtol=1e-2)

        actual = fp.table["norm_scan"][0]
        assert_allclose(actual, [0.2, 1, 5])

        actual = fp.table["stat_scan"][0] - fp.table["stat"][0]
        assert_allclose(actual, [1.816254e02, 2.037276e-02, 1.992444e03], rtol=1e-2)

    @staticmethod
    @requires_dependency("iminuit")
    @requires_data()
    def test_run_map_pwl_reoptimize(fpe_map_pwl_reoptimize):
        fp = fpe_map_pwl_reoptimize.run(steps=["err", "norm-scan", "ts"])

        actual = fp.table["norm"].data
        assert_allclose(actual, 0.932607, rtol=1e-3)

        actual = fp.table["norm_err"].data
        assert_allclose(actual, 0.052935, rtol=1e-2)

        actual = fp.table["sqrt_ts"].data
        assert_allclose(actual, 24.927798, rtol=1e-2)

        actual = fp.table["norm_scan"][0]
        assert_allclose(actual, 1)

        actual = fp.table["stat_scan"][0] - fp.table["stat"][0]
        assert_allclose(actual, 1.578256, rtol=1e-2)


def test_no_likelihood_contribution():
    dataset = simulate_spectrum_dataset(
        SkyModel(spectral_model=PowerLawSpectralModel(), name="source")
    )
    dataset.mask_safe = RegionNDMap.from_geom(dataset.counts.geom, dtype=bool)

    fpe = FluxPointsEstimator([dataset], e_edges=[1, 3, 10] * u.TeV, source="source")
    fp = fpe.run()

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

    fpe = FluxPointsEstimator(
        datasets=[dataset_2, dataset_1], e_edges=[1, 10] * u.TeV, source="source"
    )

    fp = fpe.run()

    assert_allclose(fp.table["counts"], 0)
