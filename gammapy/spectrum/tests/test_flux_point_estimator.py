# Licensed under a 3-clause BSD style license - see LICENSE.rst
import pytest
import numpy as np
from numpy.testing import assert_allclose
from astropy import units as u
from astropy.coordinates import SkyCoord
from ...utils.testing import requires_dependency
from ...irf import EffectiveAreaTable, load_cta_irfs
from ..models import PowerLaw, ExponentialCutoffPowerLaw
from ..simulation import SpectrumSimulation
from ..flux_point import FluxPointsEstimator
from ...cube import simulate_dataset
from ...cube.models import SkyModel
from ...image.models import SkyGaussian
from ...maps import MapAxis, WcsGeom


# TODO: use pregenerate data instead
def simulate_spectrum_dataset(model):
    energy = np.logspace(-0.5, 1.5, 21) * u.TeV
    aeff = EffectiveAreaTable.from_parametrization(energy=energy)
    bkg_model = PowerLaw(index=2.5, amplitude="1e-12 cm-2 s-1 TeV-1")
    sim = SpectrumSimulation(
        aeff=aeff,
        source_model=model,
        livetime=100 * u.h,
        background_model=bkg_model,
        alpha=0.2,
    )
    sim.run(seed=[0])
    obs = sim.result[0]
    return obs.to_spectrum_dataset()


def create_fpe(model):
    dataset = simulate_spectrum_dataset(model)
    e_edges = [0.1, 1, 10, 100] * u.TeV
    dataset.model = model
    return FluxPointsEstimator(datasets=[dataset], e_edges=e_edges, norm_n_values=11)


def simulate_map_dataset():
    filename = (
        "$GAMMAPY_DATA/cta-1dc/caldb/data/cta/1dc/bcf/South_z20_50h/irf_file.fits"
    )
    irfs = load_cta_irfs(filename)

    skydir = SkyCoord("0 deg", "0 deg", frame="galactic")
    edges = np.logspace(-1, 2, 15) * u.TeV
    energy_axis = MapAxis.from_edges(edges=edges, name="energy")

    geom = WcsGeom.create(skydir=skydir, width=(4, 4), binsz=0.1, axes=[energy_axis], coordsys="GAL")

    gauss = SkyGaussian("0 deg", "0 deg", "0.4 deg", frame="galactic")
    pwl = PowerLaw(amplitude="1e-11 cm-2 s-1 TeV-1")
    skymodel = SkyModel(spatial_model=gauss, spectral_model=pwl, name="source")
    dataset = simulate_dataset(skymodel=skymodel, geom=geom, pointing=skydir, irfs=irfs, random_state=0)
    return dataset


@pytest.fixture(scope="session")
def fpe_map_pwl():
    dataset = simulate_map_dataset()
    e_edges = [0.1, 1, 10, 100] * u.TeV
    return FluxPointsEstimator(datasets=[dataset], e_edges=e_edges, norm_n_values=3, source="source")


@pytest.fixture(scope="session")
def fpe_map_pwl_reoptimize():
    dataset = simulate_map_dataset()
    e_edges = [1, 10] * u.TeV
    return FluxPointsEstimator(datasets=[dataset], e_edges=e_edges, norm_values=[1], reoptimize=True, source="source")


@pytest.fixture(scope="session")
def fpe_pwl():
    return create_fpe(PowerLaw())


@pytest.fixture(scope="session")
def fpe_ecpl():
    return create_fpe(ExponentialCutoffPowerLaw(lambda_="1 TeV-1"))


class TestFluxPointsEstimator:
    @staticmethod
    def test_str(fpe_pwl):
        assert "FluxPointsEstimator" in str(fpe_pwl)

    @staticmethod
    @requires_dependency("iminuit")
    def test_run_pwl(fpe_pwl):
        fp = fpe_pwl.run()
        actual = fp.table["norm"].data
        assert_allclose(actual, [1.080933, 0.910776, 0.922278], rtol=1e-5)

        actual = fp.table["norm_err"].data
        assert_allclose(actual, [0.066364, 0.061025, 0.179742], rtol=1e-5)

        actual = fp.table["norm_errn"].data
        assert_allclose(actual, [0.065305, 0.060409, 0.17148], rtol=1e-5)

        actual = fp.table["norm_errp"].data
        assert_allclose(actual, [0.067454, 0.061646, 0.188288], rtol=1e-5)

        actual = fp.table["norm_ul"].data
        assert_allclose(actual, [1.219995, 1.037478, 1.321045], rtol=1e-5)

        actual = fp.table["sqrt_ts"].data
        assert_allclose(actual, [18.568429, 18.054651, 7.057121], rtol=1e-5)

        actual = fp.table["norm_scan"][0][[0, 5, -1]]
        assert_allclose(actual, [0.2, 1, 5], rtol=1e-5)

        actual = fp.table["dloglike_scan"][0][[0, 5, -1]]
        assert_allclose(actual, [220.368653, 4.301011, 1881.626454], rtol=1e-5)

    @staticmethod
    @requires_dependency("iminuit")
    def test_run_ecpl(fpe_ecpl):
        fp = fpe_ecpl.estimate_flux_point(fpe_ecpl.e_groups[1])
        assert_allclose(fp["norm"], 1, rtol=1e-1)


    @staticmethod
    @requires_dependency("iminuit")
    def test_run_map_pwl(fpe_map_pwl):
        fp = fpe_map_pwl.run(steps=["err", "norm-scan", "ts"])

        actual = fp.table["norm"].data
        assert_allclose(actual, [0.97922 , 0.94081 , 1.074426], rtol=1e-3)

        actual = fp.table["norm_err"].data
        assert_allclose(actual, [0.069967, 0.052631, 0.093025], rtol=1e-3)

        actual = fp.table["sqrt_ts"].data
        assert_allclose(actual, [16.165811, 27.121425, 22.040969], rtol=1e-3)

        actual = fp.table["norm_scan"][0]
        assert_allclose(actual, [0.2, 1, 5], rtol=1e-3)

        actual = fp.table["dloglike_scan"][0] - fp.table["loglike"][0]
        assert_allclose(actual, [1.536460e+02, 8.756689e-02, 1.883420e+03], rtol=1e-3)

    @staticmethod
    @requires_dependency("iminuit")
    def test_run_map_pwl_reoptimize(fpe_map_pwl_reoptimize):
        fp = fpe_map_pwl_reoptimize.run(steps=["err", "norm-scan", "ts"])

        actual = fp.table["norm"].data
        assert_allclose(actual, 0.882532, rtol=1e-3)

        actual = fp.table["norm_err"].data
        assert_allclose(actual, 0.057878, rtol=1e-3)

        actual = fp.table["sqrt_ts"].data
        assert_allclose(actual, 26.580089, rtol=1e-3)

        actual = fp.table["norm_scan"][0]
        assert_allclose(actual, 1, rtol=1e-3)

        actual = fp.table["dloglike_scan"][0] - fp.table["loglike"][0]
        assert_allclose(actual, 3.847814, rtol=1e-3)
