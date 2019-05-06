# Licensed under a 3-clause BSD style license - see LICENSE.rst
import pytest
import numpy as np
from numpy.testing import assert_allclose
from astropy import units as u
from ...utils.testing import requires_dependency
from ...irf import EffectiveAreaTable
from ..models import PowerLaw, ExponentialCutoffPowerLaw
from ..simulation import SpectrumSimulation
from ..flux_point import FluxPointEstimator


# TODO: use pregenerate data instead
def simulate_dataset(model):
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
    dataset = simulate_dataset(model)
    e_edges = [0.1, 1, 10, 100] * u.TeV
    dataset.model = model
    return FluxPointEstimator(datasets=[dataset], e_edges=e_edges, norm_n_values=3)


@pytest.fixture(scope="session")
def fpe_pwl():
    return create_fpe(PowerLaw())


@pytest.fixture(scope="session")
def fpe_ecpl():
    return create_fpe(ExponentialCutoffPowerLaw(lambda_="1 TeV-1"))


class TestFluxPointEstimator:
    @staticmethod
    def test_str(fpe_pwl):
        assert "FluxPointEstimator" in str(fpe_pwl)

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
        assert_allclose(actual, [1.216227, 1.035472, 1.316878], rtol=1e-5)

        actual = fp.table["sqrt_ts"].data
        assert_allclose(actual, [18.568429, 18.054651, 7.057121], rtol=1e-5)

        actual = fp.table["norm_scan"][0]
        assert_allclose(actual, [0.2, 1, 5], rtol=1e-5)

        actual = fp.table["dloglike_scan"][0]
        assert_allclose(actual, [220.368653, 4.301011, 1881.626454], rtol=1e-5)

    @staticmethod
    @requires_dependency("iminuit")
    def test_run_ecpl(fpe_ecpl):
        fp = fpe_ecpl.estimate_flux_point(fpe_ecpl.e_groups[1])
        assert_allclose(fp["norm"], 1, rtol=1e-1)
