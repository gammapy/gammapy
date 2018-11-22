# Licensed under a 3-clause BSD style license - see LICENSE.rst
from __future__ import absolute_import, division, print_function, unicode_literals
import pytest
import numpy as np
from numpy.testing import assert_allclose
from astropy import units as u
from ...utils.testing import assert_quantity_allclose, requires_dependency
from ...irf import EffectiveAreaTable
from ..energy_group import SpectrumEnergyGroupMaker
from ..models import PowerLaw, ExponentialCutoffPowerLaw
from ..simulation import SpectrumSimulation
from ..flux_point import FluxPointEstimator


# TODO: use pregenerate data instead
def simulate_obs(model):
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
    return sim.result[0]


def define_energy_groups(obs):
    # the energy bounds ar choosen such, that one flux point is
    ebounds = [0.1, 1, 10, 100] * u.TeV
    segm = SpectrumEnergyGroupMaker(obs=obs)
    segm.compute_groups_fixed(ebounds=ebounds)
    return segm.groups


def create_fpe(model):
    obs = simulate_obs(model)
    groups = define_energy_groups(obs)
    return FluxPointEstimator(obs=obs, model=model, groups=groups, norm_n_values=3)


@pytest.fixture(scope="session")
def fpe_pwl():
    return create_fpe(PowerLaw())


@pytest.fixture(scope="session")
def fpe_ecpl():
    return create_fpe(ExponentialCutoffPowerLaw(lambda_="1 TeV-1"))


class TestFluxPointEstimator:
    def test_str(self, fpe_pwl):
        assert "FluxPointEstimator" in str(fpe_pwl)

    @requires_dependency("iminuit")
    def test_energy_range(self, fpe_pwl):
        group = fpe_pwl.groups[1]
        fpe_pwl.estimate_flux_point(group)
        fit_range = fpe_pwl.fit.true_fit_range[0]
        assert_quantity_allclose(fit_range[0], group.energy_min)
        assert_quantity_allclose(fit_range[1], group.energy_max)

    @requires_dependency("iminuit")
    def test_run_pwl(self, fpe_pwl):
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

    @requires_dependency("iminuit")
    def test_run_ecpl(self, fpe_ecpl):
        fp = fpe_ecpl.estimate_flux_point(fpe_ecpl.groups[1])
        assert_allclose(fp["norm"], 1, rtol=1e-1)
