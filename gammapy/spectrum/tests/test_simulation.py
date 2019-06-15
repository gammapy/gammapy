# Licensed under a 3-clause BSD style license - see LICENSE.rst
import numpy as np
from numpy.testing import assert_allclose
import astropy.units as u
from ...utils.energy import energy_logspace
from ...irf import EnergyDispersion, EffectiveAreaTable
from .. import SpectrumExtraction, SpectrumSimulation
from ..models import PowerLaw


class TestSpectrumSimulation:
    def setup(self):
        e_true = SpectrumExtraction.DEFAULT_TRUE_ENERGY
        e_reco = SpectrumExtraction.DEFAULT_RECO_ENERGY

        edisp = EnergyDispersion.from_gauss(
            e_true=e_true, e_reco=e_reco, sigma=0.2, bias=0
        )

        aeff = EffectiveAreaTable.from_parametrization(energy=e_true)

        self.source_model = PowerLaw(
            index=2.3, amplitude="2.5e-12 cm-2 s-1 TeV-1", reference="1 TeV"
        )
        self.background_model = PowerLaw(
            index=3, amplitude="3e-12 cm-2 s-1 TeV-1", reference="1 TeV"
        )
        self.alpha = 1.0 / 3

        # Minimal setup
        self.sim = SpectrumSimulation(
            aeff=aeff, edisp=edisp, source_model=self.source_model, livetime=4 * u.h
        )

    def test_without_background(self):
        self.sim.simulate_obs(seed=23, obs_id=23)
        assert self.sim.obs.counts.total_counts == 160

    def test_with_background(self):
        self.sim.background_model = self.background_model
        self.sim.alpha = self.alpha
        self.sim.simulate_obs(seed=23, obs_id=23)
        assert self.sim.obs.counts.total_counts == 530
        assert self.sim.obs.counts_off.total_counts == 1112

    def test_observations_list(self):
        seeds = np.arange(5)
        self.sim.run(seed=seeds)
        assert (np.array([_.obs_id for _ in self.sim.result]) == seeds).all()
        assert self.sim.result[0].counts.total_counts == 158
        assert self.sim.result[1].counts.total_counts == 158
        assert self.sim.result[2].counts.total_counts == 161
        assert self.sim.result[3].counts.total_counts == 168
        assert self.sim.result[4].counts.total_counts == 186

    def test_without_edisp(self):
        sim = SpectrumSimulation(
            aeff=self.sim.aeff, source_model=self.sim.source_model, livetime=4 * u.h
        )
        sim.simulate_obs(seed=23, obs_id=23)
        assert sim.obs.counts.total_counts == 161
        # The test value is taken from the test with edisp
        assert_allclose(
            np.sum(sim.npred_source.data.data.value), 167.467572145, rtol=0.01
        )

    def test_without_aeff(self):
        e_true = energy_logspace(1, 10, 5, u.TeV)
        self.source_model.amplitude.unit = "TeV-1 s-1"
        self.source_model.amplitude.value = 1
        sim = SpectrumSimulation(
            source_model=self.source_model, livetime=4 * u.h, e_true=e_true
        )
        sim.simulate_obs(seed=23, obs_id=23)
        assert sim.obs.counts.total_counts == 10509
