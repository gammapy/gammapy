# Licensed under a 3-clause BSD style license - see LICENSE.rst
import pytest
from numpy.testing import assert_allclose
import astropy.units as u
import numpy as np
from ...utils.testing import requires_data, requires_dependency
from ...utils.random import get_random_state
from ...irf import EffectiveAreaTable, EnergyDispersion
from ...utils.fitting import Fit
from ..models import PowerLaw, ConstantModel
from ...spectrum import (
    PHACountsSpectrum,
    ONOFFSpectrumDataset
)



class Test_ONOFFSpectrumDataset:
    """ Test ON OFF SpectrumDataset"""
    def setup(self):

        etrue = np.logspace(-1,1,10)*u.TeV
        self.e_true = etrue
        ereco = np.logspace(-1,1,5)*u.TeV
        elo = ereco[:-1]
        ehi = ereco[1:]

        self.aeff = EffectiveAreaTable(etrue[:-1],etrue[1:], np.ones(9)*u.cm**2)
        self.edisp = EnergyDispersion.from_diagonal_response(etrue, ereco)

        self.on_counts = PHACountsSpectrum(elo, ehi, np.ones_like(elo), backscal=np.ones_like(elo))
        self.off_counts = PHACountsSpectrum(elo, ehi, np.ones_like(elo)*10, backscal=np.ones_like(elo)*10)

        self.livetime = 1000*u.s

    def test_init_no_model(self):
        dataset = ONOFFSpectrumDataset(ONcounts=self.on_counts, OFFcounts=self.off_counts,
                         aeff=self.aeff, edisp=self.edisp, livetime = self.livetime)

        with pytest.raises(AttributeError):
            dataset.npred()

    def test_alpha(self):
        dataset = ONOFFSpectrumDataset(ONcounts=self.on_counts, OFFcounts=self.off_counts,
                         aeff=self.aeff, edisp=self.edisp, livetime = self.livetime)

        assert dataset.alpha.shape == (4,)
        assert_allclose(dataset.alpha, 0.1)

    def test_init_no_edisp(self):
        const = 1 / u.TeV / u.cm ** 2 / u.s
        model = ConstantModel(const)
        livetime = 1*u.s
        dataset = ONOFFSpectrumDataset(ONcounts=self.on_counts, OFFcounts=self.off_counts,
                         aeff=self.aeff, model=model, livetime = livetime)

        expected = self.aeff.data.data[0]*(self.aeff.energy.hi[-1]-self.aeff.energy.lo[0])*const*livetime

        assert_allclose(dataset.npred().sum(), expected.value)



