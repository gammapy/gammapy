# Licensed under a 3-clause BSD style license - see LICENSE.rst
from __future__ import absolute_import, division, print_function, unicode_literals
import pytest
import numpy as np
from numpy.testing import assert_allclose, assert_equal
import astropy.units as u
from ...utils.testing import assert_quantity_allclose
from ...utils.testing import requires_data, requires_dependency
from ..core import PHACountsSpectrum
from ..observation import SpectrumObservation, SpectrumObservationList
from ..energy_group_new import FluxPointBinMaker, FluxPointBins

class TestFluxPointBinMaker:

    @pytest.fixture(scope='session')
    def obs(self):
        """An example SpectrumObservation object for tests."""
        pha_ebounds = np.arange(1, 11) * u.TeV
        on_vector = PHACountsSpectrum(
            energy_lo=pha_ebounds[:-1],
            energy_hi=pha_ebounds[1:],
            data=np.zeros(len(pha_ebounds) - 1),
            livetime=99 * u.s
        )
        return SpectrumObservation(on_vector=on_vector)

    @pytest.mark.parametrize('data', [
        {'ebounds':[1.25, 5.5, 7.5] * u.TeV, 'low':[1,5], 'high':[4,6]},
        {'ebounds':[2, 6, 8] * u.TeV, 'low':[1,5], 'high':[4,6]},
        {'ebounds': [-1, 6, 100] * u.TeV, 'low': [0, 5], 'high': [4, 9]}
        ]
   )
    def test_groups_fixed(self, obs, data):
        fpbmaker = FluxPointBinMaker(obs=obs)
        fpbmaker.compute_bins_fixed(data['ebounds'])
        fpbins = fpbmaker.indices

        print("lower bounds = ",fpbins.lower_bounds())
        print("upper bounds = ",fpbins.upper_bounds())
        print("energies = ", obs.on_vector.energy)
        assert (fpbins.lower_bounds() == data['low']).all()
        assert (fpbins.upper_bounds() == data['high']).all()

