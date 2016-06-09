# Licensed under a 3-clause BSD style license - see LICENSE.rst
from __future__ import absolute_import, division, print_function, unicode_literals
import numpy as np
from numpy.testing import assert_allclose
from astropy.coordinates import SkyCoord
from ...datasets import gammapy_extra
from ...utils.testing import requires_data
from ...utils.energy import EnergyBounds
from ...data import DataStore, ObservationList


@requires_data('gammapy-extra')
def test_make_psftable():
    """Test creating a datastore as subset of another datastore"""
    center = SkyCoord(83.63, 22.01, unit='deg')
    store = gammapy_extra.filename("datasets/hess-crab4-hd-hap-prod2")
    data_store = DataStore.from_dir(store)
    list = [data_store.obs(id) for id in data_store.obs_table["OBS_ID"]]
    obslist = ObservationList(list)
    energy = EnergyBounds.equal_log_spacing(1, 10, 100, "TeV")
    obslist.make_psftable(source_position=center, energy=energy, spectral_index=2.3)
