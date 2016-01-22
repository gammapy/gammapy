# Licensed under a 3-clause BSD style license - see LICENSE.rst
from __future__ import absolute_import, division, print_function, unicode_literals
import numpy as np
from ..energy_offset_array import EnergyOffsetArray
from ...data import DataStore, EventList
from ...datasets import gammapy_extra
from ...utils.testing import requires_dependency, requires_data
from ...utils.energy import EnergyBounds


@requires_dependency('matplotlib')
@requires_data('gammapy-extra')
def test_energy_offset_array():
    dir = str(gammapy_extra.dir) + '/datasets/hess-crab4'
    data_store = DataStore.from_dir(dir)
    ev_list = data_store.load_all('events')
    ebounds = EnergyBounds.equal_log_spacing(0.1, 100, 100, 'TeV')
    offset = np.linspace(0, 2.5, 100)
    Array = EnergyOffsetArray(ebounds, offset)
    Array.fill_events(ev_list)
    Array.plot_image()
