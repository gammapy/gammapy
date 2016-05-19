# Licensed under a 3-clause BSD style license - see LICENSE.rst
from __future__ import absolute_import, division, print_function, unicode_literals

import pytest

from astropy.coordinates import SkyCoord

from ...data import DataStore, ObservationTable, ObservationTableSummary

from ...utils.testing import data_manager, requires_data, requires_dependency

@requires_data('gammapy-extra')
@pytest.fixture
def summary():    
    data_store = DataStore.from_dir('$GAMMAPY_EXTRA/datasets/hess-crab4-hd-hap-prod2/')
    target_pos = SkyCoord(83.633083, 22.0145, unit='deg')
    return ObservationTableSummary(data_store.obs_table, target_pos)

@requires_data('gammapy-extra')
def test_str():
    text = str(summary())
    assert 'Observation summary' in text

@requires_data('gammapy-extra')
def test_offset():
    offset = summary().offset
    assert ((offset.degree.mean() - 1.0) <1.e-3 and (offset.degree.std() - 0.5) <1.e-3)

@requires_data('gammapy-extra')
@requires_dependency('matplotlib')
def test_plot_zenith():
    summary().plot_zenith_distribution()

@requires_data('gammapy-extra')
@requires_dependency('matplotlib')
def test_plot_offset():
    summary().plot_offset_distribution()
