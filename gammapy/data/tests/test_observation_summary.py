# Licensed under a 3-clause BSD style license - see LICENSE.rst
from __future__ import absolute_import, division, print_function, unicode_literals

import pytest

from astropy.coordinates import SkyCoord

from ...data import DataStore, ObservationTable, ObservationTableSummary

from ...utils.testing import data_manager, requires_data, requires_dependency

from numpy.testing import assert_allclose

@requires_data('gammapy-extra')
@pytest.fixture
def summary():    
    data_store = DataStore.from_dir('$GAMMAPY_EXTRA/datasets/hess-crab4-hd-hap-prod2/')
    target_pos = SkyCoord(83.633083, 22.0145, unit='deg')
    return ObservationTableSummary(data_store.obs_table, target_pos)

def test_str(summary):
    text = str(summary)
    assert('Observation summary' in text)

def test_offset(summary):
    offset = summary.offset
    assert_allclose(offset.degree.mean(),1.,rtol=1.e-2,atol=0.)
    assert_allclose(offset.degree.std(),0.5,rtol=1.e-2,atol=0.)

@requires_dependency('matplotlib')
def test_plot_zenith(summary):
    summary.plot_zenith_distribution()

@requires_dependency('matplotlib')
def test_plot_offset(summary):
    summary.plot_offset_distribution()
