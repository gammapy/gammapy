# Licensed under a 3-clause BSD style license - see LICENSE.rst
from __future__ import absolute_import, division, print_function, unicode_literals
from astropy.coordinates import SkyCoord

from ...data import DataStore, ObservationTable, ObservationTableSummary

from ...utils.testing import data_manager, requires_data, requires_dependency

def init_summary():
    """Init summary table with gammapy-extra data

    Returns
    -------
    summary = `gammapy.data.ObservationTableSummary`
        Summary table 
    """
    
    data_store = DataStore.from_dir('$GAMMAPY_EXTRA/datasets/hess-crab4-hd-hap-prod2/')
    observations = data_store.obs_table
    summary = ObservationTableSummary(observations,SkyCoord.from_name('crab'))
    return summary

@requires_data('gammapy-extra')
def test_str():
    """Test if str is well computed"""
    summary = init_summary()
    text = str(summary)
    assert 'Observation summary' in text

@requires_data('gammapy-extra')
def test_offset():
    """Test if offset is well computed"""
    summary = init_summary()
    offset = summary.offset

    assert ((offset.degree.mean() - 1.0) <1.e-3 and (offset.degree.std() - 0.5) <1.e-3)

@requires_data('gammapy-extra')
@requires_dependency('matplotlib')
def test_plot_zenith():
    """Test if zenith plot is done"""
    
    summary = init_summary()
    summary.plot_zenith_distribution()

@requires_data('gammapy-extra')
@requires_dependency('matplotlib')
def test_plot_offset():
    """Test if offset plot is done"""
    
    summary = init_summary()
    summary.plot_offset_distribution()
