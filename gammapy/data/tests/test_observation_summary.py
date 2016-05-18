# Licensed under a 3-clause BSD style license - see LICENSE.rst
from __future__ import absolute_import, division, print_function, unicode_literals
from astropy.coordinates import Angle, SkyCoord

from gammapy.data import DataStore
from gammapy.data import ObservationTable
from gammapy.data import ObservationTableSummary

from ...utils.testing import data_manager, requires_data, requires_dependency

from ...datasets import gammapy_extra

def init_summary():
    """Init summary table with test gammapy-extra data

    Returns
    -------
    summary = `gammapy.data.ObservationTableSummary`
        Summary table 
    """
    
    datas_tore = DataStore.from_dir('$GAMMAPY_EXTRA/datasets/hess-crab4-hd-hap-prod2/')
    observations = datastore.obs_table
    summary = ObservationTableSummary(observations,SkyCoord.from_name('crab'))
    return summary

@requires_data('gammapy-extra')
def test_offset():
    """Test if offset is well computed"""
    summary = init_summary()
    offset = summary.offset

    assert ((offset.mean() - 1.0) <1.e-6 and (offset.std() - 0.5) <1.e-6)

@requires_data('gammapy-extra')
def test_plot():
    """Test if plots are done"""
    import matplotlib.pyplot as plt
    
    summary = init_summary()

    plt.figure()
    summary.plot_zenith_distribution()
    plt.savefig('output/plot_zenith_distribution.pdf')

    plt.figure()
    summary.plot_offset_distribution()
    plt.savefig('output/plot_offset_distribution.pdf')

