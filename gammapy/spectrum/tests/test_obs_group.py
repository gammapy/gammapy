# Licensed under a 3-clause BSD style license - see LICENSE.rst
from __future__ import absolute_import, division, print_function, unicode_literals
from astropy.tests.helper import assert_quantity_allclose
import pytest
from ...utils.testing import requires_data, requires_dependency
from ...data import ObservationTable
from ...datasets import gammapy_extra
from ..obs_group import group_obs_table, SpectrumObservationGrouping
from ..fit import SpectrumFit


@pytest.mark.xfail
@requires_dependency('sherpa')
@requires_data('gammapy-extra')
def test_define_groups_and_stack(tmpdir):
    obs_table_file = gammapy_extra.filename(
        'datasets/hess-crab4_pha/observation_table.fits')

    obs_table = ObservationTable.read(obs_table_file)

    fit = SpectrumFit.from_observation_table(obs_table)
    fit.model = 'PL'
    fit.energy_threshold_low = '1 TeV'
    fit.energy_threshold_high = '10 TeV'
    fit.run(method='sherpa')

    # Use each obs in one group
    obs_table1 = group_obs_table(obs_table, eff_range=[90, 95], n_eff_bin=5)
    obs_table1.write('grouped_table1_debug.fits', overwrite=True)

    grouping = SpectrumObservationGrouping(obs_table1)
    grouping.run()

    fit_band2 = SpectrumFit.from_observation_table(grouping.stacked_obs_table)
    fit_band2.model = 'PL'
    fit_band2.energy_threshold_low = '1 TeV'
    fit_band2.energy_threshold_high = '10 TeV'
    fit_band2.run(method='sherpa')

    assert_quantity_allclose(fit.result.parameters["index"],
                             fit_band2.result.parameters["index"], rtol=1e-5)
    assert_quantity_allclose(fit.result.parameters["norm"],
                             fit_band2.result.parameters["norm"], rtol=1e-5)

    # Put all runs in one group
    obs_table2 = group_obs_table(obs_table, n_eff_bin=1, n_off_bin=1, n_zen_bin=1)
    obs_table2.write('grouped_table2_debug.fits', overwrite=True)

    grouping = SpectrumObservationGrouping(obs_table2)
    grouping.run()

    fit_band3 = SpectrumFit.from_observation_table(grouping.stacked_obs_table)
    fit_band3.model = 'PL'
    fit_band3.energy_threshold_low = '100 GeV'
    fit_band3.energy_threshold_high = '10 TeV'
    fit_band3.run(method='sherpa')

    # Todo: check why the difference is so large
    assert_quantity_allclose(fit.result.parameters["index"],
                             fit_band3.result.parameters["index"], rtol=1e0)
    assert_quantity_allclose(fit_band3.result.parameters["norm"],
                             fit.result.parameters["norm"], rtol=1e0)
