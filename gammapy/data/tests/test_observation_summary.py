# Licensed under a 3-clause BSD style license - see LICENSE.rst
from __future__ import absolute_import, division, print_function, unicode_literals
from numpy.testing import assert_allclose
from astropy.tests.helper import pytest
from astropy.coordinates import SkyCoord
import astropy.units as u
from regions import CircleSkyRegion
from ...data import DataStore, ObservationTableSummary, ObservationSummary
from ...data import ObservationStats, ObservationStatsList, ObservationList
from ...data import Target
from ...utils.testing import requires_data, requires_dependency
from ...background import reflected_regions_background_estimate as refl
from ...image import SkyMask


def table_summary():
    data_store = DataStore.from_dir('$GAMMAPY_EXTRA/datasets/hess-crab4-hd-hap-prod2/')
    target_pos = SkyCoord(83.633083, 22.0145, unit='deg')
    return ObservationTableSummary(data_store.obs_table, target_pos)


@requires_data('gammapy-extra')
class TestObservationSummaryTable:
    def setup(self):
        self.table_summary = table_summary()

    def test_str(self):
        text = str(self.table_summary)
        assert 'Observation summary' in text

    def test_offset(self):
        offset = self.table_summary.offset
        assert_allclose(offset.degree.mean(), 1., rtol=1.e-2)
        assert_allclose(offset.degree.std(), 0.5, rtol=1.e-2)

    @requires_dependency('matplotlib')
    def test_plot_zenith(self):
        self.table_summary.plot_zenith_distribution()

    @requires_dependency('matplotlib')
    def test_plot_offset(self):
        self.table_summary.plot_offset_distribution()


def obs_summary():
    datastore = DataStore.from_dir('$GAMMAPY_EXTRA/datasets/hess-crab4-hd-hap-prod2/')
    run_list = [23523, 23526, 23559, 23592]

    pos = SkyCoord(83.63 * u.deg, 22.01 * u.deg, frame='icrs')
    on_size = 0.3 * u.deg
    on_region = CircleSkyRegion(pos, on_size)

    target = Target(position=pos, on_region=on_region,
                    name='Crab Nebula', tag='crab')

    mask = SkyMask.read('$GAMMAPY_EXTRA/datasets/exclusion_masks/tevcat_exclusion.fits')

    obs_list = ObservationList([datastore.obs(_) for _ in run_list])
    obs_stats = ObservationStatsList()

    for index, run in enumerate(obs_list):
        bkg = refl(on_region, run.pointing_radec, mask, run.events)

        obs_stats.append(ObservationStats.from_target(run, target, bkg))

    summary = ObservationSummary(obs_stats)

    return summary


@requires_data('gammapy-extra')
@requires_dependency('scipy')
@requires_dependency('matplotlib')
class TestObservationSummary:
    """
    Test observation summary.
    """

    def setup(self):
        self.obs_summary = obs_summary()

    def test_plot_significance(self):
        self.obs_summary.plot_significance_vs_livetime()

    def test_plot_excess(self):
        self.obs_summary.plot_excess_vs_livetime()

    def test_plot_background(self):
        self.obs_summary.plot_background_vs_livetime()

    def test_plot_gamma_rate(self):
        self.obs_summary.plot_gamma_rate()

    def test_plot_background_rate(self):
        self.obs_summary.plot_background_rate()

    def test_obs_str(self):
        text = str(self.obs_summary)
        assert 'Observation summary' in text
