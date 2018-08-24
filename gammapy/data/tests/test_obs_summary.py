# Licensed under a 3-clause BSD style license - see LICENSE.rst
from __future__ import absolute_import, division, print_function, unicode_literals
from numpy.testing import assert_allclose
import pytest
from astropy.coordinates import SkyCoord
import astropy.units as u
from regions import CircleSkyRegion
from ...data import DataStore, ObservationTableSummary, ObservationSummary
from ...data import ObservationStats
from ...utils.testing import requires_data, requires_dependency
from ...background import ReflectedRegionsBackgroundEstimator


@requires_data('gammapy-extra')
class TestObservationSummaryTable:
    @staticmethod
    def make_observation_summary_table():
        data_store = DataStore.from_dir('$GAMMAPY_EXTRA/datasets/hess-crab4-hd-hap-prod2/')
        target_pos = SkyCoord(83.633083, 22.0145, unit='deg')
        return ObservationTableSummary(data_store.obs_table, target_pos)

    @classmethod
    def setup_class(cls):
        cls.table_summary = cls.make_observation_summary_table()

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


@requires_data('gammapy-extra')
@requires_dependency('scipy')
class TestObservationSummary:
    """
    Test observation summary.
    """

    @staticmethod
    def make_observation_summary():
        datastore = DataStore.from_dir('$GAMMAPY_EXTRA/datasets/hess-crab4-hd-hap-prod2/')
        obs_ids = [23523, 23526]

        pos = SkyCoord(83.63 * u.deg, 22.01 * u.deg, frame='icrs')
        on_size = 0.3 * u.deg
        on_region = CircleSkyRegion(pos, on_size)

        obs_stats_list = []
        for obs_id in obs_ids:
            obs = datastore.obs(obs_id)
            bkg = ReflectedRegionsBackgroundEstimator(
                on_region=on_region,
                obs_list=[obs],
            )
            bkg.run()
            bg_estimate = bkg.result[0]

            obs_stats = ObservationStats.from_obs(obs, bg_estimate)
            obs_stats_list.append(obs_stats)

        return ObservationSummary(obs_stats_list)

    @classmethod
    def setup_class(cls):
        cls.obs_summary = cls.make_observation_summary()

    @pytest.mark.xfail
    def test_results(self):
        # TODO: add test with assert on result numbers yet!!!
        # from pprint import pprint
        # pprint(self.obs_summary.__dict__)
        assert 0

    def test_obs_str(self):
        text = str(self.obs_summary)
        assert 'Observation summary' in text

    @requires_dependency('matplotlib')
    def test_plot_significance(self):
        self.obs_summary.plot_significance_vs_livetime()

    @requires_dependency('matplotlib')
    def test_plot_excess(self):
        self.obs_summary.plot_excess_vs_livetime()

    @requires_dependency('matplotlib')
    def test_plot_background(self):
        self.obs_summary.plot_background_vs_livetime()

    @requires_dependency('matplotlib')
    def test_plot_gamma_rate(self):
        self.obs_summary.plot_gamma_rate()

    @requires_dependency('matplotlib')
    def test_plot_background_rate(self):
        self.obs_summary.plot_background_rate()
