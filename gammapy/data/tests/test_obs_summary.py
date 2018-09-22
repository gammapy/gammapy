# Licensed under a 3-clause BSD style license - see LICENSE.rst
from __future__ import absolute_import, division, print_function, unicode_literals
from numpy.testing import assert_allclose
import pytest
from astropy.coordinates import SkyCoord
import astropy.units as u
from regions import CircleSkyRegion
from ...data import DataStore, ObservationTableSummary, ObservationSummary
from ...data import ObservationStats
from ...utils.testing import requires_data, requires_dependency, mpl_plot_check
from ...background import ReflectedRegionsBackgroundEstimator


@requires_data("gammapy-extra")
class TestObservationSummaryTable:
    @classmethod
    def setup_class(cls):
        data_store = DataStore.from_dir("$GAMMAPY_EXTRA/datasets/hess-dl3-dr1/")
        obs_table = data_store.obs_table
        obs_table = obs_table[obs_table["TARGET_NAME"] == "Crab"]
        target_pos = SkyCoord(83.633083, 22.0145, unit="deg")
        cls.table_summary = ObservationTableSummary(obs_table, target_pos)

    def test_str(self):
        text = str(self.table_summary)
        assert "Observation summary" in text

    def test_offset(self):
        offset = self.table_summary.offset
        assert len(offset) == 4
        assert_allclose(offset.degree.mean(), 1., rtol=0.01)
        assert_allclose(offset.degree.std(), 0.5, rtol=0.01)

    @requires_dependency("matplotlib")
    def test_plot_zenith(self):
        with mpl_plot_check():
            self.table_summary.plot_zenith_distribution()

    @requires_dependency("matplotlib")
    def test_plot_offset(self):
        with mpl_plot_check():
            self.table_summary.plot_offset_distribution()


@requires_data("gammapy-extra")
@requires_dependency("scipy")
class TestObservationSummary:
    """
    Test observation summary.
    """

    @classmethod
    def setup_class(cls):
        datastore = DataStore.from_dir("$GAMMAPY_EXTRA/datasets/hess-dl3-dr1/")
        obs_ids = [23523, 23526]

        on_region = CircleSkyRegion(
            SkyCoord(83.63 * u.deg, 22.01 * u.deg, frame="icrs"), 0.3 * u.deg
        )

        obs_stats_list = []
        for obs_id in obs_ids:
            obs = datastore.obs(obs_id)
            bkg = ReflectedRegionsBackgroundEstimator(
                on_region=on_region, obs_list=[obs]
            )
            bkg.run()
            bg_estimate = bkg.result[0]

            obs_stats = ObservationStats.from_obs(obs, bg_estimate)
            obs_stats_list.append(obs_stats)

        cls.obs_summary = ObservationSummary(obs_stats_list)

    @pytest.mark.xfail
    def test_results(self):
        # TODO: add test with assert on result numbers yet!!!
        # from pprint import pprint
        # pprint(self.obs_summary.__dict__)
        assert 0

    def test_obs_str(self):
        text = str(self.obs_summary)
        assert "Observation summary" in text

    @requires_dependency("matplotlib")
    def test_plot_significance(self):
        with mpl_plot_check():
            self.obs_summary.plot_significance_vs_livetime()

    @requires_dependency("matplotlib")
    def test_plot_excess(self):
        with mpl_plot_check():
            self.obs_summary.plot_excess_vs_livetime()

    @requires_dependency("matplotlib")
    def test_plot_background(self):
        with mpl_plot_check():
            self.obs_summary.plot_background_vs_livetime()

    @requires_dependency("matplotlib")
    def test_plot_gamma_rate(self):
        with mpl_plot_check():
            self.obs_summary.plot_gamma_rate()

    @requires_dependency("matplotlib")
    def test_plot_background_rate(self):
        with mpl_plot_check():
            self.obs_summary.plot_background_rate()
