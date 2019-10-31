# Licensed under a 3-clause BSD style license - see LICENSE.rst
from numpy.testing import assert_allclose
from astropy.coordinates import SkyCoord
from gammapy.data import DataStore, ObservationTableSummary
from gammapy.utils.testing import mpl_plot_check, requires_data, requires_dependency


@requires_data()
class TestObservationSummaryTable:
    @classmethod
    def setup_class(cls):
        data_store = DataStore.from_dir("$GAMMAPY_DATA/hess-dl3-dr1/")
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
        assert_allclose(offset.degree.mean(), 1.0, rtol=0.01)
        assert_allclose(offset.degree.std(), 0.5, rtol=0.01)

    @requires_dependency("matplotlib")
    def test_plot_zenith(self):
        with mpl_plot_check():
            self.table_summary.plot_zenith_distribution()

    @requires_dependency("matplotlib")
    def test_plot_offset(self):
        with mpl_plot_check():
            self.table_summary.plot_offset_distribution()
