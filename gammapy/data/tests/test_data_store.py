# Licensed under a 3-clause BSD style license - see LICENSE.rst
import os
from pathlib import Path
import pytest
from gammapy.data import DataStore
from gammapy.utils.testing import requires_data


@pytest.fixture(scope="session")
def data_store():
    return DataStore.from_dir("$GAMMAPY_DATA/hess-dl3-dr1/")


@requires_data()
def test_datastore_hd_hap(data_store):
    """Test HESS HAP-HD data access."""
    obs = data_store.obs(obs_id=23523)

    assert obs.events.__class__.__name__ == "EventList"
    assert obs.gti.__class__.__name__ == "GTI"
    assert obs.aeff.__class__.__name__ == "EffectiveAreaTable2D"
    assert obs.edisp.__class__.__name__ == "EnergyDispersion2D"
    assert obs.psf.__class__.__name__ == "PSF3D"


@requires_data()
def test_datastore_from_dir():
    """Test the `from_dir` method."""
    data_store_rel_path = DataStore.from_dir(
        "$GAMMAPY_DATA/hess-dl3-dr1/", "hdu-index.fits.gz", "obs-index.fits.gz"
    )

    data_store_abs_path = DataStore.from_dir(
        "$GAMMAPY_DATA/hess-dl3-dr1/",
        "$GAMMAPY_DATA/hess-dl3-dr1/hdu-index.fits.gz",
        "$GAMMAPY_DATA/hess-dl3-dr1/obs-index.fits.gz",
    )

    assert "Data store" in data_store_rel_path.info(show=False)
    assert "Data store" in data_store_abs_path.info(show=False)


@requires_data()
def test_datastore_from_file():
    filename = "$GAMMAPY_DATA/hess-dl3-dr1/hdu-index.fits.gz"
    data_store = DataStore.from_file(filename)
    obs = data_store.obs(obs_id=23523)
    # Check that things can be loaded:
    obs.events
    obs.bkg


@requires_data()
def test_datastore_from_events():
    # Test that `DataStore.from_events_files` works.
    # The real tests for `DataStoreMaker` are below.
    path = "$GAMMAPY_DATA/cta-1dc/data/baseline/gps/gps_baseline_110380.fits"
    data_store = DataStore.from_events_files([path])
    assert len(data_store.obs_table) == 1
    assert len(data_store.hdu_table) == 6


@requires_data()
def test_datastore_get_observations(data_store):
    """Test loading data and IRF files via the DataStore"""
    observations = data_store.get_observations([23523, 23592])
    assert observations[0].obs_id == 23523

    # Test that default is all observations
    observations = data_store.get_observations()
    assert len(observations) == 105

    with pytest.raises(ValueError):
        data_store.get_observations([11111, 23592])

    observations = data_store.get_observations([11111, 23523], skip_missing=True)
    assert observations[0].obs_id == 23523


@requires_data()
def test_datastore_copy_obs(tmp_path, data_store):
    data_store.copy_obs([23523, 23592], tmp_path, overwrite=True)

    substore = DataStore.from_dir(tmp_path)

    assert str(substore.hdu_table.base_dir) == str(tmp_path)
    assert len(substore.obs_table) == 2

    desired = data_store.obs(23523)
    actual = substore.obs(23523)

    assert str(actual.events.table) == str(desired.events.table)


@requires_data()
def test_datastore_copy_obs_subset(tmp_path, data_store):
    # Copy only certain HDU classes
    data_store.copy_obs([23523, 23592], tmp_path, hdu_class=["events"])

    substore = DataStore.from_dir(tmp_path)
    assert len(substore.hdu_table) == 2


@requires_data()
class TestDataStoreChecker:
    def setup(self):
        self.data_store = DataStore.from_dir("$GAMMAPY_DATA/cta-1dc/index/gps")

    def test_check_all(self):
        records = list(self.data_store.check())
        assert len(records) == 32


@requires_data("gammapy-data")
class TestDataStoreMaker:
    def setup(self):
        paths = [
            f"$GAMMAPY_DATA/cta-1dc/data/baseline/gps/gps_baseline_{obs_id:06d}.fits"
            for obs_id in [110380, 111140, 111630, 111159]
        ]
        self.data_store = DataStore.from_events_files(paths)

        # Useful for debugging:
        # self.data_store.hdu_table.write("hdu-index.fits.gz", overwrite=True)
        # self.data_store.obs_table.write("obs-index.fits.gz", overwrite=True)

    def test_obs_table(self):
        table = self.data_store.obs_table
        assert table.__class__.__name__ == "ObservationTable"
        assert len(table) == 4
        assert len(table.colnames) == 24

        # TODO: implement https://github.com/gammapy/gammapy/issues/1218 and add tests here
        # assert table.time_start[0].iso == "spam"
        # assert table.time_start[-1].iso == "spam"

    def test_hdu_table(self):
        table = self.data_store.hdu_table
        assert table.__class__.__name__ == "HDUIndexTable"
        assert len(table) == 24
        hdu_class = ["events", "gti", "aeff_2d", "edisp_2d", "psf_3gauss", "bkg_3d"]
        assert list(self.data_store.hdu_table["HDU_CLASS"]) == 4 * hdu_class

        assert table["FILE_DIR"][2] == "$CALDB/data/cta/1dc/bcf/South_z20_50h"

    def test_observation(self, monkeypatch):
        """Check that one observation can be accessed OK"""
        obs = self.data_store.obs(110380)
        assert obs.obs_id == 110380

        assert obs.events.time[0].iso == "2021-01-21 12:00:03.045"
        assert obs.gti.time_start[0].iso == "2021-01-21 12:00:00.000"

        # Note: IRF access requires the CALDB env var
        caldb_path = Path(os.environ["GAMMAPY_DATA"]) / Path("cta-1dc/caldb")
        monkeypatch.setenv("CALDB", str(caldb_path))

        assert obs.aeff.__class__.__name__ == "EffectiveAreaTable2D"
        assert obs.bkg.__class__.__name__ == "Background3D"
        assert obs.edisp.__class__.__name__ == "EnergyDispersion2D"
        assert obs.psf.__class__.__name__ == "EnergyDependentMultiGaussPSF"
