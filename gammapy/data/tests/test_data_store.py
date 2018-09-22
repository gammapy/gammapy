# Licensed under a 3-clause BSD style license - see LICENSE.rst
from __future__ import absolute_import, division, print_function, unicode_literals
import pytest
from ...data import DataStore
from ...utils.testing import requires_data

pytest.importorskip("scipy")


@pytest.fixture(scope="session")
def data_store():
    return DataStore.from_dir("$GAMMAPY_EXTRA/datasets/hess-dl3-dr1/")


@requires_data("gammapy-extra")
def test_datastore_hd_hap(data_store):
    """Test HESS HAP-HD data access."""
    obs = data_store.obs(obs_id=23523)

    assert str(type(obs.events)) == "<class 'gammapy.data.event_list.EventList'>"
    assert str(type(obs.gti)) == "<class 'gammapy.data.gti.GTI'>"
    assert (
        str(type(obs.aeff))
        == "<class 'gammapy.irf.effective_area.EffectiveAreaTable2D'>"
    )
    assert (
        str(type(obs.edisp))
        == "<class 'gammapy.irf.energy_dispersion.EnergyDispersion2D'>"
    )
    assert str(type(obs.psf)) == "<class 'gammapy.irf.psf_3d.PSF3D'>"


@requires_data("gammapy-extra")
def test_datastore_from_dir():
    """Test the `from_dir` method."""
    data_store_rel_path = DataStore.from_dir(
        "$GAMMAPY_EXTRA/datasets/hess-dl3-dr1/",
        "hdu-index.fits.gz",
        "obs-index.fits.gz",
    )

    data_store_abs_path = DataStore.from_dir(
        "$GAMMAPY_EXTRA/datasets/hess-dl3-dr1/",
        "$GAMMAPY_EXTRA/datasets/hess-dl3-dr1/hdu-index.fits.gz",
        "$GAMMAPY_EXTRA/datasets/hess-dl3-dr1/obs-index.fits.gz",
    )

    assert "Data store" in data_store_rel_path.info(show=False)
    assert "Data store" in data_store_abs_path.info(show=False)


@requires_data("gammapy-extra")
def test_datastore_from_file():
    filename = "$GAMMAPY_DATA/hess-dl3-dr1/hess-dl3-dr3-with-background.fits.gz"
    data_store = DataStore.from_file(filename)
    obs = data_store.obs(obs_id=23523)
    # Check that things can be loaded:
    obs.events
    obs.bkg


@requires_data("gammapy-extra")
def test_datastore_pa():
    """Test HESS ParisAnalysis data access."""
    data_store = DataStore.from_dir("$GAMMAPY_EXTRA/datasets/hess-crab4-pa")

    obs = data_store.obs(obs_id=23523)
    filename = str(obs.location(hdu_type="bkg").path(abs_path=False))
    assert filename == "background/bgmodel_alt7_az0.fits.gz"

    assert (
        str(type(obs.aeff))
        == "<class 'gammapy.irf.effective_area.EffectiveAreaTable2D'>"
    )
    assert (
        str(type(obs.edisp))
        == "<class 'gammapy.irf.energy_dispersion.EnergyDispersion2D'>"
    )
    assert str(type(obs.psf)) == "<class 'gammapy.irf.psf_king.PSFKing'>"
    assert str(type(obs.bkg)) == "<class 'gammapy.irf.background.Background3D'>"


@requires_data("gammapy-extra")
def test_datastore_obslist(data_store):
    """Test loading data and IRF files via the DataStore"""
    obslist = data_store.obs_list([23523, 23592])
    assert obslist[0].obs_id == 23523

    with pytest.raises(ValueError):
        data_store.obs_list([11111, 23592])

    obslist = data_store.obs_list([11111, 23523], skip_missing=True)
    assert obslist[0].obs_id == 23523


@requires_data("gammapy-extra")
def test_datastore_subset(tmpdir, data_store):
    """Test creating a datastore as subset of another datastore"""
    selected_obs = data_store.obs_table.select_obs_id([23523, 23592])
    storedir = tmpdir / "substore"
    data_store.copy_obs(selected_obs, storedir)
    obs_id = [23523, 23592]
    data_store.copy_obs(obs_id, storedir, overwrite=True)

    substore = DataStore.from_dir(storedir)

    assert str(substore.hdu_table.base_dir) == str(storedir)
    assert len(substore.obs_table) == 2

    desired = data_store.obs(23523)
    actual = substore.obs(23523)

    assert str(actual.events.table) == str(desired.events.table)

    # Copy only certain HDU classes
    storedir = tmpdir / "substore2"
    data_store.copy_obs(obs_id, storedir, hdu_class=["events"])

    substore = DataStore.from_dir(storedir)
    assert len(substore.hdu_table) == 2


@requires_data("gammapy-extra")
class TestDataStoreChecker:
    def setup(self):
        self.data_store = DataStore.from_dir(
            "$GAMMAPY_EXTRA/datasets/cta-1dc/index/gps"
        )

    def test_check_all(self):
        records = list(self.data_store.check())
        assert len(records) == 48
