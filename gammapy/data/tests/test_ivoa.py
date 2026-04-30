# Licensed under a 3-clause BSD style license - see LICENSE.rst
from numpy.testing import assert_allclose
from gammapy.data.ivoa import (
    empty_obscore_table,
    to_obscore_table,
    make_obs_table,
    ObsTableRow,
)
from gammapy.utils.testing import requires_data
from gammapy.utils.scripts import make_path

from astropy.table import Table
from astropy.io.votable import from_table
from pyvo.dal.tap import TAPResults
from pytest import raises


def test_obscore_structure():
    obscore_default_tab = empty_obscore_table()
    assert len(obscore_default_tab.columns) == 29
    assert len(obscore_default_tab["dataproduct_type"]) == 0
    assert obscore_default_tab.colnames[0] == "dataproduct_type"
    assert obscore_default_tab.columns[0].dtype == "<U10"
    assert obscore_default_tab.columns[0].meta["UCD"] == "meta.id"
    assert obscore_default_tab.colnames[28] == "instrument_name"
    assert (
        obscore_default_tab.columns[28].meta["Utype"]
        == "Provenance.ObsConfig.Instrument.name"
    )


@requires_data()
def test_make_obs_table():
    root_path = make_path("$GAMMAPY_DATA/")
    tap_res = TAPResults(
        from_table(Table.read(root_path / "tests" / "obscore_table.fits.gz"))
    )
    obs_table = make_obs_table(tap_res)

    assert len(obs_table["OBS_MODE"]) == 4
    assert obs_table["OBS_ID"][0] == 23523
    assert obs_table["DATE-OBS"][3] == "2004-12-08"

    ivoa_tab = from_table(
        Table.read(root_path / "tests" / "minimal_datastore" / "obs-index.fits.gz")
    )
    with raises(KeyError):
        obs_table = make_obs_table(TAPResults(ivoa_tab))


@requires_data()
def test_ObsTableRow():
    mandatory_cols = [
        "obs_id",
        "obs_mode",
        "ra_pnt",
        "dec_pnt",
        "alt_pnt",
        "az_pnt",
        "tstart",
        "tstop",
    ]
    root_path = make_path("$GAMMAPY_DATA/")
    obscore_tab = Table.read(root_path / "tests" / "obscore_table.fits.gz")

    testrow = obscore_tab[0]

    # Test from_row
    otr = ObsTableRow.from_row(testrow[mandatory_cols])
    assert otr.obs_id == 23523
    assert otr.obs_mode == "wobble"
    assert otr.tstart == 123890826.0

    # Test from_table
    obs_tab = otr.to_obs_table()
    assert len(mandatory_cols) == len(obs_tab.columns)

    # Test from_row
    otr = ObsTableRow.from_row(testrow)
    assert otr.obs_id == 23523
    assert otr.date_obs == "2004-12-04"
    assert otr.ontime is None

    # Test from_table
    obs_tab = otr.to_obs_table()
    assert obs_tab["OBS_ID"] == 23523
    assert obs_tab["DATE-OBS"] == "2004-12-04"
    assert len(obs_tab.columns) < len(testrow.columns)

    testrow = obscore_tab[1]
    otr = ObsTableRow.from_row(testrow)
    assert otr.obs_id == 23526

    with raises(KeyError):
        otr = ObsTableRow.from_row(testrow[mandatory_cols[1:]])


@requires_data()
def test_to_obscore_table():
    """Test to_obscore_table with two Obs_IDs:[20136, 47828] from HESS data.
    I check the number of columns and three parameters of different origin in the table. One read from the data,
    one given by the user and one that is fixed.
    """
    path = "$GAMMAPY_DATA/hess-dl3-dr1/"
    obscore_tab = to_obscore_table(
        path, [20136, 47828], obs_publisher_did="ivo://padc.obspm/hess"
    )
    assert len(obscore_tab["dataproduct_type"]) == 2
    assert obscore_tab["obs_id"][0] == "20136"
    assert obscore_tab["obs_id"][1] == "47828"
    assert obscore_tab["calib_level"][0] == 2
    assert obscore_tab["target_name"][0] == "MSH15-52"
    assert obscore_tab["obs_publisher_did"][1] == "ivo://padc.obspm/hess#47828"
    assert_allclose(obscore_tab["t_resolution"][1], 0.0)

    obscore_tab = to_obscore_table(
        path,
        [20136],
        obs_publisher_did="ivo://padc.obspm/hess",
        obscore_template={"obs_collection": "DL4"},
    )
    assert obscore_tab["obs_collection"][0] == "DL4"
    assert obscore_tab["calib_level"][0] == 2
