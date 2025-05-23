# Licensed under a 3-clause BSD style license - see LICENSE.rst

from gammapy.data.ivoa import empty_obscore_table, to_obscore_table
from gammapy.utils.testing import requires_data


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
    assert obscore_tab["t_resolution"][1] == 0.0

    obscore_tab = to_obscore_table(
        path,
        [20136],
        obs_publisher_did="ivo://padc.obspm/hess",
        obscore_template={"obs_collection": "DL4"},
    )
    assert obscore_tab["obs_collection"][0] == "DL4"
    assert obscore_tab["calib_level"][0] == 2
