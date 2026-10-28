# Licensed under a 3-clause BSD style license - see LICENSE.rst
import re

import pytest
from astropy.io.votable import from_table as from_tablevo
from astropy.io.votable import parse as parsevo
from astropy.table import Table
from numpy.testing import assert_allclose
from pyvo.dal.adhoc import DatalinkResults
from pyvo.dal.tap import TAPResults

from gammapy.data.ivoa import (
    ObsTableRow,
    empty_obscore_table,
    fetch_files,
    make_fetch_list,
    make_hdu_table,
    make_obs_table,
    to_obscore_table,
)
from gammapy.utils.scripts import make_path
from gammapy.utils.testing import requires_data


def download_mock(url, out_path):
    pass


@pytest.fixture()
def get_result_rows():
    class result_row:
        def __init__(self, datalink):
            self.datalink = datalink
            self.id = datalink.getcolumn("ID")[0]

        def __getitem__(self, item):
            return self.id

        def getdatalink(self):
            return self.datalink

    root_path = make_path("$GAMMAPY_DATA/")

    def _make_results(pattern):
        data_link_resources = (root_path / "tests" / "ivoa").rglob(pattern)
        result = []
        for fil in data_link_resources:
            vot = parsevo(fil)
            dl = DatalinkResults(votable=vot)
            result.append(result_row(dl))
        return result

    return _make_results


@pytest.fixture(scope="function", params=["datalink*.xml", "split_datalink*.xml"])
def fetch_list(get_result_rows, request):
    glob = request.param
    results = get_result_rows(glob)
    return make_fetch_list(results)


@pytest.fixture(scope="function", params=["datalink*.xml", "split_datalink*.xml"])
def fetched_list(get_result_rows, monkeypatch, request):
    glob = request.param
    results = get_result_rows(glob)
    ft_list = make_fetch_list(results)

    save_dir = make_path("$GAMMAPY_DATA/") / "tests" / "ivoa"
    monkeypatch.setattr("gammapy.data.ivoa.progress_download", download_mock)
    return fetch_files(ft_list, save_dir)


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
def test_make_fetch_list(get_result_rows):
    out_tag_dict = {
        "bkgrate": "bkg",
        "event-list": "event-list",
        "aeff": "aeff",
        "edisp": "edisp",
        "psf": "psf",
    }
    reg = re.compile(r"TapResult-\d+-(.*).fits.gz")
    result_rows = get_result_rows("datalink*.xml")
    fl = make_fetch_list(result_rows)
    assert len(fl) == len(result_rows)
    for idx in range(len(result_rows)):
        dl_tab = result_rows[idx].getdatalink().to_table()
        res_row = dl_tab[dl_tab["semantics"] == "#package"]

        out_tag = reg.match(fl[0][1]).group(1)
        assert fl[idx][0] == res_row["access_url"]
        assert fl[idx][2] == res_row["ID"]
        assert out_tag == res_row["content_qualifier"]

    result_rows = get_result_rows("split_datalink*.xml")
    fl = make_fetch_list(result_rows)
    assert len(fl) == 5 * len(result_rows)
    fl_split = [fl[:5], fl[5:]]
    for idx in range(len(result_rows)):
        dl_tab = result_rows[idx].getdatalink().to_table()
        res_row = dl_tab[dl_tab["semantics"] == "#this"]
        out_tag = reg.match(fl[0][1]).group(1)
        assert fl_split[idx][0][0] == res_row["access_url"]
        assert fl_split[idx][0][2] == res_row["ID"]
        assert out_tag == res_row["content_qualifier"]
        res_rows = dl_tab[dl_tab["semantics"] == "#calibration"]
        irf_files = fl_split[idx][1:]
        for row, (url, filn, obsid) in zip(res_rows, irf_files):
            out_tag = reg.match(filn).group(1)
            assert url == row["access_url"]
            assert obsid == row["ID"]
            assert out_tag == out_tag_dict[row["content_qualifier"]]


@requires_data()
def test_fetch_files(monkeypatch, fetch_list):
    save_dir = "tmp_test_dir"
    monkeypatch.setattr("gammapy.data.ivoa.progress_download", download_mock)
    fetched = fetch_files(fetch_list, make_path(save_dir))
    assert len(fetch_list) == len(fetched)
    out_path, file_name = fetched[0][1].parts
    assert save_dir == out_path
    assert file_name == fetch_list[0][1]


@requires_data()
def test_make_hdu_table(fetched_list):
    hdu_tab = make_hdu_table(fetched_list)

    if "aeff" in str(fetched_list[1][1]):
        assert len(hdu_tab) == (len(fetched_list) + 2)
    else:
        assert len(hdu_tab) == 6 * len(fetched_list)


@requires_data()
def test_make_obs_table():
    root_path = make_path("$GAMMAPY_DATA/")
    tap_res = TAPResults(
        from_tablevo(Table.read(root_path / "tests" / "ivoa" / "obscore_table.fits.gz"))
    )
    obs_table = make_obs_table(tap_res)

    assert len(obs_table["OBS_MODE"]) == 4
    assert obs_table["OBS_ID"][0] == 23523
    assert obs_table["DATE-OBS"][3] == "2004-12-08"

    ivoa_tab = from_tablevo(
        Table.read(root_path / "tests" / "minimal_datastore" / "obs-index.fits.gz")
    )
    with pytest.raises(KeyError):
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
    obscore_tab = Table.read(root_path / "tests" / "ivoa" / "obscore_table.fits.gz")

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

    with pytest.raises(KeyError):
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
