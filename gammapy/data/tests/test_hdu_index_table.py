# Licensed under a 3-clause BSD style license - see LICENSE.rst
from pathlib import Path
import pytest
from ...utils.scripts import make_path
from ...utils.testing import requires_data
from ..hdu_index_table import HDUIndexTable


@pytest.fixture(scope="session")
def hdu_index_table():
    table = HDUIndexTable(
        rows=[
            {
                "OBS_ID": 42,
                "HDU_TYPE": "events",
                "HDU_CLASS": "spam42",
                "FILE_DIR": "a",
                "FILE_NAME": "b",
                "HDU_NAME": "c",
            }
        ]
    )
    table.meta["BASE_DIR"] = "spam"
    return table


def test_hdu_index_table(hdu_index_table):
    """
    This test ensures that the HDUIndexTable works in a case-insensitive
    way concerning the values in the ``HDU_CLASS`` and ``HDU_TYPE`` columns.

    We ended up with this, because initially the HDU index table spec used
    lower-case, but then the ``HDUCLAS`` header keys to all the HDUs
    (e.g. EVENTS, IRFs, ...) were defined in upper-case.

    So for consistency we changed to all upper-case in the spec also for the HDU
    index table, just with a mention that values should be treated in a case-insensitive
    manner for backward compatibility with existing index tables.

    See https://github.com/open-gamma-ray-astro/gamma-astro-data-formats/issues/118
    """
    location = hdu_index_table.hdu_location(obs_id=42, hdu_type="events")
    assert location.path().as_posix() == "spam/a/b"

    assert hdu_index_table.summary().startswith("HDU index table")


@requires_data("gammapy-data")
def test_hdu_index_table_hd_hap():
    """Test HESS HAP-HD data access."""
    hdu_index = HDUIndexTable.read("$GAMMAPY_DATA/hess-dl3-dr1/hdu-index.fits.gz")

    assert "BASE_DIR" in hdu_index.meta
    assert hdu_index.base_dir == make_path("$GAMMAPY_DATA/hess-dl3-dr1")

    # A few valid queries

    location = hdu_index.hdu_location(obs_id=23523, hdu_type="events")
    hdu = location.get_hdu()
    assert hdu.name == "EVENTS"

    # The next line is to check if the HDU is still accessible
    # See https://github.com/gammapy/gammapy/issues/1775
    assert hdu.filebytes() == 224640

    assert location.path(abs_path=False) == Path(
        "data/hess_dl3_dr1_obs_id_023523.fits.gz"
    )
    path1 = str(location.path(abs_path=True))
    path2 = str(location.path(abs_path=False))
    assert path1.endswith(path2)

    location = hdu_index.hdu_location(obs_id=23523, hdu_class="psf_table")
    assert location.path(abs_path=False) == Path(
        "data/hess_dl3_dr1_obs_id_023523.fits.gz"
    )

    location = hdu_index.hdu_location(obs_id=23523, hdu_type="psf")
    assert location.path(abs_path=False) == Path(
        "data/hess_dl3_dr1_obs_id_023523.fits.gz"
    )

    # A few invalid queries

    with pytest.raises(IndexError) as exc:
        hdu_index.hdu_location(obs_id=42, hdu_class="psf_3gauss")
    msg = "No entry available with OBS_ID = 42"
    assert exc.value.args[0] == msg

    with pytest.raises(IndexError) as exc:
        hdu_index.hdu_location(obs_id=23523, hdu_type="bkg")
    msg = "No HDU found matching: OBS_ID = 23523, HDU_TYPE = bkg, HDU_CLASS = None"
    assert exc.value.args[0] == msg

    with pytest.raises(ValueError) as exc:
        hdu_index.hdu_location(obs_id=23523)
    msg = "You have to specify `hdu_type` or `hdu_class`."
    assert exc.value.args[0] == msg

    with pytest.raises(ValueError) as exc:
        hdu_index.hdu_location(obs_id=23523, hdu_type="invalid")
    msg = "Invalid hdu_type: invalid. Valid values are: ['events', 'gti', 'aeff', 'edisp', 'psf', 'bkg']"
    assert exc.value.args[0] == msg

    with pytest.raises(ValueError) as exc:
        hdu_index.hdu_location(obs_id=23523, hdu_class="invalid")
    msg = "Invalid hdu_class: invalid. Valid values are: ['events', 'gti', 'aeff_2d', 'edisp_2d', 'psf_table', 'psf_3gauss', 'psf_king', 'bkg_2d', 'bkg_3d']"
    assert exc.value.args[0] == msg
