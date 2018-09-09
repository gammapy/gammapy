# Licensed under a 3-clause BSD style license - see LICENSE.rst
from __future__ import absolute_import, division, print_function, unicode_literals
import numpy as np
import pytest
from astropy.io import fits
from astropy.table import Table, Column
from ..fits import SmartHDUList


def make_test_hdu_list():
    return fits.HDUList(
        [
            fits.PrimaryHDU(),
            fits.BinTableHDU(name="TABLE1"),
            fits.ImageHDU(name="IMAGE1", data=np.zeros(shape=(1, 2, 3))),
            fits.BinTableHDU(name="TABLE2"),
            fits.ImageHDU(name="IMAGE2", data=np.zeros(shape=(4, 5))),
        ]
    )


# TODO: merge this fixture with the one in `test_table.py`.
# Need to move to conftest or can import?
@pytest.fixture()
def table():
    t = Table(meta={"version": 42})
    t["a"] = np.array([1, 2], dtype=np.int32)
    t["b"] = Column(np.array([1, 2], dtype=np.int64), unit="m", description="Velocity")
    t["b"].meta["ucd"] = "spam"
    t["c"] = Column(["x", "yy"], "c")
    return t


class TestSmartHDUList:
    def setup(self):
        self.hdus = SmartHDUList(hdu_list=make_test_hdu_list())

        self.names = ["PRIMARY", "TABLE1", "IMAGE1", "TABLE2", "IMAGE2"]
        self.numbers = list(range(5))

    def test_names(self):
        assert self.hdus.names == self.names

    def test_fits_get_hdu(self):
        def g(hdu=None, hdu_type=None):
            """Short helper function, to save some typing."""
            return self.hdus.get_hdu(hdu, hdu_type).name

        # Make a few valid queries, and assert that the right result comes back

        for number, name in zip(self.numbers, self.names):
            assert g(hdu=name) == name
            assert g(hdu=name.lower()) == name
            assert g(hdu=number) == name

        assert g(hdu_type="image") == "IMAGE1"
        assert g(hdu_type="table") == "TABLE1"

        # Call the method incorrectly, and assert that ValueError is raised:

        with pytest.raises(ValueError) as exc:
            g()
        assert (
            str(exc.value)
            == "Must give either `hdu` or `hdu_type`. Got `None` for both."
        )

        with pytest.raises(ValueError) as exc:
            g(hdu_type="bad value")
        assert str(exc.value) == "Invalid hdu_type=bad value"

        # Query for non-existent HDUs, and assert that KeyError is raised:

        with pytest.raises(KeyError):
            g(hdu=["bad", "type"])

        with pytest.raises(KeyError):
            g(hdu="kronka lonka")

        with pytest.raises(KeyError):
            g(hdu=42)

    def test_fits_get_hdu_index(self):
        # We test almost everything above via `test_fits_get_hdu`
        # Here we just add a single test for `get_hdu_index` to
        # make sure it returns an int index all right.
        assert self.hdus.get_hdu_index(hdu="TABLE2") == 3

    def test_read_write(self, tmpdir):
        filename = str(tmpdir / "data.fits")
        self.hdus.write(filename)
        hdus2 = SmartHDUList.open(filename)
        assert self.hdus.names == hdus2.names


def test_table_fits_io_astropy(table):
    """Test `astropy.table.Table` FITS I/O in Astropy.

    Having these tests in Gammapy is to check / ensure that the features
    we rely on work properly for all Astropy versions we support in CI
    (currently Astropy 1.3 and up)

    This is useful, because Table FITS I/O was pretty shaky for a while
    and incrementally improved over time.

    These are the same examples that we have in the docstring
    at the top of `gammapy/utils/fits.py`.
    """
    # Check Table -> BinTableHDU
    hdu = fits.BinTableHDU(table)
    assert hdu.header["TTYPE2"] == "b"
    assert hdu.header["TFORM2"] == "K"
    assert hdu.header["TUNIT2"] == "m"

    # Check BinTableHDU -> Table
    table2 = Table.read(hdu)
    assert isinstance(table2.meta, dict)
    assert table2.meta == {"VERSION": 42}
    assert table2["b"].unit == "m"
    # Note: description doesn't come back in older versions of Astropy
    # that we still support, so we're not asserting on that here for now.
