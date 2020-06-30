# Licensed under a 3-clause BSD style license - see LICENSE.rst
import pytest
import numpy as np
from astropy.io import fits
from astropy.table import Column, Table


# Need to move to conftest or can import?
@pytest.fixture()
def table():
    t = Table(meta={"version": 42})
    t["a"] = np.array([1, 2], dtype=np.int32)
    t["b"] = Column(np.array([1, 2], dtype=np.int64), unit="m", description="Velocity")
    t["b"].meta["ucd"] = "spam"
    t["c"] = Column(["x", "yy"], "c")
    return t


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
