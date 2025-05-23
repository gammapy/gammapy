# Licensed under a 3-clause BSD style license - see LICENSE.rst
import pytest
import numpy as np
from numpy.testing import assert_allclose
from astropy.io import fits
from astropy.table import Column, Table
from gammapy.utils.fits import earth_location_from_dict, earth_location_to_dict
from gammapy.utils.scripts import make_path
from gammapy.utils.testing import requires_data


@pytest.fixture()
def header():
    filename = make_path(
        "$GAMMAPY_DATA/hess-dl3-dr1/data/hess_dl3_dr1_obs_id_023523.fits.gz"
    )
    hdulist = fits.open(filename)
    return hdulist["EVENTS"].header


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


@requires_data()
def test_earth_location_from_dict(header):
    location = earth_location_from_dict(header)

    assert_allclose(location.lon.value, 16.50022, rtol=1e-4)
    assert_allclose(location.lat.value, -23.271777, rtol=1e-4)
    assert_allclose(location.height.value, 1834.999999, rtol=1e-4)


@requires_data()
def test_earth_location_to_dict(header):
    location = earth_location_from_dict(header)
    loc_dict = earth_location_to_dict(location)

    assert_allclose(loc_dict["GEOLON"], 16.50022, rtol=1e-4)
    assert_allclose(loc_dict["GEOLAT"], -23.271777, rtol=1e-4)
    assert_allclose(loc_dict["ALTITUDE"], 1834.999999, rtol=1e-4)
