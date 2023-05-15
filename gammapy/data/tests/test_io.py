# Licensed under a 3-clause BSD style license - see LICENSE.rst
import pytest
from numpy.testing import assert_allclose
from astropy.io import fits
from pydantic import ValidationError
from gammapy.data.io import GADFEvents, GADFEventsHeader
from gammapy.utils.scripts import make_path
from gammapy.utils.testing import requires_data


@requires_data()
def test_gadf_header_read():
    filename = make_path(
        "$GAMMAPY_DATA/hess-dl3-dr1/data/hess_dl3_dr1_obs_id_020136.fits.gz"
    )
    hdul = fits.open(filename)
    header = hdul[1].header
    gadf_hdr = GADFEventsHeader.from_header(header)

    assert gadf_hdr.OBS_ID == 20136
    assert gadf_hdr["TELESCOP"] == "HESS"
    assert gadf_hdr["RELHUM"] is None

    header_bad = header.copy()
    header_bad["HDUCLAS1"] = "EVENT"
    with pytest.raises(ValidationError):
        GADFEventsHeader.from_header(header_bad)

    header["OBS_ID"] = "test"
    with pytest.raises(ValidationError):
        GADFEventsHeader.from_header(header)


@requires_data()
def test_gadf_event_reader():
    filename = make_path(
        "$GAMMAPY_DATA/hess-dl3-dr1/data/hess_dl3_dr1_obs_id_020136.fits.gz"
    )
    events = GADFEvents.read(filename)

    assert len(events.table) == 11243
    assert_allclose(events.table["ENERGY"][0], 0.55890286)


@requires_data()
def test_gadf_event_writer():
    return False
