from ...irf.instrument_response import InstrumentResponse
from astropy.utils.data import get_pkg_data_filename
import astropy.units as u
import numpy as np
import pytest


@pytest.mark.parametrize('path', ['data/1dc_irf.fits', 'data/prod3b_irf.fits'])
def test_reading_fits(path):
    path = get_pkg_data_filename(path)
    irf = InstrumentResponse.from_fits(path, extension='EFFECTIVE AREA')
    assert len(irf.names) > 0

@pytest.mark.parametrize('path', ['data/1dc_irf.fits', 'data/prod3b_irf.fits'])
def test_reading_aeff(path):
    path = get_pkg_data_filename(path)
    irf = InstrumentResponse.from_fits(path, extension='EFFECTIVE AREA')

    assert 'ENERG' in irf.names
    assert 'THETA' in irf.names

    point = {'THETA': 3.5*u.deg, 'ENERG': 1*u.TeV}

    interpolated_result = irf.evaluate(point)
    assert not np.isnan(interpolated_result)

    assert interpolated_result.si.unit == u.m**2


@pytest.mark.parametrize('path', ['data/1dc_irf.fits', 'data/prod3b_irf.fits'])
def test_reading_edisp(path):
    path = get_pkg_data_filename(path)
    irf = InstrumentResponse.from_fits(path, extension='ENERGY DISPERSION')

    assert 'ETRUE' in irf.names
    assert 'THETA' in irf.names
    assert 'MIGRA' in irf.names

    point = {'THETA': 3.5*u.deg, 'ETRUE': 1*u.TeV, 'MIGRA': 2}

    interpolated_result = irf.evaluate(point)
    assert not np.isnan(interpolated_result)


@pytest.mark.parametrize('path', ['data/1dc_irf.fits', 'data/prod3b_irf.fits'])
def test_reading_background(path):
    path = get_pkg_data_filename(path)
    irf = InstrumentResponse.from_fits(path, extension='BACKGROUND')

    assert 'DETX' in irf.names
    assert 'DETY' in irf.names
    assert 'ENERG' in irf.names

    point = {'DETX': 3.5*u.deg, 'DETY': 3.5*u.deg, 'ENERG': 1*u.TeV,}

    interpolated_result = irf.evaluate(point)
    assert not np.isnan(interpolated_result)
    assert interpolated_result.unit == u.Unit('1/(MeV * sr * s)')
