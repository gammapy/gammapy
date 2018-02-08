# Licensed under a 3-clause BSD style license - see LICENSE.rst
from __future__ import absolute_import, division, print_function, unicode_literals
import numpy as np
from numpy.testing import assert_allclose
import astropy.units as u
from ..instrument_response import InstrumentResponse, Background3D


def test_reading_aeff():
    path = '$GAMMAPY_EXTRA/datasets/cta-1dc/caldb/data/cta//1dc/bcf/South_z20_50h/irf_file.fits'
    irf = InstrumentResponse.read(path, hdu='EFFECTIVE AREA')

    assert irf.names == ['energy', 'theta']

    point = {'theta': 3.5 * u.deg, 'energy': 1 * u.TeV}

    interpolated_result = irf.evaluate(point)
    assert not np.isnan(interpolated_result)

    assert interpolated_result.si.unit == u.m ** 2


def test_reading_edisp():
    path = '$GAMMAPY_EXTRA/datasets/cta-1dc/caldb/data/cta//1dc/bcf/South_z20_50h/irf_file.fits'
    irf = InstrumentResponse.read(path, hdu='ENERGY DISPERSION')

    assert 'ETRUE' in irf.names
    assert 'THETA' in irf.names
    assert 'MIGRA' in irf.names

    point = {'THETA': 3.5 * u.deg, 'ETRUE': 1 * u.TeV, 'MIGRA': 2}

    interpolated_result = irf.evaluate(point)
    assert not np.isnan(interpolated_result)


def test_reading_background():
    path = '$GAMMAPY_EXTRA/datasets/cta-1dc/caldb/data/cta//1dc/bcf/South_z20_50h/irf_file.fits'
    irf = InstrumentResponse.read(path, hdu='BACKGROUND')

    assert 'DETX' in irf.names
    assert 'DETY' in irf.names
    assert 'ENERG' in irf.names

    point = {'DETX': 3.5 * u.deg, 'DETY': 3.5 * u.deg, 'ENERG': 1 * u.TeV, }

    interpolated_result = irf.evaluate(point)
    assert not np.isnan(interpolated_result)
    assert interpolated_result.unit == u.Unit('1/(MeV * sr * s)')


def test_reading_background_with_subclass():
    path = '$GAMMAPY_EXTRA/datasets/cta-1dc/caldb/data/cta//1dc/bcf/South_z20_50h/irf_file.fits'
    irf = Background3D.read(path, hdu='BACKGROUND')
    assert irf.names == ['detx', 'dety', 'energy']

    point = {'detx': 3.5 * u.deg, 'dety': 3.5 * u.deg, 'energy': 1 * u.TeV, }

    interpolated_result = irf.evaluate(**point)
    assert not np.isnan(interpolated_result)
    assert interpolated_result.unit == u.Unit('1/(MeV * sr * s)')

    bkg_rate = irf.evaluate(energy='1 TeV', detx='0.2 deg', dety='0.5 deg')
    assert_allclose(bkg_rate.value, 0.00013352689711418575)
    assert bkg_rate.unit == u.Unit('s-1 MeV-1 sr-1')
