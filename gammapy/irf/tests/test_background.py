# Licensed under a 3-clause BSD style license - see LICENSE.rst
from __future__ import absolute_import, division, print_function, unicode_literals
import pytest
import astropy.units as u
import numpy as np
from numpy.testing import assert_allclose, assert_equal
from astropy.tests.helper import assert_quantity_allclose
from astropy.coordinates import Angle
from ...utils.testing import requires_dependency, requires_data
from ..background import Background3D, Background2D
from ...utils.fits import table_to_fits_table
from ...utils.energy import EnergyBounds


@pytest.fixture(scope='session')
def bkg_3d():
    filename = '$GAMMAPY_EXTRA/datasets/cta-1dc/caldb/data/cta//1dc/bcf/South_z20_50h/irf_file.fits'
    return Background3D.read(filename, hdu='BACKGROUND')


@requires_dependency('scipy')
@requires_data('gammapy-extra')
def test_background_3d_basics(bkg_3d):
    assert 'NDDataArray summary info' in str(bkg_3d.data)

    axis = bkg_3d.data.axis('energy')
    assert axis.nbins == 21
    assert axis.unit == 'TeV'

    axis = bkg_3d.data.axis('detx')
    assert axis.nbins == 36
    assert axis.unit == 'deg'

    axis = bkg_3d.data.axis('dety')
    assert axis.nbins == 36
    assert axis.unit == 'deg'

    data = bkg_3d.data.data
    assert data.shape == (21, 36, 36)
    assert data.unit == u.Unit('s-1 MeV-1 sr-1')


@requires_dependency('scipy')
@requires_data('gammapy-extra')
def test_background_3d_evaluate(bkg_3d):
    bkg_rate = bkg_3d.data.evaluate(energy='1 TeV', detx='0.2 deg', dety='0.5 deg')
    assert_allclose(bkg_rate.value, 0.00013352689711418575)
    assert bkg_rate.unit == u.Unit('s-1 MeV-1 sr-1')


@requires_data('gammapy-extra')
def test_background_3d_write(bkg_3d):
    hdu = table_to_fits_table(bkg_3d.to_table())
    assert_equal(hdu.data['DETX_LO'][0], bkg_3d.data.axis('detx').lo.value)
    assert hdu.header['TUNIT1'] == bkg_3d.data.axis('detx').lo.unit


def make_test_array():
    # Create a dummy `Background2D`
    # Make Axes
    nE_bins = 100
    ebounds = EnergyBounds.equal_log_spacing(0.1, 100, 100, 'TeV')
    offset = Angle(np.linspace(0, 2.5, 100), "deg")
    data = np.zeros((nE_bins, len(offset) - 1))
    # Make dummy data
    # At offset=0.59343434 deg and energy=0.11885022 TeV
    data[2, 23] = 1
    # At offset=1.50252525 deg and energy= 22.64644308 TeV
    data[78, 59] = 1
    data_unit = u.Unit('s-1 MeV-1 sr-1')
    bkg_2d = Background2D(energy_lo=ebounds[:-1], energy_hi=ebounds[1:],
                          offset_lo=offset[:-1], offset_hi=offset[1:],
                          data=data * data_unit)
    return bkg_2d


def test_background2d_read_write(tmpdir):
    bkg_2d_1 = make_test_array()
    # hdu_list=bkg_2d_1.to_hdulist(name='BACKGROUND')
    # bkg_2d_2=Background2D.from_hdulist(hdulist=hdu_list, hdu='BACKGROUND')
    filename = str(tmpdir / "bkg2d.fits")
    bkg_2d_1.write(filename)
    bkg_2d_2 = Background2D.read(filename)

    axis = bkg_2d_2.data.axis('energy')
    assert axis.nbins == 100
    assert axis.unit == 'TeV'

    axis = bkg_2d_2.data.axis('offset')
    assert axis.nbins == 99
    assert axis.unit == 'deg'

    data = bkg_2d_2.data.data
    assert data.shape == (100, 99)
    assert data.unit == u.Unit('s-1 MeV-1 sr-1')


def test_background2d_evaluate():
    bkg_2d = make_test_array()
    data_unit = u.Unit('s-1 MeV-1 sr-1')
    # Check we get the bkg_2d.data.data when we interpolate at the energy
    # and offset bin of the bgk_2d axes
    off = bkg_2d.data.axis('offset').nodes
    e_reco = bkg_2d.data.axis('energy').nodes
    res = bkg_2d.evaluate(fov_offset=off, energy_reco=e_reco)
    assert_quantity_allclose(res, bkg_2d.data.data)
    # Test that evaluate works for a 2D offset array.
    off_tab = Angle(np.array([0.59343434, 1.50252525]), "deg")
    e_reco_tab = u.Quantity([0.11885022, 22.64644308], "TeV")
    res = bkg_2d.evaluate(fov_offset=off_tab[0], energy_reco=e_reco_tab[0])
    assert_quantity_allclose(res, 1 * data_unit, rtol=1e-06)
    offset_2d = np.meshgrid(off_tab, e_reco_tab)[0]
    res = bkg_2d.evaluate(fov_offset=offset_2d, energy_reco=e_reco_tab[0])
    assert_quantity_allclose(res[:, 0], 1 * data_unit, rtol=1e-06)
    assert_quantity_allclose(res[:, 1], 0 * data_unit, rtol=1e-06)
    assert res.shape == offset_2d.shape
