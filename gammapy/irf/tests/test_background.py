# Licensed under a 3-clause BSD style license - see LICENSE.rst
from __future__ import absolute_import, division, print_function, unicode_literals
import pytest
import astropy.units as u
import numpy as np
from numpy.testing import assert_allclose, assert_equal
from astropy.tests.helper import assert_quantity_allclose
from astropy.coordinates import Angle
from astropy.io import fits
from ...utils.testing import requires_dependency, requires_data
from ..background import Background3D, Background2D
from ...utils.fits import table_to_fits_table


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
    energy = [1, 10, 100] * u.TeV
    offset = [0, 1, 2, 3] * u.deg
    data = np.zeros((len(energy) - 1, len(offset) - 1)) * u.Unit('s-1 MeV-1 sr-1')
    # Data contains 2 for the bin [0,1] degrees in offset and [10,100] TeV in energy
    data.value[1, 0] = 2
    # Data contains 4 for the bin [1,2] degrees in offset and [10,100] TeV in energy
    data.value[1, 1] = 4
    return Background2D(
        energy_lo=energy[:-1], energy_hi=energy[1:],
        offset_lo=offset[:-1], offset_hi=offset[1:],
        data=data,
    )


def test_background2d_read_write(tmpdir):
    bkg_2d_1 = make_test_array()
    filename = str(tmpdir / "bkg2d.fits")
    prim_hdu = fits.PrimaryHDU()
    hdu_bkg = bkg_2d_1.to_fits()
    hdulist = fits.HDUList([prim_hdu, hdu_bkg])
    hdulist.writeto(filename)
    bkg_2d_2 = Background2D.read(filename)

    axis = bkg_2d_2.data.axis('energy')
    assert axis.nbins == 2
    assert axis.unit == 'TeV'

    axis = bkg_2d_2.data.axis('offset')
    assert axis.nbins == 3
    assert axis.unit == 'deg'

    data = bkg_2d_2.data.data
    assert data.shape == (2, 3)
    assert data.unit == u.Unit('s-1 MeV-1 sr-1')


@requires_dependency('scipy')
def test_background2d_evaluate():
    bkg_2d = make_test_array()
    data_unit = u.Unit('s-1 MeV-1 sr-1')

    off_tab = Angle(np.array([1, 0.5]), "deg")
    # log_center value of the first energy bin used to define the Background2D data
    energy_center_bin_1 = u.Quantity([3.16227766], "TeV")
    res = bkg_2d.evaluate(fov_offset=off_tab, energy_reco=energy_center_bin_1)
    assert_quantity_allclose(res[0], 0 * data_unit)
    assert_quantity_allclose(res[1], 0 * data_unit)
    # log_center value of the second energy bin used to define the Background2D data
    energy_center_bin_2 = u.Quantity([31.6227766], "TeV")
    res = bkg_2d.evaluate(fov_offset=off_tab, energy_reco=energy_center_bin_2)
    assert_quantity_allclose(res[0], 3 * data_unit)
    assert_quantity_allclose(res[1], 2 * data_unit)
    assert res.shape == (2,)

    energy_tab = u.Quantity(np.array([energy_center_bin_1.value, energy_center_bin_2.value]), "TeV")
    offset_2d = np.meshgrid(off_tab, energy_tab)[0]
    res = bkg_2d.evaluate(fov_offset=offset_2d, energy_reco=energy_center_bin_2)
    assert_quantity_allclose(res[0, 0], 3 * data_unit)
    assert_quantity_allclose(res[1, 0], 3 * data_unit)
    assert_quantity_allclose(res[0, 1], 2 * data_unit)
    assert_quantity_allclose(res[1, 1], 2 * data_unit)
    assert res.shape == (2, 2)

    off = Angle(1, "deg")
    res = bkg_2d.evaluate(fov_offset=off, energy_reco=energy_tab)
    assert_quantity_allclose(res[0], 0 * data_unit)
    assert_quantity_allclose(res[1], 3 * data_unit)
    assert res.shape == (2,)
