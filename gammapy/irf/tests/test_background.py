# Licensed under a 3-clause BSD style license - see LICENSE.rst
from __future__ import absolute_import, division, print_function, unicode_literals
import pytest
import astropy.units as u
from numpy.testing import assert_allclose, assert_equal
import numpy as np
from astropy.tests.helper import assert_quantity_allclose
from astropy.coordinates import Angle
from ...utils.testing import requires_dependency, requires_data
from ..background import Background3D, Background2D
from ...utils.fits import table_to_fits_table
from ...utils.energy import EnergyBounds, Energy


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
    data[2, 23] = 1
    data[78, 59] = 1
    data_unit = u.Unit('s-1 MeV-1 sr-1')
    bkg_2d = Background2D(energy_lo=ebounds[0:-1], energy_hi=ebounds[1:],
                          offset_lo=offset[0:-1], offset_hi=offset[1:],
                          data=data * data_unit)
    return bkg_2d


def make_test_cube():
    # Create a dummy `Background2D`
    # Make Axes
    nE_bins = 100
    ebounds = EnergyBounds.equal_log_spacing(0.1, 100, nE_bins, 'TeV')
    offset = Angle(np.linspace(0, 2.5, 100), "deg")
    bins = 2 * len(offset)
    coordx_edges = Angle(np.linspace(-offset.max().value, offset.max().value, bins + 1), "deg")
    coordy_edges = Angle(np.linspace(-offset.max().value, offset.max().value, bins + 1), "deg")
    data = np.zeros((nE_bins, bins, bins))
    # Make dummy data
    data[3, 10, 152] = 1
    data[20, 108, 52] = 1
    data_unit = u.Unit('s-1 MeV-1 sr-1')
    bkg_3d = Background3D(energy_lo=ebounds[0:-1], energy_hi=ebounds[1:], detx_lo=coordx_edges[0:-1],
                          detx_hi=coordx_edges[1:], dety_lo=coordy_edges[0:-1], dety_hi=coordy_edges[1:],
                          data=data * data_unit)
    return bkg_3d


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


def test_background3d_read_write(tmpdir):
    bkg_3d_1 = make_test_cube()
    filename = str(tmpdir / "bkg3d.fits")
    bkg_3d_1.write(filename)
    bkg_3d_2 = Background3D.read(filename)

    axis = bkg_3d_2.data.axis('energy')
    assert axis.nbins == 100
    assert axis.unit == 'TeV'

    axis = bkg_3d_2.data.axis('detx')
    assert axis.nbins == 200
    assert axis.unit == 'deg'

    axis = bkg_3d_2.data.axis('dety')
    assert axis.nbins == 200
    assert axis.unit == 'deg'

    data = bkg_3d_2.data.data
    assert data.shape == (100, 200, 200)
    assert data.unit == u.Unit('s-1 MeV-1 sr-1')


def test_background2d_evaluate():
    bkg_2d = make_test_array()
    method = 'nearest'
    data_unit = u.Unit('s-1 MeV-1 sr-1')

    # Define the offset and energy for which bkg_2d contains 1. With the "nearest" method used for the interpolation
    # we check that it is giving 1.
    offset_tab = Angle([bkg_2d.data.axis("offset").nodes[23].value, bkg_2d.data.axis("offset").nodes[59].value], "deg")
    energy_tab = u.Quantity([bkg_2d.data.axis("energy").nodes.value[2], bkg_2d.data.axis("energy").nodes.value[78]],
                            bkg_2d.data.axis("energy").nodes.unit)
    # With some values
    for off, e_reco in zip(offset_tab, energy_tab):
        res = bkg_2d.evaluate(fov_offset=off, energy_reco=e_reco, method=method)
        assert_quantity_allclose(res, 1 * data_unit)
    # Interoplate at the energy and offset bin of the bgk_2d axes
    off = bkg_2d.data.axis('offset').nodes
    e_reco = bkg_2d.data.axis('energy').nodes
    res = bkg_2d.evaluate(fov_offset=off, energy_reco=e_reco, method=method)
    assert_quantity_allclose(res, bkg_2d.data.data)
    # Test that evaluate works for a 2D offset array. The bkg_2d data contains 1 for first column of the offset 2D array
    # and energy_tab[0].
    offset_2d = np.meshgrid(offset_tab, energy_tab)[0]
    res = bkg_2d.evaluate(fov_offset=offset_2d, energy_reco=energy_tab[0], method=method)
    assert_quantity_allclose(res[:, 0], 1 * data_unit)
    assert_quantity_allclose(res[:, 1], 0 * data_unit)


def test_background3d_evaluate():
    bkg_3d = make_test_cube()
    method = 'nearest'
    data_unit = u.Unit('s-1 MeV-1 sr-1')

    # at 2D in spatial
    # Define the detx,dety and energy for which bkg_2d contains 1. With the "nearest" method used for the interpolation
    # we check that it is giving 1.
    det_x = Angle(np.array([bkg_3d.data.axis("detx").nodes[10].value, bkg_3d.data.axis("detx").nodes[108].value]),
                  "deg")
    det_y = Angle(np.array([bkg_3d.data.axis("detx").nodes[152].value, bkg_3d.data.axis("dety").nodes[52].value]),
                  "deg")
    energy_tab = [bkg_3d.data.axis("energy").nodes[3], bkg_3d.data.axis("energy").nodes[20]]
    offset_tab = np.sqrt(det_x ** 2 + det_y ** 2)
    phi_tab = np.arctan(det_y / det_x)
    phi_tab.value[np.where(det_x.value < 0)] = phi_tab.value[np.where(det_x.value < 0)] + np.pi
    # For the energy.nodes[3], only the the coordinates (0,0) of the (off,phi) array contains 1
    res = bkg_3d.evaluate(fov_offset=offset_tab, fov_phi=phi_tab, energy_reco=energy_tab[0], method=method)
    assert_quantity_allclose(res[0, 0], 1 * data_unit)
    assert_quantity_allclose(res[0, 1], 0 * data_unit)
    assert_quantity_allclose(res[1, 0], 0 * data_unit)
    assert_quantity_allclose(res[1, 1], 0 * data_unit)
    # For the energy.nodes[20], only the the coordinates (0,0) of the (off,phi) array contains 1
    res = bkg_3d.evaluate(fov_offset=offset_tab, fov_phi=phi_tab, energy_reco=energy_tab[1], method=method)
    assert_quantity_allclose(res[0, 0], 0 * data_unit)
    assert_quantity_allclose(res[0, 1], 0 * data_unit)
    assert_quantity_allclose(res[1, 0], 0 * data_unit)
    assert_quantity_allclose(res[1, 1], 1 * data_unit)


def test_background2d_integrate_on_energy_band():
    """
    We define an energy range on witch we want to compute the integral background rate that has the same boundaries as the
    energy axis of the Background2D one. The number of bins used for the integration on this range is the same than the
    energy axis of the Background2D. This way, the interpolator will evaluate at energies that are the same as the one
    that define the RegularGridInterpolator. With the method="nearest",we are sure to get 1 for the energybin where
    are located the "1" value of the dummy Background2D data (see function make_test_cube() above). I check if for the
    (offset) of these "1" value (23) and (59), we get what we expect by multiplying 1 by the energy bin width
    (bin 2 for (offset)=(23) and bin 78 for (offset)=(59)).
    """
    bkg_2d = make_test_array()
    energy_range = Energy([0.1, 100], 'TeV')
    bins = 100
    method = 'nearest'
    off = bkg_2d.data.axis('offset').nodes
    bkg_integrate = bkg_2d.integrate_on_energy_range(tab_energy_band=energy_range, energy_bins=bins, fov_offset=off,
                                                     method=method)
    data_unit = u.Unit('s-1 MeV-1 sr-1')
    axis_energy_band = (bkg_2d.data.axes[0].bins[1:] - bkg_2d.data.axes[0].bins[:-1]).to('MeV')
    assert_quantity_allclose(bkg_integrate[23],
                             1 * data_unit * axis_energy_band[2])
    assert_quantity_allclose(bkg_integrate[59],
                             1 * data_unit * axis_energy_band[78])
    # at 2D in spatial
    offset_tab = Angle([bkg_2d.data.axis("offset").nodes[23].value, bkg_2d.data.axis("offset").nodes[59].value], "deg")
    energy_tab = u.Quantity([bkg_2d.data.axis("energy").nodes.value[2], bkg_2d.data.axis("energy").nodes.value[78]],
                            bkg_2d.data.axis("energy").nodes.unit)
    offset_2d = np.meshgrid(offset_tab, energy_tab)[0]
    res = bkg_2d.integrate_on_energy_range(fov_offset=offset_2d, tab_energy_band=energy_range, energy_bins=bins,
                                           method=method)
    assert_quantity_allclose(res[:, 0], 1 * data_unit * axis_energy_band[2])
    assert_quantity_allclose(res[:, 1], 1 * data_unit * axis_energy_band[78])
