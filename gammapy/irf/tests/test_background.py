# Licensed under a 3-clause BSD style license - see LICENSE.rst
from __future__ import absolute_import, division, print_function, unicode_literals
import pytest
import astropy.units as u
from numpy.testing import assert_allclose, assert_equal
import numpy as np
from astropy.table import Table
from astropy.tests.helper import assert_quantity_allclose
from astropy.coordinates import Angle
from ...utils.testing import requires_dependency, requires_data
from ..background import Background3D, Background2D
from ...utils.fits import table_to_fits_table
from ...data import EventList
from ...utils.energy import EnergyBounds, Energy
from ...background import EnergyOffsetArray



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
    ebounds = EnergyBounds.equal_log_spacing(0.1, 100, 100, 'TeV')
    offset = Angle(np.linspace(0, 2.5, 100), "deg")
    array = EnergyOffsetArray(ebounds, offset)

    # Define an EventList with three events
    table = Table()
    table['RA'] = [0.6, 0, 2] * u.deg
    table['DEC'] = [0, 1.5, 0] * u.deg
    table['ENERGY'] = [0.12, 22, 55] * u.TeV
    table.meta['RA_PNT'] = 0
    table.meta['DEC_PNT'] = 0
    events = EventList(table)
    array.fill_events([events])
    data_unit = u.Unit('s-1 MeV-1 sr-1')
    bkg_2d = Background2D(energy_lo=ebounds[0:-1], energy_hi=ebounds[1:],
                 offset_lo=offset[0:-1], offset_hi=offset[1:],
                          data=array.data.value*data_unit)
    return bkg_2d, events.offset, events.energy



def test_background2D_read_write(tmpdir):
    bkg_2d_1, events_offset, events_energy=make_test_array()
    #hdu_list=bkg_2d_1.to_hdulist(name='BACKGROUND')
    #bkg_2d_2=Background2D.from_hdulist(hdulist=hdu_list, hdu='BACKGROUND')
    bkg_2d_1.write(tmpdir+"/bkg2d.fits")
    bkg_2d_2=Background2D.read(tmpdir+"/bkg2d.fits")

    axis = bkg_2d_2.data.axis('energy')
    assert axis.nbins == 100
    assert axis.unit == 'TeV'

    axis = bkg_2d_2.data.axis('offset')
    assert axis.nbins == 99
    assert axis.unit == 'deg'


    data = bkg_2d_2.data.data
    assert data.shape == (100, 99)
    assert data.unit == u.Unit('s-1 MeV-1 sr-1')



def test_background2D_evaluate():
    bkg_2d, events_offset, events_energy  = make_test_array()
    method='nearest'
    data_unit = u.Unit('s-1 MeV-1 sr-1')
    #With some values
    for off, e_reco in zip(events_offset, events_energy):
        res = bkg_2d.evaluate(fov_offset= off, energy_reco=e_reco, method=method)
        assert_quantity_allclose(res, 1*data_unit)
    #at the default one
    res = bkg_2d.evaluate(method=method)
    assert_quantity_allclose(res, bkg_2d.data.data)
    #at 2D in spatial: La premiere colone du tableau contient le bon offset
    offset_2d=np.meshgrid(events_offset,events_offset)[0]
    res = bkg_2d.evaluate(fov_offset= offset_2d, energy_reco= events_energy[0], method=method)
    assert_quantity_allclose(res[:,0], 1*data_unit)
    assert_quantity_allclose(res[:,1], 0*data_unit)
    assert_quantity_allclose(res[:,2], 0*data_unit)


def test_background2D_integrate_on_energy_band():
    """
    I define an energy range on witch I want to compute the acceptance curve that has the same boundaries as the
    energyoffsetarray.energy one and I take a Nbin for this range equal to the number of bin of the
    energyoffsetarray.energy one. This way, the interpolator will evaluate at energies that are the same as the one
    that define the RegularGridInterpolator. With the method="nearest" you are sure to get 1 for the energybin where
    are located the three events that define the energyoffsetarray. Since in this method we integrate over the energy
    and multiply by the solid angle, I check if for the offset of the three events (bin [23, 59, 79]), we get in the
    table["Acceptance"] what we expect by multiplying 1 by the solid angle and the energy bin width where is situated
    the event (bin [2, 78, 91]).
    """
    bkg_2d, events_offset, events_energy  = make_test_array()
    energy_range = Energy([0.1, 100], 'TeV')
    bins = 100
    method='nearest'
    bkg_integrate=bkg_2d.integrate_on_energy_range(tab_energy_band=energy_range, energy_bins=bins, method=method)
    data_unit = u.Unit('s-1 MeV-1 sr-1')
    axis_energy_band=(bkg_2d.data.axes[0].bins[1:]-bkg_2d.data.axes[0].bins[:-1]).to('MeV')
    assert_quantity_allclose(bkg_integrate[0,23],
                             1* data_unit * axis_energy_band[2])
    assert_quantity_allclose(bkg_integrate[0,59],
                             1 * data_unit * axis_energy_band[78])
    assert_quantity_allclose(bkg_integrate[0,79],
                             1 * data_unit * axis_energy_band[91])

    #at 2D in spatial
    offset_2d=np.meshgrid(events_offset,events_offset)[0]
    res = bkg_2d.integrate_on_energy_range(fov_offset= offset_2d, tab_energy_band=energy_range, energy_bins=bins, method=method)
    assert_quantity_allclose(res[0,:,0], 1* data_unit * axis_energy_band[2])
    assert_quantity_allclose(res[0,:,1], 1 * data_unit * axis_energy_band[78])
    assert_quantity_allclose(res[0,:,2], 1 * data_unit * axis_energy_band[91])


def make_test_cube():
    nE_bins=100
    ebounds = EnergyBounds.equal_log_spacing(0.1, 100, nE_bins, 'TeV')
    offset = Angle(np.linspace(0, 2.5, 100), "deg")

    bins = 2 * len(offset)
    coordx_edges = Angle(np.linspace(-offset.max().value, offset.max().value, bins+1), "deg")
    coordy_edges = Angle(np.linspace(-offset.max().value, offset.max().value, bins+1), "deg")
    data=np.zeros((nE_bins,bins,bins))
    #events at energy bin=[0.123,0.132] TeV, 0.12735031 TeV
    data[3,104,152]=1
    #events at energy bin=[0.398,0.426] TeV, 0.41209752 TeV
    data[20,108,172]=1
    data_unit = u.Unit('s-1 MeV-1 sr-1')
    bkg_3d=Background3D(energy_lo=ebounds[0:-1], energy_hi=ebounds[1:],detx_lo=coordx_edges[0:-1], detx_hi=coordx_edges[1:], dety_lo=coordy_edges[0:-1], dety_hi=coordy_edges[1:],data=data*data_unit)
    return bkg_3d

def test_background3D_read_write(tmpdir):
    bkg_3d_1=make_test_cube()
    bkg_3d_1.write(tmpdir+"/bkg3d.fits")
    bkg_3d_2=Background3D.read(tmpdir+"/bkg3d.fits")

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

def test_background3D_evaluate():
    bkg_3d = make_test_cube()
    method='nearest'
    data_unit = u.Unit('s-1 MeV-1 sr-1')

    det_x=Angle(np.array([bkg_3d.data.axis("detx").nodes[104].value,bkg_3d.data.axis("detx").nodes[108].value]),"deg")
    det_y=Angle(np.array([bkg_3d.data.axis("detx").nodes[152].value,bkg_3d.data.axis("dety").nodes[172].value]),"deg")
    energy_tab=[bkg_3d.data.axis("energy").nodes[3],bkg_3d.data.axis("energy").nodes[20]]
    offset_tab=np.sqrt(det_x**2+det_y**2)
    phi_tab=np.arctan(det_y/det_x)
    #at the default one
    res = bkg_3d.evaluate(method=method)
    assert_quantity_allclose(res, bkg_3d.data.data)
    #at 2D in spatial
    res = bkg_3d.evaluate(fov_offset= offset_tab,fov_phi= phi_tab, energy_reco= bkg_3d.data.axis("energy").nodes[3], method=method)
    assert_quantity_allclose(res[0,0], 1*data_unit)
    assert_quantity_allclose(res[0,1], 0*data_unit)
    assert_quantity_allclose(res[1,0], 0*data_unit)
    assert_quantity_allclose(res[1,1], 0*data_unit)
    res = bkg_3d.evaluate(fov_offset= offset_tab,fov_phi= phi_tab, energy_reco= bkg_3d.data.axis("energy").nodes[20], method=method)
    assert_quantity_allclose(res[0,0], 0*data_unit)
    assert_quantity_allclose(res[0,1], 0*data_unit)
    assert_quantity_allclose(res[1,0], 0*data_unit)
    assert_quantity_allclose(res[1,1], 1*data_unit)

def test_background3D_integrate_on_energy_band():
    """
    I define an energy range on witch I want to compute the acceptance curve that has the same boundaries as the
    energyoffsetarray.energy one and I take a Nbin for this range equal to the number of bin of the
    energyoffsetarray.energy one. This way, the interpolator will evaluate at energies that are the same as the one
    that define the RegularGridInterpolator. With the method="nearest" you are sure to get 1 for the energybin where
    are located the three events that define the energyoffsetarray. Since in this method we integrate over the energy
    and multiply by the solid angle, I check if for the offset of the three events (bin [23, 59, 79]), we get in the
    table["Acceptance"] what we expect by multiplying 1 by the solid angle and the energy bin width where is situated
    the event (bin [2, 78, 91]).
    """
    bkg_3d = make_test_cube()
    method='nearest'
    data_unit = u.Unit('s-1 MeV-1 sr-1')

    det_x=Angle(np.array([bkg_3d.data.axis("detx").nodes[104].value,bkg_3d.data.axis("detx").nodes[108].value]),"deg")
    det_y=Angle(np.array([bkg_3d.data.axis("detx").nodes[152].value,bkg_3d.data.axis("dety").nodes[172].value]),"deg")
    energy_tab=[bkg_3d.data.axis("energy").nodes[3],bkg_3d.data.axis("energy").nodes[20]]
    offset_tab=np.sqrt(det_x**2+det_y**2)
    phi_tab=np.arctan(det_y/det_x)

    energy_range = Energy([0.1, 100], 'TeV')
    bins = 100
    method='nearest'
    import IPython; IPython.embed()
    bkg_integrate=bkg_3d.integrate_on_energy_range(tab_energy_band=energy_range, energy_bins=bins, method=method)
    data_unit = u.Unit('s-1 MeV-1 sr-1')
    axis_energy_band=(bkg_2d.data.axes[0].bins[1:]-bkg_2d.data.axes[0].bins[:-1]).to('MeV')
    assert_quantity_allclose(bkg_integrate[0,104,152],
                             1* data_unit * axis_energy_band[2])
    assert_quantity_allclose(bkg_integrate[0,108,172],
                             1 * data_unit * axis_energy_band[78])
    bkg_integrate=bkg_3d.integrate_on_energy_range(fov_offset= offset_tab,fov_phi= phi_tab, tab_energy_band=energy_range, energy_bins=bins, method=method)

    """
    #at 2D in spatial
    offset_2d=np.meshgrid(events_offset,events_offset)[0]
    res = bkg_2d.integrate_on_energy_range(fov_offset= offset_2d, tab_energy_band=energy_range, energy_bins=bins, method=method)
    assert_quantity_allclose(res[0,:,0], 1* data_unit * axis_energy_band[2])
    assert_quantity_allclose(res[0,:,1], 1 * data_unit * axis_energy_band[78])
    assert_quantity_allclose(res[0,:,2], 1 * data_unit * axis_energy_band[91])
    """


