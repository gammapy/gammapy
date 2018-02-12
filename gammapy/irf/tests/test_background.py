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
    for off, e_reco in zip(events_offset, events_energy):
        res = bkg_2d.evaluate(fov_offset= off, energy_reco=e_reco, method=method)
        data_unit = u.Unit('s-1 MeV-1 sr-1')
        assert_quantity_allclose(res, 1*data_unit)


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
    assert_quantity_allclose(bkg_integrate[23],
                             1* data_unit * bkg_2d.data.axes[0].bins[2].to('MeV'))
    assert_quantity_allclose(bkg_integrate[59],
                             1 * data_unit * bkg_2d.data.axes[0].bins[78].to('MeV'))
    assert_quantity_allclose(bkg_integrate[79],
                             1 * data_unit * bkg_2d.data.axes[0].bins[91].to('MeV'))


