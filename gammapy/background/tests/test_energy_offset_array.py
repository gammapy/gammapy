# Licensed under a 3-clause BSD style license - see LICENSE.rst
from __future__ import absolute_import, division, print_function, unicode_literals
from numpy.testing import assert_equal
import astropy.units as u
from astropy.tests.helper import assert_quantity_allclose
import numpy as np
from astropy.table import Table
from astropy.coordinates import Angle
from ...data import DataStore
from ...datasets import gammapy_extra
from ...utils.testing import requires_dependency, requires_data
from ...utils.energy import EnergyBounds, Energy
from ..energy_offset_array import EnergyOffsetArray
from ..cube import Cube
from ...data import EventList


def make_test_array(dummy_data=False):
    ebounds = EnergyBounds.equal_log_spacing(0.1, 100, 100, 'TeV')
    offset = Angle(np.linspace(0, 2.5, 100), "deg")
    array = EnergyOffsetArray(ebounds, offset)
    if dummy_data is True:
        # Define an EventList with three events
        table = Table()
        table['RA'] = [0.6, 0, 2]
        table['DEC'] = [0, 1.5, 0]
        table['ENERGY'] = [0.12, 22, 55]
        table.meta['RA_PNT'] = 0
        table.meta['DEC_PNT'] = 0
        table.meta['EUNIT'] = 'TeV'
        events = EventList(table)
        ev_list = [events]
        # Fill the array with these three events
        array.fill_events(ev_list)
        return array, events.offset, events.energy
    else:
        return array


def make_empty_Cube():
    array = make_test_array()
    offmax = array.offset.max() / 2.
    offmin = array.offset.min()
    Nbin = 2 * len(array.offset)
    coordx_edges = Angle(np.linspace(offmax.value, offmin.value, Nbin), "deg")
    coordy_edges = Angle(np.linspace(offmax.value, offmin.value, Nbin), "deg")
    energy_edges = array.energy
    empty_cube = Cube(coordx_edges, coordy_edges, energy_edges)
    return empty_cube


@requires_data('gammapy-extra')
def test_energy_offset_array_fill():
    dir = str(gammapy_extra.dir) + '/datasets/hess-crab4-hd-hap-prod2'
    data_store = DataStore.from_dir(dir)
    ev_list = data_store.load_all('events')

    array = make_test_array()
    array.fill_events(ev_list)

    # TODO: add some assert, e.g. counts in some bin with non-zero entries.


@requires_dependency('scipy')
def test_energy_offset_array_fill_evaluate():
    array, offset, energy = make_test_array(True)
    # Test if the array is filled correctly
    bin_E = np.array([2, 78, 91])
    bin_off = np.array([23, 59, 79])
    ind = np.where(array.data.value == 1)
    assert_equal(bin_E, ind[0])
    assert_equal(bin_off, ind[1])
    # Test the evaluate method
    interpol_param = dict(method='nearest', bounds_error=False)
    for off, E in zip(offset, energy):
        res = array.evaluate(E, off, interpol_param)
        res_GeV = array.evaluate(E.to('GeV'), off, interpol_param)
        assert_equal(res, 1)
        assert_equal(res_GeV, 1)


@requires_dependency('matplotlib')
def test_energy_offset_array_plot():
    array = make_test_array()
    array.plot_image()


def test_energy_offset_array_read_write(tmpdir):
    array = make_test_array()

    filename = str(tmpdir / 'data.fits')
    array.write(filename)
    array2 = EnergyOffsetArray.read(filename)

    assert_equal(array.data, array2.data)
    assert_equal(array.energy, array2.energy)
    assert_equal(array.offset, array2.offset)


def test_energy_offset_array_bin_volume(tmpdir):
    array = make_test_array()
    energy_bin = array.energy.bands
    offset_bin = np.pi * (array.offset[1:] ** 2 - array.offset[:-1] ** 2)
    expected_volume = energy_bin[3] * offset_bin[4].to('sr')
    bin_volume = array.bin_volume
    assert_quantity_allclose(expected_volume, bin_volume[3, 4])


def test_evaluate_at_energy():
    array, offset, energy = make_test_array(True)
    e_eval = energy[0]
    bin_E = np.array([2, 78, 91])
    bin_off = np.array([23, 59, 79])
    interpol_param = dict(method='nearest', bounds_error=False)
    table_energy = array.evaluate_at_energy(e_eval, interpol_param)
    assert_quantity_allclose(table_energy["offset"], array.offset_bin_center)
    assert_equal(table_energy["value"][23], 1)

def test_evaluate_at_offset():
    array, offset, energy = make_test_array(True)
    off_eval = offset[0]
    bin_E = np.array([2, 78, 91])
    bin_off = np.array([23, 59, 79])
    interpol_param = dict(method='nearest', bounds_error=False)
    table_offset = array.evaluate_at_offset(off_eval, interpol_param)
    assert_quantity_allclose(table_offset["energy"], array.energy.log_centers)
    assert_equal(table_offset["value"][2], 1)

def test_acceptance_curve_in_energy_band():

    array, offset, energy = make_test_array(True)
    bin_E = np.array([2, 78, 91])
    bin_off = np.array([23, 59, 79])
    #Test for a range in energy that is the same as array.energy and that has the same number of bin as the array.energy. I know where are my three events in offset and energy thus I know where I will get one for the interpolation with the method="nearest". then by mulyiplying by the solid_angle and the enrgy_bin width where are my three events I can use assert
    Erange = Energy([0.1, 100], 'TeV')
    Nbin = 100
    interpol_param = dict(method='nearest', bounds_error=False)
    table_energy = array.acceptance_curve_in_energy_band(Erange, Nbin, interpol_param)
    assert_quantity_allclose(table_energy["offset"], array.offset_bin_center)
    assert_quantity_allclose(table_energy["Acceptance"][23]*table_energy["Acceptance"].unit, 1*array.solid_angle[23].to('sr')*array.energy.bands[2].to('MeV'))
    assert_quantity_allclose(table_energy["Acceptance"][59]*table_energy["Acceptance"].unit, 1*array.solid_angle[59].to('sr')*array.energy.bands[78].to('MeV'))
    assert_quantity_allclose(table_energy["Acceptance"][79]*table_energy["Acceptance"].unit, 1*array.solid_angle[79].to('sr')*array.energy.bands[91].to('MeV'))

def test_to_cube():
    array, offset, energy = make_test_array(True)
    bin_E = np.array([2, 78, 91])
    bin_off = np.array([23, 59, 79])
    Cube = make_empty_Cube()
    # energy bin differen from the enrgyoffsetarray just for the test
    E = EnergyBounds.equal_log_spacing(0.1, 100, 10, 'TeV')
    array.to_multi_Cube(Cube.coordx_edges, Cube.coordy_edges, E)

def test_to_cube():
    array, offset, energy = make_test_array(True)
    bin_E = np.array([2, 78, 91])
    bin_off = np.array([23, 59, 79])
    Cube = make_empty_Cube()
    interpol_param = dict(method='nearest', bounds_error=False)
    CubeModel=array.to_multi_Cube(Cube.coordx_edges, Cube.coordy_edges, Cube.energy_edges, interpol_param)
    i=np.where(CubeModel.data[2,:,:]==1)
    x=Cube.coordx_edges
    y=Cube.coordy_edges
    XX,YY=np.meshgrid(x,y)
    dist=np.sqrt(XX**2+YY**2)
    assert_quantity_allclose(dist[i],0.6*u.deg, atol=0.1*u.deg)



"""
def test_curve():
    array = test_energy_offset_array_fill()
    energy = Energy(4, 'TeV')
    off = Angle(1.2, 'deg')
    table_energy = evaluate.curve_at_energy(energy)
    table_band = evaluate.curve_at_offset(off)
    Erange = Energy([1, 10], 'TeV')
    Nbin = 10
    table_offset = array.acceptance_curve_in_energy_band(Erange, Nbin)


def test_to_cube():
    array = test_energy_offset_array_fill()
    Cube = make_empty_Cube()
    # energy bin differen from the enrgyoffsetarray just for the test
    E = EnergyBounds.equal_log_spacing(0.1, 100, 10, 'TeV')
    array.to_multi_Cube(Cube.coordx_edges, Cube.coordy_edges, E)
"""