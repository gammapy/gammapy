# Licensed under a 3-clause BSD style license - see LICENSE.rst
from __future__ import absolute_import, division, print_function, unicode_literals
from numpy.testing import assert_equal
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
        table['RA'] = [83, 85.2, 86]
        table['DEC'] = [21, 22.8, 22.5]
        table['ENERGY'] = [0.12, 22, 55]
        table.meta['RA_PNT'] = 85
        table.meta['DEC_PNT'] = 22
        table.meta['EUNIT'] = 'TeV'
        events = EventList(table)
        ev_list = [events]
        # Fill the array with these three events
        array.fill_events(ev_list)
        return array, events.offset, events.energy
    else:
        return array

def make_empty_Cube():
    array=make_test_array()
    offmax = array.offset.max() / 2.
    offmin = array.offset.min() / 2.
    Nbin = 2 * len(array.offset)
    coordx_edges = Angle(np.linspace(offmax.value, offmin.value, Nbin),"deg")
    coordy_edges = Angle(np.linspace(offmax.value, offmin.value, Nbin),"deg")
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
    return array
    # TODO: add some assert, e.g. counts in some bin with non-zero entries.


@requires_dependency('scipy')
def test_energy_offset_array_fill_evaluate():
    array, offset, energy = make_test_array(True)
    # Test if the array is filled correctly
    bin_E = np.array([2, 78, 91])
    bin_off = np.array([83, 32, 41])
    ind = np.where(array.data.value == 1)
    assert_equal(bin_E, ind[0])
    assert_equal(bin_off, ind[1])
    # Test the evaluate method
    interpol_param = dict(method='nearest', fill_value=None)
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

def test_curve():
    array=test_energy_offset_array_fill()
    energy= Energy(4, 'TeV')
    off=Angle(1.2, 'deg')
    table_energy=array.curve_at_energy(energy)
    table_band=array.curve_at_offset(off)
    Erange=Energy([1,10],'TeV')
    Nbin=10
    table_offset=array.acceptance_curve_in_energy_band(Erange, Nbin)

def test_to_cube():
    array=test_energy_offset_array_fill()
    Cube= make_empty_Cube()
    #energy bin differen from the enrgyoffsetarray just for the test
    E=EnergyBounds.equal_log_spacing(0.1, 100, 10, 'TeV')
    array.to_multi_Cube(Cube.coordx_edges, Cube.coordy_edges, E)
