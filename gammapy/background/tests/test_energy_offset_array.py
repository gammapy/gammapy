# Licensed under a 3-clause BSD style license - see LICENSE.rst
from __future__ import absolute_import, division, print_function, unicode_literals
import numpy as np
from numpy.testing import assert_equal
import astropy.units as u
from astropy.tests.helper import assert_quantity_allclose
from astropy.table import Table
from astropy.coordinates import Angle
from ...data import DataStore, EventList
from ...utils.testing import requires_dependency, requires_data
from ...utils.energy import EnergyBounds, Energy
from ..energy_offset_array import EnergyOffsetArray
from ..fov_cube import FOVCube


def make_test_array(dummy_data=False):
    ebounds = EnergyBounds.equal_log_spacing(0.1, 100, 100, 'TeV')
    offset = Angle(np.linspace(0, 2.5, 100), "deg")
    array = EnergyOffsetArray(ebounds, offset)
    if dummy_data is True:
        # Define an EventList with three events
        table = Table()
        table['RA'] = [0.6, 0, 2]
        table['RA'].unit = 'deg'
        table['DEC'] = [0, 1.5, 0] * u.deg
        table['ENERGY'] = [0.12, 22, 55] * u.TeV
        table.meta['RA_PNT'] = 0
        table.meta['DEC_PNT'] = 0
        events = EventList(table)
        array.fill_events([events])
        return array, events.offset, events.energy
    else:
        return array


def make_empty_cube():
    array = make_test_array()
    offmax = array.offset.max() / 2.
    offmin = array.offset.min()
    bins = 2 * len(array.offset)
    coordx_edges = Angle(np.linspace(offmax.value, offmin.value, bins), "deg")
    coordy_edges = Angle(np.linspace(offmax.value, offmin.value, bins), "deg")
    energy_edges = array.energy
    empty_cube = FOVCube(coordx_edges, coordy_edges, energy_edges)
    return empty_cube


@requires_data('gammapy-extra')
def test_energy_offset_array_fill():
    dir = '$GAMMAPY_EXTRA/datasets/hess-crab4-hd-hap-prod2'
    data_store = DataStore.from_dir(dir)
    ev_list = data_store.load_all('events')
    array = make_test_array()
    array.fill_events(ev_list)
    # TODO: add some assert, e.g. counts in some bin with non-zero entries.


@requires_dependency('scipy')
def test_energy_offset_array_fill_evaluate():
    array, offset, energy = make_test_array(True)
    # Test if the array is filled correctly
    bin_e = np.array([2, 78, 91])
    bin_off = np.array([23, 59, 79])
    ind = np.where(array.data.value == 1)
    assert_equal(bin_e, ind[0])
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
    array.plot()


def test_energy_offset_array_read_write(tmpdir):
    array = make_test_array()

    filename = str(tmpdir / 'data.fits')
    array.write(filename)
    array2 = EnergyOffsetArray.read(filename)

    assert_equal(array.data, array2.data)
    assert_equal(array.energy, array2.energy)
    assert_equal(array.offset, array2.offset)

    # Test if the data in the EnergyOffsetArray have an associated error
    array, event_off, event_energy = make_test_array(True)
    filename = str(tmpdir / 'data2.fits')
    array.write(filename)
    array2 = EnergyOffsetArray.read(filename)
    assert_equal(array.data_err, array2.data_err)
    assert_equal(array.data, array2.data)
    assert_equal(array.energy, array2.energy)
    assert_equal(array.offset, array2.offset)


def test_energy_offset_array_bin_volume():
    array = make_test_array()
    energy_bin = array.energy.bands
    offset_bin = np.pi * (array.offset[1:] ** 2 - array.offset[:-1] ** 2)
    expected_volume = energy_bin[3] * offset_bin[4].to('sr')
    bin_volume = array.bin_volume
    assert_quantity_allclose(expected_volume, bin_volume[3, 4])


@requires_dependency('scipy')
def test_evaluate_at_energy():
    array, offset, energy = make_test_array(True)
    e_eval = energy[0]
    interpol_param = dict(method='nearest', bounds_error=False)
    table_energy = array.evaluate_at_energy(e_eval, interpol_param)
    assert_quantity_allclose(table_energy["offset"], array.offset_bin_center)
    # Offset bin for the first event is 23
    assert_equal(table_energy["value"][23], 1)


@requires_dependency('scipy')
def test_evaluate_at_offset():
    array, offset, energy = make_test_array(True)
    off_eval = offset[0]
    interpol_param = dict(method='nearest', bounds_error=False)
    table_offset = array.evaluate_at_offset(off_eval, interpol_param)
    assert_quantity_allclose(table_offset["energy"], array.energy.log_centers)
    # Energy bin for the first event is 2
    assert_equal(table_offset["value"][2], 1)


@requires_dependency('scipy')
def test_acceptance_curve_in_energy_band():
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
    array, offset, energy = make_test_array(True)
    energ_range = Energy([0.1, 100], 'TeV')
    bins = 100
    interpol_param = dict(method='nearest', bounds_error=False)
    table_energy = array.acceptance_curve_in_energy_band(energ_range, bins, interpol_param)
    assert_quantity_allclose(table_energy["offset"], array.offset_bin_center)
    assert_quantity_allclose(table_energy["Acceptance"][23] * table_energy["Acceptance"].unit,
                             1 * array.energy.bands[2].to('MeV'))
    assert_quantity_allclose(table_energy["Acceptance"][59] * table_energy["Acceptance"].unit,
                             1 * array.energy.bands[78].to('MeV'))
    assert_quantity_allclose(table_energy["Acceptance"][79] * table_energy["Acceptance"].unit,
                             1 * array.energy.bands[91].to('MeV'))


@requires_dependency('scipy')
def test_to_cube():
    """
    There are three events in the energyoffsetarray at three offset and energies. I define a FOVCube with the same energy
    bin as the energyoffsetarray.energy. For the event in the offset bin 23 (=0.6 degre) and in the energy bin 2
    (0.12 Tev), I check if after calling the to_cube() method, all the x and y of the new FOVCube matching with an offset
    equal to 0.6+/-0.1 are filled with 1.
    """
    array, offset, energy = make_test_array(True)
    cube = make_empty_cube()
    interpol_param = dict(method='nearest', bounds_error=False)
    cube_model = array.to_cube(cube.coordx_edges, cube.coordy_edges, cube.energy_edges, interpol_param)
    i = np.where(cube_model.data[2, :, :] == 1)
    x = cube_model.coordx_edges
    y = cube_model.coordy_edges
    xx, yy = np.meshgrid(x, y)
    dist = np.sqrt(xx ** 2 + yy ** 2)
    assert_quantity_allclose(dist[i], 0.6 * u.deg, atol=0.1 * u.deg)
