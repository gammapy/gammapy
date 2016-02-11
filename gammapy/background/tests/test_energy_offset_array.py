# Licensed under a 3-clause BSD style license - see LICENSE.rst
from __future__ import absolute_import, division, print_function, unicode_literals
from numpy.testing import assert_equal
import numpy as np
from ...data import DataStore
from ...datasets import gammapy_extra
from ...utils.testing import requires_dependency, requires_data
from ...utils.energy import EnergyBounds
from ..energy_offset_array import EnergyOffsetArray


def make_test_array():
    ebounds = EnergyBounds.equal_log_spacing(0.1, 100, 20, 'TeV')
    offset = np.linspace(0, 2.5, 10)
    array = EnergyOffsetArray(ebounds, offset)
    # put some dummy example data
    # TODO: this could be made optional or split out into a
    # separate utility function later on when lookup / interpolation methods are added to the class.
    # array.data[5, 3] = 42
    return array


@requires_dependency('matplotlib')
@requires_data('gammapy-extra')
def test_energy_offset_array_fill():
    dir = str(gammapy_extra.dir) + '/datasets/hess-crab4'
    data_store = DataStore.from_dir(dir)
    ev_list = data_store.load_all('events')

    array = make_test_array()
    array.fill_events(ev_list)

    # TODO: add some assert, e.g. counts in some bin with non-zero entries.


def test_energy_offset_array_plot():
    array = make_test_array()
    array.plot_image()


def test_energy_offset_array_read_write(tmpdir):
    array = make_test_array()

    filename = str(tmpdir / 'data.fits')
    array.write(filename)
    array2 = array.read(filename)

    assert_equal(array.data, array2.data)
    assert_equal(array.energy, array2.energy)
    assert_equal(array.offset, array2.offset)
   