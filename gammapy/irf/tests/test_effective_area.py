# Licensed under a 3-clause BSD style license - see LICENSE.rst
from __future__ import print_function, division
from numpy.testing import assert_allclose

from astropy.units import Quantity
from astropy.utils.data import get_pkg_data_filename

from ..effective_area import *


def test_abramowski_effective_area():
    energy = Quantity(100, 'GeV')
    area_ref = Quantity(1.65469579e+07, 'cm^2')

    area = abramowski_effective_area(energy, 'HESS')
    assert_allclose(area, area_ref)
    assert area.unit == area_ref.unit

    energy = Quantity([0.1, 2], 'TeV')
    area_ref = Quantity([1.65469579e+07, 1.46451957e+09], 'cm^2')

    area = abramowski_effective_area(energy, 'HESS')
    assert_allclose(area, area_ref)
    assert area.unit == area_ref.unit


def test_EnergyDependentTableARF():
    filename = get_pkg_data_filename('data/arf_info.txt')
    info_str = open(filename, 'r').read()
    filename = get_pkg_data_filename('data/arf.fits')
    arf = EnergyDependentTableARF.read(filename)
    assert arf.info() == info_str