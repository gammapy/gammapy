# Licensed under a 3-clause BSD style license - see LICENSE.rst
from __future__ import absolute_import, division, print_function, unicode_literals
from astropy.tests.helper import pytest
from ...background import Maps, ReflectedRegionMaker


@pytest.mark.xfail
def test_TestReflectedBgMaker():
    runs = 'TODO'
    maps = Maps('maps.fits')
    reflected_bg_maker = ReflectedRegionMaker(runs, maps, psi=2, theta=0.1)
    total_maps = Maps('total_maps.fits')
    for run in runs:
        run_map = total_maps.cutout(run)
        reflected_bg_maker.make_n_reflected_map(run, run_map)
        total_maps.add(run_map)
    total_maps.save('n_reflected.fits')
