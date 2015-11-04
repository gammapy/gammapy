# Licensed under a 3-clause BSD style license - see LICENSE.rst
from __future__ import absolute_import, division, print_function, unicode_literals
from astropy.tests.helper import pytest
from astropy.io import fits
from astropy.coordinates import SkyCoord, Angle
from .. import ReflectedRegionMaker, find_reflected_regions_for_event_list
from ...utils.testing import requires_data
from ...datasets import gammapy_extra

@requires_data('gammapy-extra')
def test_ReflectedRegionMaker():
    exclfile = gammapy_extra.filename('test_datasets/spectrum/dummy_exclusion.fits')
    exclusion = fits.open(exclfile, hdu = 0)[0]
    fov = SkyCoord(82.87, 23.24, unit='deg')
    maker = ReflectedRegionMaker(exclusion, fov)
    target = SkyCoord(80.2, 23.5, unit='deg')
    r_on = Angle(0.3, 'deg')
    regions = maker.compute(target, r_on)
    assert regions.number_of_regions == 19

@requires_data('gammapy-extra')
def test_find_reflected_regions_for_event_list():
    from gammapy.data import EventList 

    exclfile = gammapy_extra.filename('test_datasets/spectrum/dummy_exclusion.fits')
    filename = gammapy_extra.filename('test_datasets/unbundled/hess/run_0023037_hard_eventlist.fits.gz')

    exclusion = fits.open(exclfile, hdu = 0)[0]
    event_list = EventList.read(filename, hdu='EVENTS') 
    on_rad = Angle(0.3, 'deg')

    regions = find_reflected_regions_for_event_list(event_list, exclusion, on_rad)
    assert regions.number_of_regions == 3
