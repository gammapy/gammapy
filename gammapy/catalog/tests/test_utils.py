# Licensed under a 3-clause BSD style license - see LICENSE.rst
from __future__ import print_function, division
from astropy.coordinates import ICRS
from .. import make_source_designation


def test_make_source_designation():
    coordinate = ICRS('05h34m31.93830s +22d00m52.1758s')
    designation = make_source_designation(coordinate, ra_digits=4, acronym='HESS')
    assert designation == 'HESS J0534+220'
        
    coordinate = ICRS('21h58m52.06511s -30d13m32.1182s')
    designation = make_source_designation(coordinate, ra_digits=5, acronym='')
    assert designation == 'J21588.7-3013'
