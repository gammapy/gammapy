# Licensed under a 3-clause BSD style license - see LICENSE.rst
from __future__ import print_function, division
from numpy.testing import assert_allclose
from astropy.tests.helper import pytest
from ..inverse_compton import InverseComptonSpectrum

# TODO: implement
@pytest.mark.xfail
def test_InverseComptonSpectrum():
    spectrum = InverseComptonSpectrum()
