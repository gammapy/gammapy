# Licensed under a 3-clause BSD style license - see LICENSE.rst
from __future__ import print_function, division
import numpy as np
import pytest

try:
    import ROOT
    from ... import root
    HAS_ROOT = True
except ImportError:
    HAS_ROOT = False


def make_test_TH2():
    """Generate an example TH2 we use to test TH2_to_FITS(),
    corresponding approximately to the HESS survey region."""
    name, title = 'test_image', 'My Test Image'
    nbinsx, xlow, xup = 1400, -80, 60
    nbinsx, xlow, xup = 1400, 60, -80
    nbinsy, ylow, yup = 100, -5, 5
    h = ROOT.TH2F(name, title, nbinsx, xlow, xup, nbinsy, ylow, yup)
    # Just for fun:
    # Fill with distance to Galactic center, to have something to look at
    for ix in range(nbinsx):
        for iy in range(nbinsy):
            x = h.GetXaxis().GetBinCenter(ix)
            y = h.GetYaxis().GetBinCenter(iy)
            value = np.sqrt(x * x + y * y)
            h.SetBinContent(ix, iy, value)
    return h


@pytest.mark.skipif('not HAS_ROOT')
def _test_TH2_to_FITS():
    h = make_test_TH2()
    h.Print('base')
    f = root.TH2_to_FITS(h)
    from pprint import pprint
    pprint(f.header2classic())
    filename = 'TH2_to_FITS.fits'
    print('Writing %s' % filename)
    f.writetofits(filename, clobber=True)
