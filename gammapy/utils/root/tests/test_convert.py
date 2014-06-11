# Licensed under a 3-clause BSD style license - see LICENSE.rst
from __future__ import print_function, division
import numpy as np
from numpy.testing import assert_allclose
from astropy.tests.helper import pytest

HAS_ROOT = False

try:
    import ROOT
    from ... import root
    HAS_ROOT = True
except:
    HAS_ROOT = False


@pytest.mark.skipif('not HAS_ROOT')
def test_graph1d_to_table():
    x = np.array([-0.22, 0.05, 0.25, 0.35])
    y = np.array([1, 2.9, 5.6, 7.4])
    ex = np.array([.05, .1, .07, .07])
    ey = np.array([.8, .7, .6, .5])
    n = len(x)

    graph = ROOT.TGraphErrors(n, x, y, ex, ey)
    table = root.graph1d_to_table(graph)
    assert_allclose(table['x'], x)

    graph = ROOT.TGraph(n, x, y)
    table = root.graph1d_to_table(graph)
    assert_allclose(table['x'], x)


@pytest.mark.skipif('not HAS_ROOT')
def test_hist1d_to_table():
    hist = ROOT.TH1F('name', 'title', 4, -10, 10)
    hist.Fill(3)

    table = root.hist1d_to_table(hist)
    assert_allclose(table['x'], [-7.5, -2.5,  2.5,  7.5])
    assert_allclose(table['y'], [0., 0., 1., 0.])


@pytest.mark.skipif('not HAS_ROOT')
def make_test_TH2():
    """Generate an example TH2 we use to test TH2_to_FITS(),
    corresponding approximately to the HESS survey region."""
    name, title = 'test_image', 'My Test Image'
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


@pytest.mark.xfail
@pytest.mark.skipif('not HAS_ROOT')
def test_TH2_to_FITS():
    h = make_test_TH2()
    h.Print('base')
    f = root.TH2_to_FITS(h)
    from pprint import pprint
    pprint(f.header2classic())
    filename = 'TH2_to_FITS.fits'
    print('Writing {0}'.format(filename))
    f.writetofits(filename, clobber=True)
